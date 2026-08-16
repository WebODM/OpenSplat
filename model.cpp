#include <filesystem>
#include <random>
#include <numeric>
#include <algorithm>
#include <load-spz.h>
#include "model.hpp"
#include "constants.hpp"
#include "splat-types.h"
#include "tile_bounds.hpp"
#include "project_gaussians.hpp"
#include "rasterize_gaussians.hpp"
#include "tensor_math.hpp"
#include "gsplat.hpp"
#include "utils.hpp"
#include "rad.hpp"

#ifdef USE_MPS
#include <torch/mps.h>
#endif

#ifdef USE_HIP
#include <c10/hip/HIPCachingAllocator.h>
#elif defined(USE_CUDA)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace fs = std::filesystem;

torch::Tensor randomQuatTensor(long long n){
    torch::Tensor u = torch::rand(n);
    torch::Tensor v = torch::rand(n);
    torch::Tensor w = torch::rand(n);
    return torch::stack({
        torch::sqrt(1 - u) * torch::sin(2 * PI * v),
        torch::sqrt(1 - u) * torch::cos(2 * PI * v),
        torch::sqrt(u) * torch::sin(2 * PI * w),
        torch::sqrt(u) * torch::cos(2 * PI * w)
    }, -1);
}

torch::Tensor identityQuatTensor(long long n){
    torch::Tensor q = torch::zeros({n, 4});
    q.index_put_({Slice(), 0}, 1.0f);
    return q;
}

// MRNF scale init (port of LichtFeld compute_mrnf_knn_log_scales):
// isotropic log-scale from the mean of the 2 nearest neighbor distances,
// clamped by a percentile-based scene size
torch::Tensor mrnfKnnLogScales(const torch::Tensor &xyz){
    long long n = xyz.size(0);
    torch::Tensor lo = std::get<0>(xyz.kthvalue((std::max)(static_cast<long long>(1), static_cast<long long>(0.125 * n)), 0));
    torch::Tensor hi = std::get<0>(xyz.kthvalue((std::min)(n, static_cast<long long>(0.875 * n) + 1), 0));
    torch::Tensor extents = (hi - lo) / 2.0f;
    float medianSize = (std::max)(2.0f * extents.median().item<float>(), 0.01f);
    float maxScale = 0.1f * medianSize;

    PointsTensor pt(xyz);
    KdTreeTensor *index = pt.getIndex<KdTreeTensor>();
    torch::Tensor dists = torch::zeros({n, 1}, torch::kFloat32);
    auto acc = xyz.accessor<float, 2>();
    auto dAcc = dists.accessor<float, 2>();
    for (long long i = 0; i < n; i++){
        float query[3] = { acc[i][0], acc[i][1], acc[i][2] };
        size_t retIndex[3];
        float sqDist[3];
        nanoflann::KNNResultSet<float, size_t> resultSet(3);
        resultSet.init(retIndex, sqDist);
        index->findNeighbors(resultSet, query);
        // Skip self (distance 0), average the next two
        float d = (std::sqrt(sqDist[1]) + std::sqrt(sqDist[2])) * 0.25f;
        dAcc[i][0] = std::clamp(d, 1e-3f, maxScale);
    }
    pt.freeIndex<KdTreeTensor>();
    return dists.log();
}

torch::Tensor gumbelTopK(const torch::Tensor &weights, int k){
    if (k <= 0) return torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(weights.device()));
    torch::Tensor u = torch::rand_like(weights).clamp(1e-10f, 1.0f - 1e-7f);
    torch::Tensor keys = torch::where(weights > 0,
        torch::log(weights.clamp_min(1e-30f)) - torch::log(-torch::log(u)),
        torch::full_like(weights, -1e30f));
    return std::get<1>(keys.topk(k));
}

torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device){
    // OpenGL perspective projection matrix
    float t = zNear * std::tan(0.5f * fovY);
    float b = -t;
    float r = zNear * std::tan(0.5f * fovX);
    float l = -r;
    return torch::tensor({
        {2.0f * zNear / (r - l), 0.0f, (r + l) / (r - l), 0.0f},
        {0.0f, 2 * zNear / (t - b), (t + b) / (t - b), 0.0f},
        {0.0f, 0.0f, (zFar + zNear) / (zFar - zNear), -1.0f * zFar * zNear / (zFar - zNear)},
        {0.0f, 0.0f, 1.0f, 0.0f}
    }, device);
}

torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt){
    torch::Tensor mse = (rendered - gt).pow(2).mean();
    return (10.f * torch::log10(1.0 / mse));
}

torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt){
    return torch::abs(gt - rendered).mean();
}

template<typename T>
std::vector<T> tensor_to_vector(const torch::Tensor t){
    return std::vector<T>(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
}

void Model::setupOptimizers(){
    releaseOptimizers();

    // FastGS learning rates
    const double eps = 1e-15;
    meansOpt = new torch::optim::Adam({means}, torch::optim::AdamOptions(1.6e-4 * spatialLrScale).eps(eps));
    scalesOpt = new torch::optim::Adam({scales}, torch::optim::AdamOptions(5e-3).eps(eps));
    quatsOpt = new torch::optim::Adam({quats}, torch::optim::AdamOptions(1e-3).eps(eps));
    featuresDcOpt = new torch::optim::Adam({featuresDc}, torch::optim::AdamOptions(2.5e-3).eps(eps));
    featuresRestOpt = new torch::optim::Adam({featuresRest}, torch::optim::AdamOptions(2.5e-4).eps(eps)); // highfeature_lr / 20
    opacitiesOpt = new torch::optim::Adam({opacities}, torch::optim::AdamOptions(0.025).eps(eps));
}

void Model::releaseOptimizers(){
    RELEASE_SAFELY(meansOpt);
    RELEASE_SAFELY(scalesOpt);
    RELEASE_SAFELY(quatsOpt);
    RELEASE_SAFELY(featuresDcOpt);
    RELEASE_SAFELY(featuresRestOpt);
    RELEASE_SAFELY(opacitiesOpt);
}


torch::Tensor Model::forward(Camera& cam, int step){

    const float scaleFactor = getDownscaleFactor(step);
    const float fx = cam.fx / scaleFactor;
    const float fy = cam.fy / scaleFactor;
    const float cx = cam.cx / scaleFactor;
    const float cy = cam.cy / scaleFactor;
    const int height = static_cast<int>(static_cast<float>(cam.height) / scaleFactor);
    const int width = static_cast<int>(static_cast<float>(cam.width) / scaleFactor);

    torch::Tensor R = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
    torch::Tensor T = cam.camToWorld.index({Slice(None, 3), Slice(3,4)});

    // Flip the z and y axes to align with gsplat conventions
    R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, R.device())));

    // worldToCam
    torch::Tensor Rinv = R.transpose(0, 1);
    torch::Tensor Tinv = torch::matmul(-Rinv, T);

    lastHeight = height;
    lastWidth = width;

    torch::Tensor viewMat = torch::eye(4, device);
    viewMat.index_put_({Slice(None, 3), Slice(None, 3)}, Rinv);
    viewMat.index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);
        
    float fovX = 2.0f * std::atan(width / (2.0f * fx));
    float fovY = 2.0f * std::atan(height / (2.0f * fy));

    torch::Tensor projMat = projectionMatrix(0.001f, 1000.0f, fovX, fovY, device);
    torch::Tensor colors =  torch::cat({featuresDc.index({Slice(), None, Slice()}), featuresRest}, 1);

    torch::Tensor conics;
    torch::Tensor depths; // GPU-only
    torch::Tensor numTilesHit; // GPU-only
    torch::Tensor cov2d; // CPU-only
    torch::Tensor camDepths; // CPU-only
    torch::Tensor rgb;

    if (device == torch::kCPU){
        auto p = ProjectGaussiansCPU::apply(means, 
                                torch::exp(scales), 
                                1, 
                                quats / quats.norm(2, {-1}, true), 
                                viewMat, 
                                torch::matmul(projMat, viewMat),
                                fx, 
                                fy,
                                cx,
                                cy,
                                height,
                                width);
        xys = p[0];
        radii = p[1];
        conics = p[2];
        cov2d = p[3];
        camDepths = p[4];
    }else{
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)

        TileBounds tileBounds = std::make_tuple((width + BLOCK_X - 1) / BLOCK_X,
                        (height + BLOCK_Y - 1) / BLOCK_Y,
                        1);
        auto p = ProjectGaussians::apply(means, 
                        torch::exp(scales), 
                        1, 
                        quats / quats.norm(2, {-1}, true), 
                        viewMat, 
                        torch::matmul(projMat, viewMat),
                        fx, 
                        fy,
                        cx,
                        cy,
                        height,
                        width,
                        tileBounds);

        xys = p[0];
        depths = p[1];
        radii = p[2];
        conics = p[3];
        numTilesHit = p[4];
        #else
            throw std::runtime_error("GPU support not built, use --cpu");
        #endif
    }
    
    xys.retain_grad();

    if (radii.sum().item<float>() == 0.0f){
        lastAlpha = torch::zeros({height, width}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        errorMap = torch::Tensor();
        return backgroundColor.repeat({height, width, 1});
    }

    torch::Tensor viewDirs = means.detach() - T.transpose(0, 1).to(device);
    viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
    int degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);
    torch::Tensor rgbs;
    
    if (device == torch::kCPU){
        rgbs = SphericalHarmonicsCPU::apply(degreesToUse, viewDirs, colors);
    }else{
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        #ifdef USE_MPS
        torch::mps::synchronize();
        #endif
        rgbs = SphericalHarmonics::apply(degreesToUse, viewDirs, colors);
        #endif
    }
    
    rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f);

    auto fOpts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor camEdgeMap = torch::empty({0}, fOpts);
    if (!scoringPass){
        errorMap = torch::empty({0}, fOpts);
        densificationInfo = torch::empty({0}, fOpts);
        // Abs-GS screen-gradient accumulation while densification is active
        xyAbsGrad = step <= densifyUntilIter ? torch::zeros({means.size(0), 2}, fOpts)
                                             : torch::empty({0}, fOpts);
    }else if (edgeGuidance){
        // Scoring pass: edge-weighted blending accumulates in densification_info row 2
        camEdgeMap = cam.getEdgeMapGpu(getDownscaleFactor(step), device);
    }
    // During scoring passes, errorMap/densificationInfo are managed by computeMultiViewScores

    if (device == torch::kCPU){
        auto rast = RasterizeGaussiansCPU::apply(
                xys,
                radii,
                conics,
                rgbs,
                torch::sigmoid(opacities),
                cov2d,
                camDepths,
                height,
                width,
                backgroundColor,
                errorMap,
                camEdgeMap,
                densificationInfo,
                xyAbsGrad);
        rgb = rast[0];
        lastAlpha = rast[1];
    }else{
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        auto rast = RasterizeGaussians::apply(
                xys,
                depths,
                radii,
                conics,
                numTilesHit,
                rgbs,
                torch::sigmoid(opacities),
                height,
                width,
                backgroundColor,
                errorMap,
                camEdgeMap,
                densificationInfo,
                xyAbsGrad);
        rgb = rast[0];
        lastAlpha = rast[1];
        #endif
    }

    rgb = torch::clamp_max(rgb, 1.0f);

    return rgb;
}

void Model::optimizersZeroGrad(){
  meansOpt->zero_grad();
  scalesOpt->zero_grad();
  quatsOpt->zero_grad();
  featuresDcOpt->zero_grad();
  featuresRestOpt->zero_grad();
  opacitiesOpt->zero_grad();
}

void Model::optimizersStep(){
  meansOpt->step();
  scalesOpt->step();
  quatsOpt->step();
  featuresDcOpt->step();
  featuresRestOpt->step();
  opacitiesOpt->step();
}

static void setOptimizerLr(torch::optim::Adam *opt, double lr){
    static_cast<torch::optim::AdamOptions&>(opt->param_groups()[0].options()).set_lr(lr);
}

void Model::schedulersStep(int step){
    // Exponential position LR decay 1.6e-4 -> 1.6e-6 (x spatialLrScale)
    double t = std::clamp(static_cast<double>(step) / maxSteps, 0.0, 1.0);
    double lr = std::exp(std::log(1.6e-4) * (1.0 - t) + std::log(1.6e-6) * t) * spatialLrScale;
    setOptimizerLr(meansOpt, lr);
}

// FastGS optimizer cadence: SH-rest at 1/16 before densifyUntil; everything at
// 1/32 until 20k, then 1/64. Gradients accumulate between steps.
void Model::optimizerStepCadence(int step){
    auto stepAndZero = [](torch::optim::Adam *opt){
        opt->step();
        opt->zero_grad(true);
    };
    int lateStart = (std::min)(20000, maxSteps * 2 / 3);
    if (step <= densifyUntilIter){
        for (torch::optim::Adam *opt : {meansOpt, scalesOpt, quatsOpt, featuresDcOpt, opacitiesOpt}){
            stepAndZero(opt);
        }
        if (step % 16 == 0) stepAndZero(featuresRestOpt);
    }else if (step <= lateStart){
        if (step % 32 == 0){
            for (torch::optim::Adam *opt : {meansOpt, scalesOpt, quatsOpt, featuresDcOpt, opacitiesOpt, featuresRestOpt}){
                stepAndZero(opt);
            }
        }
    }else{
        if (step % 64 == 0){
            for (torch::optim::Adam *opt : {meansOpt, scalesOpt, quatsOpt, featuresDcOpt, opacitiesOpt, featuresRestOpt}){
                stepAndZero(opt);
            }
        }
    }
}

int Model::getDownscaleFactor(int step){
    return std::pow(2, (std::max<int>)(numDownscales - step / resolutionSchedule, 0));
}

void Model::addToOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &idcs, int nSamples){
    torch::Tensor param = optimizer->param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(*optimizer->state()[pId]));
    
    std::vector<int64_t> repeats;
    repeats.push_back(nSamples);
    for (long int i = 0; i < paramState->exp_avg().dim() - 1; i++){
        repeats.push_back(1);
    }

    paramState->exp_avg(torch::cat({
        paramState->exp_avg(), 
        torch::zeros_like(paramState->exp_avg().index({idcs.squeeze()})).repeat(repeats)
    }, 0));
    
    paramState->exp_avg_sq(torch::cat({
        paramState->exp_avg_sq(), 
        torch::zeros_like(paramState->exp_avg_sq().index({idcs.squeeze()})).repeat(repeats)
    }, 0));

    optimizer->state().erase(pId);

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto newPId = newParam.unsafeGetTensorImpl();
#else
    auto newPId = c10::guts::to_string(newParam.unsafeGetTensorImpl());
#endif    
    optimizer->state()[newPId] = std::move(paramState);
    optimizer->param_groups()[0].params()[0] = newParam;
}

void Model::removeFromOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &deletedMask){
    torch::Tensor param = optimizer->param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(*optimizer->state()[pId]));

    paramState->exp_avg(paramState->exp_avg().index({~deletedMask}));
    paramState->exp_avg_sq(paramState->exp_avg_sq().index({~deletedMask}));

    optimizer->state().erase(pId);
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto newPId = newParam.unsafeGetTensorImpl();
#else
    auto newPId = c10::guts::to_string(newParam.unsafeGetTensorImpl());
#endif
    optimizer->param_groups()[0].params()[0] = newParam;
    optimizer->state()[newPId] = std::move(paramState);
}

void Model::zeroOptimizerRows(torch::optim::Adam *optimizer, const torch::Tensor &idcs){
    if (idcs.numel() == 0) return;
    torch::Tensor param = optimizer->param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    auto it = optimizer->state().find(pId);
    if (it == optimizer->state().end()) return;
    auto &s = static_cast<torch::optim::AdamParamState&>(*it->second);
    s.exp_avg().index_put_({idcs}, torch::zeros_like(s.exp_avg().index({idcs})));
    s.exp_avg_sq().index_put_({idcs}, torch::zeros_like(s.exp_avg_sq().index({idcs})));
}


// FastGS (arXiv 2511.04283) multi-view consistency scoring:
// render sampled views, threshold the min-max normalized per-pixel L1 map,
// count high-error pixels within each gaussian's blended footprint
std::tuple<torch::Tensor, torch::Tensor> Model::computeMultiViewScores(int step, bool densify){
    long long N = means.size(0);
    auto fOpts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor fullCounts = torch::zeros({N}, fOpts);
    torch::Tensor fullScore = torch::zeros({N}, fOpts);
    torch::Tensor edgeScores = torch::zeros({N}, fOpts);

    std::vector<size_t> indices(trainCams->size());
    std::iota(indices.begin(), indices.end(), 0);
    static thread_local std::mt19937 rng(42 + step);
    std::shuffle(indices.begin(), indices.end(), rng);
    size_t numViews = (std::min)(static_cast<size_t>(numScoreViews), indices.size());

    scoringPass = true;
    for (size_t v = 0; v < numViews; v++){
        Camera &cam = (*trainCams)[indices[v]];
        int ds = getDownscaleFactor(step);
        torch::Tensor gt = cam.getImageGpu(ds, device);

        errorMap = torch::zeros({gt.size(0), gt.size(1)}, fOpts);
        densificationInfo = torch::zeros({4, N}, fOpts);
        xyAbsGrad = torch::empty({0}, fOpts);

        torch::Tensor rgb = forward(cam, step);

        {
            torch::NoGradGuard noGrad;
            torch::Tensor l1Map = (rgb.detach() - gt).abs().mean(-1);
            float lo = l1Map.min().item<float>();
            float hi = l1Map.max().item<float>();
            torch::Tensor norm = (l1Map - lo) / (std::max)(hi - lo, 1e-8f);
            errorMap.copy_((norm > lossThresh).to(torch::kFloat32));
        }

        // Zero-gradient backward runs the rasterizer backward, which fills
        // densificationInfo row 3 with the per-gaussian high-error pixel count
        (rgb.sum() * 0.0f).backward();

        torch::NoGradGuard noGrad;
        torch::Tensor ssimLoss = 1.0f - ssim.eval(rgb.detach(), gt);
        torch::Tensor l1Loss = torch::abs(gt - rgb.detach()).mean();
        float photometric = (0.8f * l1Loss + 0.2f * ssimLoss).item<float>();

        torch::Tensor counts = densificationInfo[3];
        fullCounts += counts;
        fullScore += photometric * counts;
        if (edgeGuidance) edgeScores += densificationInfo[2];
    }
    scoringPass = false;
    errorMap = torch::empty({0}, fOpts);
    densificationInfo = torch::empty({0}, fOpts);

    torch::NoGradGuard noGrad;
    torch::Tensor importance;
    if (densify){
        importance = (fullCounts / static_cast<float>(numViews)).floor();
        if (edgeGuidance){
            // Bias densification toward image edges (combats blur):
            // factor = 1 + 0.25 * median-normalized edge score
            torch::Tensor pos = edgeScores.index({edgeScores > 0});
            if (pos.numel() > 0){
                importance = importance * (1.0f + 0.25f * edgeScores / pos.median().clamp_min(1e-12f));
            }
        }
    }
    float lo = fullScore.min().item<float>();
    float hi = fullScore.max().item<float>();
    torch::Tensor pruningScore = (fullScore - lo) / (std::max)(hi - lo, 1e-8f);
    return std::make_tuple(importance, pruningScore);
}

void Model::resetOpacity(float value){
    torch::NoGradGuard noGrad;
    float cap = torch::logit(torch::tensor(value)).item<float>();
    opacities.clamp_max_(cap);
    torch::Tensor allIdx = torch::arange(opacities.size(0), torch::TensorOptions().dtype(torch::kLong).device(device));
    zeroOptimizerRows(opacitiesOpt, allIdx);
}

// Escaped or exploded splats relative to a percentile scene AABB
// (robust to the floaters it is meant to catch)
static torch::Tensor spatialSanityMask(const torch::Tensor &means, const torch::Tensor &scales, const torch::Device &device){
    torch::NoGradGuard noGrad;
    torch::Tensor mc = means.detach().cpu();
    long long n = mc.size(0);
    if (n < 8) return torch::zeros({n}, torch::TensorOptions().dtype(torch::kBool).device(device));
    torch::Tensor lo = std::get<0>(mc.kthvalue((std::max)(static_cast<long long>(1), static_cast<long long>(0.1 * n)), 0));
    torch::Tensor hi = std::get<0>(mc.kthvalue((std::min)(n, static_cast<long long>(0.9 * n) + 1), 0));
    torch::Tensor center = ((lo + hi) / 2.0f).to(device);
    float maxExtent = (std::max)(((hi - lo) / 2.0f).max().item<float>(), 1e-6f);
    torch::Tensor escaped = std::get<0>((means.detach() - center).abs().max(-1)) > 100.0f * maxExtent;
    torch::Tensor exploded = std::get<0>(scales.detach().exp().max(-1)) > 100.0f * maxExtent;
    return escaped | exploded;
}

void Model::densifyAndPrune(int step, const torch::Tensor &importanceScore, const torch::Tensor &pruningScore){
    torch::NoGradGuard noGrad;
    long long numPointsBefore = means.size(0);

    float gradScale = 0.5f * static_cast<float>((std::max)(lastWidth, lastHeight));
    torch::Tensor grads = (xyzGradAccum / gradDenom.clamp_min(1.0f)) * gradScale;
    torch::Tensor gradsAbs = (xyzGradAbsAccum / gradDenom.clamp_min(1.0f)) * gradScale;
    torch::Tensor maxScale = std::get<0>(scales.exp().max(-1));
    torch::Tensor metricMask = importanceScore > 5.0f;

    torch::Tensor cloneMask = (maxScale <= denseThresh * spatialLrScale) & (grads >= gradThresh) & metricMask;
    torch::Tensor splitMask = (maxScale > denseThresh * spatialLrScale) & (gradsAbs >= gradAbsThresh) & metricMask;

    if (maxGaussians > 0){
        long long budget = maxGaussians - numPointsBefore;
        long long requested = cloneMask.sum().item<long long>() + 2 * splitMask.sum().item<long long>();
        if (requested > budget){
            cloneMask &= importanceScore > (budget > 0 ? 5.0f : 1e30f);
            if (budget <= 0){
                splitMask &= torch::zeros_like(splitMask);
            }
        }
    }

    // Clone: duplicate small high-error gaussians
    torch::Tensor cloneIdx = torch::where(cloneMask)[0];
    if (cloneIdx.numel() > 0){
        means = torch::cat({means.detach(), means.detach().index({cloneIdx})}, 0).requires_grad_();
        scales = torch::cat({scales.detach(), scales.detach().index({cloneIdx})}, 0).requires_grad_();
        quats = torch::cat({quats.detach(), quats.detach().index({cloneIdx})}, 0).requires_grad_();
        featuresDc = torch::cat({featuresDc.detach(), featuresDc.detach().index({cloneIdx})}, 0).requires_grad_();
        featuresRest = torch::cat({featuresRest.detach(), featuresRest.detach().index({cloneIdx})}, 0).requires_grad_();
        opacities = torch::cat({opacities.detach(), opacities.detach().index({cloneIdx})}, 0).requires_grad_();

        addToOptimizer(meansOpt, means, cloneIdx, 1);
        addToOptimizer(scalesOpt, scales, cloneIdx, 1);
        addToOptimizer(quatsOpt, quats, cloneIdx, 1);
        addToOptimizer(featuresDcOpt, featuresDc, cloneIdx, 1);
        addToOptimizer(featuresRestOpt, featuresRest, cloneIdx, 1);
        addToOptimizer(opacitiesOpt, opacities, cloneIdx, 1);
    }

    // Split: two samples from the gaussian, scale / 1.6, parent pruned below
    long long nAfterClone = means.size(0);
    torch::Tensor splitIdx = torch::where(splitMask)[0];
    long long nSplits = splitIdx.numel();
    if (nSplits > 0){
        const int nSamples = 2;
        torch::Tensor sampled = torch::randn({nSamples * nSplits, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        torch::Tensor stds = scales.detach().index({splitIdx}).exp().repeat({nSamples, 1});
        torch::Tensor qs = quats.detach().index({splitIdx});
        qs = qs / qs.norm(2, {-1}, true).clamp_min(1e-12);
        torch::Tensor rots = quatToRotMat(qs.repeat({nSamples, 1}));
        torch::Tensor offsets = torch::bmm(rots, (sampled * stds).unsqueeze(-1)).squeeze(-1);
        torch::Tensor newMeans = offsets + means.detach().index({splitIdx}).repeat({nSamples, 1});
        torch::Tensor newScales = torch::log(stds / 1.6f);
        torch::Tensor newQuats = quats.detach().index({splitIdx}).repeat({nSamples, 1});
        torch::Tensor newFDc = featuresDc.detach().index({splitIdx}).repeat({nSamples, 1});
        torch::Tensor newFRest = featuresRest.detach().index({splitIdx}).repeat({nSamples, 1, 1});
        torch::Tensor newOpac = opacities.detach().index({splitIdx}).repeat({nSamples, 1});

        means = torch::cat({means.detach(), newMeans}, 0).requires_grad_();
        scales = torch::cat({scales.detach(), newScales}, 0).requires_grad_();
        quats = torch::cat({quats.detach(), newQuats}, 0).requires_grad_();
        featuresDc = torch::cat({featuresDc.detach(), newFDc}, 0).requires_grad_();
        featuresRest = torch::cat({featuresRest.detach(), newFRest}, 0).requires_grad_();
        opacities = torch::cat({opacities.detach(), newOpac}, 0).requires_grad_();

        addToOptimizer(meansOpt, means, splitIdx, nSamples);
        addToOptimizer(scalesOpt, scales, splitIdx, nSamples);
        addToOptimizer(quatsOpt, quats, splitIdx, nSamples);
        addToOptimizer(featuresDcOpt, featuresDc, splitIdx, nSamples);
        addToOptimizer(featuresRestOpt, featuresRest, splitIdx, nSamples);
        addToOptimizer(opacitiesOpt, opacities, splitIdx, nSamples);
    }

    long long N = means.size(0);
    auto boolOpts = torch::TensorOptions().dtype(torch::kBool).device(device);

    // Split parents must go; escaped/exploded splats are pruned unconditionally
    torch::Tensor parentMask = torch::zeros({N}, boolOpts);
    if (nSplits > 0) parentMask.index_put_({splitIdx}, true);
    torch::Tensor sanityMask = spatialSanityMask(means, scales, device);

    // Targeted pruning: opacity/size candidates, remove only half of them,
    // sampled by inverse (1 - pruning_score)
    torch::Tensor pruneMask = (torch::sigmoid(opacities.squeeze(-1)) < 0.005f);
    if (step > opacityResetInterval && maxRadii2D.defined() && maxRadii2D.size(0) >= numPointsBefore){
        torch::Tensor big2D = torch::zeros({N}, boolOpts);
        big2D.index_put_({Slice(None, numPointsBefore)}, maxRadii2D.index({Slice(None, numPointsBefore)}) > 20.0f);
        pruneMask |= big2D | (std::get<0>(scales.exp().max(-1)) > 0.1f * spatialLrScale);
    }
    pruneMask &= ~parentMask; // parents handled separately

    long long removeBudget = pruneMask.sum().item<long long>() / 2;
    torch::Tensor finalPrune = parentMask | sanityMask;
    if (removeBudget > 0){
        torch::Tensor weights = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        long long S = (std::min)(pruningScore.size(0), N);
        weights.index_put_({Slice(None, S)}, 1.0f / (1e-6f + 1.0f - pruningScore.index({Slice(None, S)})));
        torch::Tensor sampledIdx = torch::multinomial(weights.cpu(), removeBudget, false).to(device);
        torch::Tensor sampledMask = torch::zeros({N}, boolOpts);
        sampledMask.index_put_({sampledIdx}, true);
        finalPrune |= (pruneMask & sampledMask);
    }

    long long cullCount = finalPrune.sum().item<long long>();
    if (cullCount > 0){
        torch::Tensor keep = ~finalPrune;
        means = means.index({keep}).detach().requires_grad_();
        scales = scales.index({keep}).detach().requires_grad_();
        quats = quats.index({keep}).detach().requires_grad_();
        featuresDc = featuresDc.index({keep}).detach().requires_grad_();
        featuresRest = featuresRest.index({keep}).detach().requires_grad_();
        opacities = opacities.index({keep}).detach().requires_grad_();

        removeFromOptimizer(meansOpt, means, finalPrune);
        removeFromOptimizer(scalesOpt, scales, finalPrune);
        removeFromOptimizer(quatsOpt, quats, finalPrune);
        removeFromOptimizer(featuresDcOpt, featuresDc, finalPrune);
        removeFromOptimizer(featuresRestOpt, featuresRest, finalPrune);
        removeFromOptimizer(opacitiesOpt, opacities, finalPrune);
    }

    // Opacity cap at 0.8 after every densification event
    float cap = torch::logit(torch::tensor(0.8f)).item<float>();
    opacities.clamp_max_(cap);
    torch::Tensor allIdx = torch::arange(opacities.size(0), torch::TensorOptions().dtype(torch::kLong).device(device));
    zeroOptimizerRows(opacitiesOpt, allIdx);

    // Reset accumulators
    xyzGradAccum = torch::Tensor();
    xyzGradAbsAccum = torch::Tensor();
    gradDenom = torch::Tensor();
    maxRadii2D = torch::Tensor();

    std::cout << "Densify " << step << ": +clone " << cloneIdx.numel() << " +split " << 2 * nSplits
              << " -prune " << cullCount << ", total " << means.size(0) << std::endl;
}

bool Model::afterTrain(int step){
    bool restructured = false;
    long long N = means.size(0);
    auto fOpts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    if (step < densifyUntilIter){
        if (xys.grad().defined()){
            torch::NoGradGuard noGrad;
            torch::Tensor visible = (radii > 0).flatten();
            if (!xyzGradAccum.defined() || xyzGradAccum.size(0) != N){
                xyzGradAccum = torch::zeros({N}, fOpts);
                xyzGradAbsAccum = torch::zeros({N}, fOpts);
                gradDenom = torch::zeros({N}, fOpts);
                maxRadii2D = torch::zeros({N}, fOpts);
            }
            torch::Tensor g = torch::linalg_vector_norm(xys.grad().detach(), 2, { -1 }, false, torch::kFloat32);
            xyzGradAccum += g * visible.to(torch::kFloat32);
            if (xyAbsGrad.defined() && xyAbsGrad.numel() == 2 * N){
                torch::Tensor ga = torch::linalg_vector_norm(xyAbsGrad, 2, { -1 }, false, torch::kFloat32);
                xyzGradAbsAccum += ga * visible.to(torch::kFloat32);
            }
            gradDenom += visible.to(torch::kFloat32);
            maxRadii2D = torch::maximum(maxRadii2D, radii.detach().to(torch::kFloat32) * visible.to(torch::kFloat32));
        }

        if (step > densifyFromIter && step % densificationInterval == 0 && trainCams != nullptr){
            auto scores = computeMultiViewScores(step, true);
            densifyAndPrune(step, std::get<0>(scores), std::get<1>(scores));
            restructured = true;
        }

        if (step % opacityResetInterval == 0){
            resetOpacity(0.01f);
            std::cout << "Opacity reset" << std::endl;
        }
    }else if (step % 3000 == 0 && step > densifyUntilIter && step < maxSteps && trainCams != nullptr){
        // Final multi-view pruning: the model has converged enough for
        // aggressive removal
        auto scores = computeMultiViewScores(step, false);
        torch::NoGradGuard noGrad;
        torch::Tensor pruningScore = std::get<1>(scores);
        torch::Tensor pruneMask = (torch::sigmoid(opacities.squeeze(-1)) < 0.1f) | (pruningScore > 0.9f)
                                | spatialSanityMask(means, scales, device);
        long long cullCount = pruneMask.sum().item<long long>();
        if (cullCount > 0 && cullCount < means.size(0)){
            torch::Tensor keep = ~pruneMask;
            means = means.index({keep}).detach().requires_grad_();
            scales = scales.index({keep}).detach().requires_grad_();
            quats = quats.index({keep}).detach().requires_grad_();
            featuresDc = featuresDc.index({keep}).detach().requires_grad_();
            featuresRest = featuresRest.index({keep}).detach().requires_grad_();
            opacities = opacities.index({keep}).detach().requires_grad_();

            removeFromOptimizer(meansOpt, means, pruneMask);
            removeFromOptimizer(scalesOpt, scales, pruneMask);
            removeFromOptimizer(quatsOpt, quats, pruneMask);
            removeFromOptimizer(featuresDcOpt, featuresDc, pruneMask);
            removeFromOptimizer(featuresRestOpt, featuresRest, pruneMask);
            removeFromOptimizer(opacitiesOpt, opacities, pruneMask);
            std::cout << "Final prune " << step << ": -" << cullCount << ", remaining " << means.size(0) << std::endl;
        }
        restructured = true;
    }

    if (restructured && device != torch::kCPU){
        #ifdef USE_HIP
                c10::hip::HIPCachingAllocator::emptyCache();
        #elif defined(USE_CUDA)
                c10::cuda::CUDACachingAllocator::emptyCache();
        #endif
    }
    return restructured;
}


void Model::save(const std::string &filename, int step){
    std::string extension = fs::path(filename).extension().string(); 
    if (extension == ".splat"){
        saveSplat(filename);
        std::cout << "Wrote " << filename << std::endl;
    }
    else if (extension == ".ply") {
        savePly(filename, step);
        std::cout << "Wrote " << filename << std::endl;
    }
    else if (extension == ".rad") {
        if (saveRad(filename)) {
            std::cout << "Wrote " << filename << std::endl;
        }
        else {
            std::cerr << "Failed to write " << filename << ", aborting save." << std::endl;
        }
    }
    else {
        bool success = saveSpz(filename);
        if (success) {
            std::cout << "Wrote " << filename << std::endl;
        }
        else {
            std::cerr << "Failed to write " << filename << ", aborting save." << std::endl;
        }
    }
}

void Model::savePly(const std::string &filename, int step){
    std::ofstream o(filename, std::ios::binary);
    int numPoints = means.size(0);

    o << "ply" << std::endl;
    o << "format binary_little_endian 1.0" << std::endl;
    o << "comment Generated by opensplat at iteration " << step << std::endl;
    o << "element vertex " << numPoints << std::endl;
    o << "property float x" << std::endl;
    o << "property float y" << std::endl;
    o << "property float z" << std::endl;
    o << "property float nx" << std::endl;
    o << "property float ny" << std::endl;
    o << "property float nz" << std::endl;

    for (int i = 0; i < featuresDc.size(1); i++){
        o << "property float f_dc_" << i << std::endl;
    }

    // Match Inria's version
    torch::Tensor featuresRestCpu = featuresRest.cpu().transpose(1, 2).reshape({numPoints, -1});
    for (int i = 0; i < featuresRestCpu.size(1); i++){
        o << "property float f_rest_" << i << std::endl;
    }

    o << "property float opacity" << std::endl;

    o << "property float scale_0" << std::endl;
    o << "property float scale_1" << std::endl;
    o << "property float scale_2" << std::endl;

    o << "property float rot_0" << std::endl;
    o << "property float rot_1" << std::endl;
    o << "property float rot_2" << std::endl;
    o << "property float rot_3" << std::endl;
    
    o << "end_header" << std::endl;

    float zeros[] = { 0.0f, 0.0f, 0.0f };

    torch::Tensor meansCpu = keepCrs ? (means.cpu() / scale) + translation : means.cpu();
    torch::Tensor featuresDcCpu = featuresDc.cpu();
    torch::Tensor opacitiesCpu = opacities.cpu();
    torch::Tensor scalesCpu = keepCrs ? torch::log((torch::exp(scales.cpu()) / scale)) : scales.cpu();
    torch::Tensor quatsCpu = quats.cpu();

    for (size_t i = 0; i < numPoints; i++) {
        o.write(reinterpret_cast<const char *>(meansCpu[i].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(zeros), sizeof(float) * 3); // TODO: do we need to write zero normals?
        o.write(reinterpret_cast<const char *>(featuresDcCpu[i].data_ptr()), sizeof(float) * featuresDcCpu.size(1));
        o.write(reinterpret_cast<const char *>(featuresRestCpu[i].data_ptr()), sizeof(float) * featuresRestCpu.size(1));
        o.write(reinterpret_cast<const char *>(opacitiesCpu[i].data_ptr()), sizeof(float) * 1);
        o.write(reinterpret_cast<const char *>(scalesCpu[i].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(quatsCpu[i].data_ptr()), sizeof(float) * 4);
    }

    o.close();
}

void Model::saveSplat(const std::string &filename){
    std::ofstream o(filename, std::ios::binary);
    int numPoints = means.size(0);

    torch::Tensor meansCpu = keepCrs ? (means.cpu() / scale) + translation : means.cpu();
    torch::Tensor scalesCpu = keepCrs ? (torch::exp(scales.cpu()) / scale) : torch::exp(scales.cpu());
    torch::Tensor rgbsCpu = (sh2rgb(featuresDc.cpu()) * 255.0f).toType(torch::kUInt8);
    torch::Tensor opac = (1.0f + torch::exp(-opacities.cpu()));
    torch::Tensor opacitiesCpu = torch::clamp(((1.0f / opac) * 255.0f), 0.0f, 255.0f).toType(torch::kUInt8);
    torch::Tensor quatsCpu = torch::clamp(quats.cpu() * 128.0f + 128.0f, 0.0f, 255.0f).toType(torch::kUInt8);

    std::vector< size_t > splatIndices( numPoints );
    std::iota( splatIndices.begin(), splatIndices.end(), 0 );
    torch::Tensor order = (scalesCpu.index({"...", 0}) + 
                            scalesCpu.index({"...", 1}) + 
                            scalesCpu.index({"...", 2})) / 
                            opac.index({"...", 0});
    float *orderPtr = reinterpret_cast<float *>(order.data_ptr());

    std::sort(splatIndices.begin(), splatIndices.end(), 
        [&orderPtr](size_t const &a, size_t const &b) {
            return orderPtr[a] > orderPtr[b];
        });

    for (int i = 0; i < numPoints; i++){
        size_t idx = splatIndices[i];

        o.write(reinterpret_cast<const char *>(meansCpu[idx].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(scalesCpu[idx].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(rgbsCpu[idx].data_ptr()), sizeof(uint8_t) * 3);
        o.write(reinterpret_cast<const char *>(opacitiesCpu[idx].data_ptr()), sizeof(uint8_t) * 1);
        o.write(reinterpret_cast<const char *>(quatsCpu[idx].data_ptr()), sizeof(uint8_t) * 4);
    }
    o.close();
}

bool Model::saveSpz(const std::string &filename){
    auto numPoints = means.size(0);


    torch::Tensor meansCpu = keepCrs ? (means.cpu() / scale) + translation : means.cpu();
    torch::Tensor scalesCpu = keepCrs ? torch::log(torch::exp(scales.cpu()) / scale) : scales.cpu();

    torch::Tensor meansFlat = meansCpu.flatten();
    torch::Tensor scalesFlat = scalesCpu.flatten();
    torch::Tensor colorsFlat = featuresDc.cpu().flatten(); // raw DC coefficients
    torch::Tensor opacFlat = opacities.flatten().cpu();
    torch::Tensor quatsFlat = torch::roll(quats.cpu(), -1, 1).flatten();
    torch::Tensor shRestFlat = featuresRest.cpu().flatten();
    
    spz::GaussianCloud gaussians;
    gaussians.numPoints = meansCpu.size(0);
    gaussians.shDegree = shDegree;
    gaussians.antialiased = false;
    gaussians.positions = tensor_to_vector<float>(meansFlat);
    gaussians.scales = tensor_to_vector<float>(scalesFlat);
    gaussians.rotations = tensor_to_vector<float>(quatsFlat);
    gaussians.alphas = tensor_to_vector<float>(opacFlat);
    gaussians.colors = tensor_to_vector<float>(colorsFlat);
    gaussians.sh = tensor_to_vector<float>(shRestFlat);

    spz::PackOptions options;
    options.version = 3;       // V4 available but not handled by many viewers
    options.from = spz::CoordinateSystem::RUB;
    options.sh1Bits = 6;
    options.shRestBits = 5;
    bool success = spz::saveSpz(gaussians, options, filename);
    return success;
}

bool Model::saveRad(const std::string &filename){
    size_t numPoints = means.size(0);

    // Same value preparation as savePly; rad.cpp expects exactly the values
    // a saved PLY would contain
    torch::Tensor meansCpu = (keepCrs ? (means.cpu() / scale) + translation : means.cpu()).contiguous();
    torch::Tensor featuresDcCpu = featuresDc.cpu().contiguous();
    // featuresRest [N, K, 3] coefficient-major matches rad's SH layout; no transpose
    torch::Tensor featuresRestCpu = featuresRest.cpu().contiguous();
    torch::Tensor opacitiesCpu = opacities.cpu().contiguous();
    torch::Tensor scalesCpu = (keepCrs ? torch::log(torch::exp(scales.cpu()) / scale) : scales.cpu()).contiguous();
    torch::Tensor quatsCpu = quats.cpu().contiguous();

    rad::SplatData data;
    data.numPoints = numPoints;
    data.numRestCoeffs = featuresRest.size(1);
    data.means = tensor_to_vector<float>(meansCpu);
    data.featuresDc = tensor_to_vector<float>(featuresDcCpu);
    data.featuresRest = tensor_to_vector<float>(featuresRestCpu);
    data.opacities = tensor_to_vector<float>(opacitiesCpu);
    data.scales = tensor_to_vector<float>(scalesCpu);
    data.quats = tensor_to_vector<float>(quatsCpu);

    return rad::saveRad(filename, data);
}

void Model::saveDebugPly(const std::string &filename, int step){
    // A standard PLY
    std::ofstream o(filename, std::ios::binary);
    int numPoints = means.size(0);

    o << "ply" << std::endl;
    o << "format binary_little_endian 1.0" << std::endl;
    o << "comment Generated by opensplat at iteration " << step << std::endl;
    o << "element vertex " << numPoints << std::endl;
    o << "property float x" << std::endl;
    o << "property float y" << std::endl;
    o << "property float z" << std::endl;
    o << "property uchar red" << std::endl;
    o << "property uchar green" << std::endl;
    o << "property uchar blue" << std::endl;
    o << "end_header" << std::endl;

    torch::Tensor meansCpu = keepCrs ? (means.cpu() / scale) + translation : means.cpu();
    torch::Tensor rgbsCpu = (sh2rgb(featuresDc.cpu()) * 255.0f).toType(torch::kUInt8);

    for (size_t i = 0; i < numPoints; i++) {
        o.write(reinterpret_cast<const char *>(meansCpu[i].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(rgbsCpu[i].data_ptr()), sizeof(uint8_t) * 3);
    }

    o.close();
    std::cout << "Wrote " << filename << std::endl;
}

int Model::loadPly(const std::string &filename){
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Invalid PLY file");

    // Ensure we have a valid ply file
    std::string line;
    int numPoints;
    int step;
    size_t bytesRead = 0;

    std::getline(f, line);
    bytesRead += f.gcount();

    if (line == "ply"){
        std::getline(f, line);
        bytesRead += f.gcount();
        if (line == "format binary_little_endian 1.0"){
            std::getline(f, line);
            bytesRead += f.gcount();
            const std::string pattern = "comment Generated by opensplat at iteration ";

            if (line.rfind(pattern, 0) == 0){
                step = std::stoi(line.substr(pattern.length()));
                if (step >= 0){
                    std::getline(f, line);
                    bytesRead += f.gcount();
                    const std::string pattern = "element vertex ";

                    if (line.rfind(pattern, 0) == 0){
                        const int numPoints = std::stoi(line.substr(pattern.length()));
                        
                        const char *requiredProps[] = {
                            "property float x",
                            "property float y",
                            "property float z",
                            "property float nx",
                            "property float ny",
                            "property float nz",
                            "property float f_dc_"
                            "property float f_rest_",
                            "property float opacity",
                            "property float scale_0",
                            "property float scale_1",
                            "property float scale_2",
                            "property float rot_0",
                            "property float rot_1",
                            "property float rot_2",
                            "property float rot_3",
                            "end_header"
                        };

                        for (int i = 0; i < 6; i++){
                            std::getline(f, line);
                            bytesRead += f.gcount();
                            if (line != requiredProps[i]){
                                throw std::runtime_error(std::string("PLY file's header does not contain required property: ") + requiredProps[i]);
                            }
                        }
                        std::getline(f, line);
                        bytesRead += f.gcount();

                        auto countPrefixes = [&f, &line](const char *prefix){
                                int n = 0;
                                while(true){
                                    if (line.rfind(prefix, 0) == 0){
                                        ++n;
                                        std::getline(f, line);
                                    } else {
                                        break;
                                    }
                                }
                                return n;
                        };
                        int featuresDcSize = countPrefixes("property float f_dc_");
                        int featuresRestSize = countPrefixes("property float f_rest_");
                        
                        bool foundEnd = false;
                        for (int i = 8; i < std::size(requiredProps); i++){
                            std::getline(f, line);
                            bytesRead += f.gcount();

                            if (line != requiredProps[i]){
                                throw std::runtime_error(std::string("PLY file's header does not contain required property: ") + requiredProps[i]);
                            }

                            if (line == "end_header"){
                                foundEnd = true;
                                break;
                            }
                        }

                        if (!foundEnd){
                            throw std::runtime_error("PLY file header does not contain header end");
                        } 

                        const size_t bytesPerPoint = sizeof(float) * (14 + featuresDcSize + featuresRestSize);
                        const size_t remainingFileSize = fs::file_size(filename) - bytesRead;
                        if (remainingFileSize != bytesPerPoint * numPoints){
                            std::cout << "Loading PLY..." << std::endl;
                            
                            float zeros[3];

                            torch::Tensor meansCpu = torch::zeros({numPoints, 3}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor featuresDcCpu = torch::zeros({numPoints, featuresDcSize}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor featuresRestCpu = torch::zeros({numPoints, featuresRestSize}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor opacitiesCpu = torch::zeros({numPoints, 1}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor scalesCpu = torch::zeros({numPoints, 3}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor quatsCpu = torch::zeros({numPoints, 4}, torch::TensorOptions().dtype(torch::kFloat32));

                            for (size_t i = 0; i < numPoints; i++){
                                f.read(reinterpret_cast<char *>(meansCpu[i].data_ptr()), sizeof(float) * 3);
                                f.read(reinterpret_cast<char *>(&zeros[0]), sizeof(float) * 3);
                                f.read(reinterpret_cast<char *>(featuresDcCpu[i].data_ptr()), sizeof(float) * featuresDcSize);
                                f.read(reinterpret_cast<char *>(featuresRestCpu[i].data_ptr()), sizeof(float) * featuresRestSize);
                                f.read(reinterpret_cast<char *>(opacitiesCpu[i].data_ptr()), sizeof(float) * 1);
                                f.read(reinterpret_cast<char *>(scalesCpu[i].data_ptr()), sizeof(float) * 3);
                                f.read(reinterpret_cast<char *>(quatsCpu[i].data_ptr()), sizeof(float) * 4);
                            }
                            if (keepCrs){
                                meansCpu = (meansCpu - translation) * scale;
                                scalesCpu = torch::log(scale * torch::exp(scalesCpu));
                            }
                            
                            means = meansCpu.to(device).requires_grad_();
                            featuresDc = featuresDcCpu.to(device).requires_grad_();
                            featuresRest = featuresRestCpu.reshape({numPoints, 3, featuresRestSize/3}).transpose(2, 1).to(device).requires_grad_();
                            opacities = opacitiesCpu.to(device).requires_grad_();
                            scales = scalesCpu.to(device).requires_grad_();
                            quats = quatsCpu.to(device).requires_grad_();
                            
                            std::cerr << "Loaded " << means.size(0) << " gaussians" << std::endl;
                            
                            setupOptimizers();
                            
                            f.close();
                            return step;
                        } else {
                            throw std::runtime_error("PLY file's data section is wrong size");
                        }
                    }
                } else {
                    throw std::runtime_error("PLY file failed sanity check: iteration count should not begin at 0");
                }
            } else if (line.rfind("comment Generated by opensplat")){
                throw std::runtime_error("PLY file does not contain iteration count metadata. You can edit the file to add this metadata manually, by changing \"comment Generated by opensplat\" to \"comment Generated by opensplat at iteration 12345\", changing 12345 to the desired value.");
            }
        }
    }
    throw std::runtime_error("Invalid PLY file");
}

torch::Tensor Model::mainLoss(torch::Tensor &rgb, torch::Tensor &gt, torch::Tensor &mask, float ssimWeight){
    bool hasMask = mask.defined() && mask.numel() > 0;
    // Without a mask the weights are all ones, so skip materializing them and
    // the full-image multiplies they would add to both passes.
    torch::Tensor w = hasMask ? mask : torch::Tensor();

    torch::Tensor absDiff = torch::abs(gt - rgb);
    torch::Tensor l1Loss = hasMask
        ? (w.unsqueeze(-1) * absDiff).sum() / (w.sum() * gt.size(2) + 1e-8f)
        : absDiff.sum() / (static_cast<float>(gt.numel()) + 1e-8f);

    torch::Tensor ssimMap = ssim.map(rgb, gt);
    int pad = ssim.getWindowSize() / 2;
    torch::Tensor sv = ssimMap.index({Slice(pad, ssimMap.size(0) - pad), Slice(pad, ssimMap.size(1) - pad)});
    torch::Tensor dssim;
    if (hasMask){
        torch::Tensor wv = w.index({Slice(pad, w.size(0) - pad), Slice(pad, w.size(1) - pad)});
        dssim = (wv * (1.0f - sv)).sum() / (wv.sum() + 1e-8f);
    }else{
        dssim = (1.0f - sv).sum() / (static_cast<float>(sv.numel()) + 1e-8f);
    }

    torch::Tensor loss = (1.0f - ssimWeight) * l1Loss + ssimWeight * dssim;

    // Segment-mode opacity penalty: push alpha to 0 in masked-out areas
    if (hasMask && lastAlpha.defined() && lastAlpha.numel() == w.numel()){
        loss = loss + (lastAlpha * (1.0f - w).pow(2.0f)).mean();
    }

    // Global opacity regularization: taxes haze and unsupported floaters
    if (opacityReg > 0.0f){
        loss = loss + opacityReg * torch::sigmoid(opacities).mean();
    }

    return loss;
}