#include <filesystem>
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

    // MRNF learning rates (LichtFeld mrnf_defaults)
    const double eps = 1e-15;
    meanLrUnscaled = 2e-5;
    scaleLrCurrent = 7e-3;
    meanLrGamma = std::pow(2e-7 / 2e-5, 1.0 / maxSteps);
    scaleLrGamma = std::pow(5e-3 / 7e-3, 1.0 / maxSteps);

    meansOpt = new torch::optim::Adam({means}, torch::optim::AdamOptions(meanLrUnscaled).eps(eps));
    scalesOpt = new torch::optim::Adam({scales}, torch::optim::AdamOptions(scaleLrCurrent).eps(eps));
    quatsOpt = new torch::optim::Adam({quats}, torch::optim::AdamOptions(2e-3).eps(eps));
    featuresDcOpt = new torch::optim::Adam({featuresDc}, torch::optim::AdamOptions(2e-3).eps(eps));
    featuresRestOpt = new torch::optim::Adam({featuresRest}, torch::optim::AdamOptions(1e-4).eps(eps)); // sh0 / 20
    opacitiesOpt = new torch::optim::Adam({opacities}, torch::optim::AdamOptions(0.012).eps(eps));
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
    torch::Tensor camEdgeMap;
    if (step < stopRefine){
        // Filled in-place by mainLoss before backward; the rasterizer backward reads it
        errorMap = torch::zeros({height, width}, fOpts);
        densificationInfo = torch::zeros({3, means.size(0)}, fOpts);
        if (edgeGuidance) camEdgeMap = cam.getEdgeMap(getDownscaleFactor(step)).to(device).contiguous();
    }else{
        errorMap = torch::empty({0}, fOpts);
        densificationInfo = torch::empty({0}, fOpts);
    }
    if (!camEdgeMap.defined()) camEdgeMap = torch::empty({0}, fOpts);

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
                densificationInfo);
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
                densificationInfo);
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
    meanLrUnscaled *= meanLrGamma;
    scaleLrCurrent *= scaleLrGamma;
    setOptimizerLr(scalesOpt, scaleLrCurrent);
    if (boundsValid) setOptimizerLr(meansOpt, meanLrUnscaled * boundsMedianSize);
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

void Model::computeBounds(){
    torch::NoGradGuard noGrad;
    torch::Tensor m = means.detach();
    if (freeMask.defined() && freeMask.numel() == m.size(0)){
        torch::Tensor active = ~freeMask;
        if (active.sum().item<long long>() > 1) m = m.index({active});
    }
    long long n = m.size(0);
    if (n < 2){ boundsValid = false; return; }
    torch::Tensor mc = m.cpu();
    // 80th-percentile AABB: per-axis 10th / 90th percentile
    torch::Tensor lo = std::get<0>(mc.kthvalue((std::max)(static_cast<long long>(1), static_cast<long long>(0.1 * n)), 0));
    torch::Tensor hi = std::get<0>(mc.kthvalue((std::min)(n, static_cast<long long>(0.9 * n) + 1), 0));
    torch::Tensor center = (lo + hi) / 2.0f;
    torch::Tensor extent = (hi - lo) / 2.0f;
    for (int i = 0; i < 3; i++) boundsCenter[i] = center[i].item<float>();
    boundsMedianSize = (std::max)(2.0f * extent.median().item<float>(), 1e-6f);
    boundsMaxExtent = (std::max)(extent.max().item<float>(), 1e-6f);
    boundsValid = true;
    refinesSinceBounds = 0;
    setOptimizerLr(meansOpt, meanLrUnscaled * boundsMedianSize);
}

void Model::injectNoise(){
    torch::NoGradGuard noGrad;
    torch::Tensor w = torch::pow(1.0f - torch::sigmoid(opacities.squeeze(-1)), 150.0f)
        * static_cast<float>(meanLrUnscaled * boundsMedianSize) * 50.0f;
    w = w * (visCount > 0).to(torch::kFloat32);
    torch::Tensor noise = (torch::randn_like(means) * w.unsqueeze(-1)).clamp(-boundsMedianSize, boundsMedianSize);
    means.add_(noise);
}

bool Model::afterTrain(int step){
    torch::NoGradGuard noGrad;

    if (step >= stopRefine) return false;

    long long N = means.size(0);
    auto fOpts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    if (!freeMask.defined() || freeMask.size(0) != N){
        freeMask = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
    }
    if (!visCount.defined() || visCount.size(0) != N){
        visCount = torch::zeros({N}, fOpts);
        refineWeightMax = torch::zeros({N}, fOpts);
        edgeScoreSum = torch::zeros({N}, fOpts);
        edgeSampleCount = 0;
    }
    if (densificationInfo.defined() && densificationInfo.numel() == 3 * N){
        visCount += densificationInfo[0];
        refineWeightMax = torch::maximum(refineWeightMax, densificationInfo[1]);
        if (edgeGuidance){
            edgeScoreSum += densificationInfo[2];
            edgeSampleCount++;
        }
    }

    if (boundsValid) injectNoise();

    if (step > 0 && step % refineEvery == 0) return refine(step);
    return false;
}

bool Model::refine(int step){
    bool restructured = false;

    if (!boundsValid || ++refinesSinceBounds >= 5) computeBounds();

    // Soft prune: mark dead splats as free slots
    torch::Tensor active = ~freeMask;
    torch::Tensor prune = (opacities.squeeze(-1) < -5.54126358f) // logit(1/255)
        | (quats.pow(2).sum(-1) < 1e-8f)
        | (std::get<0>(scales.min(-1)) < -23.0258509f); // log(1e-10)
    if (boundsValid){
        torch::Tensor center = torch::tensor({boundsCenter[0], boundsCenter[1], boundsCenter[2]}, means.options());
        prune = prune | (std::get<0>(scales.max(-1)) > std::log(100.0f * boundsMaxExtent))
                      | (std::get<0>((means - center).abs().max(-1)) > 100.0f * boundsMaxExtent);
    }
    prune = prune & active;
    int prunedCount = prune.sum().item<int>();
    if (prunedCount > 0){
        torch::Tensor pruneIdx = torch::where(prune)[0];
        freeMask.index_put_({pruneIdx}, true);
        quats.index_put_({pruneIdx}, torch::zeros({prunedCount, 4}, quats.options()));
        for (torch::optim::Adam *opt : {meansOpt, scalesOpt, quatsOpt, featuresDcOpt, featuresRestOpt, opacitiesOpt}){
            zeroOptimizerRows(opt, pruneIdx);
        }
    }

    // Grow and split (Long-Axis Split, Gumbel top-k selection)
    active = ~freeMask;
    torch::Tensor activeF = active.to(torch::kFloat32);
    torch::Tensor visF = (visCount > 0).to(torch::kFloat32);
    torch::Tensor candidates = (refineWeightMax > 0.003f) & (visCount > 0) & active;
    long long activeCount = active.sum().item<long long>();
    int budget = maxGaussians > 0 ? (std::max)(0, maxGaussians - static_cast<int>(activeCount)) : std::numeric_limits<int>::max();

    torch::Tensor edgeFactor;
    if (edgeGuidance && edgeScoreSum.defined() && edgeScoreSum.size(0) == means.size(0) && edgeSampleCount > 0){
        torch::Tensor es = edgeScoreSum / static_cast<float>(edgeSampleCount);
        torch::Tensor pos = es.index({es > 0});
        if (pos.numel() > 0){
            edgeFactor = 1.0f + 0.25f * es / pos.median().clamp_min(1e-12f);
        }
    }

    torch::Tensor replaceW = torch::sigmoid(opacities.squeeze(-1)) * visF * activeF;
    if (edgeFactor.defined()) replaceW = replaceW * edgeFactor;
    int actualReplace = (std::min)((std::min)(prunedCount, budget), static_cast<int>((replaceW > 0).sum().item<long long>()));
    torch::Tensor replaceIdx = gumbelTopK(replaceW, actualReplace);

    torch::Tensor growIdx = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(device));
    if (step < growUntil && budget > actualReplace){
        int desired = static_cast<int>(std::lround(candidates.sum().item<long long>() * 0.07));
        int nGrow = std::clamp(desired - actualReplace, 0, budget - actualReplace);
        torch::Tensor growW = candidates.to(torch::kFloat32) * refineWeightMax;
        if (edgeFactor.defined()) growW = growW * edgeFactor;
        if (actualReplace > 0) growW.index_put_({replaceIdx}, 0.0f);
        nGrow = (std::min)(nGrow, static_cast<int>((growW > 0).sum().item<long long>()));
        growIdx = gumbelTopK(growW, nGrow);
    }
    torch::Tensor splitIdx = torch::cat({replaceIdx, growIdx});
    long long K = splitIdx.size(0);

    if (K > 0){
        torch::Tensor s = scales.index({splitIdx});
        torch::Tensor longest = s.argmax(-1);
        torch::Tensor sLong = s.gather(1, longest.unsqueeze(1));
        torch::Tensor offMag = torch::exp(sLong) * 0.5f;
        torch::Tensor qs = quats.index({splitIdx});
        torch::Tensor qn = qs / qs.norm(2, {-1}, true).clamp_min(1e-12);
        torch::Tensor R = quatToRotMat(qn);
        torch::Tensor dir = R.gather(2, longest.view({-1, 1, 1}).expand({K, 3, 1})).squeeze(-1);

        torch::Tensor newScales = s + std::log(0.85f);
        newScales.scatter_(1, longest.unsqueeze(1), sLong + std::log(0.5f));
        torch::Tensor newOpac = torch::logit((torch::sigmoid(opacities.index({splitIdx})) * 0.6f).clamp(1e-6f, 1.0f - 1e-6f));
        torch::Tensor parentMeans = means.index({splitIdx});
        torch::Tensor childMeans = parentMeans - dir * offMag;
        torch::Tensor childFeaturesDc = featuresDc.index({splitIdx});
        torch::Tensor childFeaturesRest = featuresRest.index({splitIdx});

        // Parent moves to +offset, gets the shrunk shape
        means.index_put_({splitIdx}, parentMeans + dir * offMag);
        scales.index_put_({splitIdx}, newScales);
        opacities.index_put_({splitIdx}, newOpac);

        // Stale-gradient safety: zero this iteration's grads at parent rows
        for (torch::Tensor *p : {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities}){
            if (p->mutable_grad().defined()){
                p->mutable_grad().index_put_({splitIdx}, torch::zeros_like(p->mutable_grad().index({splitIdx})));
            }
        }

        // Children fill free slots first, then append
        torch::Tensor freeIdx = torch::where(freeMask)[0];
        long long nFill = (std::min)(K, freeIdx.size(0));
        if (nFill > 0){
            torch::Tensor fillIdx = freeIdx.slice(0, 0, nFill);
            means.index_put_({fillIdx}, childMeans.slice(0, 0, nFill));
            scales.index_put_({fillIdx}, newScales.slice(0, 0, nFill));
            quats.index_put_({fillIdx}, qn.slice(0, 0, nFill));
            featuresDc.index_put_({fillIdx}, childFeaturesDc.slice(0, 0, nFill));
            featuresRest.index_put_({fillIdx}, childFeaturesRest.slice(0, 0, nFill));
            opacities.index_put_({fillIdx}, newOpac.slice(0, 0, nFill));
            freeMask.index_put_({fillIdx}, false);
            for (torch::optim::Adam *opt : {meansOpt, scalesOpt, quatsOpt, featuresDcOpt, featuresRestOpt, opacitiesOpt}){
                zeroOptimizerRows(opt, fillIdx);
            }
            for (torch::Tensor *p : {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities}){
                if (p->mutable_grad().defined()){
                    p->mutable_grad().index_put_({fillIdx}, torch::zeros_like(p->mutable_grad().index({fillIdx})));
                }
            }
        }
        long long nAppend = K - nFill;
        if (nAppend > 0){
            torch::Tensor tailIdx = splitIdx.slice(0, nFill, K);
            means = torch::cat({means.detach(), childMeans.slice(0, nFill, K)}, 0).requires_grad_();
            scales = torch::cat({scales.detach(), newScales.slice(0, nFill, K)}, 0).requires_grad_();
            quats = torch::cat({quats.detach(), qn.slice(0, nFill, K)}, 0).requires_grad_();
            featuresDc = torch::cat({featuresDc.detach(), childFeaturesDc.slice(0, nFill, K)}, 0).requires_grad_();
            featuresRest = torch::cat({featuresRest.detach(), childFeaturesRest.slice(0, nFill, K)}, 0).requires_grad_();
            opacities = torch::cat({opacities.detach(), newOpac.slice(0, nFill, K)}, 0).requires_grad_();

            addToOptimizer(meansOpt, means, tailIdx, 1);
            addToOptimizer(scalesOpt, scales, tailIdx, 1);
            addToOptimizer(quatsOpt, quats, tailIdx, 1);
            addToOptimizer(featuresDcOpt, featuresDc, tailIdx, 1);
            addToOptimizer(featuresRestOpt, featuresRest, tailIdx, 1);
            addToOptimizer(opacitiesOpt, opacities, tailIdx, 1);

            freeMask = torch::cat({freeMask, torch::zeros({nAppend}, freeMask.options())}, 0);
            restructured = true;
        }
        std::cout << "Refine " << step << ": pruned " << prunedCount << ", split " << K
                  << " (" << actualReplace << " into free slots budget), total " << means.size(0)
                  << " (" << (~freeMask).sum().item<long long>() << " active)" << std::endl;
    }

    // Enforce max gaussians (keep by opacity, Gumbel top-k)
    active = ~freeMask;
    activeCount = active.sum().item<long long>();
    if (maxGaussians > 0 && activeCount > maxGaussians){
        torch::Tensor keepW = torch::sigmoid(opacities.squeeze(-1)) * active.to(torch::kFloat32);
        torch::Tensor keepIdx = gumbelTopK(keepW, maxGaussians);
        torch::Tensor keepMask = torch::zeros({means.size(0)}, torch::TensorOptions().dtype(torch::kBool).device(device));
        keepMask.index_put_({keepIdx}, true);
        torch::Tensor cullMask = ~keepMask;

        means = means.index({keepMask}).detach().requires_grad_();
        scales = scales.index({keepMask}).detach().requires_grad_();
        quats = quats.index({keepMask}).detach().requires_grad_();
        featuresDc = featuresDc.index({keepMask}).detach().requires_grad_();
        featuresRest = featuresRest.index({keepMask}).detach().requires_grad_();
        opacities = opacities.index({keepMask}).detach().requires_grad_();

        removeFromOptimizer(meansOpt, means, cullMask);
        removeFromOptimizer(scalesOpt, scales, cullMask);
        removeFromOptimizer(quatsOpt, quats, cullMask);
        removeFromOptimizer(featuresDcOpt, featuresDc, cullMask);
        removeFromOptimizer(featuresRestOpt, featuresRest, cullMask);
        removeFromOptimizer(opacitiesOpt, opacities, cullMask);

        freeMask = torch::zeros({means.size(0)}, torch::TensorOptions().dtype(torch::kBool).device(device));
        restructured = true;
        std::cout << "Max cap enforced, kept " << means.size(0) << " gaussians" << std::endl;
    }

    // Decay instead of opacity reset
    float tShrink = 1.0f - static_cast<float>(step) / maxSteps;
    torch::Tensor op = (torch::sigmoid(opacities) - 0.004f * tShrink).clamp(1e-6f, 1.0f - 1e-6f);
    opacities.copy_(torch::logit(op));
    scales.copy_(torch::log((scales.exp() * (1.0f - 0.002f * tShrink)).clamp_min(1e-12f)));

    // Reset per-window stats
    visCount = torch::Tensor();
    refineWeightMax = torch::Tensor();
    edgeScoreSum = torch::Tensor();
    edgeSampleCount = 0;

    if (device != torch::kCPU){
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
    torch::Tensor w = hasMask ? mask : torch::ones({gt.size(0), gt.size(1)}, gt.options());

    torch::Tensor l1Loss = (w.unsqueeze(-1) * torch::abs(gt - rgb)).sum() / (w.sum() * gt.size(2) + 1e-8f);

    torch::Tensor ssimMap = ssim.map(rgb, gt);
    int pad = ssim.getWindowSize() / 2;
    torch::Tensor wv = w.index({Slice(pad, w.size(0) - pad), Slice(pad, w.size(1) - pad)});
    torch::Tensor sv = ssimMap.index({Slice(pad, ssimMap.size(0) - pad), Slice(pad, ssimMap.size(1) - pad)});
    torch::Tensor dssim = (wv * (1.0f - sv)).sum() / (wv.sum() + 1e-8f);

    torch::Tensor loss = (1.0f - ssimWeight) * l1Loss + ssimWeight * dssim;

    // Segment-mode opacity penalty: push alpha to 0 in masked-out areas
    if (hasMask && lastAlpha.defined() && lastAlpha.numel() == w.numel()){
        loss = loss + (lastAlpha * (1.0f - w).pow(2.0f)).mean();
    }

    // Densification error map (per-pixel DSSIM, mean-normalized)
    if (errorMap.defined() && errorMap.numel() == ssimMap.numel()){
        torch::NoGradGuard noGrad;
        torch::Tensor err = 1.0f - ssimMap.detach();
        if (hasMask) err = err * w;
        errorMap.copy_(err / (err.mean() + 1e-6f));
    }

    return loss;
}
