// Originally started from https://github.com/nerfstudio-project/gsplat
// This implementation has been substantially changed and optimized 
// Licensed under the AGPLv3
// Piero Toffanin - 2024

#include "bindings.h"
#include "../gsplat/config.h"

#include <cstdio>
#include <iostream>
#include <cmath>
#include <tuple>
#include <thread>
#include <atomic>

using namespace torch::indexing;

namespace {

int rasterWorkers(int height, size_t floatsPerWorker = 0){
    int workers = static_cast<int>((std::max)(1u, std::thread::hardware_concurrency()));
    workers = (std::min)(workers, (std::max)(1, height));
    if (floatsPerWorker > 0){
        const size_t budget = 64ull << 20;
        const int maxWorkers = (std::max)(1, static_cast<int>(budget / (floatsPerWorker * sizeof(float) + 1)));
        workers = (std::min)(workers, maxWorkers);
    }
    return workers;
}

// Runs fn(chunk, slot) for every chunk in [0, numChunks)
template <typename F>
void parallelChunks(int numChunks, int numWorkers, const F &fn){
    if (numWorkers <= 1){
        for (int c = 0; c < numChunks; c++) fn(c, 0);
        return;
    }
    std::atomic<int> next(0);
    auto worker = [&](int slot){
        int c;
        while ((c = next.fetch_add(1, std::memory_order_relaxed)) < numChunks) fn(c, slot);
    };
    std::vector<std::thread> threads;
    threads.reserve(numWorkers - 1);
    for (int t = 1; t < numWorkers; t++){
        threads.emplace_back([&worker, t](){ worker(t); });
    }
    worker(0);
    for (std::thread &t : threads) t.join();
}

const int CHUNKS_PER_WORKER = 4;

}

torch::Tensor quatToRot(const torch::Tensor &quat){
    auto u = torch::unbind(torch::nn::functional::normalize(quat, torch::nn::functional::NormalizeFuncOptions().dim(-1)), -1);
    torch::Tensor w = u[0];
    torch::Tensor x = u[1];
    torch::Tensor y = u[2];
    torch::Tensor z = u[3];
    return torch::stack({
        torch::stack({
            1.0 - 2.0 * (y.pow(2) + z.pow(2)),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y)
        }, -1),
        torch::stack({
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x.pow(2) + z.pow(2)),
            2.0 * (y * z - w * x)
        }, -1),
        torch::stack({
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x.pow(2) + y.pow(2))
        }, -1)
    }, -2);
    
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor_cpu(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    torch::Tensor &projmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    const float clip_thresh
){
    float fovx = 0.5f * static_cast<float>(img_width) / fx;
    float fovy = 0.5f * static_cast<float>(img_height) / fy;
    
    // clip_near_plane
    torch::Tensor Rclip = viewmat.index({"...", Slice(None, 3), Slice(None, 3)}); 
    torch::Tensor Tclip = viewmat.index({"...", Slice(None, 3), 3});
    torch::Tensor pView = torch::matmul(Rclip, means3d.index({"...", None})).index({"...", 0}) + Tclip;
    // torch::Tensor isClose = pView.index({"...", 2}) < clip_thresh;

    // scale_rot_to_cov3d
    torch::Tensor R = quatToRot(quats);
    torch::Tensor M = R * glob_scale * scales.index({"...", None, Slice()});
    torch::Tensor cov3d = torch::matmul(M, M.transpose(-1, -2));

    // project_cov3d_ewa
    torch::Tensor limX = 1.3f * torch::tensor({fovx}, means3d.device());
    torch::Tensor limY = 1.3f * torch::tensor({fovy}, means3d.device());
    
    torch::Tensor minLimX = pView.index({"...", 2}) * torch::min(limX, torch::max(-limX, pView.index({"...", 0}) / pView.index({"...", 2})));
    torch::Tensor minLimY = pView.index({"...", 2}) * torch::min(limY, torch::max(-limY, pView.index({"...", 1}) / pView.index({"...", 2})));
    
    torch::Tensor t = torch::cat({minLimX.index({"...", None}), minLimY.index({"...", None}), pView.index({"...", 2, None})}, -1);
    torch::Tensor rz = 1.0f / t.index({"...", 2});
    torch::Tensor rz2 = rz.pow(2);

    torch::Tensor J = torch::stack({
        torch::stack({fx * rz, torch::zeros_like(rz), -fx * t.index({"...", 0}) * rz2}, -1),
        torch::stack({torch::zeros_like(rz), fy * rz, -fy * t.index({"...", 1}) * rz2}, -1)
    }, -2);

    torch::Tensor T = torch::matmul(J, Rclip);
    torch::Tensor cov2d = torch::matmul(T, torch::matmul(cov3d, T.transpose(-1, -2)));

    // Add blur along axes
    cov2d.index_put_({"...", 0, 0}, cov2d.index({"...", 0, 0}) + 0.3f);
    cov2d.index_put_({"...", 1, 1}, cov2d.index({"...", 1, 1}) + 0.3f);
     
    // compute_cov2d_bounds
    float eps = 1e-6f;
    torch::Tensor det = cov2d.index({"...", 0, 0}) * cov2d.index({"...", 1, 1}) - cov2d.index({"...", 0, 1}).pow(2);
    det = torch::clamp_min(det, eps);
    torch::Tensor conic = torch::stack({
            cov2d.index({"...", 1, 1}) / det,
            -cov2d.index({"...", 0, 1}) / det,
            cov2d.index({"...", 0, 0}) / det
        }, -1);

    torch::Tensor b = (cov2d.index({"...", 0, 0}) + cov2d.index({"...", 1, 1})) / 2.0f;
    torch::Tensor sq = torch::sqrt(torch::clamp_min(b.pow(2) - det, 0.1f));
    torch::Tensor v1 = b + sq;
    torch::Tensor v2 = b - sq;
    torch::Tensor radius = torch::ceil(3.0f * torch::sqrt(torch::max(v1, v2)));
    // torch::Tensor detValid = det > eps;

    // project_pix
    torch::Tensor pHom = torch::nn::functional::pad(means3d, torch::nn::functional::PadFuncOptions({0, 1}).mode(torch::kConstant).value(1.0f));
    pHom = torch::einsum("...ij,...j->...i", {projmat, pHom});
    torch::Tensor rw = 1.0f / torch::clamp_min(pHom.index({"...", 3}), eps);
    torch::Tensor pProj = pHom.index({"...", Slice(None, 3)}) * rw.index({"...", None});
    torch::Tensor u = 0.5f * ((pProj.index({"...", 0}) + 1.0f) * static_cast<float>(img_width) - 1.0f);
    torch::Tensor v = 0.5f * ((pProj.index({"...", 1}) + 1.0f) * static_cast<float>(img_height) - 1.0f);
    torch::Tensor xys = torch::stack({u, v}, -1); // center

    torch::Tensor radii = radius.to(torch::kInt32);
    torch::Tensor camDepths = pView.index({"...", 2}).contiguous();

    return std::make_tuple(xys, radii, conic, cov2d, camDepths);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    std::vector<int32_t> *
> rasterize_forward_tensor_cpu(
    const int width,
    const int height,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background,
    const torch::Tensor &cov2d,
    const torch::Tensor &camDepths
){
    torch::NoGradGuard noGrad;

    int channels = colors.size(1);
    int numPoints = xys.size(0);
    float *pDepths = static_cast<float *>(camDepths.data_ptr());
    std::vector<int32_t> *px2gid = new std::vector<int32_t>[width * height];

    std::vector< size_t > gIndices( numPoints );
    std::iota( gIndices.begin(), gIndices.end(), 0 );
    std::sort(gIndices.begin(), gIndices.end(), [&pDepths](int a, int b){
        return pDepths[a] < pDepths[b];
    });

    torch::Device device = xys.device();

    torch::Tensor outImg = torch::zeros({height, width, channels}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor finalTs = torch::ones({height, width}, torch::TensorOptions().dtype(torch::kFloat32).device(device));   
    torch::Tensor done = torch::zeros({height, width}, torch::TensorOptions().dtype(torch::kBool).device(device));   

    torch::Tensor sqCov2dX = 3.0f * torch::sqrt(cov2d.index({"...", 0, 0}));
    torch::Tensor sqCov2dY = 3.0f * torch::sqrt(cov2d.index({"...", 1, 1}));
    
    float *pConics = static_cast<float *>(conics.data_ptr());
    float *pCenters = static_cast<float *>(xys.data_ptr());
    float *pSqCov2dX = static_cast<float *>(sqCov2dX.data_ptr());
    float *pSqCov2dY = static_cast<float *>(sqCov2dY.data_ptr());
    float *pOpacities = static_cast<float *>(opacities.data_ptr());

    float *pOutImg = static_cast<float *>(outImg.data_ptr());
    float *pFinalTs = static_cast<float *>(finalTs.data_ptr());
    bool *pDone = static_cast<bool *>(done.data_ptr());

    float *pColors = static_cast<float *>(colors.data_ptr());
    
    float bgX = background[0].item<float>();
    float bgY = background[1].item<float>();
    float bgZ = background[2].item<float>();

    const float alphaThresh = 1.0f / 255.0f;

    const int numWorkers = rasterWorkers(height);
    const int numBands = (std::min)(height, numWorkers * CHUNKS_PER_WORKER);

    parallelChunks(numBands, numWorkers, [&](int band, int){
            for (int idx = 0; idx < numPoints; idx++){
                int32_t gaussianId = gIndices[idx];

                float sqy = pSqCov2dY[gaussianId];
                float gY = pCenters[gaussianId * 2 + 1];

                int minx = (std::max)(0, static_cast<int>(std::floor(gY - sqy)) - 2);
                int maxx = (std::min)(height, static_cast<int>(std::ceil(gY + sqy)) + 2);
                // first row >= minx that belongs to this band
                minx += ((band - minx) % numBands + numBands) % numBands;
                if (minx >= maxx) continue;

                float sqx = pSqCov2dX[gaussianId];
                float gX = pCenters[gaussianId * 2 + 0];

                int miny = (std::max)(0, static_cast<int>(std::floor(gX - sqx)) - 2);
                int maxy = (std::min)(width, static_cast<int>(std::ceil(gX + sqx)) + 2);
                if (miny >= maxy) continue;

                float A = pConics[gaussianId * 3 + 0];
                float B = pConics[gaussianId * 3 + 1];
                float C = pConics[gaussianId * 3 + 2];

                const float opacity = pOpacities[gaussianId];
                const float r = pColors[gaussianId * 3 + 0];
                const float g = pColors[gaussianId * 3 + 1];
                const float b = pColors[gaussianId * 3 + 2];
                const float sigmaCut = opacity > 0.0f
                    ? std::log(255.0f * opacity) + 1e-3f
                    : -1.0f;

                for (int i = minx; i < maxx; i += numBands){
                    for (int j = miny; j < maxy; j++){
                        size_t pixIdx = (i * width + j);
                        if (pDone[pixIdx]) continue;

                        float xCam = gX - j;
                        float yCam = gY - i;
                        float sigma = (
                            0.5f
                            * (A * xCam * xCam + C * yCam * yCam)
                            + B * xCam * yCam
                        );

                        if (sigma < 0.0f) continue;
                        if (sigma > sigmaCut) continue;
                        float alpha = (std::min)(0.999f, (opacity * std::exp(-sigma)));
                        if (alpha < alphaThresh) continue;

                        float T = pFinalTs[pixIdx];
                        float nextT = T * (1.0f - alpha);
                        if (nextT <= 1e-4f) { // this pixel is done
                            pDone[pixIdx] = true;
                            continue;
                        }

                        float vis = alpha * T;

                        pOutImg[pixIdx * 3 + 0] += vis * r;
                        pOutImg[pixIdx * 3 + 1] += vis * g;
                        pOutImg[pixIdx * 3 + 2] += vis * b;

                        pFinalTs[pixIdx] = nextT;
                        px2gid[pixIdx].push_back(gaussianId);
                    }
                }
            }

            // Background
            for (int i = band; i < height; i += numBands){
                for (int j = 0; j < width; j++){
                    size_t pixIdx = (i * width + j);
                    float T = pFinalTs[pixIdx];

                    pOutImg[pixIdx * 3 + 0] += T * bgX;
                    pOutImg[pixIdx * 3 + 1] += T * bgY;
                    pOutImg[pixIdx * 3 + 2] += T * bgZ;
                }
            }
    });

    return std::make_tuple(outImg, finalTs, px2gid);
}


std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor_cpu(
        const int height,
        const int width,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &cov2d,
        const torch::Tensor &camDepths,        
        const torch::Tensor &final_Ts,
        const std::vector<int32_t> *px2gid,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha,
        const torch::Tensor &error_map,
        const torch::Tensor &edge_map,
        const torch::Tensor &densification_info,
        const torch::Tensor &v_xy_abs
    ){
    torch::NoGradGuard noGrad;

    int numPoints = xys.size(0);
    int channels = colors.size(1);
    torch::Device device = xys.device();

    float *pColors = static_cast<float *>(colors.data_ptr());
    float *pv_output = static_cast<float *>(v_output.data_ptr());
    float *pv_outputAlpha = static_cast<float *>(v_output_alpha.data_ptr());
    float *pConics = static_cast<float *>(conics.data_ptr());
    float *pCenters = static_cast<float *>(xys.data_ptr());
    float *pOpacities = static_cast<float *>(opacities.data_ptr());

    float bgX = background[0].item<float>();
    float bgY = background[1].item<float>();
    float bgZ = background[2].item<float>();

    float *pFinalTs = static_cast<float *>(final_Ts.data_ptr());
    const bool hasDinfo = densification_info.numel() > 0;
    float *pErr = error_map.numel() > 0 ? static_cast<float *>(error_map.data_ptr()) : nullptr;
    float *pEdge = edge_map.numel() > 0 ? static_cast<float *>(edge_map.data_ptr()) : nullptr;
    const bool hasXyAbs = v_xy_abs.numel() > 0;

    const float alphaThresh = 1.0f / 255.0f;

    auto fOpts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const size_t floatsPerWorker = static_cast<size_t>(numPoints) *
        (2 + 3 + channels + 1 + (hasDinfo ? 4 : 0) + (hasXyAbs ? 2 : 0));
    const int numWorkers = rasterWorkers(height, floatsPerWorker);
    const int numBands = (std::min)(height, numWorkers * CHUNKS_PER_WORKER);

    torch::Tensor v_xy_b = torch::zeros({numWorkers, numPoints, 2}, fOpts);
    torch::Tensor v_conic_b = torch::zeros({numWorkers, numPoints, 3}, fOpts);
    torch::Tensor v_colors_b = torch::zeros({numWorkers, numPoints, channels}, fOpts);
    torch::Tensor v_opacity_b = torch::zeros({numWorkers, numPoints, 1}, fOpts);
    torch::Tensor dinfo_b = torch::zeros({numWorkers, hasDinfo ? 4 : 0, numPoints}, fOpts);
    torch::Tensor xyAbs_b = torch::zeros({numWorkers, hasXyAbs ? numPoints : 0, 2}, fOpts);

    float *pv_xy_b = static_cast<float *>(v_xy_b.data_ptr());
    float *pv_conic_b = static_cast<float *>(v_conic_b.data_ptr());
    float *pv_colors_b = static_cast<float *>(v_colors_b.data_ptr());
    float *pv_opacity_b = static_cast<float *>(v_opacity_b.data_ptr());
    float *pDinfo_b = hasDinfo ? static_cast<float *>(dinfo_b.data_ptr()) : nullptr;
    float *pXyAbs_b = hasXyAbs ? static_cast<float *>(xyAbs_b.data_ptr()) : nullptr;

    parallelChunks(numBands, numWorkers, [&](int band, int slot){
        float *pv_xy = pv_xy_b + static_cast<size_t>(slot) * numPoints * 2;
        float *pv_conic = pv_conic_b + static_cast<size_t>(slot) * numPoints * 3;
        float *pv_colors = pv_colors_b + static_cast<size_t>(slot) * numPoints * channels;
        float *pv_opacity = pv_opacity_b + static_cast<size_t>(slot) * numPoints;
        float *pDinfo = pDinfo_b ? pDinfo_b + static_cast<size_t>(slot) * numPoints * 4 : nullptr;
        float *pXyAbs = pXyAbs_b ? pXyAbs_b + static_cast<size_t>(slot) * numPoints * 2 : nullptr;

        for (int i = static_cast<int>(band); i < height; i += numBands){
        for (int j = 0; j < width; j++){
            size_t pixIdx = (i * width + j);
            float Tfinal = pFinalTs[pixIdx];
            float T = Tfinal;
            float buffer[3] = {0.0f, 0.0f, 0.0f};

            const std::vector<int32_t> &gids = px2gid[pixIdx];
            for (auto it = gids.rbegin(); it != gids.rend(); ++it){
                const int32_t gaussianId = *it;
                float A = pConics[gaussianId * 3 + 0];
                float B = pConics[gaussianId * 3 + 1];
                float C = pConics[gaussianId * 3 + 2];

                float gX = pCenters[gaussianId * 2 + 0];
                float gY = pCenters[gaussianId * 2 + 1];

                float xCam = gX - j;
                float yCam = gY - i;
                float sigma = (
                    0.5f
                    * (A * xCam * xCam + C * yCam * yCam)
                    + B * xCam * yCam
                );

                if (sigma < 0.0f) continue;
                float vis = std::exp(-sigma);
                float alpha = (std::min)(0.999f, pOpacities[gaussianId] * vis);
                if (alpha < alphaThresh) continue;

                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                float fac = alpha * T;

                if (pDinfo){
                    float err = pErr ? pErr[pixIdx] : 1.0f;
                    pDinfo[gaussianId] += fac;
                    pDinfo[numPoints + gaussianId] += fac * err;
                    if (pEdge) pDinfo[2 * numPoints + gaussianId] += fac * pEdge[pixIdx];
                    if (err > 0.5f) pDinfo[3 * numPoints + gaussianId] += 1.0f;
                }

                pv_colors[gaussianId * 3 + 0] += fac * pv_output[pixIdx * 3 + 0];
                pv_colors[gaussianId * 3 + 1] += fac * pv_output[pixIdx * 3 + 1];
                pv_colors[gaussianId * 3 + 2] += fac * pv_output[pixIdx * 3 + 2];

                float v_alpha = ((pColors[gaussianId * 3 + 0] * T - buffer[0] * ra) * pv_output[pixIdx * 3 + 0]) +
                                ((pColors[gaussianId * 3 + 1] * T - buffer[1] * ra) * pv_output[pixIdx * 3 + 1]) +
                                ((pColors[gaussianId * 3 + 2] * T - buffer[2] * ra) * pv_output[pixIdx * 3 + 2]) +
                                (Tfinal * ra * pv_outputAlpha[pixIdx]) +

                                (-Tfinal * ra * bgX * pv_output[pixIdx * 3 + 0]) +
                                (-Tfinal * ra * bgY * pv_output[pixIdx * 3 + 1]) +
                                (-Tfinal * ra * bgZ * pv_output[pixIdx * 3 + 2]);

                buffer[0] += pColors[gaussianId * 3 + 0] * fac;
                buffer[1] += pColors[gaussianId * 3 + 1] * fac;
                buffer[2] += pColors[gaussianId * 3 + 2] * fac;
                
                float v_sigma = -pOpacities[gaussianId] * vis * v_alpha;
                pv_conic[gaussianId * 3 + 0] += 0.5f * v_sigma * xCam * xCam;
                pv_conic[gaussianId * 3 + 1] += 0.5f * v_sigma * xCam * yCam;
                pv_conic[gaussianId * 3 + 2] += 0.5f * v_sigma * yCam * yCam;

                pv_xy[gaussianId * 2 + 0] += v_sigma * (A * xCam + B * yCam);
                pv_xy[gaussianId * 2 + 1] += v_sigma * (B * xCam + C * yCam);
                if (pXyAbs){
                    pXyAbs[gaussianId * 2 + 0] += std::fabs(v_sigma * (A * xCam + B * yCam));
                    pXyAbs[gaussianId * 2 + 1] += std::fabs(v_sigma * (B * xCam + C * yCam));
                }

                pv_opacity[gaussianId] += vis * v_alpha;
            }
        }
        }
    });

    torch::Tensor v_xy = v_xy_b.sum(0);
    torch::Tensor v_conic = v_conic_b.sum(0);
    torch::Tensor v_colors = v_colors_b.sum(0);
    torch::Tensor v_opacity = v_opacity_b.sum(0);
    if (hasDinfo) densification_info.add_(dinfo_b.sum(0));
    if (hasXyAbs) v_xy_abs.add_(xyAbs_b.sum(0));

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}


const float SH_C0 = 0.28209479177387814f;
const float SH_C1 = 0.4886025119029199f;
const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};
const float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304f,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f
};

int numShBases(int degree){
    switch(degree){
        case 0:
            return 1;
        case 1:
            return 4;
        case 2:
            return 9;
        case 3:
            return 16;
        default:
            return 25;
    }
}

torch::Tensor compute_sh_forward_tensor_cpu(
    const int num_points,
    const int degree,
    const int degrees_to_use,
    const torch::Tensor &viewdirs,
    const torch::Tensor &coeffs
) {
    const int numChannels = 3;
    unsigned numBases = numShBases(degrees_to_use);

    torch::Tensor result = torch::zeros({viewdirs.size(0), numShBases(degree)}, torch::TensorOptions().dtype(torch::kFloat32).device(viewdirs.device()));   
    
    result.index_put_({"...", 0}, SH_C0);
    if (numBases > 1){
        std::vector<torch::Tensor> xyz = viewdirs.unbind(-1); 
        torch::Tensor x = xyz[0];
        torch::Tensor y = xyz[1];
        torch::Tensor z = xyz[2];
        result.index_put_({"...", 1}, SH_C1 * -y);
        result.index_put_({"...", 2}, SH_C1 * z);
        result.index_put_({"...", 3}, SH_C1 * -x);

        if (numBases > 4){
            torch::Tensor xx = x * x;
            torch::Tensor yy = y * y;
            torch::Tensor zz = z * z;
            torch::Tensor xy = x * y;
            torch::Tensor yz = y * z;
            torch::Tensor xz = x * z;

            result.index_put_({"...", 4}, SH_C2[0] * xy);
            result.index_put_({"...", 5}, SH_C2[1] * yz);
            result.index_put_({"...", 6}, SH_C2[2] * (2.0f * zz - xx - yy));
            result.index_put_({"...", 7}, SH_C2[3] * xz);
            result.index_put_({"...", 8}, SH_C2[4] * (xx - yy));

            if (numBases > 9){
                result.index_put_({"...", 9},  SH_C3[0] * y * (3 * xx - yy));
                result.index_put_({"...", 10}, SH_C3[1] * xy * z);
                result.index_put_({"...", 11}, SH_C3[2] * y * (4 * zz - xx - yy));
                result.index_put_({"...", 12}, SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy));
                result.index_put_({"...", 13}, SH_C3[4] * x * (4 * zz - xx - yy) );
                result.index_put_({"...", 14}, SH_C3[5] * z * (xx - yy));
                result.index_put_({"...", 15}, SH_C3[6] * x * (xx - 3 * yy));
                
                if (numBases > 16){
                    result.index_put_({"...", 16}, SH_C4[0] * xy * (xx - yy));
                    result.index_put_({"...", 17}, SH_C4[1] * yz * (3 * xx - yy));
                    result.index_put_({"...", 18}, SH_C4[2] * xy * (7 * zz - 1));
                    result.index_put_({"...", 19}, SH_C4[3] * yz * (7 * zz - 3));
                    result.index_put_({"...", 20}, SH_C4[4] * (zz * (35 * zz - 30) + 3));
                    result.index_put_({"...", 21}, SH_C4[5] * xz * (7 * zz - 3));
                    result.index_put_({"...", 22}, SH_C4[6] * (xx - yy) * (7 * zz - 1));
                    result.index_put_({"...", 23}, SH_C4[7] * xz * (xx - 3 * yy));
                    result.index_put_({"...", 24}, SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)));
                        
                }
            }
        }             
    }
    
    return (result.index({"...", None}) * coeffs).sum(-2);
}