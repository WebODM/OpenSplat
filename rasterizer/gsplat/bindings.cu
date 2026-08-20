#include "backward.cuh"
#include "bindings.h"
#include "forward.cuh"
#include "helpers.cuh"
#include "sh.cuh"

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#else
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#endif

#include <cstdio>
#include <iostream>
#include <math.h>
#include <tuple>

namespace cg = cooperative_groups;

// Fused L1 + DSSIM loss over [H,W,C] images. One kernel computes the
// SSIM map and the closed-form partials in a shared-memory tile (two-pass
// separable 11-tap blur), one reduces to the scalar loss on-device, and one
// produces dL/dimage directly.

#define LOSS_BX 16
#define LOSS_BY 16
#define LOSS_HALO 5
#define LOSS_SX (LOSS_BX + 2 * LOSS_HALO)
#define LOSS_SY (LOSS_BY + 2 * LOSS_HALO)
#define LOSS_C1 0.0001f
#define LOSS_C2 0.0009f

// 11-tap gaussian (sigma 1.5), matches the SSIM reference window
__device__ __constant__ float lossGauss[11] = {
    0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f,
    0.10936068743467331f, 0.21300552785396576f, 0.26601171493530273f,
    0.21300552785396576f, 0.10936068743467331f, 0.036000773310661316f,
    0.0075987582094967365f, 0.001028380123898387f};

__device__ __forceinline__ float loss_pix(
    const float* __restrict__ img, int y, int x, int c, int H, int W, int C
){
    if (x < 0 || x >= W || y < 0 || y >= H) return 0.0f;
    return img[(y * W + x) * C + c];
}

// A pixel participates in the unmasked loss only away from the blur border
__device__ __forceinline__ bool loss_valid(int y, int x, int H, int W, bool validPad){
    if (!validPad || H <= 10 || W <= 10) return true;
    return x >= LOSS_HALO && x < W - LOSS_HALO && y >= LOSS_HALO && y < H - LOSS_HALO;
}

__global__ void fused_loss_fwd_kernel(
    const int H, const int W, const int C,
    const float* __restrict__ rendered,
    const float* __restrict__ gt,
    float* __restrict__ ssimMap, // [H,W] channel mean
    __half* __restrict__ pMu,    // [H,W,C] each, or nullptr
    __half* __restrict__ pS1,
    __half* __restrict__ pS12
){
    const int px = blockIdx.x * LOSS_BX + threadIdx.x;
    const int py = blockIdx.y * LOSS_BY + threadIdx.y;
    const int tileX = blockIdx.x * LOSS_BX;
    const int tileY = blockIdx.y * LOSS_BY;

    __shared__ float sTile[LOSS_SY][LOSS_SX][2];
    __shared__ float sConv[LOSS_SY][LOSS_BX][5];

    float ssimSum = 0.0f;
    for (int c = 0; c < C; c++){
        // Load tile + halo
        {
            const int tileSize = LOSS_SY * LOSS_SX;
            const int threads = LOSS_BX * LOSS_BY;
            const int tRank = threadIdx.y * LOSS_BX + threadIdx.x;
            for (int tid = tRank; tid < tileSize; tid += threads){
                const int ly = tid / LOSS_SX;
                const int lx = tid % LOSS_SX;
                const int gy = tileY + ly - LOSS_HALO;
                const int gx = tileX + lx - LOSS_HALO;
                sTile[ly][lx][0] = loss_pix(rendered, gy, gx, c, H, W, C);
                sTile[ly][lx][1] = loss_pix(gt, gy, gx, c, H, W, C);
            }
        }
        __syncthreads();

        // Horizontal pass: accumulate moments; each thread covers two rows
        {
            const int lx = threadIdx.x + LOSS_HALO;
            for (int pass = 0; pass < 2; pass++){
                const int ly = threadIdx.y + pass * LOSS_BY;
                if (ly >= LOSS_SY) break;
                float sX = 0.f, sX2 = 0.f, sY = 0.f, sY2 = 0.f, sXY = 0.f;
                #pragma unroll
                for (int d = -LOSS_HALO; d <= LOSS_HALO; d++){
                    const float w = lossGauss[LOSS_HALO + d];
                    const float x = sTile[ly][lx + d][0];
                    const float y = sTile[ly][lx + d][1];
                    sX += x * w;
                    sX2 += x * x * w;
                    sY += y * w;
                    sY2 += y * y * w;
                    sXY += x * y * w;
                }
                sConv[ly][threadIdx.x][0] = sX;
                sConv[ly][threadIdx.x][1] = sX2;
                sConv[ly][threadIdx.x][2] = sY;
                sConv[ly][threadIdx.x][3] = sY2;
                sConv[ly][threadIdx.x][4] = sXY;
            }
        }
        __syncthreads();

        // Vertical pass + SSIM + partials
        if (px < W && py < H){
            const int ly = threadIdx.y + LOSS_HALO;
            const int lx = threadIdx.x;
            float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f, m4 = 0.f;
            #pragma unroll
            for (int d = -LOSS_HALO; d <= LOSS_HALO; d++){
                const float w = lossGauss[LOSS_HALO + d];
                const float* row = sConv[ly + d][lx];
                m0 += row[0] * w;
                m1 += row[1] * w;
                m2 += row[2] * w;
                m3 += row[3] * w;
                m4 += row[4] * w;
            }
            const float muX = m0;
            const float muY = m2;
            const float sigmaX = m1 - muX * muX;
            const float sigmaY = m3 - muY * muY;
            const float sigmaXY = m4 - muX * muY;

            const float A = muX * muX + muY * muY + LOSS_C1;
            const float B = sigmaX + sigmaY + LOSS_C2;
            const float Cc = 2.f * muX * muY + LOSS_C1;
            const float Dc = 2.f * sigmaXY + LOSS_C2;
            const float s = (Cc * Dc) / (A * B);
            ssimSum += s;

            if (pMu){
                const int idx = (py * W + px) * C + c;
                const float dMu = (muY * 2.f * Dc) / (A * B) - (muY * 2.f * Cc) / (A * B)
                                - (muX * 2.f * Cc * Dc) / (A * A * B) + (muX * 2.f * Cc * Dc) / (A * B * B);
                pMu[idx] = __float2half(dMu);
                pS1[idx] = __float2half((-Cc * Dc) / (A * B * B));
                pS12[idx] = __float2half((2.f * Cc) / (A * B));
            }
        }
        __syncthreads();
    }

    if (px < W && py < H){
        ssimMap[py * W + px] = ssimSum / static_cast<float>(C);
    }
}

// Reduces the combined loss over pixels: out[0] += sum of gate * ((1-w)*sum_c|d_c| + C*w*(1-ssim)),
// out[1] += sum of gate. The gate is the mask value or the valid-padding indicator.
__global__ void fused_loss_reduce_kernel(
    const int H, const int W, const int C,
    const float* __restrict__ rendered,
    const float* __restrict__ gt,
    const float* __restrict__ ssimMap,
    const float* __restrict__ mask, // nullptr when unmasked
    const float ssimWeight,
    const bool validPad,
    float* __restrict__ out
){
    const int numPix = H * W;
    float lossSum = 0.0f;
    float gateSum = 0.0f;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < numPix; p += blockDim.x * gridDim.x){
        const int y = p / W;
        const int x = p % W;
        float gate;
        if (mask){
            gate = mask[p];
        }else{
            gate = loss_valid(y, x, H, W, validPad) ? 1.0f : 0.0f;
        }
        if (gate != 0.0f){
            float l1 = 0.0f;
            for (int c = 0; c < C; c++){
                l1 += fabsf(rendered[p * C + c] - gt[p * C + c]);
            }
            const float contrib = (1.0f - ssimWeight) * l1
                                + static_cast<float>(C) * ssimWeight * (1.0f - ssimMap[p]);
            lossSum += gate * contrib;
            gateSum += gate;
        }
    }

    __shared__ float sLoss[256];
    __shared__ float sGate[256];
    sLoss[threadIdx.x] = lossSum;
    sGate[threadIdx.x] = gateSum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            sLoss[threadIdx.x] += sLoss[threadIdx.x + stride];
            sGate[threadIdx.x] += sGate[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        atomicAdd(&out[0], sLoss[0]);
        atomicAdd(&out[1], sGate[0]);
    }
}

// out[0] = loss, out[1] = normalization denominator (gateSum * C)
__global__ void fused_loss_finalize_kernel(const int C, float* __restrict__ out){
    const float denom = out[1] * static_cast<float>(C) + 1e-8f;
    out[0] = out[0] / denom;
    out[1] = denom;
}

__global__ void fused_loss_bwd_kernel(
    const int H, const int W, const int C,
    const float ssimWeight,
    const bool validPad,
    const float* __restrict__ rendered,
    const float* __restrict__ gt,
    const float* __restrict__ mask, // nullptr when unmasked
    const __half* __restrict__ pMu,
    const __half* __restrict__ pS1,
    const __half* __restrict__ pS12,
    const float* __restrict__ stats, // stats[1] = denominator
    const float* __restrict__ vLoss,
    float* __restrict__ vRendered
){
    const int px = blockIdx.x * LOSS_BX + threadIdx.x;
    const int py = blockIdx.y * LOSS_BY + threadIdx.y;
    const int tileX = blockIdx.x * LOSS_BX;
    const int tileY = blockIdx.y * LOSS_BY;
    const float chainScale = vLoss[0] / stats[1];

    __shared__ float sData[LOSS_SY][LOSS_SX][3];
    __shared__ float sConv[LOSS_SY][LOSS_BX][3];

    for (int c = 0; c < C; c++){
        float p1 = 0.f, p2 = 0.f;
        if (px < W && py < H){
            p1 = rendered[(py * W + px) * C + c];
            p2 = gt[(py * W + px) * C + c];
        }

        // Load the chain-weighted partials for the tile + halo
        {
            const int tileSize = LOSS_SY * LOSS_SX;
            const int threads = LOSS_BX * LOSS_BY;
            const int tRank = threadIdx.y * LOSS_BX + threadIdx.x;
            for (int tid = tRank; tid < tileSize; tid += threads){
                const int ly = tid / LOSS_SX;
                const int lx = tid % LOSS_SX;
                const int gy = tileY + ly - LOSS_HALO;
                const int gx = tileX + lx - LOSS_HALO;
                float chain = 0.0f;
                if (gx >= 0 && gx < W && gy >= 0 && gy < H){
                    const float gate = mask ? mask[gy * W + gx]
                                            : (loss_valid(gy, gx, H, W, validPad) ? 1.0f : 0.0f);
                    chain = -ssimWeight * gate * chainScale;
                }
                const int idx = (gy * W + gx) * C + c;
                const bool inside = gx >= 0 && gx < W && gy >= 0 && gy < H;
                sData[ly][lx][0] = inside ? __half2float(pMu[idx]) * chain : 0.0f;
                sData[ly][lx][1] = inside ? __half2float(pS1[idx]) * chain : 0.0f;
                sData[ly][lx][2] = inside ? __half2float(pS12[idx]) * chain : 0.0f;
            }
        }
        __syncthreads();

        // Horizontal pass
        {
            const int lx = threadIdx.x + LOSS_HALO;
            for (int pass = 0; pass < 2; pass++){
                const int ly = threadIdx.y + pass * LOSS_BY;
                if (ly >= LOSS_SY) break;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f;
                #pragma unroll
                for (int d = -LOSS_HALO; d <= LOSS_HALO; d++){
                    const float w = lossGauss[LOSS_HALO + d];
                    a0 += sData[ly][lx + d][0] * w;
                    a1 += sData[ly][lx + d][1] * w;
                    a2 += sData[ly][lx + d][2] * w;
                }
                sConv[ly][threadIdx.x][0] = a0;
                sConv[ly][threadIdx.x][1] = a1;
                sConv[ly][threadIdx.x][2] = a2;
            }
        }
        __syncthreads();

        // Vertical pass + L1 term
        if (px < W && py < H){
            const int ly = threadIdx.y + LOSS_HALO;
            const int lx = threadIdx.x;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f;
            #pragma unroll
            for (int d = -LOSS_HALO; d <= LOSS_HALO; d++){
                const float w = lossGauss[LOSS_HALO + d];
                const float* row = sConv[ly + d][lx];
                s0 += row[0] * w;
                s1 += row[1] * w;
                s2 += row[2] * w;
            }
            const float gradSsim = s0 + 2.f * p1 * s1 + p2 * s2;

            const float gate = mask ? mask[py * W + px]
                                    : (loss_valid(py, px, H, W, validPad) ? 1.0f : 0.0f);
            const float sign = (p1 == p2) ? 0.0f : copysignf(1.0f, p1 - p2);
            const float gradL1 = (1.0f - ssimWeight) * sign * gate * chainScale;

            vRendered[(py * W + px) * C + c] = gradSsim + gradL1;
        }
        __syncthreads();
    }
}

std::tuple<torch::Tensor, torch::Tensor> fused_loss_forward_tensor(
    const torch::Tensor &rendered,
    const torch::Tensor &gt,
    const torch::Tensor &mask,
    const float ssim_weight,
    const bool valid_padding,
    const bool want_grad
){
    CHECK_INPUT(rendered);
    CHECK_INPUT(gt);
    const int H = rendered.size(0);
    const int W = rendered.size(1);
    const int C = rendered.size(2);
    const bool hasMask = mask.defined() && mask.numel() > 0;
    if (hasMask){ CHECK_INPUT(mask); }

    auto opts = rendered.options();
    torch::Tensor ssimMap = torch::empty({H, W}, opts);
    torch::Tensor partials = want_grad
        ? torch::empty({3, static_cast<long long>(H) * W * C}, opts.dtype(torch::kHalf))
        : torch::empty({0}, opts.dtype(torch::kHalf));
    __half* pBase = want_grad ? reinterpret_cast<__half*>(partials.data_ptr<at::Half>()) : nullptr;
    const long long planeSize = static_cast<long long>(H) * W * C;

    const dim3 block(LOSS_BX, LOSS_BY);
    const dim3 grid((W + LOSS_BX - 1) / LOSS_BX, (H + LOSS_BY - 1) / LOSS_BY);
    fused_loss_fwd_kernel<<<grid, block>>>(
        H, W, C,
        rendered.data_ptr<float>(), gt.data_ptr<float>(),
        ssimMap.data_ptr<float>(),
        pBase, pBase ? pBase + planeSize : nullptr, pBase ? pBase + 2 * planeSize : nullptr
    );

    torch::Tensor stats = torch::zeros({2}, opts);
    const int numPix = H * W;
    const int reduceBlocks = (std::min)(1024, (numPix + 255) / 256);
    fused_loss_reduce_kernel<<<reduceBlocks, 256>>>(
        H, W, C,
        rendered.data_ptr<float>(), gt.data_ptr<float>(),
        ssimMap.data_ptr<float>(),
        hasMask ? mask.data_ptr<float>() : nullptr,
        ssim_weight, valid_padding,
        stats.data_ptr<float>()
    );
    fused_loss_finalize_kernel<<<1, 1>>>(C, stats.data_ptr<float>());

    return std::make_tuple(stats, partials);
}

torch::Tensor fused_loss_backward_tensor(
    const torch::Tensor &rendered,
    const torch::Tensor &gt,
    const torch::Tensor &mask,
    const torch::Tensor &partials,
    const torch::Tensor &stats,
    const torch::Tensor &v_loss,
    const float ssim_weight,
    const bool valid_padding
){
    const int H = rendered.size(0);
    const int W = rendered.size(1);
    const int C = rendered.size(2);
    const bool hasMask = mask.defined() && mask.numel() > 0;

    torch::Tensor vRendered = torch::empty_like(rendered);
    const __half* pBase = reinterpret_cast<const __half*>(partials.data_ptr<at::Half>());
    const long long planeSize = static_cast<long long>(H) * W * C;

    const dim3 block(LOSS_BX, LOSS_BY);
    const dim3 grid((W + LOSS_BX - 1) / LOSS_BX, (H + LOSS_BY - 1) / LOSS_BY);
    fused_loss_bwd_kernel<<<grid, block>>>(
        H, W, C, ssim_weight, valid_padding,
        rendered.data_ptr<float>(), gt.data_ptr<float>(),
        hasMask ? mask.data_ptr<float>() : nullptr,
        pBase, pBase + planeSize, pBase + 2 * planeSize,
        stats.data_ptr<float>(),
        v_loss.data_ptr<float>(),
        vRendered.data_ptr<float>()
    );
    return vRendered;
}

__global__ void compute_cov2d_bounds_kernel(
    const unsigned num_pts, const float* __restrict__ covs2d, float* __restrict__ conics, float* __restrict__ radii
) {
    unsigned row = cg::this_grid().thread_rank();
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float radius;
    float3 cov2d{
        (float)covs2d[index], (float)covs2d[index + 1], (float)covs2d[index + 2]
    };
    compute_cov2d_bounds(cov2d, conic, radius);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &covs2d) {
    CHECK_INPUT(covs2d);
    torch::Tensor conics = torch::zeros(
        {num_pts, covs2d.size(1)}, covs2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor radii =
        torch::zeros({num_pts, 1}, covs2d.options().dtype(torch::kFloat32));

    int blocks = (num_pts + N_THREADS - 1) / N_THREADS;

    compute_cov2d_bounds_kernel<<<blocks, N_THREADS>>>(
        num_pts,
        covs2d.contiguous().data_ptr<float>(),
        conics.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<float>()
    );
    return std::make_tuple(conics, radii);
}

torch::Tensor compute_sh_forward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
) {
    unsigned num_bases = num_sh_bases(degree);
    if (coeffs.ndimension() != 3 || coeffs.size(0) != num_points ||
        coeffs.size(1) != num_bases || coeffs.size(2) != 3) {
        AT_ERROR("coeffs must have dimensions (N, D, 3)");
    }
    torch::Tensor colors = torch::empty({num_points, 3}, coeffs.options());
    compute_sh_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        coeffs.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>()
    );
    return colors;
}

torch::Tensor compute_sh_backward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
) {
    if (viewdirs.ndimension() != 2 || viewdirs.size(0) != num_points ||
        viewdirs.size(1) != 3) {
        AT_ERROR("viewdirs must have dimensions (N, 3)");
    }
    if (v_colors.ndimension() != 2 || v_colors.size(0) != num_points ||
        v_colors.size(1) != 3) {
        AT_ERROR("v_colors must have dimensions (N, 3)");
    }
    unsigned num_bases = num_sh_bases(degree);
    torch::Tensor v_coeffs =
        torch::zeros({num_points, num_bases, 3}, v_colors.options());
    compute_sh_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_coeffs.contiguous().data_ptr<float>()
    );
    return v_coeffs;
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
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
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    float4 intrins = {fx, fy, cx, cy};

    // Triangular covariance.
    torch::Tensor cov3d_d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));

    project_gaussians_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        projmat.contiguous().data_ptr<float>(),
        intrins,
        img_size_dim3,
        tile_bounds_dim3,
        clip_thresh,
        // Outputs.
        cov3d_d.contiguous().data_ptr<float>(),
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        cov3d_d, xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
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
    torch::Tensor &cov3d,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    float4 intrins = {fx, fy, cx, cy};

    const auto num_cov3d = num_points * 6;

    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_cov3d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean3d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_quat =
        torch::zeros({num_points, 4}, means3d.options().dtype(torch::kFloat32));

    project_gaussians_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        projmat.contiguous().data_ptr<float>(),
        intrins,
        img_size_dim3,
        cov3d.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        v_cov3d.contiguous().data_ptr<float>(),
        (float3 *)v_mean3d.contiguous().data_ptr<float>(),
        (float3 *)v_scale.contiguous().data_ptr<float>(),
        (float4 *)v_quat.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat);
}

std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        cum_tiles_hit.contiguous().data_ptr<int32_t>(),
        tile_bounds_dim3,
        // Outputs.
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects, const torch::Tensor &isect_ids_sorted
) {
    CHECK_INPUT(isect_ids_sorted);
    torch::Tensor tile_bins = torch::zeros(
        {num_intersects, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    return tile_bins;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    rasterize_forward<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)out_img.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, final_Ts, final_idx);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
nd_rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    nd_rasterize_forward<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        channels,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        out_img.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, final_Ts, final_idx);
}



std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    nd_rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha // dL_dout_alpha
    ) {

    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1
    };
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    torch::Tensor workspace;
    if (channels > 3) {
        workspace = torch::zeros(
            {img_height, img_width, channels},
            xys.options().dtype(torch::kFloat32)
        );
    } else {
        workspace = torch::zeros({0}, xys.options().dtype(torch::kFloat32));
    }

    nd_rasterize_backward_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        channels,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        v_output.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        workspace.data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha, // dL_dout_alpha
        const torch::Tensor &error_map, // [H,W] or empty
        const torch::Tensor &edge_map, // [H,W] or empty
        const torch::Tensor &densification_info, // [4,N] accumulated in place, or empty
        const torch::Tensor &v_xy_abs // [N,2] accumulated in place, or empty
    ) {

    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2 || colors.size(1) != 3) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);

    if (densification_info.numel() > 0){
        CHECK_INPUT(densification_info);
        if (densification_info.size(0) != 4 || densification_info.size(1) != num_points){
            AT_ERROR("densification_info must have dimensions (4, num_points)");
        }
    }
    if (v_xy_abs.numel() > 0){
        CHECK_INPUT(v_xy_abs);
        if (v_xy_abs.size(0) != num_points || v_xy_abs.size(1) != 2){
            AT_ERROR("v_xy_abs must have dimensions (num_points, 2)");
        }
    }
    if (error_map.numel() > 0){
        CHECK_INPUT(error_map);
        if (error_map.numel() != img_height * img_width){
            AT_ERROR("error_map must have img_height * img_width elements");
        }
    }
    const dim3 tile_bounds = {
        (img_width + BLOCK_X - 1) / BLOCK_X,
        (img_height + BLOCK_Y - 1) / BLOCK_Y,
        1
    };
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    rasterize_backward_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        num_points,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)v_output.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        error_map.numel() > 0 ? error_map.data_ptr<float>() : nullptr,
        edge_map.numel() > 0 ? edge_map.data_ptr<float>() : nullptr,
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        v_xy_abs.numel() > 0 ? (float2 *)v_xy_abs.data_ptr<float>() : nullptr,
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        densification_info.numel() > 0 ? densification_info.data_ptr<float>() : nullptr
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}