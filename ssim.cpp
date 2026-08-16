// Ported from https://github.com/Po-Hsun-Su/pytorch-ssim
// MIT
// Closed-form backward following the fused-ssim formulation.

#include "ssim.hpp"
#include "gsplat.hpp"

using namespace torch::indexing;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_MPS)
#define SSIM_HAS_FUSED_KERNELS 1
#else
#define SSIM_HAS_FUSED_KERNELS 0
#endif

#if defined(USE_MPS)
#define SSIM_IS_FUSED_DEVICE(t) ((t).is_mps())
#else
#define SSIM_IS_FUSED_DEVICE(t) ((t).is_cuda())
#endif

namespace {

// Batched separable gaussian blur over [N,1,H,W]
torch::Tensor blurBatched(const torch::Tensor &t, const torch::Tensor &wV, const torch::Tensor &wH, int windowSize){
    namespace F = torch::nn::functional;
    torch::Tensor v = F::conv2d(t, wV, F::Conv2dFuncOptions().padding({windowSize / 2, 0}));
    return F::conv2d(v, wH, F::Conv2dFuncOptions().padding({0, windowSize / 2}));
}

const float C1 = 0.01f * 0.01f;
const float C2 = 0.03f * 0.03f;

// S = (A1*A2)/(B1*B2); custom backward avoids replaying the conv graph
class SSIMMapFunction : public torch::autograd::Function<SSIMMapFunction>{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor rendered, // [C,1,H,W], needs grad
                                 torch::Tensor gt,       // [C,1,H,W], constant
                                 torch::Tensor wV, torch::Tensor wH,
                                 int64_t windowSize, int64_t channel){
        torch::Tensor x = gt;
        torch::Tensor y = rendered;
        torch::Tensor stacked;
#if SSIM_HAS_FUSED_KERNELS
        if (SSIM_IS_FUSED_DEVICE(rendered)){
            stacked = fused_ssim_stack_tensor(x.contiguous(), y.contiguous());
        }else
#endif
        {
            stacked = torch::cat({x, y, x * x, y * y, x * y}, 0);
        }
        torch::Tensor blurred = blurBatched(stacked, wV, wH, windowSize);
        auto parts = blurred.chunk(5, 0);
        torch::Tensor muX = parts[0];
        torch::Tensor muY = parts[1];

        torch::Tensor S, m1, m2, m3;
#if SSIM_HAS_FUSED_KERNELS
        if (SSIM_IS_FUSED_DEVICE(rendered)){
            torch::Tensor sigmaX = parts[2] - muX * muX;
            auto fused = fused_ssim_pointwise_fwd_tensor(muX.contiguous(), muY.contiguous(),
                                                         parts[3].contiguous(), parts[4].contiguous(),
                                                         sigmaX.contiguous());
            S = std::get<0>(fused);
            ctx->save_for_backward({ x, y, std::get<1>(fused), std::get<2>(fused), std::get<3>(fused), wV, wH });
            ctx->saved_data["windowSize"] = windowSize;
            ctx->saved_data["fused"] = true;
            return S.squeeze(1).mean(0);
        }
#endif
        torch::Tensor sigmaX = parts[2] - muX * muX;
        torch::Tensor sigmaY = parts[3] - muY * muY;
        torch::Tensor sigmaXY = parts[4] - muX * muY;

        torch::Tensor A1 = 2.0f * muX * muY + C1;
        torch::Tensor A2 = 2.0f * sigmaXY + C2;
        torch::Tensor B1 = muX * muX + muY * muY + C1;
        torch::Tensor B2 = sigmaX + sigmaY + C2;
        S = (A1 * A2) / (B1 * B2);

        ctx->save_for_backward({ x, y, muX, muY, A1, A2, B1, B2, S, wV, wH });
        ctx->saved_data["windowSize"] = windowSize;
        ctx->saved_data["fused"] = false;

        return S.squeeze(1).mean(0); // [H,W] channel mean
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list gradOutputs){
        torch::autograd::variable_list saved = ctx->get_saved_variables();
        torch::Tensor none;
        int windowSize = ctx->saved_data["windowSize"].toInt();

#if SSIM_HAS_FUSED_KERNELS
        if (ctx->saved_data["fused"].toBool()){
            torch::Tensor x = saved[0], y = saved[1];
            torch::Tensor m1 = saved[2], m2 = saved[3], m3 = saved[4];
            torch::Tensor wV = saved[5], wH = saved[6];
            int channel = x.size(0);
            torch::Tensor g = (gradOutputs[0] / static_cast<float>(channel)).contiguous();
            torch::Tensor stacked = fused_ssim_pointwise_bwd_pre_tensor(g, m1, m2, m3);
            torch::Tensor blurred = blurBatched(stacked, wV, wH, windowSize);
            auto parts = blurred.chunk(3, 0);
            torch::Tensor gradY = fused_ssim_pointwise_bwd_post_tensor(
                parts[0].contiguous(), parts[1].contiguous(), parts[2].contiguous(),
                x.contiguous(), y.contiguous());
            return { gradY, none, none, none, none, none };
        }
#endif
        torch::Tensor x = saved[0], y = saved[1], muX = saved[2], muY = saved[3];
        torch::Tensor A1 = saved[4], A2 = saved[5], B1 = saved[6], B2 = saved[7], S = saved[8];
        torch::Tensor wV = saved[9], wH = saved[10];
        int channel = x.size(0);

        // dL/dS per channel; output was the channel mean
        torch::Tensor g = (gradOutputs[0] / static_cast<float>(channel)).unsqueeze(0).unsqueeze(0);

        torch::Tensor dSdMuY = 2.0f * A2 * (muX * B1 - muY * A1) / (B1 * B1 * B2);
        torch::Tensor dSdSigmaY = -S / B2;
        torch::Tensor dSdSigmaXY = 2.0f * A1 / (B1 * B2);

        torch::Tensor stacked = torch::cat({
            g * (dSdMuY - 2.0f * muY * dSdSigmaY - muX * dSdSigmaXY),
            g * dSdSigmaY,
            g * dSdSigmaXY
        }, 0);
        torch::Tensor blurred = blurBatched(stacked, wV, wH, windowSize);
        auto parts = blurred.chunk(3, 0);
        torch::Tensor gradY = parts[0] + 2.0f * y * parts[1] + x * parts[2];

        return { gradY, none, none, none, none, none };
    }
};

}

torch::Tensor SSIM::eval(const torch::Tensor& rendered, const torch::Tensor& gt) {
    return map(rendered, gt).mean();
}

// Per-pixel channel-mean SSIM [H,W]
torch::Tensor SSIM::map(const torch::Tensor& rendered, const torch::Tensor& gt) {
    torch::Tensor img1 = gt.permute({2, 0, 1}).unsqueeze(1).contiguous();      // [C,1,H,W]
    torch::Tensor img2 = rendered.permute({2, 0, 1}).unsqueeze(1).contiguous();

    if (img1.device() != windowV.device()){
        windowV = windowV.to(img1.device());
        windowH = windowH.to(img1.device());
    }

    return SSIMMapFunction::apply(img2, img1, windowV, windowH, static_cast<int64_t>(windowSize), static_cast<int64_t>(channel));
}

void SSIM::createWindow(){
    torch::Tensor _1DWindow = gaussian(1.5f);
    windowV = _1DWindow.view({1, 1, windowSize, 1}).contiguous();
    windowH = _1DWindow.view({1, 1, 1, windowSize}).contiguous();
}

torch::Tensor SSIM::gaussian(float sigma) {
    torch::Tensor gauss = torch::zeros(windowSize);
    for (int i = 0; i < windowSize; i++) {
        gauss[i] = std::exp(-(std::pow(static_cast<float>(i - windowSize / 2), 2.0f)) / (2.0f * sigma * sigma));
    }
    return gauss / gauss.sum();
}
