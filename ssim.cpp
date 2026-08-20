// Fused (1-w)*L1 + w*DSSIM photometric loss with a closed-form
// backward

#include "ssim.hpp"
#include "gsplat.hpp"

namespace {

std::tuple<torch::Tensor, torch::Tensor> fusedLossForwardDispatch(
    const torch::Tensor &rendered, const torch::Tensor &gt, const torch::Tensor &mask,
    float ssimWeight, bool validPadding, bool wantGrad){
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_MPS)
    if (!rendered.is_cpu()){
        return fused_loss_forward_tensor(rendered, gt, mask, ssimWeight, validPadding, wantGrad);
    }
#endif
    return fused_loss_forward_tensor_cpu(rendered, gt, mask, ssimWeight, validPadding, wantGrad);
}

torch::Tensor fusedLossBackwardDispatch(
    const torch::Tensor &rendered, const torch::Tensor &gt, const torch::Tensor &mask,
    const torch::Tensor &partials, const torch::Tensor &stats, const torch::Tensor &vLoss,
    float ssimWeight, bool validPadding){
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_MPS)
    if (!rendered.is_cpu()){
        return fused_loss_backward_tensor(rendered, gt, mask, partials, stats, vLoss, ssimWeight, validPadding);
    }
#endif
    return fused_loss_backward_tensor_cpu(rendered, gt, mask, partials, stats, vLoss, ssimWeight, validPadding);
}

class FusedL1SsimLossFunction : public torch::autograd::Function<FusedL1SsimLossFunction>{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor rendered, // [H,W,C]
                                 torch::Tensor gt,       // [H,W,C]
                                 torch::Tensor mask,     // [H,W] or empty
                                 double ssimWeight, bool validPadding){
        auto r = fusedLossForwardDispatch(rendered.contiguous(), gt, mask,
                                          static_cast<float>(ssimWeight), validPadding, true);
        torch::Tensor stats = std::get<0>(r);
        ctx->save_for_backward({ rendered, gt, mask, std::get<1>(r), stats });
        ctx->saved_data["ssimWeight"] = ssimWeight;
        ctx->saved_data["validPadding"] = validPadding;
        return stats.index({0});
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list gradOutputs){
        torch::autograd::variable_list saved = ctx->get_saved_variables();
        torch::Tensor vRendered = fusedLossBackwardDispatch(
            saved[0], saved[1], saved[2], saved[3], saved[4],
            gradOutputs[0].contiguous(),
            static_cast<float>(ctx->saved_data["ssimWeight"].toDouble()),
            ctx->saved_data["validPadding"].toBool());
        torch::Tensor none;
        return { vRendered, none, none, none, none };
    }
};

}

torch::Tensor fusedL1SsimLoss(const torch::Tensor &rendered, const torch::Tensor &gt,
                              const torch::Tensor &mask, float ssimWeight, bool validPadding){
    return FusedL1SsimLossFunction::apply(rendered, gt, mask,
                                          static_cast<double>(ssimWeight), validPadding);
}

torch::Tensor fusedL1SsimLossValue(const torch::Tensor &rendered, const torch::Tensor &gt,
                                   float ssimWeight){
    torch::Tensor empty;
    auto r = fusedLossForwardDispatch(rendered.contiguous(), gt.contiguous(), empty,
                                      ssimWeight, false, false);
    return std::get<0>(r).index({0});
}
