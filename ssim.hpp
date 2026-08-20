#ifndef SSIM_H
#define SSIM_H

#include <torch/torch.h>

// Fused (1-ssimWeight)*L1 + ssimWeight*DSSIM loss with autograd support.
// mask may be an empty tensor; validPadding crops the blur border from the
// unmasked loss
torch::Tensor fusedL1SsimLoss(const torch::Tensor &rendered, const torch::Tensor &gt,
                              const torch::Tensor &mask, float ssimWeight, bool validPadding);

// Loss value only
torch::Tensor fusedL1SsimLossValue(const torch::Tensor &rendered, const torch::Tensor &gt,
                                   float ssimWeight);

#endif
