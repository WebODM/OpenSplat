// Ported from https://github.com/Po-Hsun-Su/pytorch-ssim
// MIT

#include "ssim.hpp"

using namespace torch::indexing;

torch::Tensor SSIM::eval(const torch::Tensor& rendered, const torch::Tensor& gt, const torch::Tensor& mask) {
    torch::Tensor img1 = gt.permute({2, 0, 1}).index({None, "..."});
    torch::Tensor img2 = rendered.permute({2, 0, 1}).index({None, "..."});

    if (img1.device() != window.device()){
        window = window.to(img1.device());
    }
    torch::Tensor mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(windowSize / 2).groups(channel));
    torch::Tensor mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(windowSize / 2).groups(channel));

    torch::Tensor mu1Sq = mu1.pow(2);
    torch::Tensor mu2Sq = mu2.pow(2);
    torch::Tensor mu1mu2 = mu1 * mu2;

    torch::Tensor sigma1Sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(windowSize / 2).groups(channel)) - mu1Sq;
    torch::Tensor sigma2Sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(windowSize / 2).groups(channel)) - mu2Sq;
    torch::Tensor sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(windowSize / 2).groups(channel)) - mu1mu2;

    const float C1 = 0.01 * 0.01;
    const float C2 = 0.03 * 0.03;

    torch::Tensor ssimMap = ((2.0f * mu1mu2 + C1) * (2.0f * sigma12 + C2)) / ((mu1Sq + mu2Sq + C1) * (sigma1Sq + sigma2Sq + C2));

    if (mask.numel() > 0){
        // ssimMap is [1, C, H, W], mask is [H, W, 1]
        // Permute mask to [1, 1, H, W] and expand to match ssimMap channels
        torch::Tensor ssimMask = mask.permute({2, 0, 1}).index({None, "..."});  // [1, 1, H, W]
        ssimMask = ssimMask.expand_as(ssimMap);  // [1, C, H, W]
        torch::Tensor maskedSsim = ssimMap * ssimMask;
        return maskedSsim.sum() / (ssimMask.sum() + 1e-8f);
    }

    return ssimMap.mean();
}

torch::Tensor SSIM::createWindow(){
    torch::Tensor _1DWindow = gaussian(1.5f).unsqueeze(1);
    torch::Tensor _2DWindow = _1DWindow.mm(_1DWindow.t()).unsqueeze(0).unsqueeze(0);
    return _2DWindow.expand({channel, 1, windowSize, windowSize}).contiguous();
}

torch::Tensor SSIM::gaussian(float sigma) {
    torch::Tensor gauss = torch::zeros(windowSize);
    for (int i = 0; i < windowSize; i++) {
        gauss[i] = std::exp(-(std::pow(std::floor(static_cast<float>(i - windowSize) / 2.0f), 2.0f)) / (2.0f * sigma * sigma));
    }
    return gauss / gauss.sum();
}