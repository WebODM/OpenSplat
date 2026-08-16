#ifndef SSIM_H
#define SSIM_H

#include <torch/torch.h>

// Ported from https://github.com/Po-Hsun-Su/pytorch-ssim
// MIT

class SSIM{
public:
    SSIM(int windowSize, int channel) : windowSize(windowSize), channel(channel){
        createWindow();
    };

    torch::Tensor eval(const torch::Tensor& rendered, const torch::Tensor& gt);
    torch::Tensor map(const torch::Tensor& rendered, const torch::Tensor& gt);
    int getWindowSize() const { return windowSize; }
private:
    void createWindow();
    torch::Tensor gaussian(float sigma);

    int windowSize;
    int channel;
    torch::Tensor windowV;
    torch::Tensor windowH;
};


#endif