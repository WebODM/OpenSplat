#ifndef SSIM_H
#define SSIM_H

#include <torch/torch.h>

// Ported from https://github.com/Po-Hsun-Su/pytorch-ssim
// MIT

class SSIM{
public:
    SSIM(int windowSize, int channel) : windowSize(windowSize), channel(channel){
        window = createWindow();
    };

    // mask: optional (H, W) or (H, W, 1) tensor in [0, 1], same spatial size
    // as rendered/gt. Pixels outside the masked-in (white) region don't
    // contribute to the returned score -- an undefined (default-constructed)
    // mask behaves exactly as before (every pixel counted).
    torch::Tensor eval(const torch::Tensor& rendered, const torch::Tensor& gt,
                       const torch::Tensor& mask = torch::Tensor());
private:
    torch::Tensor createWindow();
    torch::Tensor gaussian(float sigma);

    int windowSize;
    int channel;
    torch::Tensor window;
};


#endif