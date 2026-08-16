#ifndef RASTERIZE_GAUSSIANS_H
#define RASTERIZE_GAUSSIANS_H

#include <torch/torch.h>
#include "tile_bounds.hpp"

using namespace torch::autograd;

#if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)

std::tuple<torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor> binAndSortGaussians(int numPoints, int numIntersects,
                                            torch::Tensor xys,
                                            torch::Tensor depths,
                                            torch::Tensor radii,
                                            torch::Tensor cumTilesHit,
                                            TileBounds tileBounds);


class RasterizeGaussians : public Function<RasterizeGaussians>{
public:
    static variable_list forward(AutogradContext *ctx,
            torch::Tensor xys,
            torch::Tensor depths,
            torch::Tensor radii,
            torch::Tensor conics,
            torch::Tensor numTilesHit,
            torch::Tensor colors,
            torch::Tensor opacity,
            int imgHeight,
            int imgWidth,
            torch::Tensor background,
            torch::Tensor errorMap = torch::Tensor(),
            torch::Tensor edgeMap = torch::Tensor(),
            torch::Tensor densificationInfo = torch::Tensor());
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

#endif

class RasterizeGaussiansCPU : public Function<RasterizeGaussiansCPU>{
public:
    static variable_list forward(AutogradContext *ctx,
            torch::Tensor xys,
            torch::Tensor radii,
            torch::Tensor conics,
            torch::Tensor colors,
            torch::Tensor opacity,
            torch::Tensor cov2d,
            torch::Tensor camDepths,
            int imgHeight,
            int imgWidth,
            torch::Tensor background,
            torch::Tensor errorMap = torch::Tensor(),
            torch::Tensor edgeMap = torch::Tensor(),
            torch::Tensor densificationInfo = torch::Tensor());
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

#endif