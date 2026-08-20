// Originally based on https://github.com/nerfstudio-project/gsplat
// This implementation has been substantially changed and optimized 
// Licensed under the AGPLv3
// Piero Toffanin - 2024

#include <cstdio>
#include <iostream>
#include <vector>
#include <math.h>
#include <tuple>
#include <torch/all.h>

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
);

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
);

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
    );

// Fused L1 + DSSIM loss over [H,W,C] float images (same contract as
// the GPU backends): stats[0] = loss, stats[1] = normalization denominator;
// partials holds the SSIM derivative maps (empty when want_grad is false).
std::tuple<torch::Tensor, torch::Tensor> fused_loss_forward_tensor_cpu(
    const torch::Tensor &rendered,
    const torch::Tensor &gt,
    const torch::Tensor &mask, // [H,W] float or empty
    const float ssim_weight,
    const bool valid_padding,
    const bool want_grad);

torch::Tensor fused_loss_backward_tensor_cpu(
    const torch::Tensor &rendered,
    const torch::Tensor &gt,
    const torch::Tensor &mask,
    const torch::Tensor &partials,
    const torch::Tensor &stats,
    const torch::Tensor &v_loss,
    const float ssim_weight,
    const bool valid_padding);

int numShBases(int degree);

torch::Tensor compute_sh_forward_tensor_cpu(
    const int num_points,
    const int degree,
    const int degrees_to_use,
    const torch::Tensor &viewdirs,
    const torch::Tensor &coeffs
);