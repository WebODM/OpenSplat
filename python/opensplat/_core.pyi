"""Type stubs for opensplat._core (internal C-extension)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class Camera:
    id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    file_path: str
    cam_to_world: torch.Tensor

    def load_image(self, downscale_factor: float) -> None: ...
    def get_image(self, downscale_factor: int) -> torch.Tensor: ...


class Points:
    xyz: torch.Tensor
    rgb: torch.Tensor


class InputData:
    cameras: List[Camera]
    scale: float
    translation: torch.Tensor
    points: Points

    def get_cameras(
        self, validate: bool = False, val_image: str = "random"
    ) -> Tuple[List[Camera], Optional[Camera]]: ...
    def save_cameras(self, filename: str, keep_crs: bool) -> None: ...


class Model:
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    opacities: torch.Tensor

    def __init__(
        self,
        input_data: InputData,
        num_cameras: int,
        num_downscales: int,
        resolution_schedule: int,
        sh_degree: int,
        sh_degree_interval: int,
        refine_every: int,
        warmup_length: int,
        reset_alpha_every: int,
        densify_grad_thresh: float,
        densify_size_thresh: float,
        stop_screen_size_at: int,
        split_screen_size: float,
        max_steps: int,
        keep_crs: bool,
        device: torch.device,
    ) -> None: ...

    def forward(self, cam: Camera, step: int) -> torch.Tensor: ...
    def main_loss(self, rendered: torch.Tensor, gt: torch.Tensor, ssim_weight: float) -> torch.Tensor: ...
    def optimizers_zero_grad(self) -> None: ...
    def optimizers_step(self) -> None: ...
    def schedulers_step(self, step: int) -> None: ...
    def get_downscale_factor(self, step: int) -> int: ...
    def after_train(self, step: int) -> None: ...
    def save(self, filename: str, step: int) -> None: ...
    def save_ply(self, filename: str, step: int) -> None: ...
    def save_splat(self, filename: str) -> None: ...
    def load_ply(self, filename: str) -> int: ...


def input_data_from_path(project_root: str, colmap_image_source_path: str = "") -> InputData: ...
