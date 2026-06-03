"""Canonical kwarg defaults — mirrors opensplat.cpp CLI flag defaults exactly."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerKwargs:
    """All knobs exposed by the Python API. Field names are the kwarg names."""
    input: str
    output: str | None = None
    num_iters: int = 30000
    save_every: int = -1                # CLI: --save-every  (default -1 = disabled)
    device: str | None = None
    downscale_factor: float = 1.0       # CLI: --downscale-factor
    num_downscales: int = 2
    resolution_schedule: int = 3000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_weight: float = 0.2
    refine_every: int = 100
    warmup_length: int = 500
    reset_alpha_every: int = 30
    densify_grad_thresh: float = 0.0002
    densify_size_thresh: float = 0.01
    stop_screen_size_at: int = 4000
    split_screen_size: float = 0.05
    keep_crs: bool = False
    val: bool = False
    val_image: str = "random"
    val_render: str | None = None
    colmap_image_path: str = ""


def validate(k: TrainerKwargs) -> None:
    """Raise ValueError on conflicting or impossible kwargs.

    Permissive checks only — bad enum values get caught by libtorch / loaders.
    """
    if k.num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {k.num_iters}")
    if k.val_render is not None and not k.val:
        raise ValueError(
            "val_render is set but val=False; the CLI implicitly enables --val "
            "when --val-render is given. Pass val=True explicitly."
        )
    if k.output is not None:
        ext = k.output.rsplit(".", 1)[-1].lower() if "." in k.output else ""
        if ext not in ("ply", "splat"):
            raise ValueError(f"output extension must be .ply or .splat, got '{k.output}'")
