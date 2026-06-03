"""Trainer — drives the per-step training loop on top of opensplat._core.Model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from opensplat import _core
from opensplat._device import resolve_device
from opensplat._kwargs import TrainerKwargs, validate


@dataclass(frozen=True)
class StepResult:
    """One training step's observable result. Yielded by Trainer iteration."""
    step: int
    loss: float
    num_gaussians: int


class Trainer:
    """Iterable trainer. See `opensplat.train()` for the one-shot convenience form."""

    def __init__(self, **kwargs: Any) -> None:
        # Construct kwargs dataclass — surfaces TypeError on unknown args.
        self._kw = TrainerKwargs(**kwargs)
        validate(self._kw)
        self.device = resolve_device(self._kw.device)
        self.num_iters = self._kw.num_iters

        self._input_data = _core.input_data_from_path(
            str(self._kw.input), self._kw.colmap_image_path
        )
        cams, val_cam = self._input_data.get_cameras(
            self._kw.val, self._kw.val_image,
        )
        self._cameras = cams
        self._val_cam = val_cam

        self._model = _core.Model(
            self._input_data, len(cams),
            self._kw.num_downscales,
            self._kw.resolution_schedule,
            self._kw.sh_degree,
            self._kw.sh_degree_interval,
            self._kw.refine_every,
            self._kw.warmup_length,
            self._kw.reset_alpha_every,
            self._kw.densify_grad_thresh,
            self._kw.densify_size_thresh,
            self._kw.stop_screen_size_at,
            self._kw.split_screen_size,
            self._kw.num_iters,
            self._kw.keep_crs,
            self.device,
        )
        self._step = 0
