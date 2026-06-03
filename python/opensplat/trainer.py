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
        import random
        self._rng = random.Random(42)

    def __iter__(self) -> "Trainer":
        return self

    def __next__(self) -> StepResult:
        if self._step >= self.num_iters:
            raise StopIteration
        cam = self._rng.choice(self._cameras)
        downscale = self._model.get_downscale_factor(self._step)
        cam.load_image(float(downscale))
        rendered = self._model.forward(cam, self._step)
        gt = cam.get_image(int(downscale))
        loss = self._model.main_loss(rendered, gt, self._kw.ssim_weight)
        self._model.optimizers_zero_grad()
        loss.backward()
        self._model.optimizers_step()
        self._model.schedulers_step(self._step)
        self._model.after_train(self._step)

        result = StepResult(
            step=self._step,
            loss=float(loss.item()),
            num_gaussians=int(self._model.means.size(0)),
        )
        self._step += 1
        return result

    def run(self) -> None:
        """Drive the iterator to completion. Equivalent to `for _ in self: pass`."""
        for _ in self:
            pass
