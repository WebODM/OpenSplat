"""Trainer — drives the per-step training loop on top of opensplat._core.Model."""
from __future__ import annotations

import random
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


class _InfiniteShuffleIterator:
    """No-replacement infinite iterator: emits each item exactly once per pass.

    Mirrors the C++ ``InfiniteRandomIterator`` (utils.hpp): seed=42, Fisher-Yates
    shuffle per pass, draw sequentially, reshuffle when exhausted.
    """

    def __init__(self, items, seed: int = 42) -> None:
        self._items = list(items)
        self._rng = random.Random(seed)
        self._queue: list = []
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._queue = self._items.copy()
        self._rng.shuffle(self._queue)
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._items:
            raise StopIteration
        item = self._queue[self._idx]
        self._idx += 1
        if self._idx >= len(self._queue):
            self._reshuffle()
        return item


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

        # CLI parity: opensplat.cpp loads ALL cameras once with the base
        # downscale_factor before partitioning out the validation cam. The
        # per-step scheduler downscale is layered on top via Camera.get_image
        # (which caches its own image pyramid). Camera::loadImage is destructive
        # and is documented as call-once.
        base_downscale = max(float(self._kw.downscale_factor), 1.0)
        for cam in self._input_data.cameras:
            cam.load_image(base_downscale)

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
        # CLI parity: opensplat.cpp uses 1-indexed steps (1..numIters).
        self._step = 1
        self._cam_iter = _InfiniteShuffleIterator(self._cameras)

    def __iter__(self):
        try:
            while self._step <= self.num_iters:
                cam = next(self._cam_iter)
                downscale = self._model.get_downscale_factor(self._step)
                rendered = self._model.forward(cam, self._step)
                gt = cam.get_image(int(downscale))
                loss = self._model.main_loss(rendered, gt, self._kw.ssim_weight)
                self._model.optimizers_zero_grad()
                loss.backward()
                self._model.optimizers_step()
                self._model.schedulers_step(self._step)
                self._model.after_train(self._step)

                yield StepResult(
                    step=self._step,
                    loss=float(loss.item()),
                    num_gaussians=int(self._model.means.size(0)),
                )

                # Mid-training periodic save.
                if (self._kw.save_every > 0
                    and self._step % self._kw.save_every == 0
                    and self._kw.output is not None):
                    self._model.save(str(self._kw.output), self._step)

                self._step += 1
        finally:
            if self._kw.output is not None:
                self._model.save(str(self._kw.output), self._step)

    def run(self) -> None:
        """Drive the iterator to completion. Equivalent to `for _ in self: pass`."""
        for _ in self:
            pass
