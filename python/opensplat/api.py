"""Top-level convenience function. Equivalent to `Trainer(**kwargs).run()`."""
from __future__ import annotations

from typing import Any

from opensplat.trainer import Trainer


def train(**kwargs: Any) -> None:
    """Train a 3D Gaussian Splatting scene from a project directory.

    Mirrors the opensplat CLI. Required: input=. Most users also want output=.

    See opensplat._kwargs.TrainerKwargs for the full kwarg list and defaults.
    """
    Trainer(**kwargs).run()
