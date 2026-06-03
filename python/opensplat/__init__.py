"""OpenSplat — Python bindings for 3D Gaussian Splatting training.

Public API (stable from v0.1.0):
    opensplat.train(...)
    opensplat.Trainer(...)
    opensplat.StepResult

Anything imported from opensplat._core is internal and subject to change.
"""
import torch as _torch  # noqa: F401 — must be imported before _core so libtorch is resolved
from opensplat._version import __version__
from opensplat.api import train
from opensplat.trainer import Trainer, StepResult

__all__ = ["__version__", "train", "Trainer", "StepResult"]
