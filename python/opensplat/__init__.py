"""OpenSplat — Python bindings for 3D Gaussian Splatting training.

Public API (stable from v0.1.0):
    opensplat.train(...)
    opensplat.Trainer(...)
    opensplat.StepResult

Anything imported from opensplat._core is internal and subject to change.
"""
from opensplat._version import __version__

__all__ = ["__version__"]
