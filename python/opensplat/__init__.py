"""OpenSplat — Python bindings for 3D Gaussian Splatting training.

Public API (stable from v0.1.0):
    opensplat.train(...)
    opensplat.Trainer(...)
    opensplat.StepResult

Anything imported from opensplat._core is internal and subject to change.
"""
# NOTE: when a future change imports anything from opensplat._core here,
# add `import torch as _torch  # noqa: F401` ABOVE it — libtorch must be
# loaded into the process before the C-extension's first dlopen, otherwise
# the @rpath references inside _core resolve before torch's libs are mapped.
from opensplat._version import __version__

__all__ = ["__version__"]
