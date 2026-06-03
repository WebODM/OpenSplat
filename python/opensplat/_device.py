"""Device auto-detection — mirrors opensplat.cpp's selection logic."""
from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve the device kwarg into a concrete torch.device.

    Order, matching opensplat.cpp:
        explicit override > CUDA > MPS > CPU
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
