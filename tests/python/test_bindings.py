"""Tier 1 — binding smoke tests for opensplat._core."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_input_data_from_path_loads_colmap(colmap_mini: Path) -> None:
    from opensplat import _core
    data = _core.input_data_from_path(str(colmap_mini), "")
    assert len(data.cameras) == 8
    assert data.scale > 0


def test_camera_fields(colmap_mini: Path) -> None:
    from opensplat import _core
    data = _core.input_data_from_path(str(colmap_mini), "")
    cam = data.cameras[0]
    assert cam.width == 64
    assert cam.height == 64
    assert cam.fx == pytest.approx(60.0)
    assert cam.fy == pytest.approx(60.0)
    assert cam.file_path.endswith(".png")
