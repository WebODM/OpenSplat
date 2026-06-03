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


def test_get_cameras_validate_false(colmap_mini: Path) -> None:
    """validate=False: all cameras returned, val_cam is None."""
    from opensplat import _core
    data = _core.input_data_from_path(str(colmap_mini), "")
    cams, val_cam = data.get_cameras(validate=False)
    assert len(cams) == 8
    assert val_cam is None


def test_get_cameras_validate_true(colmap_mini: Path) -> None:
    """validate=True: 7 train cameras + 1 held-out val camera."""
    from opensplat import _core
    data = _core.input_data_from_path(str(colmap_mini), "")
    cams, val_cam = data.get_cameras(validate=True)
    assert len(cams) == 7
    assert val_cam is not None
    assert val_cam.width == 64
    assert val_cam.file_path.endswith(".png")


def test_input_data_points(colmap_mini: Path) -> None:
    """Loader populates the 3D point cloud."""
    from opensplat import _core
    data = _core.input_data_from_path(str(colmap_mini), "")
    pts = data.points
    assert pts.xyz.shape == (500, 3)
    assert pts.rgb.shape == (500, 3)
