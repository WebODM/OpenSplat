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


def test_model_constructs_from_input_data(colmap_mini: Path) -> None:
    import torch
    from opensplat import _core
    data = _core.input_data_from_path(str(colmap_mini), "")
    model = _core.Model(
        data, len(data.cameras),
        2,        # num_downscales
        3000,     # resolution_schedule
        3,        # sh_degree
        1000,     # sh_degree_interval
        100,      # refine_every
        500,      # warmup_length
        30,       # reset_alpha_every
        0.0002,   # densify_grad_thresh
        0.01,     # densify_size_thresh
        4000,     # stop_screen_size_at
        0.05,     # split_screen_size
        200,      # max_steps
        False,    # keep_crs
        torch.device("cpu"),
    )
    # 500 points in fixture; means must match
    assert model.means.shape == (500, 3)
    assert model.scales.shape == (500, 3)
    assert model.quats.shape == (500, 4)
    assert model.opacities.shape == (500, 1)
    assert model.features_dc.shape == (500, 3)
    # features_rest shape depends on sh_degree: (N, (sh+1)^2 - 1, 3)
    assert model.features_rest.shape == (500, (3 + 1) ** 2 - 1, 3)


def test_model_one_training_step(colmap_mini: Path, tmp_output_ply: Path) -> None:
    """One full training step: forward, loss, backward, optimizer step, save."""
    import torch
    from opensplat import _core

    data = _core.input_data_from_path(str(colmap_mini), "")
    cams, _val = data.get_cameras(False, "random")
    model = _core.Model(
        data, len(cams), 2, 3000, 3, 1000, 100, 500, 30,
        0.0002, 0.01, 4000, 0.05, 200, False,
        torch.device("cpu"),
    )
    cam = cams[0]
    downscale = model.get_downscale_factor(0)
    cam.load_image(downscale)

    rgb = model.forward(cam, 0)
    assert rgb.dim() == 3 and rgb.shape[2] == 3  # H, W, 3
    gt = cam.get_image(downscale)
    loss = model.main_loss(rgb, gt, 0.2)
    assert float(loss.item()) > 0

    model.optimizers_zero_grad()
    loss.backward()
    model.optimizers_step()
    model.schedulers_step(0)
    model.after_train(0)

    model.save_ply(str(tmp_output_ply), 1)
    assert tmp_output_ply.is_file() and tmp_output_ply.stat().st_size > 0
