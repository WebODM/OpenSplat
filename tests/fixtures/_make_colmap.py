"""Generate the colmap_mini fixture.

Run this once: python tests/fixtures/_make_colmap.py

Commits a deterministic ~8-camera ring + 500-point synthetic COLMAP project
under tests/fixtures/colmap_mini/ for use in binding/E2E tests.

Note: the OpenSplat C++ COLMAP loader (colmap.cpp / point_io.cpp) only
supports the *binary* COLMAP format (cameras.bin / images.bin /
points3D.bin), so this script emits the binary variant rather than .txt.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def _lookat_world_to_cam_quat_t(
    cam_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Build a COLMAP world-to-camera (R, t) pair for a camera at ``cam_pos``
    looking at ``target``, then return it as (qw, qx, qy, qz), (tx, ty, tz).

    COLMAP camera convention: +X right, +Y down, +Z forward.
    """
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    # Right = forward x up_hint (then we re-derive a consistent up).
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # +Y points down in COLMAP camera frame
    # Rows of R_w2c are the camera basis vectors expressed in world coords.
    R_w2c = np.stack([right, down, forward], axis=0)
    t = -R_w2c @ cam_pos

    # Rotation matrix -> quaternion (qw, qx, qy, qz), standard formula.
    m = R_w2c
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    return (qw, qx, qy, qz), (float(t[0]), float(t[1]), float(t[2]))

ROOT = Path(__file__).parent / "colmap_mini"
NUM_CAMERAS = 8
IMAGE_W, IMAGE_H = 64, 64
FOCAL = 60.0
NUM_POINTS = 500
RADIUS = 3.0

# Camera model IDs (see colmap.hpp CameraModel enum / official COLMAP docs):
#   SimplePinhole = 0, Pinhole = 1, SimpleRadial = 2, ...
PINHOLE_MODEL_ID = 1


def write_cameras_bin(path: Path) -> None:
    """One shared PINHOLE camera (fx, fy, cx, cy)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))  # num cameras
        f.write(struct.pack("<I", 1))  # camera_id
        f.write(struct.pack("<i", PINHOLE_MODEL_ID))  # model
        f.write(struct.pack("<Q", IMAGE_W))
        f.write(struct.pack("<Q", IMAGE_H))
        # PINHOLE params: fx, fy, cx, cy (4 doubles)
        f.write(struct.pack("<dddd", FOCAL, FOCAL, IMAGE_W / 2.0, IMAGE_H / 2.0))


def write_images_bin(path: Path) -> list[str]:
    """Ring of cameras facing the origin. Returns the per-image filenames."""
    names: list[str] = []
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", NUM_CAMERAS))  # num images
        for i in range(NUM_CAMERAS):
            angle = 2 * math.pi * i / NUM_CAMERAS
            cam_pos = np.array(
                [RADIUS * math.cos(angle), 0.0, RADIUS * math.sin(angle)],
                dtype=np.float64,
            )
            # Look-at origin with world-up = +Y. Produces a proper
            # world-to-camera (R, t) consistent with COLMAP convention.
            (qw, qx, qy, qz), (tx, ty, tz) = _lookat_world_to_cam_quat_t(
                cam_pos,
                np.zeros(3, dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            )

            name = f"frame_{i:03d}.png"
            names.append(name)

            f.write(struct.pack("<I", i + 1))  # image_id
            f.write(struct.pack("<dddd", qw, qx, qy, qz))
            f.write(struct.pack("<ddd", tx, ty, tz))
            f.write(struct.pack("<I", 1))  # camera_id
            f.write(name.encode("utf-8") + b"\x00")  # null-terminated name
            # No 2D point observations.
            f.write(struct.pack("<Q", 0))
    return names


def write_points3d_bin(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", points.shape[0]))
        for pid, (xyz, rgb) in enumerate(zip(points, colors), start=1):
            f.write(struct.pack("<Q", pid))
            f.write(struct.pack("<ddd", float(xyz[0]), float(xyz[1]), float(xyz[2])))
            f.write(struct.pack("<BBB", int(rgb[0]), int(rgb[1]), int(rgb[2])))
            f.write(struct.pack("<d", 0.5))  # reprojection error
            f.write(struct.pack("<Q", 0))  # empty track


def main() -> None:
    sparse = ROOT / "sparse" / "0"
    images_dir = ROOT / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=0)
    points = rng.uniform(-0.5, 0.5, size=(NUM_POINTS, 3))
    colors = rng.integers(0, 256, size=(NUM_POINTS, 3), dtype=np.uint8)

    write_cameras_bin(sparse / "cameras.bin")
    names = write_images_bin(sparse / "images.bin")
    write_points3d_bin(sparse / "points3D.bin", points, colors)

    for i, name in enumerate(names):
        # Solid-color image, distinct per camera so tests can tell them apart.
        color = ((i * 31) % 256, (i * 67) % 256, (i * 113) % 256)
        Image.new("RGB", (IMAGE_W, IMAGE_H), color).save(images_dir / name)

    print(f"Wrote {NUM_CAMERAS} images + {NUM_POINTS} points to {ROOT}")


if __name__ == "__main__":
    main()
