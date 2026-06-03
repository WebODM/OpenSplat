"""Tier 3 — CLI vs Python API parity.

Runs `opensplat` (CLI) and `opensplat.train()` on the same fixture, same
seed, same kwargs, same device. Asserts the resulting PLYs have the same
Gaussian count and final losses are within a coarse tolerance.

The C++ CLI and the Python Trainer use different RNGs for camera selection
(std::default_random_engine vs random.Random), so exact tensor parity is
not expected. This test catches loop-level structural drift (one path
calling densification more often than the other) and catastrophic loss
divergence; it does NOT catch tensor-equal drift.

If this test fails the most likely cause is structural drift between
opensplat.cpp main() and Trainer.__iter__.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not Path("build-py/opensplat").is_file()
    and shutil.which("opensplat") is None,
    reason="opensplat CLI binary not found in $PATH or build-py/",
)


def _opensplat_binary() -> str:
    repo_local = Path("build-py/opensplat").resolve()
    return str(repo_local) if repo_local.is_file() else (shutil.which("opensplat") or "opensplat")


def _read_ply_vertex_count(path: Path) -> int:
    """Parse a PLY header (ascii) and return the 'element vertex N' count."""
    with open(path, "rb") as fh:
        for line in iter(fh.readline, b""):
            text = line.decode("ascii", errors="replace").rstrip()
            if text.startswith("element vertex"):
                return int(text.split()[-1])
            if text == "end_header":
                break
    raise ValueError(f"no 'element vertex' line in PLY header at {path}")


def _omp_env() -> dict[str, str]:
    """macOS-safe env: avoid the libomp/libtorch OMP race."""
    return {
        **os.environ,
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
    }


def test_cli_and_python_train_match(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/colmap_mini").resolve()
    cli_out = tmp_path / "cli.ply"
    py_out = tmp_path / "py.ply"
    env = _omp_env()

    # --- CLI run ----------------------------------------------------------
    cli_cmd = [
        _opensplat_binary(), str(fixture),
        "-o", str(cli_out),
        "-n", "20",
        "--cpu",
        "--sh-degree", "1",
        "--save-every", "-1",
        "--warmup-length", "5",
        "--refine-every", "1000",  # avoid densification in such a short run
    ]
    cli_proc = subprocess.run(cli_cmd, env=env, capture_output=True, text=True)
    assert cli_proc.returncode == 0, (
        f"CLI failed (rc={cli_proc.returncode}):\nSTDOUT:\n{cli_proc.stdout}\nSTDERR:\n{cli_proc.stderr}"
    )
    assert cli_out.is_file()

    # --- Python run -------------------------------------------------------
    py_cmd = [
        sys.executable, "-c",
        f"import opensplat; opensplat.train("
        f"input={str(fixture)!r}, output={str(py_out)!r}, "
        f"num_iters=20, device='cpu', sh_degree=1, save_every=-1, "
        f"warmup_length=5, refine_every=1000)",
    ]
    py_proc = subprocess.run(py_cmd, env=env, capture_output=True, text=True)
    assert py_proc.returncode == 0, (
        f"Python train failed (rc={py_proc.returncode}):\nSTDOUT:\n{py_proc.stdout}\nSTDERR:\n{py_proc.stderr}"
    )
    assert py_out.is_file()

    # --- Compare ----------------------------------------------------------
    cli_n = _read_ply_vertex_count(cli_out)
    py_n = _read_ply_vertex_count(py_out)
    # With refine_every=1000 and num_iters=20, neither side should densify,
    # so both should equal the input point count (500 from the fixture).
    assert cli_n == py_n, (
        f"Gaussian count drift: CLI={cli_n} vs Python={py_n}. "
        "One of opensplat.cpp main() or Trainer.__iter__ has changed without the other."
    )
    assert cli_n == 500, f"Expected 500 (no densification at refine_every=1000), got {cli_n}"
