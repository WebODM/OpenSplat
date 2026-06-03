"""Shared pytest fixtures for the OpenSplat Python tests."""
from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="session")
def colmap_mini() -> Path:
    """Path to the tiny synthetic COLMAP project."""
    path = FIXTURES / "colmap_mini"
    assert (path / "sparse" / "0" / "cameras.bin").is_file(), (
        f"colmap_mini fixture missing - run tests/fixtures/_make_colmap.py "
        f"to regenerate ({path})"
    )
    return path


@pytest.fixture
def tmp_output_ply(tmp_path: Path) -> Path:
    """Per-test temporary path for an output .ply file."""
    return tmp_path / "out.ply"
