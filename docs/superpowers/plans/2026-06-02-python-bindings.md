# OpenSplat Python Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pip install opensplat` that wraps the existing C++ training pipeline behind a Python `train()` function and `Trainer` iterator, with prebuilt wheels for Linux+CUDA, Linux+CPU, macOS+Metal, Windows+CUDA.

**Architecture:** Refactor existing C++ sources into a SHARED `libopensplat` library, leave the `opensplat` CLI executable behavior unchanged, add a pybind11 C-extension `opensplat._core` that wraps `Model` / `InputData` / `Camera` / input loaders, and write the per-step training loop in Python inside a `Trainer` class. See spec `docs/superpowers/specs/2026-06-02-python-bindings-design.md` for the full design.

**Tech Stack:** C++17, CMake 3.21, libtorch (existing), pybind11 ≥2.13, scikit-build-core ≥0.10, cibuildwheel, pytest.

---

## Spec Corrections

Three CLI defaults in the spec's kwargs table don't match `opensplat.cpp`. Use the corrected values from this plan; the spec will be updated as part of Task 13.

| Kwarg | Spec said | Truth (`opensplat.cpp`) |
|---|---|---|
| `save_every` | `7000` | `-1` (disabled) |
| `downscale_factor` (missing) | — | `1.0`, CLI `--downscale-factor` |
| `colmap_image_path` (missing) | — | `""`, CLI `--colmap-image-path` |

Also: in the CLI, `val=True` is implied if `val_render` is non-empty. The Python API keeps these flat kwargs — if `val_render` is set but `val=False`, raise `ValueError` per the spec's error-handling section. Document this in `Trainer.__init__` docstring.

---

## File Structure

Files this plan creates or modifies. Each is sized to one clear responsibility — they should mostly stay short.

**Created:**

```
pyproject.toml                         # PEP 517 build config (scikit-build-core)
.github/workflows/wheels.yml           # cibuildwheel matrix
python/bindings.cpp                    # pybind11 module — wraps Model, InputData, Camera, loaders
python/opensplat/__init__.py           # public re-exports: train, Trainer, StepResult
python/opensplat/_core.pyi             # type stubs for the C-extension
python/opensplat/_device.py            # device auto-detect helper
python/opensplat/_kwargs.py            # kwarg defaults + validation
python/opensplat/trainer.py            # Trainer class — the for-loop
python/opensplat/api.py                # train() convenience function
python/opensplat/_version.py           # __version__
tests/fixtures/colmap_mini/            # tiny COLMAP project (script-generated)
tests/fixtures/nerfstudio_mini/        # tiny nerfstudio project (script-generated)
tests/fixtures/_make_colmap.py         # one-shot generator for colmap_mini
tests/fixtures/_make_nerfstudio.py     # one-shot generator for nerfstudio_mini
tests/python/conftest.py               # pytest fixtures (paths, tmp dirs)
tests/python/test_bindings.py          # Tier 1: binding smoke tests
tests/python/test_training.py          # Tier 2: E2E training
tests/python/test_cli_parity.py        # Tier 3: CLI vs Python parity
```

**Modified:**

```
CMakeLists.txt                         # add libopensplat SHARED + OPENSPLAT_BUILD_PYTHON_BINDINGS
```

`opensplat.cpp`, `model.{hpp,cpp}`, `input_data.{hpp,cpp}`, and all other existing C++ sources are NOT modified. They just get compiled into the shared library instead of the executable.

---

## Task 1: Refactor build into libopensplat SHARED + opensplat exe

Pull all existing C++ sources (except `opensplat.cpp`) into a new SHARED library target. The `opensplat` executable becomes a thin link against it. Behavior of the executable must not change.

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Establish a baseline.** Build today's main and capture the executable behavior so we can verify nothing changed.

```bash
cd /path/to/OpenSplat
# Set OPENCV_DIR + Torch_DIR per your existing local setup. Example below uses CPU build.
cmake -S . -B build-baseline -DGPU_RUNTIME=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build-baseline -j
./build-baseline/opensplat --version | tee /tmp/opensplat-version-before.txt
```

Expected: prints the version string defined by `VERSION` + git rev. Save the output for comparison after the refactor.

- [ ] **Step 2: Edit `CMakeLists.txt` — replace the existing `add_executable(opensplat ...)` block.**

Find this block (currently around lines 254–295):

```cmake
set(OPENSPLAT_SRC_FILES opensplat.cpp point_io.cpp nerfstudio.cpp model.cpp
kdtree_tensor.cpp spherical_harmonics.cpp cv_utils.cpp utils.cpp project_gaussians.cpp
rasterize_gaussians.cpp ssim.cpp optim_scheduler.cpp colmap.cpp opensfm.cpp openmvg.cpp input_data.cpp
tensor_math.cpp)

if (OPENSPLAT_BUILD_VISUALIZER)
    if (Pangolin_FOUND)
        message(STATUS "Found Pangolin. Building visualizer (beta)")
        list(APPEND OPENSPLAT_SRC_FILES visualizer.cpp)
        add_definitions(-DUSE_VISUALIZATION)
    else()
        message(FATAL "Pangolin not found. Cannot build visualizer (beta)")
    endif()
endif()

add_executable(opensplat ${OPENSPLAT_SRC_FILES})

install(TARGETS opensplat DESTINATION bin)
set_property(TARGET opensplat PROPERTY CXX_STANDARD 17)
target_include_directories(opensplat PRIVATE
    ${PROJECT_SOURCE_DIR}/rasterizer
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(opensplat PUBLIC ${STDPPFS_LIBRARY} ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
if (Pangolin_FOUND)
    target_link_libraries(opensplat PUBLIC ${Pangolin_LIBRARIES})
endif()
target_link_libraries(opensplat PRIVATE
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    nanoflann::nanoflann
)
if (NOT WIN32)
    target_link_libraries(opensplat PUBLIC pthread)
endif()
if(GPU_RUNTIME STREQUAL "HIP")
    target_compile_definitions(opensplat PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
elseif(GPU_RUNTIME STREQUAL "CUDA")
    target_compile_definitions(opensplat PRIVATE USE_CUDA)
elseif(GPU_RUNTIME STREQUAL "MPS")
    target_compile_definitions(opensplat PRIVATE USE_MPS)
endif()
```

Replace it with the following. The split is: every existing source except `opensplat.cpp` (and the optional `visualizer.cpp`) becomes part of `libopensplat`; the executable target just builds `opensplat.cpp` and links the library.

```cmake
# Source files that make up the core library (everything except the CLI entry point)
set(LIBOPENSPLAT_SRC_FILES
    point_io.cpp nerfstudio.cpp model.cpp
    kdtree_tensor.cpp spherical_harmonics.cpp cv_utils.cpp utils.cpp project_gaussians.cpp
    rasterize_gaussians.cpp ssim.cpp optim_scheduler.cpp colmap.cpp opensfm.cpp openmvg.cpp input_data.cpp
    tensor_math.cpp
)

if (OPENSPLAT_BUILD_VISUALIZER)
    if (Pangolin_FOUND)
        message(STATUS "Found Pangolin. Building visualizer (beta)")
        list(APPEND LIBOPENSPLAT_SRC_FILES visualizer.cpp)
        add_definitions(-DUSE_VISUALIZATION)
    else()
        message(FATAL "Pangolin not found. Cannot build visualizer (beta)")
    endif()
endif()

# Core library — SHARED so both the CLI and the future Python C-extension can link it
add_library(opensplat_core SHARED ${LIBOPENSPLAT_SRC_FILES})
set_target_properties(opensplat_core PROPERTIES
    OUTPUT_NAME "opensplat"
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(opensplat_core PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/rasterizer
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(opensplat_core PUBLIC ${STDPPFS_LIBRARY} ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
if (Pangolin_FOUND)
    target_link_libraries(opensplat_core PUBLIC ${Pangolin_LIBRARIES})
endif()
target_link_libraries(opensplat_core PUBLIC
    nlohmann_json::nlohmann_json
    nanoflann::nanoflann
)
if (NOT WIN32)
    target_link_libraries(opensplat_core PUBLIC pthread)
endif()
if(GPU_RUNTIME STREQUAL "HIP")
    target_compile_definitions(opensplat_core PUBLIC USE_HIP __HIP_PLATFORM_AMD__)
elseif(GPU_RUNTIME STREQUAL "CUDA")
    target_compile_definitions(opensplat_core PUBLIC USE_CUDA)
elseif(GPU_RUNTIME STREQUAL "MPS")
    target_compile_definitions(opensplat_core PUBLIC USE_MPS)
endif()

# CLI executable — just the entry point, links the core library
add_executable(opensplat opensplat.cpp)
set_property(TARGET opensplat PROPERTY CXX_STANDARD 17)
target_link_libraries(opensplat PRIVATE opensplat_core cxxopts::cxxopts)
install(TARGETS opensplat DESTINATION bin)
install(TARGETS opensplat_core LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
```

Notes on the changes:
- `cxxopts` is only used inside `opensplat.cpp` (the CLI parser), so it stays linked to the executable, not the library.
- `nlohmann_json` and `nanoflann` are used by core sources (input loaders, kdtree) so they move to the library as PUBLIC dependencies.
- The Windows DLL-copy block at the bottom of `CMakeLists.txt` (`if (MSVC) file(GLOB TORCH_DLLS ...)`) still copies into `$<TARGET_FILE_DIR:opensplat>` — leave it as-is; it works for the executable's directory and the shared lib lands beside it.

- [ ] **Step 3: Rebuild and verify CLI behavior is identical.**

```bash
rm -rf build-refactored
cmake -S . -B build-refactored -DGPU_RUNTIME=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build-refactored -j
./build-refactored/opensplat --version | tee /tmp/opensplat-version-after.txt
diff /tmp/opensplat-version-before.txt /tmp/opensplat-version-after.txt
```

Expected: `diff` produces no output (identical).

- [ ] **Step 4: Verify the shared library exists and the executable links to it.**

```bash
ls build-refactored/libopensplat.*       # macOS: .dylib   Linux: .so   (Windows: opensplat.dll in bin/)
file build-refactored/opensplat          # confirms it's an executable
# on linux/macOS, confirm the link:
otool -L build-refactored/opensplat 2>/dev/null || ldd build-refactored/opensplat
# Expected: line referencing libopensplat.dylib / libopensplat.so
```

Expected: a `libopensplat.*` file exists; `otool`/`ldd` shows it among the executable's dependencies.

- [ ] **Step 5: Smoke-run the CLI's help and make sure it still parses options.**

```bash
./build-refactored/opensplat --help | head -30
```

Expected: cxxopts help output identical to before — same flags, same defaults.

- [ ] **Step 6: Commit.**

```bash
git add CMakeLists.txt
git commit -m "build: extract libopensplat shared library from CLI executable

Prepares the codebase for Python bindings by giving us a shared library
both the existing opensplat CLI and a future pybind11 C-extension can
link against. No CLI behavior change."
```

---

## Task 2: Add OPENSPLAT_BUILD_PYTHON_BINDINGS option + empty `_core` module

Add the CMake plumbing for the pybind11 C-extension. Produces a buildable but empty Python module `opensplat._core` so we can verify the toolchain before writing any bindings.

**Files:**
- Modify: `CMakeLists.txt`
- Create: `python/bindings.cpp`

- [ ] **Step 1: Append the pybind11 + extension target block to `CMakeLists.txt`** (at the very end of the file, before the `add_compile_definitions(GLOG_USE_GLOG_EXPORT)` line — or after it, position doesn't matter):

```cmake
# ---------------------------------------------------------------------------
# Python bindings (opensplat._core C-extension)
# ---------------------------------------------------------------------------
option(OPENSPLAT_BUILD_PYTHON_BINDINGS "Build Python bindings (opensplat._core)" OFF)

if(OPENSPLAT_BUILD_PYTHON_BINDINGS)
    # Locate the Python interpreter scikit-build-core / the build env points us at.
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

    if(FETCH_DEPENDENCIES)
        FetchContent_Declare(pybind11
            URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
        )
        FetchContent_MakeAvailable(pybind11)
    else()
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    pybind11_add_module(_core MODULE python/bindings.cpp)
    set_target_properties(_core PROPERTIES
        CXX_STANDARD 17
        # Output location matches the package layout so editable installs work.
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/opensplat"
    )
    target_link_libraries(_core PRIVATE opensplat_core)
    target_include_directories(_core PRIVATE ${PROJECT_SOURCE_DIR})

    # Install rule for scikit-build-core: place the .so/.pyd inside the opensplat package.
    install(TARGETS _core LIBRARY DESTINATION opensplat)
endif()
```

- [ ] **Step 2: Create the minimal `python/bindings.cpp`.** Empty module body — just defines the symbol so import works.

```cpp
// python/bindings.cpp
//
// pybind11 module exposing OpenSplat's C++ training surface to Python.
// Wraps Model, InputData, Camera, and the input-format loaders. The Python
// package opensplat (python/opensplat/) writes the training loop on top.

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "OpenSplat C++ bindings (internal). Use the opensplat package, not _core, directly.";
}
```

- [ ] **Step 3: Configure with bindings enabled and build.**

```bash
rm -rf build-py
cmake -S . -B build-py -DGPU_RUNTIME=CPU -DCMAKE_BUILD_TYPE=Release \
    -DOPENSPLAT_BUILD_PYTHON_BINDINGS=ON \
    -DPython_EXECUTABLE=$(which python3)
cmake --build build-py -j
```

Expected: succeeds. Produces `build-py/python/opensplat/_core.*.so` (Linux/macOS) or `_core.*.pyd` (Windows).

- [ ] **Step 4: Verify the extension is importable.**

```bash
PYTHONPATH="$PWD/build-py/python" python3 -c "from opensplat import _core; print(_core.__doc__)"
```

Expected (the doc string from `bindings.cpp`):
```
OpenSplat C++ bindings (internal). Use the opensplat package, not _core, directly.
```

If this fails with a libtorch / dyld error: the issue is rpath — verify that `_core.*.so` was linked against the `libopensplat.*` we built (use `otool -L` / `ldd`). Add `set_target_properties(_core PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}")` if needed.

- [ ] **Step 5: Confirm the CLI still builds with the option ON.**

```bash
./build-py/opensplat --version
```

Expected: prints the version. (We turned on the Python option; the executable should still build.)

- [ ] **Step 6: Commit.**

```bash
git add CMakeLists.txt python/bindings.cpp
git commit -m "build: add OPENSPLAT_BUILD_PYTHON_BINDINGS option + empty _core module

Wires up pybind11 (fetched at configure time) and produces an empty
opensplat._core C-extension. Verifies the toolchain end-to-end before
we start adding actual bindings."
```

---

## Task 3: pyproject.toml + Python package skeleton + editable install

Create the `pyproject.toml` and bare `opensplat/` Python package so `pip install -e .` produces an importable (still empty) module.

**Files:**
- Create: `pyproject.toml`
- Create: `python/opensplat/__init__.py`
- Create: `python/opensplat/_version.py`

- [ ] **Step 1: Write `pyproject.toml`** at the repo root.

```toml
[build-system]
requires = [
    "scikit-build-core>=0.10",
    "pybind11>=2.13",
    "torch",
]
build-backend = "scikit_build_core.build"

[project]
name = "opensplat"
description = "Python bindings for OpenSplat — open-source 3D Gaussian Splatting"
readme = "README.md"
license = { file = "LICENSE.md" }
requires-python = ">=3.10"
dynamic = ["version"]
authors = [{ name = "OpenSplat contributors" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Affero General Public License v3",
    "Programming Language :: C++",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Image Processing",
]
dependencies = [
    "torch",         # exact lower bound pinned at release time
    "numpy",
]

[project.urls]
Homepage = "https://github.com/pierotofy/OpenSplat"
Issues = "https://github.com/pierotofy/OpenSplat/issues"

[tool.scikit-build]
cmake.version = ">=3.21"
cmake.args = ["-DOPENSPLAT_BUILD_PYTHON_BINDINGS=ON"]
wheel.packages = ["python/opensplat"]
wheel.py-api = "cp310"  # placeholder; cibuildwheel overrides per-version

[tool.scikit-build.metadata.version]
provider = "scikit_build_core.metadata.regex"
input = "python/opensplat/_version.py"
regex = '^__version__ = "(?P<value>[^"]+)"$'

[tool.pytest.ini_options]
testpaths = ["tests/python"]
```

- [ ] **Step 2: Write `python/opensplat/_version.py`.**

```python
__version__ = "0.1.0.dev0"
```

- [ ] **Step 3: Write a placeholder `python/opensplat/__init__.py`.**

```python
"""OpenSplat — Python bindings for 3D Gaussian Splatting training.

Public API (stable from v0.1.0):
    opensplat.train(...)
    opensplat.Trainer(...)
    opensplat.StepResult

Anything imported from opensplat._core is internal and subject to change.
"""
from opensplat._version import __version__

__all__ = ["__version__"]
```

- [ ] **Step 4: Editable install + smoke test.** In a fresh virtual environment with `torch` already installed:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e . --no-build-isolation -v
python3 -c "import opensplat; print(opensplat.__version__)"
```

Expected: prints `0.1.0.dev0`. `--no-build-isolation` is important — it lets the build see the system `torch` so `find_package(Torch REQUIRED)` works. Production users will use isolated builds with `torch` in `[build-system].requires`; for development this is faster.

- [ ] **Step 5: Verify `_core` is importable through the editable install.**

```bash
python3 -c "from opensplat import _core; print(_core.__doc__)"
```

Expected: same doc string as Task 2.

- [ ] **Step 6: Commit.**

```bash
git add pyproject.toml python/opensplat/__init__.py python/opensplat/_version.py
git commit -m "build: add pyproject.toml + opensplat package skeleton

Bare pip-installable package via scikit-build-core. Compiles the _core
C-extension at install time; public surface still empty."
```

---

## Task 4: Test fixture — tiny COLMAP project

Generate a minimal COLMAP project under `tests/fixtures/colmap_mini/` that the binding tests can load. Use a generator script (committed alongside the fixture) so the data is reproducible.

**Files:**
- Create: `tests/fixtures/_make_colmap.py`
- Create: `tests/fixtures/colmap_mini/*` (generated)
- Create: `tests/python/conftest.py`

- [ ] **Step 1: Inspect what `colmap.cpp` expects.** Read `colmap.cpp` lines that parse `cameras.bin` / `images.bin` / `points3D.bin` (or the `.txt` equivalents) — confirm the format we're emitting.

```bash
grep -nE "cameras\.(bin|txt)|images\.(bin|txt)|points3D\.(bin|txt)" colmap.cpp | head -20
```

Expected: code path that reads either `sparse/0/*.bin` or `sparse/0/*.txt`. The `.txt` format is simpler to emit — use it.

- [ ] **Step 2: Write `tests/fixtures/_make_colmap.py`.** Generates 8 cameras in a circle around a 4-point unit cube, with 64×64 PNG images and a `sparse/0/{cameras,images,points3D}.txt` directory. Run once, output committed.

```python
"""Generate the colmap_mini fixture.

Run this once: python tests/fixtures/_make_colmap.py
Commits a deterministic ~8-camera ring + 500-point synthetic COLMAP project
under tests/fixtures/colmap_mini/ for use in binding/E2E tests.
"""
from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent / "colmap_mini"
NUM_CAMERAS = 8
IMAGE_W, IMAGE_H = 64, 64
FOCAL = 60.0
NUM_POINTS = 500
RADIUS = 3.0


def main() -> None:
    sparse = ROOT / "sparse" / "0"
    images_dir = ROOT / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=0)
    points = rng.uniform(-0.5, 0.5, size=(NUM_POINTS, 3))
    colors = rng.integers(0, 256, size=(NUM_POINTS, 3), dtype=np.uint8)

    # cameras.txt: one shared PINHOLE camera
    (sparse / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {IMAGE_W} {IMAGE_H} {FOCAL} {FOCAL} {IMAGE_W/2} {IMAGE_H/2}\n"
    )

    # images.txt: ring of cameras facing origin
    img_lines = [
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
    ]
    for i in range(NUM_CAMERAS):
        angle = 2 * math.pi * i / NUM_CAMERAS
        cx, cy, cz = RADIUS * math.cos(angle), 0.0, RADIUS * math.sin(angle)
        # Camera looks at origin; world-to-camera rotation is "look-at -cam_pos".
        # For a synthetic fixture we only need *some* valid orientation, not perfect aim.
        # Use identity quaternion (qw=1) and translation = -cam_pos (rough placeholder).
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        tx, ty, tz = -cx, -cy, -cz
        name = f"frame_{i:03d}.png"
        img_lines.append(
            f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {name}\n"
            "\n"  # empty POINTS2D line
        )
        # Solid-color image — distinct per camera so tests can tell them apart.
        color = ((i * 31) % 256, (i * 67) % 256, (i * 113) % 256)
        Image.new("RGB", (IMAGE_W, IMAGE_H), color).save(images_dir / name)
    (sparse / "images.txt").write_text("".join(img_lines))

    # points3D.txt
    pts_lines = [
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
    ]
    for pid, (xyz, rgb) in enumerate(zip(points, colors), start=1):
        pts_lines.append(
            f"{pid} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
            f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0.5\n"
        )
    (sparse / "points3D.txt").write_text("".join(pts_lines))

    print(f"Wrote {NUM_CAMERAS} images + {NUM_POINTS} points to {ROOT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the generator and commit the resulting fixture.**

```bash
python3 tests/fixtures/_make_colmap.py
ls tests/fixtures/colmap_mini/
ls tests/fixtures/colmap_mini/sparse/0/
ls tests/fixtures/colmap_mini/images/ | head -3
du -sh tests/fixtures/colmap_mini/
```

Expected: ~50 KB total. Top-level contains `images/` and `sparse/`. `images/` has 8 PNGs. `sparse/0/` has `cameras.txt`, `images.txt`, `points3D.txt`.

- [ ] **Step 4: Write `tests/python/conftest.py`** — pytest fixtures pointing at the test data.

```python
"""Shared pytest fixtures for the OpenSplat Python tests."""
from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="session")
def colmap_mini() -> Path:
    """Path to the tiny synthetic COLMAP project."""
    path = FIXTURES / "colmap_mini"
    assert (path / "sparse" / "0" / "cameras.txt").is_file(), \
        f"colmap_mini fixture missing — run tests/fixtures/_make_colmap.py to regenerate ({path})"
    return path


@pytest.fixture
def tmp_output_ply(tmp_path: Path) -> Path:
    """Per-test temporary path for an output .ply file."""
    return tmp_path / "out.ply"
```

- [ ] **Step 5: Sanity-check the fixture loads via the C++ CLI** (proves COLMAP loader recognizes it).

```bash
./build-py/opensplat tests/fixtures/colmap_mini -n 5 -o /tmp/sanity.ply 2>&1 | tail -20
```

Expected: training kicks off, runs 5 iterations, writes `/tmp/sanity.ply`. If the loader complains about the fixture's structure, fix `_make_colmap.py` and re-run before continuing — the fixture has to work end-to-end with the existing C++ pipeline.

- [ ] **Step 6: Commit.**

```bash
git add tests/fixtures/_make_colmap.py tests/fixtures/colmap_mini tests/python/conftest.py
git commit -m "test: add tiny synthetic COLMAP fixture for binding/E2E tests

8 cameras + 500 points + 64x64 images, generator script committed alongside.
Total fixture size ~50 KB."
```

---

## Task 5: Bind `InputData`, `Camera`, and `input_data_from_path` dispatcher

First real bindings. These three come together because every binding test needs a loaded `InputData` to inspect.

**Files:**
- Modify: `python/bindings.cpp`
- Create: `tests/python/test_bindings.py`

- [ ] **Step 1: Write the failing test.** Add to `tests/python/test_bindings.py`:

```python
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
```

- [ ] **Step 2: Run, confirm it fails.**

```bash
python3 -m pytest tests/python/test_bindings.py -v
```

Expected: `AttributeError: module 'opensplat._core' has no attribute 'input_data_from_path'`.

- [ ] **Step 3: Replace `python/bindings.cpp`** with the version that wraps `Camera`, `InputData`, and the dispatcher:

```cpp
// python/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "input_data.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "OpenSplat C++ bindings (internal). Use the opensplat package, not _core, directly.";

    py::class_<Camera>(m, "Camera")
        .def_readonly("id",        &Camera::id)
        .def_readonly("width",     &Camera::width)
        .def_readonly("height",    &Camera::height)
        .def_readonly("fx",        &Camera::fx)
        .def_readonly("fy",        &Camera::fy)
        .def_readonly("cx",        &Camera::cx)
        .def_readonly("cy",        &Camera::cy)
        .def_readonly("k1",        &Camera::k1)
        .def_readonly("k2",        &Camera::k2)
        .def_readonly("k3",        &Camera::k3)
        .def_readonly("p1",        &Camera::p1)
        .def_readonly("p2",        &Camera::p2)
        .def_readonly("file_path", &Camera::filePath)
        .def_readonly("cam_to_world", &Camera::camToWorld)
        .def("load_image", &Camera::loadImage, py::arg("downscale_factor"),
             py::call_guard<py::gil_scoped_release>())
        .def("get_image",  &Camera::getImage,  py::arg("downscale_factor"),
             py::call_guard<py::gil_scoped_release>());

    py::class_<Points>(m, "Points")
        .def_readonly("xyz", &Points::xyz)
        .def_readonly("rgb", &Points::rgb);

    py::class_<InputData>(m, "InputData")
        .def_readonly("cameras",     &InputData::cameras)
        .def_readonly("scale",       &InputData::scale)
        .def_readonly("translation", &InputData::translation)
        .def_readonly("points",      &InputData::points)
        .def("get_cameras", &InputData::getCameras,
             py::arg("validate"), py::arg("val_image") = std::string("random"),
             py::call_guard<py::gil_scoped_release>());

    m.def("input_data_from_path", &inputDataFromX,
          py::arg("project_root"), py::arg("colmap_image_source_path") = std::string(""),
          py::call_guard<py::gil_scoped_release>(),
          "Detect input format from project_root and load it.");
}
```

Notes:
- `torch/extension.h` makes pybind11 understand `torch::Tensor` ↔ `torch.Tensor` directly. No need to write conversion code.
- `py::call_guard<py::gil_scoped_release>()` releases the GIL during heavy work (image loading, file parsing). Cheap operations like accessors don't need it.
- `Camera::camToWorld` is a `torch::Tensor` — pybind11 hands it to Python as a `torch.Tensor` zero-copy.

- [ ] **Step 4: Rebuild and re-run the test.**

```bash
cmake --build build-py -j
python3 -m pytest tests/python/test_bindings.py -v
```

Expected: both tests PASS.

If you get an `ImportError` mentioning `c10` or `torch_cpu`: the C-extension can't find libtorch at runtime. The fix is to ensure `torch.utils.cpp_extension` symbols resolve — typically `import torch` before `import opensplat._core` works because `torch`'s `__init__.py` already loads the libs. If we need to be defensive, prepend `import torch` to `python/opensplat/__init__.py`. Add it now:

```python
# python/opensplat/__init__.py — UPDATE the existing file
"""OpenSplat — Python bindings for 3D Gaussian Splatting training."""
import torch as _torch  # noqa: F401 — must be imported before _core so libtorch is resolved
from opensplat._version import __version__

__all__ = ["__version__"]
```

- [ ] **Step 5: Commit.**

```bash
git add python/bindings.cpp python/opensplat/__init__.py tests/python/test_bindings.py
git commit -m "feat(py): bind InputData, Camera, and input_data_from_path

Two binding-smoke tests pass: loading the colmap_mini fixture, reading
camera fields. Uses torch/extension.h for zero-copy Tensor exchange."
```

---

## Task 6: Bind `Model` — constructor + tensor properties

The largest binding. Wraps `Model`'s constructor (16 parameters) and its read-only tensor fields.

**Files:**
- Modify: `python/bindings.cpp`
- Modify: `tests/python/test_bindings.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/python/test_bindings.py`:

```python
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
    # featuresRest shape depends on sh_degree: (N, (sh+1)^2 - 1, 3)
    assert model.features_rest.shape == (500, (3 + 1) ** 2 - 1, 3)
```

- [ ] **Step 2: Run, confirm it fails.**

```bash
python3 -m pytest tests/python/test_bindings.py::test_model_constructs_from_input_data -v
```

Expected: `AttributeError: module 'opensplat._core' has no attribute 'Model'`.

- [ ] **Step 3: Edit `python/bindings.cpp`.** Add `#include "model.hpp"` at the top, and add the `Model` binding block before the module's closing brace:

```cpp
#include "model.hpp"
```

```cpp
    py::class_<Model>(m, "Model")
        .def(py::init<const InputData&, int, int, int, int, int,
                      int, int, int, float, float, int, float,
                      int, bool, const torch::Device&>(),
             py::arg("input_data"),
             py::arg("num_cameras"),
             py::arg("num_downscales"),
             py::arg("resolution_schedule"),
             py::arg("sh_degree"),
             py::arg("sh_degree_interval"),
             py::arg("refine_every"),
             py::arg("warmup_length"),
             py::arg("reset_alpha_every"),
             py::arg("densify_grad_thresh"),
             py::arg("densify_size_thresh"),
             py::arg("stop_screen_size_at"),
             py::arg("split_screen_size"),
             py::arg("max_steps"),
             py::arg("keep_crs"),
             py::arg("device"))
        .def_readonly("means",         &Model::means)
        .def_readonly("scales",        &Model::scales)
        .def_readonly("quats",         &Model::quats)
        .def_readonly("features_dc",   &Model::featuresDc)
        .def_readonly("features_rest", &Model::featuresRest)
        .def_readonly("opacities",     &Model::opacities);
```

- [ ] **Step 4: Rebuild and re-run.**

```bash
cmake --build build-py -j
python3 -m pytest tests/python/test_bindings.py::test_model_constructs_from_input_data -v
```

Expected: PASS.

If construction fails with a CUDA error: the test passes a CPU device, but `Model`'s constructor may try CUDA-specific initialization. Look at `model.hpp:32-56` — the `.to(device)` calls should respect the CPU device. If they don't, that's a bug in the existing code, not the binding; the workaround is to skip this test in CPU mode and add it back once we have a CUDA fixture.

- [ ] **Step 5: Commit.**

```bash
git add python/bindings.cpp tests/python/test_bindings.py
git commit -m "feat(py): bind Model constructor and read-only tensor properties

Constructs from the colmap_mini fixture on CPU device; tensor shapes
verified for sh_degree=3, 500-point input."
```

---

## Task 7: Bind `Model` training-step methods + `save` / `load`

Wraps the methods needed to run one training iteration plus PLY/Splat persistence.

**Files:**
- Modify: `python/bindings.cpp`
- Modify: `tests/python/test_bindings.py`

- [ ] **Step 1: Write the failing test.** Append:

```python
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
```

- [ ] **Step 2: Run, confirm it fails.**

```bash
python3 -m pytest tests/python/test_bindings.py::test_model_one_training_step -v
```

Expected: `AttributeError` on the first method called (`get_downscale_factor`).

- [ ] **Step 3: Extend the `Model` binding in `python/bindings.cpp`.** Add these `.def` lines into the existing `py::class_<Model>` block (after the existing tensor-property bindings):

```cpp
        .def("forward", &Model::forward,
             py::arg("cam"), py::arg("step"),
             py::call_guard<py::gil_scoped_release>())
        .def("main_loss", &Model::mainLoss,
             py::arg("rendered"), py::arg("gt"), py::arg("ssim_weight"))
        .def("optimizers_zero_grad", &Model::optimizersZeroGrad,
             py::call_guard<py::gil_scoped_release>())
        .def("optimizers_step", &Model::optimizersStep,
             py::call_guard<py::gil_scoped_release>())
        .def("schedulers_step", &Model::schedulersStep, py::arg("step"))
        .def("get_downscale_factor", &Model::getDownscaleFactor, py::arg("step"))
        .def("after_train", &Model::afterTrain, py::arg("step"),
             py::call_guard<py::gil_scoped_release>())
        .def("save", &Model::save,
             py::arg("filename"), py::arg("step"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_ply", &Model::savePly,
             py::arg("filename"), py::arg("step"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_splat", &Model::saveSplat,
             py::arg("filename"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_ply", &Model::loadPly,
             py::arg("filename"),
             py::call_guard<py::gil_scoped_release>());
```

- [ ] **Step 4: Rebuild and run the new test.**

```bash
cmake --build build-py -j
python3 -m pytest tests/python/test_bindings.py::test_model_one_training_step -v
```

Expected: PASS. PLY file written.

- [ ] **Step 5: Run the whole binding test suite to make sure nothing regressed.**

```bash
python3 -m pytest tests/python/test_bindings.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit.**

```bash
git add python/bindings.cpp tests/python/test_bindings.py
git commit -m "feat(py): bind Model training-step methods + save/load

Full forward + backward + optimizer step + save_ply round-trips on CPU
device. GIL released around all heavy methods."
```

---

## Task 8: `StepResult` dataclass and `Trainer.__init__`

Start the Python-side training loop. This task just gets the `Trainer` constructed; the iteration logic is the next task.

**Files:**
- Create: `python/opensplat/_device.py`
- Create: `python/opensplat/_kwargs.py`
- Create: `python/opensplat/trainer.py`
- Create: `tests/python/test_training.py`

- [ ] **Step 1: Write the failing test.** Create `tests/python/test_training.py`:

```python
"""Tier 2 — end-to-end training tests."""
from __future__ import annotations

from pathlib import Path


def test_trainer_constructs(colmap_mini: Path) -> None:
    from opensplat import Trainer
    trainer = Trainer(input=str(colmap_mini), num_iters=10, save_every=-1)
    assert trainer.num_iters == 10
    # device resolution: in CI we expect CPU
    assert str(trainer.device) in ("cpu", "cuda:0", "mps")
```

- [ ] **Step 2: Run, confirm it fails.**

```bash
python3 -m pytest tests/python/test_training.py::test_trainer_constructs -v
```

Expected: `ImportError: cannot import name 'Trainer' from 'opensplat'`.

- [ ] **Step 3: Write `python/opensplat/_device.py`.**

```python
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
```

- [ ] **Step 4: Write `python/opensplat/_kwargs.py`** — the canonical list of kwargs and their CLI defaults (corrected per the "Spec Corrections" section at the top of this plan).

```python
"""Canonical kwarg defaults — mirrors opensplat.cpp CLI flag defaults exactly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainerKwargs:
    """All knobs exposed by the Python API. Field names are the kwarg names."""
    input: str
    output: str | None = None
    num_iters: int = 30000
    save_every: int = -1                # CLI: --save-every  (default -1 = disabled)
    device: str | None = None
    downscale_factor: float = 1.0       # CLI: --downscale-factor
    num_downscales: int = 2
    resolution_schedule: int = 3000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_weight: float = 0.2
    refine_every: int = 100
    warmup_length: int = 500
    reset_alpha_every: int = 30
    densify_grad_thresh: float = 0.0002
    densify_size_thresh: float = 0.01
    stop_screen_size_at: int = 4000
    split_screen_size: float = 0.05
    keep_crs: bool = False
    val: bool = False
    val_image: str = "random"
    val_render: str | None = None
    colmap_image_path: str = ""


def validate(k: TrainerKwargs) -> None:
    """Raise ValueError on conflicting or impossible kwargs.

    Permissive checks only — bad enum values get caught by libtorch / loaders.
    """
    if k.num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {k.num_iters}")
    if k.val_render is not None and not k.val:
        raise ValueError(
            "val_render is set but val=False; the CLI implicitly enables --val "
            "when --val-render is given. Pass val=True explicitly."
        )
    if k.output is not None:
        ext = k.output.rsplit(".", 1)[-1].lower() if "." in k.output else ""
        if ext not in ("ply", "splat"):
            raise ValueError(f"output extension must be .ply or .splat, got '{k.output}'")
```

- [ ] **Step 5: Write `python/opensplat/trainer.py`** — constructor only; iteration in the next task.

```python
"""Trainer — drives the per-step training loop on top of opensplat._core.Model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from opensplat import _core
from opensplat._device import resolve_device
from opensplat._kwargs import TrainerKwargs, validate


@dataclass(frozen=True)
class StepResult:
    """One training step's observable result. Yielded by Trainer iteration."""
    step: int
    loss: float
    num_gaussians: int


class Trainer:
    """Iterable trainer. See `opensplat.train()` for the one-shot convenience form."""

    def __init__(self, **kwargs: Any) -> None:
        # Construct kwargs dataclass — surfaces TypeError on unknown args.
        self._kw = TrainerKwargs(**kwargs)
        validate(self._kw)
        self.device = resolve_device(self._kw.device)
        self.num_iters = self._kw.num_iters

        self._input_data = _core.input_data_from_path(
            str(self._kw.input), self._kw.colmap_image_path
        )
        cams, val_cam = self._input_data.get_cameras(
            self._kw.val, self._kw.val_image,
        )
        self._cameras = cams
        self._val_cam = val_cam

        self._model = _core.Model(
            self._input_data, len(cams),
            self._kw.num_downscales,
            self._kw.resolution_schedule,
            self._kw.sh_degree,
            self._kw.sh_degree_interval,
            self._kw.refine_every,
            self._kw.warmup_length,
            self._kw.reset_alpha_every,
            self._kw.densify_grad_thresh,
            self._kw.densify_size_thresh,
            self._kw.stop_screen_size_at,
            self._kw.split_screen_size,
            self._kw.num_iters,
            self._kw.keep_crs,
            self.device,
        )
        self._step = 0
```

- [ ] **Step 6: Update `python/opensplat/__init__.py`** to re-export the new public surface.

```python
"""OpenSplat — Python bindings for 3D Gaussian Splatting training."""
import torch as _torch  # noqa: F401 — must be imported before _core so libtorch is resolved
from opensplat._version import __version__
from opensplat.trainer import Trainer, StepResult

__all__ = ["__version__", "Trainer", "StepResult"]
```

- [ ] **Step 7: Reinstall the package** (editable install picks up new .py files automatically, but in case of caching):

```bash
python3 -m pip install -e . --no-build-isolation
python3 -m pytest tests/python/test_training.py::test_trainer_constructs -v
```

Expected: PASS.

- [ ] **Step 8: Commit.**

```bash
git add python/opensplat/_device.py python/opensplat/_kwargs.py python/opensplat/trainer.py python/opensplat/__init__.py tests/python/test_training.py
git commit -m "feat(py): add StepResult + Trainer constructor

Trainer constructs from kwargs, validates them, resolves the device, and
builds the underlying _core.Model. Iteration not yet implemented."
```

---

## Task 9: `Trainer.__iter__` — the training loop

Add the iterator that yields one `StepResult` per step and runs the actual forward/backward.

**Files:**
- Modify: `python/opensplat/trainer.py`
- Modify: `tests/python/test_training.py`

- [ ] **Step 1: Write the failing test.** Append:

```python
def test_trainer_yields_step_results(colmap_mini: Path) -> None:
    from opensplat import Trainer, StepResult

    trainer = Trainer(
        input=str(colmap_mini), num_iters=3, save_every=-1, sh_degree=1,
    )
    results = list(trainer)
    assert len(results) == 3
    assert all(isinstance(r, StepResult) for r in results)
    assert [r.step for r in results] == [0, 1, 2]
    assert all(r.loss > 0 for r in results)
    assert all(r.num_gaussians > 0 for r in results)
```

(`sh_degree=1` makes each step cheaper.)

- [ ] **Step 2: Run, confirm it fails.**

```bash
python3 -m pytest tests/python/test_training.py::test_trainer_yields_step_results -v
```

Expected: `TypeError: 'Trainer' object is not iterable` (or similar).

- [ ] **Step 3: Edit `python/opensplat/trainer.py`** — append the iteration method to the `Trainer` class, plus an internal RNG initialized in `__init__`. Find the line `self._step = 0` at the end of `__init__` and add immediately after:

```python
        import random
        self._rng = random.Random(42)
```

Then add the iteration methods to the `Trainer` class (after `__init__`):

```python
    def __iter__(self) -> "Trainer":
        return self

    def __next__(self) -> StepResult:
        if self._step >= self.num_iters:
            raise StopIteration
        cam = self._rng.choice(self._cameras)
        downscale = float(self._model.get_downscale_factor(self._step))
        cam.load_image(downscale)
        rendered = self._model.forward(cam, self._step)
        gt = cam.get_image(downscale)
        loss = self._model.main_loss(rendered, gt, self._kw.ssim_weight)
        self._model.optimizers_zero_grad()
        loss.backward()
        self._model.optimizers_step()
        self._model.schedulers_step(self._step)
        self._model.after_train(self._step)

        result = StepResult(
            step=self._step,
            loss=float(loss.item()),
            num_gaussians=int(self._model.means.size(0)),
        )
        self._step += 1
        return result

    def run(self) -> None:
        """Drive the iterator to completion. Equivalent to `for _ in self: pass`."""
        for _ in self:
            pass
```

- [ ] **Step 4: Run the iteration test.**

```bash
python3 -m pytest tests/python/test_training.py -v
```

Expected: both training tests PASS. Step count == 3, all `StepResult` fields populated.

- [ ] **Step 5: Commit.**

```bash
git add python/opensplat/trainer.py tests/python/test_training.py
git commit -m "feat(py): implement Trainer iteration

Per-step forward/backward/optimizer-step driven from Python; yields
StepResult. Camera selection uses a seeded random.Random(42) for
reproducibility. Adds .run() for the one-shot case."
```

---

## Task 10: Save-on-exit semantics

When the iterator exits — normally, via `break`, or via exception — the final state is written to `output=` if it was set.

**Files:**
- Modify: `python/opensplat/trainer.py`
- Modify: `tests/python/test_training.py`

- [ ] **Step 1: Write the failing tests.** Append two cases:

```python
def test_trainer_saves_on_normal_completion(colmap_mini: Path, tmp_output_ply: Path) -> None:
    from opensplat import Trainer
    trainer = Trainer(
        input=str(colmap_mini), output=str(tmp_output_ply),
        num_iters=2, save_every=-1, sh_degree=1,
    )
    trainer.run()
    assert tmp_output_ply.is_file() and tmp_output_ply.stat().st_size > 0


def test_trainer_saves_on_break(colmap_mini: Path, tmp_output_ply: Path) -> None:
    from opensplat import Trainer
    trainer = Trainer(
        input=str(colmap_mini), output=str(tmp_output_ply),
        num_iters=20, save_every=-1, sh_degree=1,
    )
    for r in trainer:
        if r.step >= 1:
            break
    assert tmp_output_ply.is_file() and tmp_output_ply.stat().st_size > 0
```

- [ ] **Step 2: Run, confirm they fail.**

```bash
python3 -m pytest tests/python/test_training.py -k "saves_on" -v
```

Expected: both fail with `FileNotFoundError` / `assert` on the file existence check.

- [ ] **Step 3: Restructure `Trainer.__iter__` into a generator wrapper around `__next__`** so we can use a `finally` block. Replace `__iter__`/`__next__` with the following — note the loss-of-`__next__`-style explicit state machine in favor of a generator. (We keep `_step` so the generator restarts cleanly are caller-visible: a Trainer instance is single-use.)

Find the `def __iter__` / `def __next__` block and **replace** with:

```python
    def __iter__(self):
        try:
            while self._step < self.num_iters:
                cam = self._rng.choice(self._cameras)
                downscale = float(self._model.get_downscale_factor(self._step))
                cam.load_image(downscale)
                rendered = self._model.forward(cam, self._step)
                gt = cam.get_image(downscale)
                loss = self._model.main_loss(rendered, gt, self._kw.ssim_weight)
                self._model.optimizers_zero_grad()
                loss.backward()
                self._model.optimizers_step()
                self._model.schedulers_step(self._step)
                self._model.after_train(self._step)

                yield StepResult(
                    step=self._step,
                    loss=float(loss.item()),
                    num_gaussians=int(self._model.means.size(0)),
                )

                # Mid-training periodic save.
                if (self._kw.save_every > 0
                    and self._step > 0
                    and self._step % self._kw.save_every == 0
                    and self._kw.output is not None):
                    self._model.save(str(self._kw.output), self._step)

                self._step += 1
        finally:
            if self._kw.output is not None:
                self._model.save(str(self._kw.output), self._step)
```

Generators that exit via `break` (caller stops consuming) trigger `GeneratorExit` → the `finally` block runs. `KeyboardInterrupt` propagates through the same path. `StopIteration` is raised naturally when the loop completes.

- [ ] **Step 4: Run all training tests.**

```bash
python3 -m pytest tests/python/test_training.py -v
```

Expected: all four PASS. The `break` test verifies that the file was written after the loop exited early.

- [ ] **Step 5: Commit.**

```bash
git add python/opensplat/trainer.py tests/python/test_training.py
git commit -m "feat(py): save Trainer output on any iterator exit

Wraps the loop in try/finally so output= is always written when set —
whether the loop completes, breaks early, or raises. Adds the spec's
mid-training save_every cadence too."
```

---

## Task 11: `train()` convenience function + public `__init__.py`

Tiny task — just the sugar.

**Files:**
- Create: `python/opensplat/api.py`
- Modify: `python/opensplat/__init__.py`
- Modify: `tests/python/test_training.py`

- [ ] **Step 1: Write the failing test.** Append:

```python
def test_train_function_round_trip(colmap_mini: Path, tmp_output_ply: Path) -> None:
    import opensplat
    opensplat.train(
        input=str(colmap_mini), output=str(tmp_output_ply),
        num_iters=2, save_every=-1, sh_degree=1,
    )
    assert tmp_output_ply.is_file() and tmp_output_ply.stat().st_size > 0
```

- [ ] **Step 2: Run, confirm it fails.**

```bash
python3 -m pytest tests/python/test_training.py::test_train_function_round_trip -v
```

Expected: `AttributeError: module 'opensplat' has no attribute 'train'`.

- [ ] **Step 3: Write `python/opensplat/api.py`.**

```python
"""Top-level convenience function. Equivalent to `Trainer(**kwargs).run()`."""
from __future__ import annotations

from typing import Any

from opensplat.trainer import Trainer


def train(**kwargs: Any) -> None:
    """Train a 3D Gaussian Splatting scene from a project directory.

    Mirrors the opensplat CLI. Required: input=. Most users also want output=.

    See opensplat._kwargs.TrainerKwargs for the full kwarg list and defaults.
    """
    Trainer(**kwargs).run()
```

- [ ] **Step 4: Update `python/opensplat/__init__.py`.**

```python
"""OpenSplat — Python bindings for 3D Gaussian Splatting training."""
import torch as _torch  # noqa: F401 — must be imported before _core so libtorch is resolved
from opensplat._version import __version__
from opensplat.api import train
from opensplat.trainer import Trainer, StepResult

__all__ = ["__version__", "train", "Trainer", "StepResult"]
```

- [ ] **Step 5: Run the test.**

```bash
python3 -m pytest tests/python/test_training.py::test_train_function_round_trip -v
```

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add python/opensplat/api.py python/opensplat/__init__.py tests/python/test_training.py
git commit -m "feat(py): add opensplat.train() one-shot convenience function

Thin sugar over Trainer(**kwargs).run(). Public API surface complete:
opensplat.{train, Trainer, StepResult, __version__}."
```

---

## Task 12: Tier 3 — CLI parity test

Catches drift between `opensplat` CLI and `opensplat.train()`. The canary for the duplicated loop logic we accepted in the spec.

**Files:**
- Create: `tests/python/test_cli_parity.py`

- [ ] **Step 1: Write the test.**

```python
"""Tier 3 — CLI vs Python API parity.

Runs `opensplat` (CLI) and `opensplat.train()` on the same fixture, same
seed, same kwargs, same device. Asserts the resulting PLY has the same
Gaussian count and means/opacities tensor values match within tolerance.

If this test fails, one of (opensplat.cpp main, trainer.py __iter__) has
drifted from the other.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    shutil.which("opensplat") is None and not Path("build-py/opensplat").is_file(),
    reason="opensplat CLI binary not found in $PATH or build-py/",
)


def _opensplat_binary() -> str:
    return shutil.which("opensplat") or str(Path("build-py/opensplat").resolve())


def _read_ply_means(path: Path) -> list[tuple[float, float, float]]:
    """Parse a binary little-endian PLY produced by Model::savePly.

    Returns the means (x, y, z) of every vertex. Used to compare two saved
    splats for tensor-level parity. Format details: model.cpp Model::savePly.
    """
    import struct
    means = []
    with open(path, "rb") as fh:
        header = []
        for line in iter(fh.readline, b""):
            header.append(line.decode("ascii", errors="replace").rstrip())
            if header[-1] == "end_header":
                break
        # Find the element count
        n = 0
        for h in header:
            if h.startswith("element vertex"):
                n = int(h.split()[-1])
                break
        # Each vertex has at least x,y,z as float32 at the start.
        # Other fields follow but we only need x,y,z for parity comparison.
        # Determine the full vertex byte size from the property list.
        prop_size = 0
        for h in header:
            if h.startswith("property float"):
                prop_size += 4
            elif h.startswith("property uchar") or h.startswith("property uint8"):
                prop_size += 1
        body = fh.read(n * prop_size)
        for i in range(n):
            x, y, z = struct.unpack_from("<fff", body, i * prop_size)
            means.append((x, y, z))
    return means


def test_cli_and_python_train_match(colmap_mini: Path, tmp_path: Path) -> None:
    """Same seed, same fixture, same kwargs, same device → same output."""
    cli_out = tmp_path / "cli.ply"
    py_out = tmp_path / "py.ply"

    # CLI run — force CPU, small iter count, no validation.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    cli_cmd = [
        _opensplat_binary(), str(colmap_mini),
        "-o", str(cli_out),
        "-n", "20",
        "--cpu",
        "--sh-degree", "1",
        "--save-every", "-1",
        "--warmup-length", "5",
        "--refine-every", "1000",  # avoid densification in such a short run
    ]
    subprocess.run(cli_cmd, check=True, env=env, capture_output=True)
    assert cli_out.is_file()

    # Python run — same effective configuration.
    cmd = [
        sys.executable, "-c",
        "import opensplat; opensplat.train("
        f"input={str(colmap_mini)!r}, output={str(py_out)!r}, "
        "num_iters=20, device='cpu', sh_degree=1, save_every=-1, "
        "warmup_length=5, refine_every=1000)",
    ]
    subprocess.run(cmd, check=True, env=env, capture_output=True)
    assert py_out.is_file()

    cli_means = _read_ply_means(cli_out)
    py_means = _read_ply_means(py_out)

    # Same Gaussian count.
    assert len(cli_means) == len(py_means), (
        f"Gaussian count drift: CLI={len(cli_means)} vs Python={len(py_means)}. "
        "One of opensplat.cpp main() or Trainer.__iter__ has changed without the other."
    )
    # Means match within tolerance (1e-3 — generous; small fixtures + CPU determinism).
    for (cx, cy, cz), (px, py_, pz) in zip(cli_means, py_means):
        assert abs(cx - px) < 1e-3, f"means[x] drift: {cx} vs {px}"
        assert abs(cy - py_) < 1e-3, f"means[y] drift: {cy} vs {py_}"
        assert abs(cz - pz) < 1e-3, f"means[z] drift: {cz} vs {pz}"
```

- [ ] **Step 2: Run.**

```bash
python3 -m pytest tests/python/test_cli_parity.py -v
```

Expected: PASS. If the means drift, the most likely cause is the camera-selection RNG diverging between C++ (`std::default_random_engine` somewhere) and Python (`random.Random(42)`). If so, document the actual CLI RNG behavior and either match it in Python or relax the test to "same Gaussian count, loss within 1% after 20 steps."

- [ ] **Step 3: Commit.**

```bash
git add tests/python/test_cli_parity.py
git commit -m "test(py): add CLI vs Python parity test (Tier 3)

Runs both opensplat binary and opensplat.train() on the same fixture
with same kwargs/seed on CPU, compares the resulting PLY's mean
coordinates within 1e-3 tolerance. Canary for loop-logic drift."
```

---

## Task 13: Update the spec, write type stubs, write README

Documentation pass. Fix the spec's incorrect defaults and add the type stubs that pyright/mypy users expect.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-02-python-bindings-design.md`
- Create: `python/opensplat/_core.pyi`
- Create: `python/opensplat/py.typed`
- Modify: `README.md`

- [ ] **Step 1: Update the spec's kwargs table.** In `docs/superpowers/specs/2026-06-02-python-bindings-design.md`, find the kwargs table and:
  - Change the `save_every` default from `7000` to `-1`
  - Add rows for `downscale_factor` (float, `1.0`) and `colmap_image_path` (str, `""`)
  - Note in the `val` row that setting `val_render` without `val=True` raises `ValueError` (instead of auto-promoting, which is what the CLI does)

- [ ] **Step 2: Write `python/opensplat/_core.pyi`** — hand-written stubs for the C-extension.

```python
"""Type stubs for opensplat._core (internal C-extension)."""
from __future__ import annotations

from typing import Any, Tuple, List

import torch


class Camera:
    id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    file_path: str
    cam_to_world: torch.Tensor

    def load_image(self, downscale_factor: float) -> None: ...
    def get_image(self, downscale_factor: float) -> torch.Tensor: ...


class Points:
    xyz: torch.Tensor
    rgb: torch.Tensor


class InputData:
    cameras: List[Camera]
    scale: float
    translation: torch.Tensor
    points: Points

    def get_cameras(self, validate: bool, val_image: str = "random") -> Tuple[List[Camera], Camera | None]: ...


class Model:
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    opacities: torch.Tensor

    def __init__(
        self,
        input_data: InputData,
        num_cameras: int,
        num_downscales: int,
        resolution_schedule: int,
        sh_degree: int,
        sh_degree_interval: int,
        refine_every: int,
        warmup_length: int,
        reset_alpha_every: int,
        densify_grad_thresh: float,
        densify_size_thresh: float,
        stop_screen_size_at: int,
        split_screen_size: float,
        max_steps: int,
        keep_crs: bool,
        device: torch.device,
    ) -> None: ...

    def forward(self, cam: Camera, step: int) -> torch.Tensor: ...
    def main_loss(self, rendered: torch.Tensor, gt: torch.Tensor, ssim_weight: float) -> torch.Tensor: ...
    def optimizers_zero_grad(self) -> None: ...
    def optimizers_step(self) -> None: ...
    def schedulers_step(self, step: int) -> None: ...
    def get_downscale_factor(self, step: int) -> int: ...
    def after_train(self, step: int) -> None: ...
    def save(self, filename: str, step: int) -> None: ...
    def save_ply(self, filename: str, step: int) -> None: ...
    def save_splat(self, filename: str) -> None: ...
    def load_ply(self, filename: str) -> int: ...


def input_data_from_path(project_root: str, colmap_image_source_path: str = "") -> InputData: ...
```

- [ ] **Step 3: Create `python/opensplat/py.typed`** (empty marker file per PEP 561).

```bash
touch python/opensplat/py.typed
```

- [ ] **Step 4: Add a "Python bindings" section to `README.md`.** Find the existing usage section and append:

```markdown
## Python bindings

Install the prebuilt wheel:

```bash
pip install opensplat
```

Minimal usage:

```python
import opensplat

opensplat.train(
    input="/path/to/colmap_or_nerfstudio_project",
    output="scene.ply",
    num_iters=30000,
)

# Or iterate for progress reporting / early stopping:
trainer = opensplat.Trainer(input="...", output="scene.ply", num_iters=30000)
for result in trainer:
    print(f"step {result.step}: loss={result.loss:.4f}")
```

All CLI flags are available as kwargs (`snake_case`). See `python/opensplat/_kwargs.py` for the full list.
```

- [ ] **Step 5: Commit.**

```bash
git add docs/superpowers/specs/2026-06-02-python-bindings-design.md \
        python/opensplat/_core.pyi python/opensplat/py.typed README.md
git commit -m "docs(py): type stubs, README section, fix spec defaults

- Hand-written _core.pyi for pyright/mypy
- README section showing train() and Trainer usage
- Spec kwarg-table corrections per implementation: save_every=-1,
  downscale_factor, colmap_image_path"
```

---

## Task 14: cibuildwheel — Linux CPU wheel

Get one wheel target green end-to-end before adding the rest. Linux CPU is cheapest.

**Files:**
- Create: `.github/workflows/wheels.yml`

- [ ] **Step 1: Write `.github/workflows/wheels.yml`.** Starts with just Linux CPU; the other platforms are added in Task 15.

```yaml
name: Wheels

on:
  push:
    tags: ["v*"]
  workflow_dispatch:
  pull_request:
    paths:
      - "pyproject.toml"
      - "python/**"
      - "CMakeLists.txt"
      - ".github/workflows/wheels.yml"

jobs:
  linux-cpu:
    name: Linux x86_64 CPU
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install build dependencies
        run: pip install cibuildwheel==2.21.*

      - name: Build wheels
        env:
          CIBW_BUILD: "cp310-manylinux_x86_64 cp311-manylinux_x86_64 cp312-manylinux_x86_64"
          CIBW_MANYLINUX_X86_64_IMAGE: "quay.io/pypa/manylinux_2_28_x86_64"
          # OpenCV + nlohmann_json + nanoflann + glm — all of these fetch via FetchContent
          # in CMakeLists.txt. We only need OpenCV from the system.
          CIBW_BEFORE_ALL_LINUX: "yum install -y opencv opencv-devel"
          # Pin torch CPU for build + runtime
          CIBW_BEFORE_BUILD: "pip install torch --index-url https://download.pytorch.org/whl/cpu"
          CIBW_ENVIRONMENT: 'CMAKE_ARGS="-DGPU_RUNTIME=CPU -DOPENSPLAT_BUILD_PYTHON_BINDINGS=ON"'
          CIBW_TEST_REQUIRES: "pytest pillow numpy"
          CIBW_TEST_COMMAND: 'pytest {project}/tests/python/test_bindings.py -v'
          # repair: ensure libopensplat.so + libtorch deps are sorted out by auditwheel
          # Allow torch's libs to be linked dynamically (don't bundle them — they're huge
          # and version-coupled). This matches torchvision/torchaudio convention.
          CIBW_REPAIR_WHEEL_COMMAND_LINUX: >-
            auditwheel repair -w {dest_dir} {wheel}
            --exclude libc10.so --exclude libtorch.so --exclude libtorch_cpu.so
            --exclude libtorch_python.so
        run: python -m cibuildwheel --output-dir wheelhouse

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-linux-cpu
          path: wheelhouse/*.whl
```

- [ ] **Step 2: Smoke-build a wheel locally** in a Linux container, if available. Otherwise commit and rely on CI.

```bash
# Optional local check (requires Docker):
docker run --rm -v "$PWD":/io -w /io quay.io/pypa/manylinux_2_28_x86_64 bash -c "
    yum install -y opencv opencv-devel &&
    /opt/python/cp311-cp311/bin/pip install cibuildwheel==2.21.* &&
    /opt/python/cp311-cp311/bin/cibuildwheel --output-dir wheelhouse-local
"
ls wheelhouse-local/
```

Expected: a `.whl` file is produced. Skip if Docker isn't available; CI will catch it.

- [ ] **Step 3: Commit and push, watch CI.**

```bash
git add .github/workflows/wheels.yml
git commit -m "ci: add Linux CPU wheel build via cibuildwheel

Builds cp310/311/312 manylinux wheels against torch CPU. Excludes
libtorch.so etc. from wheel bundling per the torchvision convention —
wheels link the user's installed torch at runtime."
git push
```

Expected: GH Actions runs the `linux-cpu` job to completion. Three `.whl` artifacts uploaded.

- [ ] **Step 4: Manually smoke-test one wheel.**

```bash
gh run download --name wheels-linux-cpu
python3 -m venv /tmp/wheel-test && source /tmp/wheel-test/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install opensplat-*-cp311-*.whl
python -c "import opensplat; print(opensplat.__version__)"
deactivate
```

Expected: imports successfully, prints version.

---

## Task 15: cibuildwheel — Linux CUDA, macOS arm64, Windows CUDA

Add the remaining three wheel targets. Each follows the same pattern as Linux CPU but with platform-specific `CIBW_BEFORE_ALL` and `CMAKE_ARGS`.

**Files:**
- Modify: `.github/workflows/wheels.yml`

- [ ] **Step 1: Append the `linux-cuda` job.** After the `linux-cpu:` job in `.github/workflows/wheels.yml`:

```yaml
  linux-cuda:
    name: Linux x86_64 CUDA 12.1
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install cibuildwheel==2.21.*
      - name: Build wheels
        env:
          CIBW_BUILD: "cp310-manylinux_x86_64 cp311-manylinux_x86_64 cp312-manylinux_x86_64"
          # Image bundles CUDA 12.1 toolkit + matching nvcc + libtorch CUDA build.
          CIBW_MANYLINUX_X86_64_IMAGE: "pytorch/manylinux-cuda121"
          CIBW_BEFORE_ALL_LINUX: "yum install -y opencv opencv-devel"
          CIBW_BEFORE_BUILD: "pip install torch --index-url https://download.pytorch.org/whl/cu121"
          CIBW_ENVIRONMENT: 'CMAKE_ARGS="-DGPU_RUNTIME=CUDA -DOPENSPLAT_BUILD_PYTHON_BINDINGS=ON -DOPENSPLAT_MAX_CUDA_COMPATIBILITY=ON"'
          # No CUDA in the test runner — skip runtime tests, just confirm import works.
          CIBW_TEST_COMMAND: 'python -c "import opensplat; print(opensplat.__version__)"'
          CIBW_REPAIR_WHEEL_COMMAND_LINUX: >-
            auditwheel repair -w {dest_dir} {wheel}
            --exclude libc10.so --exclude libc10_cuda.so
            --exclude libtorch.so --exclude libtorch_cpu.so
            --exclude libtorch_cuda.so --exclude libtorch_python.so
            --exclude libcudart.so.12 --exclude libnvToolsExt.so.1
        run: python -m cibuildwheel --output-dir wheelhouse
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-linux-cuda
          path: wheelhouse/*.whl
```

- [ ] **Step 2: Append the `macos-arm64` job.**

```yaml
  macos-arm64:
    name: macOS arm64 Metal
    runs-on: macos-14   # Apple Silicon
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: brew install opencv
      - run: pip install cibuildwheel==2.21.*
      - name: Build wheels
        env:
          CIBW_BUILD: "cp310-macosx_arm64 cp311-macosx_arm64 cp312-macosx_arm64"
          CIBW_ARCHS_MACOS: arm64
          CIBW_BEFORE_BUILD: "pip install torch"
          CIBW_ENVIRONMENT: 'CMAKE_ARGS="-DGPU_RUNTIME=MPS -DOPENSPLAT_BUILD_PYTHON_BINDINGS=ON -DOPENCV_DIR=$(brew --prefix opencv)" MACOSX_DEPLOYMENT_TARGET=12.0'
          CIBW_TEST_REQUIRES: "pytest pillow numpy"
          CIBW_TEST_COMMAND: 'pytest {project}/tests/python/test_bindings.py -v'
        run: python -m cibuildwheel --output-dir wheelhouse
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-macos-arm64
          path: wheelhouse/*.whl
```

- [ ] **Step 3: Append the `windows-cuda` job.** Windows is the most fiddly; the CUDA install step is non-trivial.

```yaml
  windows-cuda:
    name: Windows x86_64 CUDA 12.1
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - uses: Jimver/cuda-toolkit@v0.2.16
        id: cuda
        with:
          cuda: "12.1.0"
          method: "network"
          sub-packages: '["nvcc", "cudart", "cublas", "cublas_dev", "thrust"]'
      - name: Install OpenCV (vcpkg)
        run: |
          vcpkg install opencv4[core,imgproc,calib3d]:x64-windows
          echo "OpenCV_DIR=$env:VCPKG_INSTALLATION_ROOT/installed/x64-windows/share/opencv4" | Out-File -FilePath $env:GITHUB_ENV -Append
        shell: pwsh
      - run: pip install cibuildwheel==2.21.*
      - name: Build wheels
        env:
          CIBW_BUILD: "cp310-win_amd64 cp311-win_amd64 cp312-win_amd64"
          CIBW_BEFORE_BUILD: "pip install torch --index-url https://download.pytorch.org/whl/cu121"
          CIBW_ENVIRONMENT_WINDOWS: 'CMAKE_ARGS="-DGPU_RUNTIME=CUDA -DOPENSPLAT_BUILD_PYTHON_BINDINGS=ON -DOPENCV_DIR=$env:OpenCV_DIR"'
          CIBW_TEST_COMMAND: 'python -c "import opensplat; print(opensplat.__version__)"'
        run: python -m cibuildwheel --output-dir wheelhouse
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-windows-cuda
          path: wheelhouse/*.whl
```

- [ ] **Step 4: Commit and push.**

```bash
git add .github/workflows/wheels.yml
git commit -m "ci: add Linux CUDA, macOS arm64, Windows CUDA wheel jobs

Completes the v1 wheel matrix: 4 platforms × 3 Python versions = 12
wheels per release. Torch coupling: each platform pins one torch minor
version for v1; a second torch version can be added by parameterizing
CIBW_BEFORE_BUILD."
git push
```

Expected: all four jobs run on CI. The first time through, expect at least one to fail on something platform-specific (most likely Windows). Debug the actual failure, don't pre-emptively rewrite anything.

- [ ] **Step 5: Wait for green CI** before marking complete. CI failure on one platform should not block the others' wheels from being usable for testing.

---

## Self-Review (post-write)

Run through this checklist before declaring the plan done.

**1. Spec coverage**
- ✅ Goal/architecture: covered by spec, summarized in plan header
- ✅ Repo layout: implemented across Tasks 1–3, 5–11
- ✅ Public API (`train`, `Trainer`, `StepResult`, kwargs table): Tasks 8–11
- ✅ Cancellation/save-on-exit: Task 10
- ✅ Device auto-detect: Task 8 (`_device.py`)
- ✅ Internal C++ surface (`Camera`, `InputData`, `Model`, loaders): Tasks 5–7
- ✅ CMake refactor + `OPENSPLAT_BUILD_PYTHON_BINDINGS`: Tasks 1–2
- ✅ pyproject.toml + scikit-build-core: Task 3
- ✅ Wheel matrix (4 platforms × 3 Pythons): Tasks 14–15
- ✅ Tier 1 binding tests: Tasks 5–7 (each binding task includes its own test)
- ✅ Tier 2 E2E tests: Tasks 8–11
- ✅ Tier 3 CLI parity: Task 12
- ⚠ Two torch minor versions per platform: NOT yet parameterized. Captured as a follow-up at the end of Task 15. Documented; safe deferred.
- ⚠ Tier 1 wheel `test-command` is set on Linux CPU + macOS but skipped on CUDA jobs (no GPU). Documented; intentional.

**2. Placeholder scan**
- No "TBD" / "TODO" / "implement later" markers in any task body.
- Generator script for the COLMAP fixture is concrete and runnable.
- Type-stub file is hand-written, not generated.

**3. Type / signature consistency**
- `_core.Camera` properties match `Camera` in `input_data.hpp` (`file_path` is `snake_case`; the C++ field is `filePath` — pybind11 binding renames it).
- `_core.Model` constructor parameter order matches `model.hpp:22-32`.
- `Trainer.__iter__` is a generator (Task 10 replaced the `__next__` version from Task 9 — note explicit replace step).
- `TrainerKwargs` fields and `_core.Model` constructor positional args line up; verified by reading Task 8 + Task 9.

**4. Known fragilities (documented in tasks, not hidden)**
- Task 5 Step 4: libtorch rpath issue addressed defensively (`import torch` before `_core`).
- Task 12: CLI parity tolerance is `1e-3` on means and `len()` equality on counts — generous; if it flakes, relax to "same count, loss within 1%."
- Task 15: First Windows run is likely to need fixes; the plan explicitly says "debug the actual failure, don't pre-emptively rewrite."

Plan is ready for execution.
