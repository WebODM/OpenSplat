# OpenSplat Python Bindings — Design

**Date:** 2026-06-02
**Related issue:** #189
**Status:** Approved for implementation planning

## Goal

Ship `pip install opensplat` as a Python package that wraps OpenSplat's existing C++ training pipeline. The primary audience is end users who today run the `opensplat` CLI and want to drive equivalent training from Python — for scripting, pipeline integration, and reproducibility in notebooks.

Non-goals for v1: custom losses, custom densification, exposing rasterization kernels as standalone PyTorch ops, in-memory camera/image input, replacing the C++ CLI.

## API surface

Two entry points: `train()` for the one-shot case, `Trainer` for loop-with-observation.

```python
import opensplat

opensplat.train(
    input="/path/to/colmap_or_nerfstudio_project",
    output="scene.ply",
    num_iters=30000,
    save_every=7000,
    device=None,           # None = auto-detect (cuda → mps → cpu)
    # ... all CLI flags exposed as kwargs (see "Kwargs" below) ...
)

trainer = opensplat.Trainer(input="...", output="scene.ply", num_iters=30000, save_every=7000)
for result in trainer:
    print(result.step, result.loss, result.num_gaussians)
    if should_stop(result):
        break    # final save still runs
```

```python
@dataclass(frozen=True)
class StepResult:
    step: int
    loss: float
    num_gaussians: int
```

### Kwargs

Flat kwargs, one per CLI flag, `snake_case`, defaults match the C++ CLI defaults exactly. The full set in v1:

| Kwarg | Type | Default | Source |
|---|---|---|---|
| `input` | `str \| os.PathLike` | required | `-i` |
| `output` | `str \| os.PathLike \| None` | `None` | `-o` |
| `num_iters` | `int` | `30000` | `-n` |
| `save_every` | `int` | `7000` | `--save-every` (0 disables) |
| `device` | `str \| None` | `None` | new (CLI uses auto) |
| `num_downscales` | `int` | `2` | `--num-downscales` |
| `resolution_schedule` | `int` | `3000` | `--resolution-schedule` |
| `sh_degree` | `int` | `3` | `--sh-degree` |
| `sh_degree_interval` | `int` | `1000` | `--sh-degree-interval` |
| `refine_every` | `int` | `100` | `--refine-every` |
| `warmup_length` | `int` | `500` | `--warmup-length` |
| `reset_alpha_every` | `int` | `30` | `--reset-alpha-every` |
| `densify_grad_thresh` | `float` | `0.0002` | `--densify-grad-thresh` |
| `densify_size_thresh` | `float` | `0.01` | `--densify-size-thresh` |
| `stop_screen_size_at` | `int` | `4000` | `--stop-screen-size-at` |
| `split_screen_size` | `float` | `0.05` | `--split-screen-size` |
| `keep_crs` | `bool` | `False` | `--keep-crs` |
| `val` | `bool` | `False` | `--val` |
| `val_image` | `str` | `"random"` | `--val-image` |
| `val_render` | `str \| None` | `None` | `--val-render` |
| `ssim_weight` | `float` | `0.2` | `--ssim-weight` |

`train(**kwargs)` is exactly `Trainer(**kwargs).run()`. No translation, no divergence.

### Semantics

- **Cancellation:** `break` out of the iterator runs the `finally` block, which saves the final output if `output=` was set. `KeyboardInterrupt` propagates the same way. `SIGKILL` leaves only the last `save_every` checkpoint (matches CLI).
- **Device auto-detect:** `cuda` → `mps` → `cpu`, picking the first available. Silent fallback (matches CLI today). Explicit `device="cuda"` with no CUDA raises libtorch's `RuntimeError`.
- **Reproducibility:** `torch.manual_seed(42)` (already in `Model` constructor) plus a Python-side `random.Random(42)` for camera selection. Deterministic-ish modulo CUDA nondeterminism.
- **Output format** inferred from extension: `.ply` → `savePly`, `.splat` → `saveSplat`. Unknown extension → `ValueError`.
- **Errors:** loader/file errors → `ValueError` / `FileNotFoundError`; libtorch errors → `RuntimeError`; bad kwargs → `TypeError`; conflicting kwargs (e.g., `val_render=` set with `val=False`, unknown output extension) → `ValueError` raised in `Trainer.__init__`.

## Architecture

Three build artifacts produced from one source tree:

1. **`libopensplat`** — SHARED library. All existing C++ sources except `opensplat.cpp` move into it.
2. **`opensplat`** — executable, unchanged behavior. Links `libopensplat`. `opensplat.cpp` source untouched.
3. **`opensplat._core`** — pybind11 C-extension. Links `libopensplat`. Exposes `Model`, `InputData`, `Camera`, and the five input loaders as Python types.

The Python package `opensplat/` contains the public `train()` / `Trainer` / `StepResult` surface. `Trainer` is the only orchestration code added — it is the Python analog of `opensplat.cpp` `main()`'s training loop.

**Where the loop lives:** in Python. `Trainer.__iter__` calls `model.forward()`, `model.main_loss()`, `loss.backward()`, `model.optimizers_step()`, etc. via the wrapped `_core.Model` methods. The per-step Python↔C++ overhead (~5–10 calls) is negligible compared to one forward+backward pass.

**Duplication:** `opensplat.cpp` `main()` and `python/opensplat/trainer.py` will contain similar loop logic. Accepted tradeoff. Tier 3 CLI-parity tests (see "Testing") are the canary for drift.

### Repo layout

```
OpenSplat/
├── CMakeLists.txt              # adds SHARED libopensplat target + optional _core target
├── opensplat.cpp               # CLI executable, unchanged, links libopensplat
├── (existing C++ sources at root, now compiled into libopensplat)
├── python/
│   ├── bindings.cpp            # pybind11 module
│   └── opensplat/
│       ├── __init__.py         # re-exports train, Trainer, StepResult
│       ├── _core.pyi           # type stubs for the C-extension
│       ├── trainer.py          # Trainer class (the for-loop)
│       └── api.py              # train() convenience function
├── pyproject.toml              # scikit-build-core
├── tests/
│   ├── fixtures/colmap_mini/   # tiny COLMAP project (~500 points, 64x64 images)
│   ├── fixtures/nerfstudio_mini/
│   └── python/
│       ├── test_bindings.py    # tier 1
│       ├── test_training.py    # tier 2
│       └── test_cli_parity.py  # tier 3
└── .github/workflows/
    └── wheels.yml              # cibuildwheel matrix
```

### pybind11 surface (internal)

`opensplat._core` is underscore-prefixed and not part of the documented v1 API. It exposes the existing C++ types as-is so the Python `Trainer` can drive the loop:

- `Camera` — `width`, `height`, `fx`, `fy` readonly; `load_image(downscale_factor)`, `get_image(downscale_factor)`
- `InputData` — `cameras`, `scale`, `translation` readonly; `get_cameras(val, val_image)`
- `input_data_from_path(path, ...)` — dispatcher matching the existing `inputDataFromX` in `input_data.cpp`. Also expose per-format loaders for testability.
- `Model` — constructor with all CLI-controlled hyperparameters + `forward`, `main_loss`, `optimizers_zero_grad`, `optimizers_step`, `schedulers_step`, `after_train`, `get_downscale_factor`, `save`, `save_ply`, `save_splat`, `load_ply`. Read-only tensor fields: `means`, `scales`, `quats`, `features_dc`, `features_rest`, `opacities`.

Heavy C++ calls (`forward`, `optimizers_step`, etc.) use `py::call_guard<py::gil_scoped_release>()`. `torch::Tensor` ↔ `torch.Tensor` exchange is zero-copy via pybind11's torch ABI integration — no conversion code needed.

## Build & packaging

### CMake changes

- Refactor existing sources: all `.cpp` files currently in the `opensplat` executable target except `opensplat.cpp` move into a new `libopensplat` SHARED target.
- `opensplat` executable target shrinks to `opensplat.cpp` + `target_link_libraries(opensplat libopensplat)`.
- New option `OPENSPLAT_BUILD_PYTHON_BINDINGS` (default `OFF`). When `ON`:
  - `FetchContent` or `find_package` pulls pybind11 (≥ 2.13)
  - Adds `_core` MODULE target compiled from `python/bindings.cpp` + linked to `libopensplat`
  - Output goes to `python/opensplat/_core.<abi>.so` for editable installs
- Existing GPU/CPU/HIP/Metal switches unchanged — Python wheels just pass them through.

### pyproject.toml

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11>=2.13", "torch"]
build-backend = "scikit_build_core.build"

[project]
name = "opensplat"
dynamic = ["version"]
requires-python = ">=3.10"
dependencies = ["torch", "numpy"]   # exact torch lower bound pinned at release time

[tool.scikit-build]
cmake.args = ["-DOPENSPLAT_BUILD_PYTHON_BINDINGS=ON"]
wheel.packages = ["python/opensplat"]
```

PyPI name: `opensplat`. Fall back to `opensplat-py` if taken.

### libtorch linkage

We do not vendor libtorch. At build time, `scikit-build-core` sets `Torch_DIR` to the build environment's `torch/share/cmake/Torch` so CMake finds the right headers/libs. At runtime, `_core.so` loads libtorch from the user's installed `torch` package. This matches the torchvision/torchaudio convention and is the source of the torch-version coupling in the wheel matrix.

### Wheel matrix

| Platform | Backend | Runner / image |
|---|---|---|
| Linux x86_64 | CUDA 12.1 | `pytorch/manylinux-cuda121` |
| Linux x86_64 | CPU | `quay.io/pypa/manylinux_2_28_x86_64` |
| macOS arm64 | Metal (MPS) | `macos-14` runner |
| Windows x86_64 | CUDA 12.1 | `windows-2022` + CUDA install step |

× Python 3.10, 3.11, 3.12
× PyTorch latest stable + one previous minor (specific minor versions pinned at release time)

= **24 wheels per release.**

Driven by `cibuildwheel` from `.github/workflows/wheels.yml`. HIP/ROCm explicitly out of scope for wheels in v1; users can `pip install opensplat --no-binary opensplat` to build from source against their local torch with any backend the CMake supports.

## Testing strategy

### Tier 1 — Binding smoke tests (`tests/python/test_bindings.py`)
Fast, run on every PR and inside every `cibuildwheel` job's `test-command`. Target: <30s total.
- `Camera` / `InputData` fields readable, types correct.
- `input_data_from_path` dispatches by directory contents (parametrized over all 5 fixture formats — binding-level only for the formats without E2E fixtures).
- `Model` constructs from a tiny in-process fixture, can save PLY, reload, fields unchanged.
- `Model.forward()` returns expected shape for fixture camera.
- Tensor exchange is zero-copy (assert `data_ptr()` equality across boundary).

### Tier 2 — End-to-end training (`tests/python/test_training.py`)
Slow. Runs on merges to `main` and on release tags. `ubuntu-latest` CPU device only — no GPU CI.
- `opensplat.train(input=fixture, output=tmp, num_iters=200)` produces a non-empty PLY.
- Same fixture + same seed → byte-equal PLY on CPU device; tensor-close on CUDA.
- `Trainer` yields up to `num_iters` `StepResult` objects with monotonically increasing `step`; full iteration produces exactly `num_iters`.
- `break` after step 50 → final save still happens.
- Simulated `KeyboardInterrupt` → final save still happens, exception propagates.

### Tier 3 — CLI parity (`tests/python/test_cli_parity.py`)
Nightly + on release. Canary for loop-logic drift between `opensplat.cpp` `main()` and `Trainer`.
- Run C++ `opensplat` and `opensplat.train()` on the same fixture, same seed, same kwargs, 200 iters, CPU. Assert final Gaussian count equal; `means` / `opacities` match within `1e-5`.

### Fixtures
- `tests/fixtures/colmap_mini/` — 8 cameras, ~500 SfM points, 64×64 images, <2 MB.
- `tests/fixtures/nerfstudio_mini/` — same scale.
- OpenSfM/ODM/OpenMVG: binding-level loader tests only in v1.

### Out of scope for v1 testing
- PSNR / quality regression tests (flaky, require held weights).
- Input loader fuzzing (already exercised via the production CLI).
- HIP/Metal-specific CI (no GH Actions runners).

## Open implementation questions

These get resolved during implementation, not in this spec:

- Exact pybind11 trampoline for `Model`'s constructor with 16+ parameters — likely use a `dict` + helper struct on the C++ side to keep the binding readable.
- Whether to ship `_core.pyi` hand-written or generated. Lean: hand-written, small surface.
- Version pinning convention in `pyproject.toml` for `torch` — `>=` minimum vs. compatible-release. Validate during first wheel build.
- Whether `cibuildwheel`'s Windows+CUDA support is mature enough or needs custom GHA steps. Validate early; fall back to "no Windows wheel in v1" if blocked.

## Future (out of scope for v1)

Tracked here so we don't paint into a corner:

- **In-memory cameras / images** — promote `_core.Camera` to public; add a `Camera.from_numpy(image, intrinsics, pose)` constructor; add `InputData.from_cameras([...])`.
- **Mid-training render hooks** — add a `Trainer.render(camera)` method and a `StepResult.render` field.
- **Standalone rasterization ops** — expose `ProjectGaussians` / `RasterizeGaussians` / `SphericalHarmonics` as `torch.autograd.Function` subclasses callable from Python independent of `Model`. Lets researchers use OpenSplat's kernels in their own gsplat models.
- **Refactor `opensplat.cpp` `main()` to call into a shared C++ `TrainSession`** — eliminates the duplication accepted in v1. Tier 3 parity test makes this a safe refactor when the duplication actually causes pain.
- **HIP/ROCm wheels** when a runner is available.

None of these require breaking changes to the v1 public API (`train` / `Trainer` / `StepResult`).
