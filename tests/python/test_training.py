"""Tier 2 — end-to-end training tests."""
from __future__ import annotations

from pathlib import Path


def test_trainer_constructs(colmap_mini: Path) -> None:
    from opensplat import Trainer
    trainer = Trainer(input=str(colmap_mini), num_iters=10, save_every=-1)
    assert trainer.num_iters == 10
    # device resolution: in CI we expect CPU
    assert str(trainer.device) in ("cpu", "cuda:0", "mps")


def test_trainer_yields_step_results(colmap_mini: Path) -> None:
    from opensplat import Trainer, StepResult

    trainer = Trainer(
        input=str(colmap_mini), num_iters=3, save_every=-1, sh_degree=1,
        device="cpu",
    )
    results = list(trainer)
    assert len(results) == 3
    assert all(isinstance(r, StepResult) for r in results)
    # CLI parity: opensplat.cpp iterates step=1..numIters (1-indexed).
    assert [r.step for r in results] == [1, 2, 3]
    assert all(r.loss > 0 for r in results)
    assert all(r.num_gaussians > 0 for r in results)


def test_camera_sampling_no_replacement() -> None:
    """The shuffle iterator must emit each item exactly once per pass.

    Mirrors the C++ InfiniteRandomIterator (utils.hpp): Fisher-Yates per pass,
    no repeats until the queue is exhausted, then reshuffle.
    """
    from opensplat.trainer import _InfiniteShuffleIterator

    items = ["a", "b", "c", "d"]
    it = _InfiniteShuffleIterator(items, seed=42)

    pass1 = [next(it) for _ in range(4)]
    assert sorted(pass1) == sorted(items), "first pass must visit each item once"
    assert len(set(pass1)) == 4, "no duplicates within a single pass"

    pass2 = [next(it) for _ in range(4)]
    assert sorted(pass2) == sorted(items), "second pass must visit each item once"
    assert len(set(pass2)) == 4, "no duplicates within a single pass"

    # Deterministic with seed.
    it2 = _InfiniteShuffleIterator(items, seed=42)
    pass1_repro = [next(it2) for _ in range(4)]
    assert pass1 == pass1_repro


def test_trainer_saves_on_normal_completion(colmap_mini: Path, tmp_output_ply: Path) -> None:
    from opensplat import Trainer
    trainer = Trainer(
        input=str(colmap_mini), output=str(tmp_output_ply),
        num_iters=2, save_every=-1, sh_degree=1, device="cpu",
    )
    trainer.run()
    assert tmp_output_ply.is_file() and tmp_output_ply.stat().st_size > 0


def test_trainer_saves_on_break(colmap_mini: Path, tmp_output_ply: Path) -> None:
    from opensplat import Trainer
    trainer = Trainer(
        input=str(colmap_mini), output=str(tmp_output_ply),
        num_iters=20, save_every=-1, sh_degree=1, device="cpu",
    )
    for r in trainer:
        if r.step >= 1:
            break
    assert tmp_output_ply.is_file() and tmp_output_ply.stat().st_size > 0


def test_train_function_round_trip(colmap_mini: Path, tmp_output_ply: Path) -> None:
    import opensplat
    opensplat.train(
        input=str(colmap_mini), output=str(tmp_output_ply),
        num_iters=2, save_every=-1, sh_degree=1, device="cpu",
    )
    assert tmp_output_ply.is_file() and tmp_output_ply.stat().st_size > 0


def test_trainer_writes_cameras_json(colmap_mini: Path, tmp_path: Path) -> None:
    """cameras.json sidecar lands next to the output PLY after training.

    CLI parity: opensplat.cpp writes `outputScene.parent_path() / "cameras.json"`
    once training completes. Downstream tools that consume OpenSplat output
    (e.g. nerfstudio-style viewers) expect this sidecar.
    """
    from opensplat import Trainer
    out = tmp_path / "scene.ply"
    trainer = Trainer(
        input=str(colmap_mini), output=str(out),
        num_iters=2, save_every=-1, sh_degree=1, device="cpu",
    )
    trainer.run()
    cameras_json = tmp_path / "cameras.json"
    assert cameras_json.is_file()
    assert cameras_json.stat().st_size > 0


def test_trainer_val_render_writes_pngs(colmap_mini: Path, tmp_path: Path) -> None:
    """When val_render is set, validation renders land in the given dir.

    CLI parity: opensplat.cpp renders the held-out validation camera every
    10 steps to `{val_render}/<step>.png`. With num_iters=10 we expect at
    least one PNG (step 10).
    """
    from opensplat import Trainer
    out = tmp_path / "scene.ply"
    val_render_dir = tmp_path / "val"
    trainer = Trainer(
        input=str(colmap_mini), output=str(out),
        num_iters=10, save_every=-1, sh_degree=1, device="cpu",
        val=True, val_render=str(val_render_dir),
    )
    trainer.run()
    pngs = list(val_render_dir.glob("*.png"))
    assert len(pngs) >= 1, f"expected at least one validation render in {val_render_dir}"
    assert (val_render_dir / "10.png").is_file()
