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
    assert [r.step for r in results] == [0, 1, 2]
    assert all(r.loss > 0 for r in results)
    assert all(r.num_gaussians > 0 for r in results)


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
