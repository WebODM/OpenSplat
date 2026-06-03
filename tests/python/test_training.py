"""Tier 2 — end-to-end training tests."""
from __future__ import annotations

from pathlib import Path


def test_trainer_constructs(colmap_mini: Path) -> None:
    from opensplat import Trainer
    trainer = Trainer(input=str(colmap_mini), num_iters=10, save_every=-1)
    assert trainer.num_iters == 10
    # device resolution: in CI we expect CPU
    assert str(trainer.device) in ("cpu", "cuda:0", "mps")
