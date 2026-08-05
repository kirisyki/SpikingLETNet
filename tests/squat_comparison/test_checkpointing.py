from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

import squat_comparison.protocol as protocol_module
from squat_comparison.checkpointing import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_scheduler_rng_and_best(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(protocol_module, "OUTPUT_ROOT", tmp_path)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    x = torch.ones(1, 2)
    model(x).sum().backward()
    optimizer.step()
    scheduler.step()
    generator = torch.Generator().manual_seed(1234)
    path = tmp_path / "run/checkpoint.pth"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=7,
        best_miou=0.42,
        protocol={"seed": 1234},
        loader_generator_state=generator.get_state(),
    )
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)
    with torch.no_grad():
        model.weight.zero_()
    payload = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=True,
    )
    assert payload["epoch"] == 2 and payload["global_step"] == 7
    assert payload["best_miou"] == 0.42
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch)

