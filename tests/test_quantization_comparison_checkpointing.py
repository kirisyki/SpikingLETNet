from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "Network"
if str(NETWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(NETWORK_ROOT))

from quantization_comparison.checkpointing import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
)


def test_state_dict_checkpoint_round_trip_and_rng_restore(tmp_path):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()

    path = tmp_path / "checkpoint_last.pth"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=7,
        global_iteration=2807,
        best_miou=0.5,
        protocol={"method": "lsq"},
        manifest={"source": "unit-test"},
    )
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)

    with torch.no_grad():
        model.weight.zero_()
    random.random()
    np.random.rand()
    torch.rand(1)

    payload = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        restore_rng=True,
    )
    assert payload["epoch"] == 7
    assert payload["global_iteration"] == 2807
    assert payload["best_miou"] == 0.5
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch)
    assert not torch.all(model.weight == 0)
