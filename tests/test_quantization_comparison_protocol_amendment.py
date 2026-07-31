from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "Network"
if str(NETWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(NETWORK_ROOT))

from quantization_comparison.checkpointing import save_checkpoint  # noqa: E402
from quantization_comparison.protocol import ExperimentProtocol  # noqa: E402
from quantization_comparison.protocol_amendment import (  # noqa: E402
    amend_checkpoint_metadata,
    is_compatible_epoch_amendment,
)


def old_protocol(method: str) -> dict[str, object]:
    protocol = ExperimentProtocol(method).to_dict()
    protocol["protocol_version"] = "w4a4-comparison-v1"
    protocol["run_epochs"] = 127
    return protocol


def test_only_exact_epoch_amendment_is_compatible():
    current = ExperimentProtocol("ste").to_dict()
    old = old_protocol("ste")
    assert is_compatible_epoch_amendment(old, current)
    changed_seed = dict(old)
    changed_seed["seed"] = 999
    assert not is_compatible_epoch_amendment(changed_seed, current)


def test_metadata_amendment_preserves_model_and_optimizer_state(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=16,
        global_iteration=16 * 401,
        best_miou=0.5,
        protocol=old_protocol("ste"),
        manifest={"old": True},
    )
    before = torch.load(path, map_location="cpu", weights_only=False)
    current = ExperimentProtocol("ste").to_dict()
    amend_checkpoint_metadata(
        path,
        protocol=current,
        manifest={"amended": True},
    )
    after = torch.load(path, map_location="cpu", weights_only=False)
    assert after["protocol"] == current
    assert after["manifest"] == {"amended": True}
    assert after["optimizer_state"] == before["optimizer_state"]
    assert after["rng_state"]["python"] == before["rng_state"]["python"]
    for key, tensor in before["model_state"].items():
        torch.testing.assert_close(after["model_state"][key], tensor)
