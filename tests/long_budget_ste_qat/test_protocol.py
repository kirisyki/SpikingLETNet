from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch.nn as nn


NETWORK_DIR = Path(__file__).resolve().parents[2] / "Network"
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))

from long_budget_ste_qat.protocol import (  # noqa: E402
    LONG_PROTOCOL_VERSION,
    LongBudgetSTEProtocol,
)
from long_budget_ste_qat.run_ste_qat_127 import (  # noqa: E402
    audited_build_quantized_model,
)


def test_long_protocol_restores_run_budget_but_not_lr_horizon() -> None:
    protocol = LongBudgetSTEProtocol("ste")
    payload = protocol.to_dict()
    assert payload["protocol_version"] == LONG_PROTOCOL_VERSION
    assert protocol.run_epochs == 127
    assert protocol.schedule_epochs == 150
    assert protocol.run_iterations == 127 * 401
    assert protocol.schedule_iterations == 150 * 401


def test_long_protocol_rejects_non_ste_method() -> None:
    with pytest.raises(ValueError, match="restricted to STE-QAT"):
        LongBudgetSTEProtocol("lsq")


def test_factory_keeps_first_image_input_and_all_weights_quantized() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Conv2d(4, 4, 1))
    with pytest.raises(RuntimeError, match="expected 71 historical QLayers"):
        audited_build_quantized_model("ste", model, expected_layer_count=None)
