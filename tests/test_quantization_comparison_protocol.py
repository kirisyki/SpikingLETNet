from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "Network"
if str(NETWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(NETWORK_ROOT))

from quantization_comparison.protocol import (  # noqa: E402
    PROTECTED_HASHES,
    ExperimentProtocol,
    validate_protected_hashes,
)


def test_frozen_protocol_iteration_budget_and_lr_anchors():
    protocol = ExperimentProtocol("ste")
    assert protocol.run_epochs == 16
    assert protocol.schedule_epochs == 150
    assert protocol.batches_per_epoch == 401
    assert protocol.run_iterations == 16 * 401
    assert protocol.schedule_iterations == 150 * 401
    assert protocol.model_lr(0) == pytest.approx(1e-3)
    assert protocol.model_lr(400) == pytest.approx(
        1e-3 * (1.0 - 400 / (150 * 401)) ** 0.9
    )


def test_ewgs_has_only_preregistered_distinct_quantizer_lr():
    ewgs = ExperimentProtocol("ewgs")
    lsq = ExperimentProtocol("lsq")
    assert ewgs.quantizer_lr(0) == pytest.approx(1e-5)
    assert lsq.quantizer_lr(0) == pytest.approx(1e-3)
    assert ewgs.model_lr(123) / ewgs.quantizer_lr(123) == pytest.approx(100.0)


def test_invalid_protocol_is_rejected():
    with pytest.raises(ValueError):
        ExperimentProtocol("unknown")
    with pytest.raises(ValueError):
        ExperimentProtocol("ste", weight_bits=3)


def test_protected_files_still_match_preregistered_hashes():
    observed = validate_protected_hashes()
    assert observed == PROTECTED_HASHES
    for relative, expected in observed.items():
        actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
