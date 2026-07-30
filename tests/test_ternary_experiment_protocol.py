from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "Network"
if str(NETWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(NETWORK_ROOT))

from QAT_snn_ternary import current_lr  # noqa: E402
from evaluate_precision_comparison import conclusion  # noqa: E402


def _results(w4: float, ternary_a4: float, ternary_a1p58: float):
    return {
        "W4/A4": {"miou": w4},
        "W1.58/A4": {"miou": ternary_a4},
        "W1.58/A1.58": {"miou": ternary_a1p58},
    }


def test_poly_lr_endpoints():
    assert current_lr(1e-3, 0, 100, 0.9) == pytest.approx(1e-3)
    assert current_lr(1e-3, 100, 100, 0.9) == pytest.approx(0.0)
    assert 0.0 < current_lr(1e-3, 50, 100, 0.9) < 1e-3


def test_pre_registered_decision_supports_w4_at_two_points():
    decision = conclusion(_results(0.62, 0.60, 0.55))
    assert decision["verdict"] == "supports_w4_accuracy_value"
    assert decision["primary_gap_miou_points_w4_minus_w1p58_a4"] == pytest.approx(
        2.0
    )


def test_pre_registered_decision_requests_seeds_when_close():
    decision = conclusion(_results(0.615, 0.60, 0.55))
    assert decision["verdict"] == "ambiguous_add_two_seeds"


def test_pre_registered_decision_can_reject_w4_hypothesis():
    decision = conclusion(_results(0.60, 0.61, 0.55))
    assert decision["verdict"] == "does_not_support_w4_accuracy_value"
