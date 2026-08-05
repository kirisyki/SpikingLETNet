from __future__ import annotations

from pathlib import Path

import pytest

import squat_comparison.protocol as protocol_module
from squat_comparison.protocol import (
    ExperimentProtocol,
    assert_writable_experiment_path,
    frozen_historical_results,
    scaled_epoch_budgets,
)


def test_frozen_qad_metrics_are_read_only_reference() -> None:
    frozen = frozen_historical_results(ExperimentProtocol())
    assert frozen["qad"]["miou"] == pytest.approx(0.6195473169172718)
    assert frozen["fp32_qif"]["miou"] == pytest.approx(0.6785024141945621)
    assert "no QAD train/eval/convert" in frozen["execution_policy"]


def test_output_guard_rejects_old_and_outside_directories() -> None:
    with pytest.raises(PermissionError):
        assert_writable_experiment_path(protocol_module.REPO_ROOT / "QAT_checkpoint/x")
    with pytest.raises(PermissionError):
        assert_writable_experiment_path(Path("/tmp/not-the-squat-output"))
    accepted = assert_writable_experiment_path(protocol_module.OUTPUT_ROOT / "fp32_snn/x")
    assert accepted.is_relative_to(protocol_module.OUTPUT_ROOT.resolve())


def test_48_hour_budget_rule() -> None:
    protocol = ExperimentProtocol()
    assert scaled_epoch_budgets(47.9, protocol) == (300, 127, 1.0)
    fp_epochs, squat_epochs, scale = scaled_epoch_budgets(88.0, protocol)
    assert (fp_epochs, squat_epochs, scale) == (150, 63, 0.5)
    assert scaled_epoch_budgets(1000.0, protocol)[:2] == (100, 40)

