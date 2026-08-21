from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "Network"
if str(NETWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(NETWORK_ROOT))

import aggregate_quantization_multiseed as aggregator  # noqa: E402
import evaluate_quantization_multiseed as evaluator  # noqa: E402
import train_qad_multiseed as qad_runner  # noqa: E402
from train_quantization_baseline_multiseed import protocol_factory  # noqa: E402


def test_baseline_protocol_factory_changes_only_seed():
    protocol = protocol_factory(2345)("lsq")
    assert protocol.seed == 2345
    assert protocol.run_epochs == 16
    assert protocol.schedule_epochs == 150
    assert protocol.train_batch_size == 64
    assert protocol.weight_bits == 4
    assert protocol.activation_bits == 4


def test_qad_protocol_preserves_long_budget_and_shared_settings():
    protocol = qad_runner.protocol_dict(3456, smoke=False)
    assert protocol["method"] == "qad"
    assert protocol["seed"] == 3456
    assert protocol["run_epochs"] == 127
    assert protocol["schedule_epochs"] == 150
    assert protocol["train_batch_size"] == 64
    assert protocol["expected_quantized_layers"] == 71
    assert protocol["weight_bits"] == 4
    assert protocol["activation_bits"] == 4


def test_qad_formal_seed_is_preregistered():
    valid = argparse.Namespace(
        seed=2345,
        smoke=False,
        max_train_iters=1,
        max_val_iters=1,
    )
    qad_runner.validate_args(valid)

    invalid = argparse.Namespace(
        seed=9999,
        smoke=False,
        max_train_iters=1,
        max_val_iters=1,
    )
    with pytest.raises(ValueError, match="formal seed"):
        qad_runner.validate_args(invalid)


def test_qad_namespace_separates_run_stop_from_lr_horizon(tmp_path):
    args = argparse.Namespace(
        seed=2345,
        output_dir=tmp_path,
        resume=None,
        smoke=False,
        max_train_iters=1,
        max_val_iters=1,
    )
    namespace = qad_runner.training_namespace(args, smoke=False)
    assert namespace.max_epochs == 150
    assert namespace.batch_size == 64
    assert namespace.max_train_iters_per_epoch == 0
    assert namespace.max_val_iters == 0


def test_multiseed_evaluator_expected_protocols_include_seed():
    assert evaluator.expected_protocol("qad", 2345)["seed"] == 2345
    assert evaluator.expected_protocol("qad", 2345)["run_epochs"] == 127
    assert evaluator.expected_protocol("ste", 3456)["seed"] == 3456
    assert evaluator.expected_protocol("ste", 3456)["run_epochs"] == 16


def test_aggregator_uses_training_seed_as_statistical_unit():
    payloads = {}
    for seed, offset in zip(aggregator.SEEDS, (0.0, 0.01, 0.02)):
        payloads[seed] = {"results": {}}
        for method, base in {
            "qad": 0.6,
            "ste": 0.5,
            "lsq": 0.4,
            "ewgs": 0.3,
        }.items():
            payloads[seed]["results"][method] = {
                "miou": base + offset,
                "per_class_iou": {
                    name: base + offset for name in aggregator.CLASS_NAMES
                },
                "checkpoint": {"epoch": 1},
            }
    summary = aggregator.summarize(payloads)
    assert summary["methods"]["qad"]["mean"] == pytest.approx(0.61)
    assert summary["methods"]["qad"]["sample_sd"] == pytest.approx(0.01)
    assert summary["paired_qad_minus_baseline"]["ste"]["mean"] == pytest.approx(0.1)
    assert summary["paired_qad_minus_baseline"]["ste"]["qad_wins"] == 3
