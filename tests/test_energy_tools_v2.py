from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_energy_reports_v2 as reports  # noqa: E402
import compare_legacy_energy_v2 as reconciliation  # noqa: E402
import estimate_spikingletnet_energy_v2 as estimator  # noqa: E402
import measure_pro6000_ann_energy_v2 as measurement  # noqa: E402
from energy_accounting import LayerModePolicy, cv_percent, strict_json_dump  # noqa: E402


ENERGY = {
    "fp": {"mac": 1.0, "ac": 1.0, "mem_read": 0.0, "mem_write": 0.0, "neuron_update": 0.0},
    "int4": {"mac": 1.0, "ac": 1.0, "mem_read": 0.0, "mem_write": 0.0, "neuron_update": 0.0},
}


class SpikePolicy:
    def classify(self, _name: str, _variant: str, _op_type: str):
        return "spike", "unit-test"


class EnergyAccountingTests(unittest.TestCase):
    def test_conv_macs_contains_leading_time_dimension(self) -> None:
        conv = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
        timestep, batch, height, width = 8, 2, 5, 7
        input_tensor = torch.empty(timestep, batch, 2, height, width)
        output_tensor = torch.empty(timestep, batch, 3, height, width)
        expected_one_step = batch * 3 * height * width * 2 * 3 * 3
        self.assertEqual(
            estimator.conv_macs(conv, input_tensor, output_tensor),
            timestep * expected_one_step,
        )

    def test_qif_cumulative_spike_count_and_firing_rate_are_equivalent(self) -> None:
        timestep = 8
        class MultiStepConv(nn.Conv2d):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return inputs

        model = nn.Sequential(MultiStepConv(1, 1, kernel_size=1, bias=False))
        collector = estimator.EnergyHookCollector(
            model=model,
            variant="max",
            energy=ENERGY,
            timestep=timestep,
            integer_eps=1e-6,
            include_extended=False,
            accounting_mode="mixed",
            classification_mode="semantic-policy",
            policy=SpikePolicy(),
        )
        spike = torch.zeros(timestep, 1, 1, 4, 4)
        spike[..., :2, :2] = 1.0  # mean cumulative spike count = 0.25
        try:
            model(spike)
        finally:
            collector.remove()
        stats = collector.stats["0"]
        row = stats.as_row("max", "fp", processed_images=1, timestep=timestep)
        self.assertEqual(row["dense_macs_executed"], 128.0)
        self.assertEqual(row["dense_macs_one_step_equivalent"], 16.0)
        self.assertEqual(row["mean_spike_count_per_input"], 0.25)
        self.assertEqual(row["configured_firing_rate"], 0.03125)
        self.assertEqual(row["sop_density_per_executed_mac"], 0.03125)
        self.assertEqual(row["sop_total"], 4.0)
        self.assertEqual(
            row["dense_macs_executed"] * row["sop_density_per_executed_mac"],
            row["dense_macs_one_step_equivalent"]
            * row["mean_spike_count_per_input"],
        )

    def test_attention_matmuls_count_both_products_and_all_chunks(self) -> None:
        class Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.reduce = nn.Linear(512, 256)
                self.num_heads = 8

        batch, tokens = 8, 625
        sizes = [156, 156, 156, 156, 1]
        expected = 2 * batch * 8 * 32 * sum(size * size for size in sizes)
        self.assertEqual(
            estimator.attention_matmul_macs(Attention(), torch.empty(batch, tokens, 512)),
            expected,
        )

    def test_attention_matmuls_unwrap_qat_reduce_layer(self) -> None:
        class QuantizedAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.reduce = estimator.QLayer(nn.Linear(64, 32), name="attention.reduce")
                self.num_heads = 8

        expected = 2 * 1 * 8 * 4 * (4 * 4 * 4)
        self.assertEqual(
            estimator.attention_matmul_macs(
                QuantizedAttention(), torch.empty(1, 16, 64)
            ),
            expected,
        )

    def test_semantic_policy_covers_all_model_variants(self) -> None:
        policy = LayerModePolicy.load(PROJECT_ROOT / "tools/energy_layer_policy_v2.yaml")
        for variant, model_class in estimator.VARIANT_CLASSES.items():
            model = model_class(classes=6, config=str(estimator.DEFAULT_CONFIG))
            collector = estimator.EnergyHookCollector(
                model=model,
                variant=variant,
                energy=ENERGY,
                timestep=8,
                integer_eps=1e-4,
                include_extended=False,
                accounting_mode="mixed",
                classification_mode="semantic-policy",
                policy=policy,
            )
            self.assertGreater(len(collector.stats), 0)
            collector.remove()

    def test_policy_variant_specific_downsample_semantics(self) -> None:
        policy = LayerModePolicy.load(PROJECT_ROOT / "tools/energy_layer_policy_v2.yaml")
        self.assertEqual(policy.classify("downsample_1.conv3x3.conv", "small", "conv2d")[0], "dense")
        self.assertEqual(policy.classify("downsample_1.conv3x3.conv", "middle", "conv2d")[0], "dense")
        self.assertEqual(policy.classify("downsample_1.conv3x3.conv", "max", "conv2d")[0], "spike")

    def test_strict_json_rejects_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                strict_json_dump({"bad": float("nan")}, Path(directory) / "bad.json")

    def test_cv_percent(self) -> None:
        self.assertEqual(cv_percent([2.0]), 0.0)
        self.assertEqual(cv_percent([0.0, 0.0]), 0.0)
        self.assertTrue(math.isclose(cv_percent([1.0, 1.0]), 0.0))


class MeasurementMathTests(unittest.TestCase):
    @staticmethod
    def sample(time_s: float, power_w: float) -> measurement.PowerSample:
        return measurement.PowerSample(
            trial=1,
            phase="active",
            rel_time_s=time_s,
            wall_time_s=time_s,
            power_w=power_w,
            temperature_c=30,
            gpu_util_pct=50,
            sm_clock_mhz=1000,
            mem_clock_mhz=1000,
        )

    def test_bounded_power_integration_interpolates_window_edges(self) -> None:
        samples = [self.sample(0.0, 100.0), self.sample(1.0, 200.0)]
        self.assertAlmostEqual(measurement.integrate_energy_window_j(samples, 0.25, 0.75), 75.0)

    def test_stable_idle_mean_uses_tail_window(self) -> None:
        samples = [self.sample(float(second), 10.0 + second) for second in range(11)]
        self.assertEqual(measurement.stable_mean_power(samples, duration_s=10.0, stable_seconds=2.0), 19.0)


class ReportBuilderTests(unittest.TestCase):
    def test_mixed_precision_energy_is_computed_per_layer(self) -> None:
        rows = [
            {
                "variant": "max", "checkpoint_kind": "qat", "precision": "fp", "mode": "dense",
                "dense_macs_charged_per_image": "10", "sop_per_image": "0", "input_elems": "0",
                "input_positive_sum": "0", "input_nonzero": "0", "layer": "attention",
            },
            {
                "variant": "max", "checkpoint_kind": "qat", "precision": "int4", "mode": "spike",
                "dense_macs_charged_per_image": "0", "sop_per_image": "20", "input_elems": "100",
                "input_positive_sum": "25", "input_firing_rate_sum": "3.125",
                "input_nonzero": "20", "layer": "conv",
            },
        ]
        summary = {
            "precision": "mixed_fp_int4", "timestep": 8, "processed_images": 1,
            "classification_mode": "semantic-policy",
            "sop_method": "qif_cumulative_spike_count_times_one_step_dense_macs_v2",
        }
        energy_sets = {"test": {"fp": {"mac": 4.0, "ac": 1.0}, "int4": {"mac": 0.2, "ac": 0.1}}}
        energy_rows, operation, sparsity = reports.summarize_group(
            "t8", ("max", "qat"), rows, summary, energy_sets, Path("layer_energy.csv")
        )
        self.assertEqual(operation["fp_dense_macs_per_image"], 10.0)
        self.assertEqual(operation["int4_sops_per_image"], 20.0)
        self.assertAlmostEqual(energy_rows[0]["total_core_energy_mj_per_image"], 42.0 / 1e9)
        self.assertEqual(sparsity["mean_spike_count_per_input"], 0.25)
        self.assertEqual(sparsity["configured_firing_rate"], 0.03125)
        self.assertEqual(sparsity["sop_density_per_executed_mac"], 0.03125)


    def test_all_dense_run_has_not_applicable_sparsity(self) -> None:
        rows = [{
            "variant": "max", "checkpoint_kind": "fp", "precision": "fp", "mode": "dense",
            "dense_macs_charged_per_image": "10", "sop_per_image": "0", "input_elems": "8",
            "input_positive_sum": "4", "input_nonzero": "4", "layer": "conv",
        }]
        summary = {
            "precision": "fp", "timestep": 1, "processed_images": 1,
            "classification_mode": "semantic-policy",
            "sop_method": "qif_cumulative_spike_count_times_one_step_dense_macs_v2",
        }
        energy_sets = {"test": {"fp": {"mac": 1.0, "ac": 1.0}, "int4": {"mac": 1.0, "ac": 1.0}}}
        _, _, sparsity = reports.summarize_group(
            "t1", ("max", "fp"), rows, summary, energy_sets, Path("layer_energy.csv")
        )
        self.assertIsNone(sparsity["mean_spike_count_per_input"])
        self.assertIsNone(sparsity["zero_ratio"])


class ReconciliationTests(unittest.TestCase):
    def test_legacy_csv_reader_strips_pretty_print_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            path.write_text(" network , variant \n FP     , max     \n")
            rows = reconciliation.read_csv(path)
            self.assertEqual(rows[0]["network"], "FP")
            self.assertEqual(rows[0]["variant"], "max")


if __name__ == "__main__":
    unittest.main()
