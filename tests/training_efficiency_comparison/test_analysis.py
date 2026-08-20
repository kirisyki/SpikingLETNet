from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

NETWORK_ROOT = Path(__file__).resolve().parents[2] / "Network"
if str(NETWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(NETWORK_ROOT))

from training_efficiency_comparison.analysis import (  # noqa: E402
    bootstrap_median_ci,
    describe,
    percentile,
    summarize_runs,
)


def test_percentile_interpolates_and_rejects_empty() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.25) == pytest.approx(1.75)
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_describe_and_bootstrap_are_deterministic() -> None:
    first = describe([1.0, 2.0, 3.0], seed=7)
    second = describe([1.0, 2.0, 3.0], seed=7)
    assert first == second
    assert first["median"] == 2.0
    assert bootstrap_median_ci([4.0]) == (4.0, 4.0)


def _run(method: str, seconds: float, peak: int) -> dict:
    return {
        "status": "ok",
        "mode": "matched",
        "method": method,
        "physical_batch": 4,
        "effective_batch": 4,
        "time_steps": 1 if method == "qad" else 8,
        "memory": {
            "peak_allocated_bytes": peak,
            "peak_reserved_bytes": peak + 10,
            "steady_allocated_bytes": 20,
        },
        "trials": [
            {
                "wall_seconds": seconds,
                "images_per_second": 4 / seconds,
                "phase_seconds": {"forward": seconds / 2},
            }
        ],
    }


def test_summarize_runs_calculates_route_ratios() -> None:
    summary = summarize_runs([_run("qad", 1.0, 100), _run("squat", 4.0, 400)])
    assert len(summary["groups"]) == 2
    comparison = summary["comparisons"][0]
    assert comparison["qad_memory_reduction_fraction"] == pytest.approx(0.75)
    assert comparison["squat_to_qad_memory_ratio"] == pytest.approx(4.0)
    assert comparison["qad_throughput_speedup"] == pytest.approx(4.0)
    assert comparison["qad_time_reduction_fraction"] == pytest.approx(0.75)
    assert math.isfinite(comparison["qad_throughput_speedup"])


def test_capacity_memory_ratio_is_omitted_for_unequal_physical_batches() -> None:
    qad = _run("qad", 1.0, 100)
    squat = _run("squat", 4.0, 400)
    qad["mode"] = squat["mode"] = "capacity"
    qad["physical_batch"] = 32
    squat["physical_batch"] = 4
    qad["effective_batch"] = squat["effective_batch"] = 64
    comparison = summarize_runs([qad, squat])["comparisons"][0]
    assert comparison["memory_comparable"] is False
    assert "qad_memory_reduction_fraction" not in comparison

