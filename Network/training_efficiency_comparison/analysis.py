"""Pure analysis helpers for training-efficiency benchmark outputs."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_ci(
    values: Iterable[float],
    *,
    seed: int = 1234,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    observed = [float(value) for value in values]
    if not observed:
        raise ValueError("cannot bootstrap an empty sequence")
    if len(observed) == 1:
        return observed[0], observed[0]
    if samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resample = [generator.choice(observed) for _ in observed]
        estimates.append(statistics.median(resample))
    tail = (1.0 - confidence) / 2.0
    return percentile(estimates, tail), percentile(estimates, 1.0 - tail)


def describe(values: Iterable[float], *, seed: int = 1234) -> dict[str, float | int]:
    observed = [float(value) for value in values]
    if not observed:
        raise ValueError("cannot describe an empty sequence")
    median = statistics.median(observed)
    lower, upper = bootstrap_median_ci(observed, seed=seed)
    mean = statistics.fmean(observed)
    stdev = statistics.stdev(observed) if len(observed) > 1 else 0.0
    return {
        "count": len(observed),
        "median": median,
        "mean": mean,
        "stdev": stdev,
        "coefficient_of_variation": stdev / mean if mean else 0.0,
        "p25": percentile(observed, 0.25),
        "p75": percentile(observed, 0.75),
        "bootstrap_median_ci95_low": lower,
        "bootstrap_median_ci95_high": upper,
    }


def describe_blocked(
    values: Iterable[float], block_values: Iterable[float], *, seed: int = 1234
) -> dict[str, float | int]:
    """Describe update-level values but bootstrap independent run-level blocks."""

    result = describe(values, seed=seed)
    blocks = [float(value) for value in block_values]
    low, high = bootstrap_median_ci(blocks, seed=seed)
    result["independent_blocks"] = len(blocks)
    result["block_bootstrap_median_ci95_low"] = low
    result["block_bootstrap_median_ci95_high"] = high
    return result


def _run_key(run: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        str(run["mode"]),
        str(run["method"]),
        int(run["physical_batch"]),
        int(run["effective_batch"]),
        int(run["time_steps"]),
    )


def summarize_runs(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    valid = [run for run in runs if run.get("status") == "ok"]
    grouped: dict[tuple[str, str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for run in valid:
        grouped[_run_key(run)].append(run)

    groups: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        mode, method, physical_batch, effective_batch, time_steps = key
        wall = [
            float(trial["wall_seconds"])
            for member in members
            for trial in member["trials"]
        ]
        throughput = [
            float(trial["images_per_second"])
            for member in members
            for trial in member["trials"]
        ]
        wall_blocks = [
            statistics.median(float(trial["wall_seconds"]) for trial in member["trials"])
            for member in members
        ]
        throughput_blocks = [
            statistics.median(
                float(trial["images_per_second"]) for trial in member["trials"]
            )
            for member in members
        ]
        phase_values: dict[str, list[float]] = defaultdict(list)
        for member in members:
            for trial in member["trials"]:
                for phase, seconds in trial.get("phase_seconds", {}).items():
                    phase_values[phase].append(float(seconds))
        peak_allocated = max(int(member["memory"]["peak_allocated_bytes"]) for member in members)
        peak_reserved = max(int(member["memory"]["peak_reserved_bytes"]) for member in members)
        steady_allocated = max(
            int(member["memory"]["steady_allocated_bytes"]) for member in members
        )
        groups.append(
            {
                "mode": mode,
                "method": method,
                "physical_batch": physical_batch,
                "effective_batch": effective_batch,
                "gradient_accumulation_steps": effective_batch // physical_batch,
                "time_steps": time_steps,
                "rounds": len(members),
                "wall_seconds": describe_blocked(wall, wall_blocks),
                "images_per_second": describe_blocked(throughput, throughput_blocks),
                "phase_seconds": {
                    phase: describe(values)
                    for phase, values in sorted(phase_values.items())
                },
                "memory": {
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "steady_allocated_bytes": steady_allocated,
                    "dynamic_peak_bytes": max(0, peak_allocated - steady_allocated),
                },
            }
        )

    comparisons: list[dict[str, Any]] = []
    by_mode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in groups:
        if group["method"] in {"qad", "squat"}:
            by_mode[str(group["mode"])][str(group["method"])] = group
    for mode, methods in sorted(by_mode.items()):
        if set(methods) != {"qad", "squat"}:
            continue
        qad = methods["qad"]
        squat = methods["squat"]
        qad_peak = float(qad["memory"]["peak_allocated_bytes"])
        squat_peak = float(squat["memory"]["peak_allocated_bytes"])
        qad_throughput = float(qad["images_per_second"]["median"])
        squat_throughput = float(squat["images_per_second"]["median"])
        qad_time = float(qad["wall_seconds"]["median"])
        squat_time = float(squat["wall_seconds"]["median"])
        memory_comparable = int(qad["physical_batch"]) == int(squat["physical_batch"])
        comparison = {
            "mode": mode,
            "memory_comparable": memory_comparable,
            "qad_peak_memory_bytes": int(qad_peak),
            "squat_peak_memory_bytes": int(squat_peak),
            "qad_throughput_speedup": qad_throughput / squat_throughput,
            "qad_time_reduction_fraction": 1.0 - qad_time / squat_time,
        }
        if memory_comparable:
            comparison.update(
                qad_memory_reduction_fraction=1.0 - qad_peak / squat_peak,
                squat_to_qad_memory_ratio=squat_peak / qad_peak,
            )
        comparisons.append(comparison)
    return {"groups": groups, "comparisons": comparisons}
