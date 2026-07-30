#!/usr/bin/env python3
"""Synthetic end-to-end regression for HWiNFO marker integration and reporting."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import generate_report
import run_hwinfo_energy
from common import read_json, write_json


def marker(
    trial_id: str,
    action: str,
    model: str | None,
    role: str,
    started: datetime,
    duration_s: float,
) -> dict:
    ended = started + timedelta(seconds=duration_s)
    iterations = 0
    if model == "fp32":
        iterations = 200
    elif model == "w8a8":
        iterations = 400
    return {
        "trial_id": trial_id,
        "action": action,
        "model": model,
        "acquisition_role": role,
        "background_cpu_percent_before": 1.0,
        "client": {
            "action": action,
            "model": model,
            "started_epoch_ns": int(started.timestamp() * 1_000_000_000),
            "ended_epoch_ns": int(ended.timestamp() * 1_000_000_000),
            "elapsed_seconds": duration_s,
            "iterations": iterations,
        },
    }


def build_markers(started: datetime, duration_s: float = 20.0) -> list[dict]:
    trials = []
    cursor = started + timedelta(seconds=10)

    for index in range(3):
        trials.append(
            marker(
                f"idle_{index + 1:02d}",
                "idle",
                None,
                "idle_baseline",
                cursor,
                duration_s,
            )
        )
        cursor += timedelta(seconds=duration_s + 5)

    for round_index in range(8):
        labels = ("fp32", "w8a8") if round_index % 2 == 0 else ("w8a8", "fp32")
        role = "base" if round_index < 5 else "supplemental_reserve"
        prefix = "round" if role == "base" else "reserve"
        display_index = (
            round_index + 1 if role == "base" else round_index - 5 + 1
        )
        for label in labels:
            trials.append(
                marker(
                    f"{prefix}_{display_index:02d}_{label}",
                    "run",
                    label,
                    role,
                    cursor,
                    duration_s,
                )
            )
            cursor += timedelta(seconds=duration_s + 5)
    return trials


def write_log(path: Path, started: datetime, trials: list[dict]) -> None:
    final_epoch = max(
        item["client"]["ended_epoch_ns"] / 1_000_000_000 for item in trials
    )
    sample_count = int(final_epoch - started.timestamp()) + 11
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Date",
                "Time",
                "CPU Package Power [W]",
                "Core Distance to TjMAX [°C]",
                "CPU Package [°C]",
                "Core Thermal Throttling",
            ]
        )
        for offset in range(sample_count):
            current = started + timedelta(seconds=offset)
            current_epoch = current.timestamp()
            power_w = 10.0
            for item in trials:
                client = item["client"]
                if (
                    client["started_epoch_ns"] / 1_000_000_000
                    <= current_epoch
                    <= client["ended_epoch_ns"] / 1_000_000_000
                ):
                    if item["model"] == "fp32":
                        power_w = 50.0
                    elif item["model"] == "w8a8":
                        power_w = 30.0
                    break
            writer.writerow(
                [
                    current.strftime("%m/%d/%Y"),
                    current.strftime("%H:%M:%S"),
                    f"{power_w:.1f}",
                    "25.0",
                    "75.0",
                    "No",
                ]
            )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results = root / "results"
        results.mkdir()
        started = datetime.now().astimezone().replace(microsecond=0)
        trials = build_markers(started)
        log_path = root / "hwinfo.csv"
        marker_path = root / "markers.json"
        write_log(log_path, started, trials)
        write_json(
            marker_path,
            {
                "record_complete": True,
                "provider": "cpu",
                "measurement_backend": "hwinfo_csv_power_integration",
                "selected_config": {
                    "affinity_name": "synthetic",
                    "cpu_ids": [0],
                    "threads": 1,
                },
                "settings": {
                    "duration_seconds": 20.0,
                    "warmup_seconds": 1.0,
                    "base_trials_per_model": 5,
                    "supplemental_reserve_trials_per_model": 3,
                    "idle_trials": 3,
                    "input_samples": 1,
                    "maximum_background_cpu_percent": 10.0,
                },
                "trials": trials,
            },
        )
        args = argparse.Namespace(
            provider="cpu",
            hwinfo_log=log_path,
            markers=marker_path,
            hwinfo_version="synthetic",
            date_order="mdy",
            power_column=None,
            minimum_sample_completeness=0.99,
            minimum_median_interval_seconds=0.8,
            maximum_median_interval_seconds=1.2,
            maximum_sampling_gap_seconds=2.5,
            maximum_integration_spread=0.01,
            minimum_thermal_headroom_c=2.0,
            maximum_temperature_c=98.0,
            maximum_background_cpu_percent=10.0,
        )

        original_pipeline_results = run_hwinfo_energy.RESULTS_DIR
        original_report_results = generate_report.RESULTS_DIR
        original_argv = sys.argv
        try:
            run_hwinfo_energy.RESULTS_DIR = results
            run_hwinfo_energy.analyze(args)
            output = read_json(results / "energy_hwinfo_cpu.json")
            assert output["stable"]
            assert output["supplemental_reserve_trials_used"] == {
                "fp32": 0,
                "w8a8": 0,
            }
            fp32 = output["models"]["fp32"]
            w8a8 = output["models"]["w8a8"]
            assert abs(fp32["hwinfo_cpu_package_j_per_image_mean"] - 5.0) < 1e-9
            assert (
                abs(fp32["hwinfo_dynamic_cpu_package_j_per_image_mean"] - 4.0)
                < 1e-9
            )
            assert abs(w8a8["hwinfo_cpu_package_j_per_image_mean"] - 1.5) < 1e-9
            assert (
                abs(w8a8["hwinfo_dynamic_cpu_package_j_per_image_mean"] - 1.0)
                < 1e-9
            )
            json.dumps(output, allow_nan=False)

            generate_report.RESULTS_DIR = results
            sys.argv = ["generate_report.py", "--providers", "cpu"]
            generate_report.main()
            report = (results / "result_summary.md").read_text(encoding="utf-8")
            assert "HWiNFO CPU Package Power CSV 梯形积分" in report
            assert "不是 PCM/RAPL 计数器差值" in report
            assert (results / "raw_hwinfo" / "cpu" / log_path.name).is_file()
        finally:
            run_hwinfo_energy.RESULTS_DIR = original_pipeline_results
            generate_report.RESULTS_DIR = original_report_results
            sys.argv = original_argv
    print("PASS: synthetic HWiNFO pipeline")


if __name__ == "__main__":
    main()
