#!/usr/bin/env python3
"""Record inference windows and integrate an externally captured HWiNFO power log."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from common import RESULTS_DIR, environment_summary, read_json, sha256_file, utc_now, write_json
from hwinfo_tools import integrate_window, read_hwinfo_csv


SCRIPT_DIR = Path(__file__).resolve().parent
LABELS = ("fp32", "w8a8")


def cv(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else float("inf")


def send_request(
    host: str,
    port: int,
    request: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout_seconds) as connection:
        connection.settimeout(timeout_seconds)
        connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
        chunks = []
        while chunk := connection.recv(65536):
            chunks.append(chunk)
    response = json.loads(b"".join(chunks).decode("utf-8"))
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "Energy server returned an unknown error"))
    return response


def wait_for_ready(path: Path, process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if path.is_file():
            return read_json(path)
        if process.poll() is not None:
            raise RuntimeError(f"Energy server exited early with code {process.returncode}")
        time.sleep(0.25)
    raise TimeoutError(f"Energy server did not become ready within {timeout} seconds")


def marker_path(provider: str) -> Path:
    return RESULTS_DIR / f"hwinfo_trial_markers_{provider}.json"


def status_path(provider: str) -> Path:
    return RESULTS_DIR / f"energy_status_hwinfo_{provider}.json"


def write_record_state(
    args: argparse.Namespace,
    selected: dict[str, Any],
    trials: list[dict[str, Any]],
    *,
    complete: bool,
) -> None:
    report = {
        "generated_at_utc": utc_now(),
        "provider": args.provider,
        "measurement_backend": "hwinfo_csv_power_integration",
        "record_complete": complete,
        "requires_external_log": True,
        "precondition": "HWiNFO Sensors-only logging must run continuously before warmup until after the last trial.",
        "selected_config": selected,
        "settings": {
            "duration_seconds": args.duration_seconds,
            "warmup_seconds": args.warmup_seconds,
            "base_trials_per_model": args.trials,
            "supplemental_reserve_trials_per_model": args.max_extra_trials,
            "idle_trials": args.idle_trials,
            "input_samples": args.input_samples,
            "maximum_background_cpu_percent": args.maximum_background_cpu_percent,
        },
        "trials": trials,
    }
    write_json(marker_path(args.provider), report)
    write_json(
        status_path(args.provider),
        {
            "generated_at_utc": utc_now(),
            "provider": args.provider,
            "energy_backend": "hwinfo_csv_power_integration",
            "status": "awaiting_hwinfo_log_analysis" if complete else "recording",
            "marker_file": str(marker_path(args.provider)),
            "completed_trials": len(trials),
        },
    )


def record_trial(
    args: argparse.Namespace,
    ready: dict[str, Any],
    action: str,
    model: str | None,
    trial_id: str,
    acquisition_role: str,
) -> dict[str, Any]:
    background_cpu = psutil.cpu_percent(interval=1.0)
    response = send_request(
        ready["host"],
        ready["port"],
        {"action": action, "model": model, "duration": args.duration_seconds},
        timeout_seconds=args.duration_seconds + 120.0,
    )
    required = ("started_epoch_ns", "ended_epoch_ns", "elapsed_seconds", "iterations")
    missing = [name for name in required if name not in response]
    if missing:
        raise RuntimeError(f"Energy server response is missing timestamp fields: {missing}")
    return {
        "trial_id": trial_id,
        "action": action,
        "model": model,
        "acquisition_role": acquisition_role,
        "background_cpu_percent_before": background_cpu,
        "client": response,
    }


def record(args: argparse.Namespace) -> None:
    tuning_path = RESULTS_DIR / f"tuning_{args.provider}.json"
    if not tuning_path.is_file():
        raise FileNotFoundError(
            f"Missing {tuning_path}; first run .\\run_all.ps1 -SkipEnergy"
        )
    selected = read_json(tuning_path)["selected"]
    ready_path = RESULTS_DIR / f"hwinfo_energy_server_{args.provider}_ready.json"
    server_log_path = RESULTS_DIR / f"hwinfo_energy_server_{args.provider}.log"
    ready_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "energy_server.py"),
        "--provider",
        args.provider,
        "--threads",
        str(selected["threads"]),
        "--cpu-ids",
        *[str(value) for value in selected["cpu_ids"]],
        "--ready-file",
        str(ready_path),
        "--warmup-seconds",
        str(args.warmup_seconds),
        "--input-samples",
        str(args.input_samples),
    ]
    server_log = server_log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
    trials: list[dict[str, Any]] = []
    write_record_state(args, selected, trials, complete=False)
    try:
        ready = wait_for_ready(
            ready_path, server, timeout=args.warmup_seconds * 2.0 + 180.0
        )
        for index in range(args.idle_trials):
            trials.append(
                record_trial(
                    args,
                    ready,
                    "idle",
                    None,
                    f"idle_{index + 1:02d}",
                    "idle_baseline",
                )
            )
            write_record_state(args, selected, trials, complete=False)

        total_rounds = args.trials + args.max_extra_trials
        for round_index in range(total_rounds):
            order = LABELS if round_index % 2 == 0 else tuple(reversed(LABELS))
            role = "base" if round_index < args.trials else "supplemental_reserve"
            prefix = "round" if role == "base" else "reserve"
            display_index = (
                round_index + 1
                if role == "base"
                else round_index - args.trials + 1
            )
            for label in order:
                trials.append(
                    record_trial(
                        args,
                        ready,
                        "run",
                        label,
                        f"{prefix}_{display_index:02d}_{label}",
                        role,
                    )
                )
                write_record_state(args, selected, trials, complete=False)
        write_record_state(args, selected, trials, complete=True)
        print(
            f"PASS: recorded {len(trials)} HWiNFO-aligned windows for {args.provider}. "
            "Stop HWiNFO logging, then run the Analyze phase."
        )
    finally:
        try:
            if ready_path.is_file():
                ready = read_json(ready_path)
                send_request(
                    ready["host"],
                    ready["port"],
                    {"action": "stop"},
                    timeout_seconds=30.0,
                )
        except Exception:
            pass
        try:
            server.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            server.terminate()
            server.wait(timeout=15.0)
        server_log.close()


def _stability(rows: list[dict[str, Any]]) -> tuple[bool, float, float]:
    if len(rows) < 5:
        return False, float("inf"), float("inf")
    energy_cv = cv([row["hwinfo_cpu_package_j_per_image"] for row in rows])
    latency_cv = cv([row["mean_latency_ms"] for row in rows])
    return energy_cv <= 0.05 and latency_cv <= 0.05, energy_cv, latency_cv


def select_trials(
    trials: list[dict[str, Any]], model: str
) -> tuple[list[dict[str, Any]], int]:
    base = [
        item
        for item in trials
        if item["model"] == model
        and item["acquisition_role"] == "base"
        and item["valid"]
    ]
    supplemental = [
        item
        for item in trials
        if item["model"] == model
        and item["acquisition_role"] == "supplemental_reserve"
        and item["valid"]
    ]
    selected = list(base)
    used = 0
    while len(selected) < 5 and used < len(supplemental):
        selected.append(supplemental[used])
        used += 1
    stable, _, _ = _stability([item["derived"] for item in selected])
    while not stable and used < len(supplemental):
        selected.append(supplemental[used])
        used += 1
        stable, _, _ = _stability([item["derived"] for item in selected])
    selected_ids = {item["trial_id"] for item in selected}
    for item in trials:
        if item["model"] == model:
            item["included_in_aggregate"] = item["trial_id"] in selected_ids
    return selected, used


def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [item["derived"] for item in selected]
    if not rows:
        return {"valid_trials": 0, "stable": False, "rows": []}
    stable, energy_cv, latency_cv = _stability(rows)
    energy = [row["hwinfo_cpu_package_j_per_image"] for row in rows]
    return {
        "valid_trials": len(rows),
        "stable": stable,
        "hwinfo_cpu_package_j_per_image_mean": statistics.fmean(energy),
        "hwinfo_cpu_package_j_per_image_median": statistics.median(energy),
        "hwinfo_cpu_package_j_per_image_cv": (
            energy_cv if len(rows) >= 5 else None
        ),
        "hwinfo_dynamic_cpu_package_j_per_image_mean": statistics.fmean(
            row["hwinfo_dynamic_cpu_package_j_per_image"] for row in rows
        ),
        "mean_latency_ms": statistics.fmean(row["mean_latency_ms"] for row in rows),
        "latency_cv": latency_cv if len(rows) >= 5 else None,
        "hwinfo_cpu_package_power_w_mean": statistics.fmean(
            row["hwinfo_cpu_package_power_w"] for row in rows
        ),
        "maximum_integration_method_relative_spread": max(
            row["integration_method_relative_spread"] for row in rows
        ),
        "rows": rows,
    }


def write_trials_csv(path: Path, trials: list[dict[str, Any]]) -> None:
    fields = [
        "trial_id",
        "action",
        "model",
        "acquisition_role",
        "included_in_aggregate",
        "valid",
        "invalid_reasons",
        "background_cpu_percent_before",
        "iterations",
        "elapsed_seconds",
        "mean_latency_ms",
        "hwinfo_cpu_package_energy_j",
        "hwinfo_cpu_package_power_w",
        "hwinfo_cpu_package_j_per_image",
        "hwinfo_dynamic_cpu_package_j_per_image",
        "sample_completeness",
        "median_interval_seconds",
        "maximum_gap_seconds",
        "integration_method_relative_spread",
        "thermal_headroom_min_c",
        "temperature_max_c",
        "thermal_throttling_observed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in trials:
            integration = item["integration"]
            derived = item.get("derived") or {}
            client = item["client"]
            writer.writerow(
                {
                    "trial_id": item["trial_id"],
                    "action": item["action"],
                    "model": item["model"],
                    "acquisition_role": item["acquisition_role"],
                    "included_in_aggregate": item.get("included_in_aggregate", False),
                    "valid": item["valid"],
                    "invalid_reasons": ";".join(item["invalid_reasons"]),
                    "background_cpu_percent_before": item["background_cpu_percent_before"],
                    "iterations": client["iterations"],
                    "elapsed_seconds": client["elapsed_seconds"],
                    "mean_latency_ms": derived.get("mean_latency_ms", ""),
                    "hwinfo_cpu_package_energy_j": integration.get("trapezoidal_energy_j", ""),
                    "hwinfo_cpu_package_power_w": derived.get("hwinfo_cpu_package_power_w", ""),
                    "hwinfo_cpu_package_j_per_image": derived.get(
                        "hwinfo_cpu_package_j_per_image", ""
                    ),
                    "hwinfo_dynamic_cpu_package_j_per_image": derived.get(
                        "hwinfo_dynamic_cpu_package_j_per_image", ""
                    ),
                    "sample_completeness": integration.get("sample_completeness", ""),
                    "median_interval_seconds": integration.get(
                        "median_interval_seconds", ""
                    ),
                    "maximum_gap_seconds": integration.get("maximum_gap_seconds", ""),
                    "integration_method_relative_spread": integration.get(
                        "integration_method_relative_spread", ""
                    ),
                    "thermal_headroom_min_c": integration.get(
                        "thermal_headroom_min_c", ""
                    ),
                    "temperature_max_c": integration.get("temperature_max_c", ""),
                    "thermal_throttling_observed": integration.get(
                        "thermal_throttling_observed", ""
                    ),
                }
            )


def analyze(args: argparse.Namespace) -> None:
    markers = read_json(args.markers or marker_path(args.provider))
    if not markers.get("record_complete"):
        raise RuntimeError("HWiNFO marker recording is incomplete")
    samples, log_metadata = read_hwinfo_csv(
        args.hwinfo_log,
        date_order=args.date_order,
        power_column=args.power_column,
    )
    raw_dir = RESULTS_DIR / "raw_hwinfo" / args.provider
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied_log = raw_dir / args.hwinfo_log.name
    if args.hwinfo_log.resolve() != copied_log.resolve():
        shutil.copy2(args.hwinfo_log, copied_log)

    trials: list[dict[str, Any]] = []
    for marker in markers["trials"]:
        client = marker["client"]
        integration = integrate_window(
            samples,
            client["started_epoch_ns"] / 1_000_000_000.0,
            client["ended_epoch_ns"] / 1_000_000_000.0,
            minimum_completeness=args.minimum_sample_completeness,
            minimum_median_interval_s=args.minimum_median_interval_seconds,
            maximum_median_interval_s=args.maximum_median_interval_seconds,
            maximum_gap_s=args.maximum_sampling_gap_seconds,
            maximum_integration_spread=args.maximum_integration_spread,
            minimum_thermal_headroom_c=args.minimum_thermal_headroom_c,
            maximum_temperature_c=args.maximum_temperature_c,
            require_thermal=True,
        )
        background_invalid = (
            marker["background_cpu_percent_before"]
            > args.maximum_background_cpu_percent
        )
        invalid_reasons = list(integration.get("quality_reasons") or [])
        invalid_reasons.extend(integration.get("thermal_invalid_reasons") or [])
        if background_invalid:
            invalid_reasons.append("background_cpu")
        trials.append(
            {
                **marker,
                "integration": integration,
                "valid": not invalid_reasons,
                "invalid_reasons": sorted(set(invalid_reasons)),
            }
        )

    idle = [item for item in trials if item["action"] == "idle" and item["valid"]]
    if len(idle) < 3:
        raise RuntimeError(
            f"Need three valid HWiNFO idle baselines, found {len(idle)}"
        )
    idle_energy = sum(item["integration"]["trapezoidal_energy_j"] for item in idle)
    idle_duration = sum(item["integration"]["duration_seconds"] for item in idle)
    idle_power_w = idle_energy / idle_duration

    for item in trials:
        client = item["client"]
        integration = item["integration"]
        if item["action"] != "run" or not item["valid"]:
            continue
        count = client["iterations"]
        elapsed = client["elapsed_seconds"]
        energy_duration = integration["duration_seconds"]
        if count <= 0:
            item["valid"] = False
            item["invalid_reasons"].append("zero_iterations")
            continue
        energy = integration["trapezoidal_energy_j"]
        item["derived"] = {
            "trial_id": item["trial_id"],
            "iterations": count,
            "elapsed_seconds": elapsed,
            "energy_window_seconds": energy_duration,
            "mean_latency_ms": elapsed / count * 1000.0,
            "hwinfo_cpu_package_energy_j": energy,
            "hwinfo_cpu_package_power_w": energy / energy_duration,
            "hwinfo_cpu_package_j_per_image": energy / count,
            "hwinfo_dynamic_cpu_package_energy_j": energy - idle_power_w * energy_duration,
            "hwinfo_dynamic_cpu_package_j_per_image": (
                energy - idle_power_w * energy_duration
            )
            / count,
            "integration_method_relative_spread": integration[
                "integration_method_relative_spread"
            ],
        }

    selected_by_model = {}
    supplemental_used = {}
    for label in LABELS:
        selected, used = select_trials(trials, label)
        selected_by_model[label] = selected
        supplemental_used[label] = used
    aggregates = {
        label: aggregate(selected_by_model[label]) for label in LABELS
    }
    fp32 = aggregates["fp32"]
    w8a8 = aggregates["w8a8"]
    comparison = None
    if (
        fp32.get("valid_trials", 0) >= 5
        and w8a8.get("valid_trials", 0) >= 5
    ):
        comparison = {
            "w8a8_hwinfo_cpu_package_energy_reduction_fraction": 1.0
            - w8a8["hwinfo_cpu_package_j_per_image_mean"]
            / fp32["hwinfo_cpu_package_j_per_image_mean"],
            "w8a8_hwinfo_dynamic_cpu_package_energy_reduction_fraction": (
                1.0
                - w8a8["hwinfo_dynamic_cpu_package_j_per_image_mean"]
                / fp32["hwinfo_dynamic_cpu_package_j_per_image_mean"]
                if fp32["hwinfo_dynamic_cpu_package_j_per_image_mean"]
                else None
            ),
            "w8a8_speedup": fp32["mean_latency_ms"] / w8a8["mean_latency_ms"],
        }

    report = {
        "generated_at_utc": utc_now(),
        "provider": args.provider,
        "measurement_backend": "hwinfo_csv_power_integration",
        "measurement_domain": "HWiNFO CPU Package Power sensor telemetry",
        "measurement_boundary": "Trapezoidal integration over server-recorded session.run/idle epoch markers",
        "not_pcm_or_rapl_counter_delta": True,
        "environment": environment_summary(),
        "hwinfo": {
            "version": args.hwinfo_version,
            "source_log": str(args.hwinfo_log.resolve()),
            "copied_log": str(copied_log.relative_to(RESULTS_DIR)),
            "copied_log_sha256": sha256_file(copied_log),
            "log_metadata": log_metadata,
        },
        "selected_config": markers["selected_config"],
        "settings": {
            **markers["settings"],
            "minimum_sample_completeness": args.minimum_sample_completeness,
            "minimum_median_interval_seconds": args.minimum_median_interval_seconds,
            "maximum_median_interval_seconds": args.maximum_median_interval_seconds,
            "maximum_sampling_gap_seconds": args.maximum_sampling_gap_seconds,
            "maximum_integration_spread": args.maximum_integration_spread,
            "minimum_thermal_headroom_c": args.minimum_thermal_headroom_c,
            "maximum_temperature_c": args.maximum_temperature_c,
            "maximum_cv": 0.05,
        },
        "idle": {
            "valid_trials": len(idle),
            "hwinfo_cpu_package_power_w": idle_power_w,
        },
        "supplemental_reserve_trials_used": supplemental_used,
        "models": aggregates,
        "comparison": comparison,
        "stable": all(value["stable"] for value in aggregates.values()),
        "trials": trials,
    }
    output = RESULTS_DIR / f"energy_hwinfo_{args.provider}.json"
    write_json(output, report)
    write_trials_csv(
        RESULTS_DIR / f"energy_trials_hwinfo_{args.provider}.csv", trials
    )
    write_json(
        status_path(args.provider),
        {
            "generated_at_utc": utc_now(),
            "provider": args.provider,
            "energy_backend": "hwinfo_csv_power_integration",
            "status": "complete" if report["stable"] else "complete_unstable",
            "result": str(output),
            "stable": report["stable"],
        },
    )
    print(f"PASS: HWiNFO energy results written to {output}; stable={report['stable']}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("cpu", "openvino"), default="cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record")
    add_common(record_parser)
    record_parser.add_argument("--duration-seconds", type=float, default=180.0)
    record_parser.add_argument("--warmup-seconds", type=float, default=30.0)
    record_parser.add_argument("--trials", type=int, default=5)
    record_parser.add_argument("--idle-trials", type=int, default=3)
    record_parser.add_argument("--max-extra-trials", type=int, default=3)
    record_parser.add_argument("--input-samples", type=int, default=100)
    record_parser.add_argument("--maximum-background-cpu-percent", type=float, default=10.0)
    record_parser.set_defaults(function=record)

    analyze_parser = subparsers.add_parser("analyze")
    add_common(analyze_parser)
    analyze_parser.add_argument("--hwinfo-log", type=Path, required=True)
    analyze_parser.add_argument("--markers", type=Path)
    analyze_parser.add_argument("--hwinfo-version", default="recorded externally")
    analyze_parser.add_argument("--date-order", choices=("auto", "ymd", "mdy", "dmy"), default="auto")
    analyze_parser.add_argument("--power-column")
    analyze_parser.add_argument("--minimum-sample-completeness", type=float, default=0.99)
    analyze_parser.add_argument("--minimum-median-interval-seconds", type=float, default=0.8)
    analyze_parser.add_argument("--maximum-median-interval-seconds", type=float, default=1.2)
    analyze_parser.add_argument("--maximum-sampling-gap-seconds", type=float, default=2.5)
    analyze_parser.add_argument("--maximum-integration-spread", type=float, default=0.01)
    analyze_parser.add_argument("--minimum-thermal-headroom-c", type=float, default=2.0)
    analyze_parser.add_argument("--maximum-temperature-c", type=float, default=98.0)
    analyze_parser.add_argument("--maximum-background-cpu-percent", type=float, default=10.0)
    analyze_parser.set_defaults(function=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "record":
        if args.duration_seconds <= 0 or args.warmup_seconds <= 0:
            raise ValueError("duration and warmup must be positive")
        if args.trials < 5:
            raise ValueError("formal HWiNFO measurement requires at least five base trials")
        if args.idle_trials < 3:
            raise ValueError("formal HWiNFO measurement requires at least three idle trials")
        if args.max_extra_trials < 0 or args.input_samples <= 0:
            raise ValueError("reserve trials must be non-negative and input samples positive")
    else:
        if not 0 < args.minimum_sample_completeness <= 1:
            raise ValueError("sample completeness threshold must be in (0, 1]")
        if not (
            0 < args.minimum_median_interval_seconds
            <= args.maximum_median_interval_seconds
        ):
            raise ValueError("invalid median sampling interval bounds")
        if args.maximum_sampling_gap_seconds <= 0:
            raise ValueError("maximum sampling gap must be positive")
        if not 0 <= args.maximum_integration_spread < 1:
            raise ValueError("integration spread threshold must be in [0, 1)")
    try:
        args.function(args)
    except Exception as exc:
        write_json(
            status_path(args.provider),
            {
                "generated_at_utc": utc_now(),
                "provider": args.provider,
                "energy_backend": "hwinfo_csv_power_integration",
                "status": f"{args.command}_failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    main()
