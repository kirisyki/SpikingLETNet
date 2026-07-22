#!/usr/bin/env python3
"""Run alternating FP32/W8A8 Intel PCM package-energy trials."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from common import RESULTS_DIR, environment_summary, read_json, utc_now, write_json
from pcm_tools import run_pcm_external


SCRIPT_DIR = Path(__file__).resolve().parent


def is_admin() -> bool:
    if os.name != "nt":
        return os.geteuid() == 0 if hasattr(os, "geteuid") else False
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def cv(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else float("inf")


def send_direct(host: str, port: int, request: dict[str, Any]) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=30.0) as connection:
        connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
        chunks = []
        while chunk := connection.recv(65536):
            chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


def wait_for_ready(path: Path, process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if path.is_file():
            return read_json(path)
        if process.poll() is not None:
            raise RuntimeError(f"Energy server exited early with code {process.returncode}")
        time.sleep(0.25)
    raise TimeoutError(f"Energy server did not become ready within {timeout} seconds")


def trial(
    args: argparse.Namespace,
    ready: dict[str, Any],
    action: str,
    model: str | None,
    trial_id: str,
) -> dict[str, Any]:
    raw_dir = RESULTS_DIR / "raw_pcm" / args.provider
    client_result = raw_dir / f"{trial_id}_client.json"
    pcm_csv = raw_dir / f"{trial_id}_pcm.csv"
    pcm_log = raw_dir / f"{trial_id}_pcm.log"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "energy_client.py"),
        "--host",
        ready["host"],
        "--port",
        str(ready["port"]),
        "--action",
        action,
        "--duration",
        str(args.duration_seconds),
        "--output",
        str(client_result),
    ]
    if model is not None:
        command.extend(["--model", model])
    background_cpu = psutil.cpu_percent(interval=1.0)
    pcm = run_pcm_external(
        args.pcm_exe,
        command,
        pcm_csv,
        pcm_log,
        timeout_seconds=args.duration_seconds + 120.0,
    )
    client = read_json(client_result)
    result = {
        "trial_id": trial_id,
        "action": action,
        "model": model,
        "background_cpu_percent_before": background_cpu,
        "package_energy_j": pcm["package_energy_j"],
        "thermal_headroom_min_c": pcm["thermal_headroom_min_c"],
        "pcm_csv": str(pcm_csv.relative_to(RESULTS_DIR)),
        "pcm_log": str(pcm_log.relative_to(RESULTS_DIR)),
        "client": client,
    }
    thermal_invalid = pcm["thermal_headroom_min_c"] is not None and pcm["thermal_headroom_min_c"] <= args.minimum_thermal_headroom_c
    background_invalid = background_cpu > args.maximum_background_cpu_percent
    result["valid"] = not thermal_invalid and not background_invalid
    result["invalid_reasons"] = [
        reason
        for condition, reason in (
            (thermal_invalid, "thermal_headroom"),
            (background_invalid, "background_cpu"),
        )
        if condition
    ]
    return result


def aggregate(trials: list[dict[str, Any]], idle_power_w: float, model: str) -> dict[str, Any]:
    selected = [item for item in trials if item["model"] == model and item["valid"]]
    rows = []
    for item in selected:
        client = item["client"]
        count = client["iterations"]
        elapsed = client["elapsed_seconds"]
        gross = item["package_energy_j"]
        dynamic = gross - idle_power_w * elapsed
        rows.append(
            {
                "trial_id": item["trial_id"],
                "iterations": count,
                "elapsed_seconds": elapsed,
                "mean_latency_ms": elapsed / count * 1000.0,
                "package_energy_j": gross,
                "package_power_w": gross / elapsed,
                "package_j_per_image": gross / count,
                "dynamic_energy_j": dynamic,
                "dynamic_j_per_image": dynamic / count,
            }
        )
    if not rows:
        return {"valid_trials": 0, "stable": False, "rows": []}
    energy_values = [row["package_j_per_image"] for row in rows]
    latency_values = [row["mean_latency_ms"] for row in rows]
    return {
        "valid_trials": len(rows),
        "stable": len(rows) >= 5 and cv(energy_values) <= 0.05 and cv(latency_values) <= 0.05,
        "package_j_per_image_mean": statistics.fmean(energy_values),
        "package_j_per_image_median": statistics.median(energy_values),
        "package_j_per_image_cv": cv(energy_values),
        "dynamic_j_per_image_mean": statistics.fmean(row["dynamic_j_per_image"] for row in rows),
        "mean_latency_ms": statistics.fmean(latency_values),
        "latency_cv": cv(latency_values),
        "package_power_w_mean": statistics.fmean(row["package_power_w"] for row in rows),
        "rows": rows,
    }


def write_trials_csv(path: Path, trials: list[dict[str, Any]], idle_power_w: float) -> None:
    fields = [
        "trial_id", "action", "model", "valid", "invalid_reasons", "background_cpu_percent_before",
        "thermal_headroom_min_c", "iterations", "elapsed_seconds", "mean_latency_ms",
        "package_energy_j", "package_power_w", "package_j_per_image", "dynamic_j_per_image",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in trials:
            client = item["client"]
            count = client["iterations"]
            elapsed = client["elapsed_seconds"]
            energy = item["package_energy_j"]
            writer.writerow(
                {
                    "trial_id": item["trial_id"],
                    "action": item["action"],
                    "model": item["model"],
                    "valid": item["valid"],
                    "invalid_reasons": ";".join(item["invalid_reasons"]),
                    "background_cpu_percent_before": item["background_cpu_percent_before"],
                    "thermal_headroom_min_c": item["thermal_headroom_min_c"],
                    "iterations": count,
                    "elapsed_seconds": elapsed,
                    "mean_latency_ms": elapsed / count * 1000.0 if count else "",
                    "package_energy_j": energy,
                    "package_power_w": energy / elapsed,
                    "package_j_per_image": energy / count if count else "",
                    "dynamic_j_per_image": (energy - idle_power_w * elapsed) / count if count else "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("cpu", "openvino"), default="cpu")
    parser.add_argument("--pcm-exe", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--idle-trials", type=int, default=3)
    parser.add_argument("--max-extra-trials", type=int, default=3)
    parser.add_argument("--minimum-thermal-headroom-c", type=float, default=2.0)
    parser.add_argument("--maximum-background-cpu-percent", type=float, default=10.0)
    parser.add_argument("--input-samples", type=int, default=100)
    args = parser.parse_args()
    if not is_admin():
        raise PermissionError("Intel PCM energy measurement must run from an Administrator PowerShell")
    tuning = read_json(RESULTS_DIR / f"tuning_{args.provider}.json")
    selected = tuning["selected"]

    ready_path = RESULTS_DIR / f"energy_server_{args.provider}_ready.json"
    server_log_path = RESULTS_DIR / f"energy_server_{args.provider}.log"
    ready_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "energy_server.py"),
        "--provider", args.provider,
        "--threads", str(selected["threads"]),
        "--cpu-ids", *[str(value) for value in selected["cpu_ids"]],
        "--ready-file", str(ready_path),
        "--warmup-seconds", str(args.warmup_seconds),
        "--input-samples", str(args.input_samples),
    ]
    server_log = server_log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
    trials: list[dict[str, Any]] = []
    try:
        ready = wait_for_ready(ready_path, server, timeout=args.warmup_seconds * 2.0 + 180.0)
        for index in range(args.idle_trials):
            trials.append(trial(args, ready, "idle", None, f"idle_{index + 1:02d}"))
        idle_valid = [item for item in trials if item["action"] == "idle" and item["valid"]]
        if not idle_valid:
            raise RuntimeError("No valid idle baseline trials")
        idle_energy = sum(item["package_energy_j"] for item in idle_valid)
        idle_seconds = sum(item["client"]["elapsed_seconds"] for item in idle_valid)
        idle_power_w = idle_energy / idle_seconds

        for round_index in range(args.trials):
            order = ("fp32", "w8a8") if round_index % 2 == 0 else ("w8a8", "fp32")
            for label in order:
                trials.append(
                    trial(args, ready, "run", label, f"round_{round_index + 1:02d}_{label}")
                )

        extra = 0
        while extra < args.max_extra_trials:
            aggregates = {label: aggregate(trials, idle_power_w, label) for label in ("fp32", "w8a8")}
            unstable = [label for label, value in aggregates.items() if not value["stable"]]
            if not unstable:
                break
            label = max(
                unstable,
                key=lambda item: max(
                    aggregates[item].get("package_j_per_image_cv", float("inf")),
                    aggregates[item].get("latency_cv", float("inf")),
                ),
            )
            extra += 1
            trials.append(trial(args, ready, "run", label, f"extra_{extra:02d}_{label}"))

        aggregates = {label: aggregate(trials, idle_power_w, label) for label in ("fp32", "w8a8")}
        fp32 = aggregates["fp32"]
        w8a8 = aggregates["w8a8"]
        comparison = None
        if fp32.get("valid_trials") and w8a8.get("valid_trials"):
            comparison = {
                "w8a8_package_energy_reduction_fraction": 1.0 - w8a8["package_j_per_image_mean"] / fp32["package_j_per_image_mean"],
                "w8a8_dynamic_energy_reduction_fraction": 1.0 - w8a8["dynamic_j_per_image_mean"] / fp32["dynamic_j_per_image_mean"] if fp32["dynamic_j_per_image_mean"] else None,
                "w8a8_speedup": fp32["mean_latency_ms"] / w8a8["mean_latency_ms"],
            }
        report = {
            "generated_at_utc": utc_now(),
            "provider": args.provider,
            "measurement_boundary": "Intel PCM CPU package energy around a client-triggered, pre-warmed session.run loop",
            "environment": environment_summary(),
            "selected_config": selected,
            "settings": vars(args) | {"pcm_exe": str(args.pcm_exe)},
            "idle": {
                "valid_trials": len(idle_valid),
                "package_power_w": idle_power_w,
            },
            "models": aggregates,
            "comparison": comparison,
            "stable": all(value["stable"] for value in aggregates.values()),
            "trials": trials,
        }
        output = RESULTS_DIR / f"energy_{args.provider}.json"
        write_json(output, report)
        write_trials_csv(RESULTS_DIR / f"energy_trials_{args.provider}.csv", trials, idle_power_w)
        print(f"PASS: energy results written to {output}; stable={report['stable']}")
    finally:
        try:
            if ready_path.is_file():
                ready = read_json(ready_path)
                send_direct(ready["host"], ready["port"], {"action": "stop"})
        except Exception:
            pass
        try:
            server.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            server.terminate()
            server.wait(timeout=15.0)
        server_log.close()


if __name__ == "__main__":
    main()
