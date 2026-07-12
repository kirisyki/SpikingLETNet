#!/usr/bin/env python3
"""Measure paired TensorRT FP32/INT8 board energy on the RTX PRO 6000."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pynvml
import tensorrt as trt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import measure_pro6000_ann_energy_v2 as protocol  # noqa: E402
from tensorrt_energy_analysis.trt_runtime import TensorRTEngine  # noqa: E402

ENGINE_DIR = PROJECT_ROOT / "tensorrt_energy_analysis/engines"
DEFAULT_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")
DEFAULT_OUTPUT = PROJECT_ROOT / "tensorrt_energy_analysis/results"
THRESHOLDS = {"throughput_cv_pct": 2.0, "gross_cv_pct": 5.0, "net_cv_pct": 10.0}


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def sample_ids(paths: Sequence[str]) -> list[str]:
    return [hashlib.sha256(path.encode()).hexdigest()[:16] for path in paths]


def preload_images(
    split_file: Path, count: int, device: torch.device
) -> tuple[list[torch.Tensor], list[str]]:
    paths = protocol.read_split(split_file, count)
    images = [
        protocol.load_image_tensor(path, (400, 400))
        .unsqueeze(0)
        .contiguous()
        .to(device, non_blocking=True)
        for path in paths
    ]
    torch.cuda.synchronize(device)
    return images, paths


def active_measurement(
    handle: Any,
    runner: TensorRTEngine,
    images: Sequence[torch.Tensor],
    count: int,
    schedule_index: int,
    precision: str,
    interval_s: float,
) -> tuple[dict[str, Any], list[protocol.PowerSample]]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    runner.synchronize()
    torch.cuda.synchronize(runner.device)
    with protocol.NvmlPowerSampler(
        handle, interval_s, schedule_index, f"{precision}_active"
    ) as sampler:
        start_rel = sampler.now()
        energy_before = protocol.read_total_energy_j(handle)
        wall_start = time.perf_counter()
        start_event.record(runner.stream)
        for index in range(count):
            runner.enqueue(images[index % len(images)])
        end_event.record(runner.stream)
        runner.synchronize()
        wall_elapsed = time.perf_counter() - wall_start
        energy_after = protocol.read_total_energy_j(handle)
        end_rel = sampler.now()
    if not sampler.samples:
        raise RuntimeError(f"NVML returned no active samples: {sampler.errors[:3]}")
    integrated_j = protocol.integrate_energy_window_j(sampler.samples, start_rel, end_rel)
    if energy_before is not None and energy_after is not None and energy_after >= energy_before:
        gross_j = energy_after - energy_before
        source = "nvml_total_energy_counter"
    else:
        gross_j = integrated_j
        source = "bounded_power_integration"
    return {
        "active_wall_s": wall_elapsed,
        "active_cuda_s": start_event.elapsed_time(end_event) / 1000.0,
        "active_window_start_s": start_rel,
        "active_window_end_s": end_rel,
        "gross_energy_j": gross_j,
        "integrated_power_energy_j": integrated_j,
        "energy_source": source,
        "counter_vs_integrated_pct": (gross_j / integrated_j - 1.0) * 100.0,
        "active_mean_power_w": integrated_j / (end_rel - start_rel),
    }, sampler.samples


def summarize(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, float]:
    return protocol.summarize_values([float(row[metric]) for row in rows])


def precision_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        metric: summarize(rows, metric)
        for metric in (
            "gross_j_per_image",
            "net_j_per_image",
            "images_per_second",
            "active_mean_power_w",
            "active_cuda_s",
        )
    }
    stable = (
        metrics["images_per_second"]["cv_pct"] <= THRESHOLDS["throughput_cv_pct"]
        and metrics["gross_j_per_image"]["cv_pct"] <= THRESHOLDS["gross_cv_pct"]
        and metrics["net_j_per_image"]["cv_pct"] <= THRESHOLDS["net_cv_pct"]
    )
    return {"status": "ready" if stable else "unstable", "metrics": metrics}


def compare(
    fp32_rows: Sequence[dict[str, Any]],
    int8_rows: Sequence[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metric = lambda precision, name: summaries[precision]["metrics"][name]["median"]
    fp_by_pair = {int(row["pair"]): row for row in fp32_rows}
    i8_by_pair = {int(row["pair"]): row for row in int8_rows}
    if set(fp_by_pair) != set(i8_by_pair):
        raise RuntimeError("FP32 and INT8 pair ids differ")
    paired = []
    for pair in sorted(fp_by_pair):
        fp, i8 = fp_by_pair[pair], i8_by_pair[pair]
        paired.append({
            "pair": pair,
            "fp32_gross_j_per_image": fp["gross_j_per_image"],
            "int8_gross_j_per_image": i8["gross_j_per_image"],
            "gross_energy_reduction": 1.0 - float(i8["gross_j_per_image"]) / float(fp["gross_j_per_image"]),
            "fp32_net_j_per_image": fp["net_j_per_image"],
            "int8_net_j_per_image": i8["net_j_per_image"],
            "net_energy_reduction": 1.0 - float(i8["net_j_per_image"]) / float(fp["net_j_per_image"]),
        })
    return {
        "primary_method": "ratio_of_precision_medians",
        "gross_energy_reduction": 1.0 - metric("int8", "gross_j_per_image") / metric("fp32", "gross_j_per_image"),
        "net_energy_reduction": 1.0 - metric("int8", "net_j_per_image") / metric("fp32", "net_j_per_image"),
        "throughput_ratio": metric("int8", "images_per_second") / metric("fp32", "images_per_second"),
        "throughput_increase": metric("int8", "images_per_second") / metric("fp32", "images_per_second") - 1.0,
        "paired_trials": paired,
        "median_paired_gross_energy_reduction": statistics.median(x["gross_energy_reduction"] for x in paired),
        "median_paired_net_energy_reduction": statistics.median(x["net_energy_reduction"] for x in paired),
    }


def report_markdown(result: dict[str, Any]) -> str:
    s, c, coverage = result["precision_summaries"], result["comparison"], result["precision_coverage"]
    lines = [
        "# PRO6000 TensorRT FP32 vs INT8 Energy", "",
        f"Status: **{result['status']}**. Gross board energy is primary; net energy is secondary.", "",
        "| Engine | Gross J/image | CV | Net J/image | CV | Images/s | CV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in ("fp32", "int8"):
        m = s[p]["metrics"]
        lines.append(
            f"| {p.upper()} | {m['gross_j_per_image']['median']:.9f} | {m['gross_j_per_image']['cv_pct']:.3f}% | "
            f"{m['net_j_per_image']['median']:.9f} | {m['net_j_per_image']['cv_pct']:.3f}% | "
            f"{m['images_per_second']['median']:.6f} | {m['images_per_second']['cv_pct']:.3f}% |"
        )
    lines += ["", "## Main comparison", "",
        f"- Gross energy reduction: **{c['gross_energy_reduction']:.4%}**",
        f"- Net energy reduction: **{c['net_energy_reduction']:.4%}**",
        f"- Throughput ratio: **{c['throughput_ratio']:.4f}x**",
        f"- Median paired gross-energy reduction: {c['median_paired_gross_energy_reduction']:.4%}",
        "", "## Actual engine precision", "",
        f"- FP32 layers touching INT8: {coverage['fp32']['layers_touching_int8']}",
        f"- INT8 layers touching INT8: {coverage['int8']['layers_touching_int8']} / {coverage['int8']['total_engine_layers']}",
    ]
    for layer_type, stats in coverage["int8"]["main_compute"].items():
        lines.append(f"- INT8 {layer_type}: {stats['int8']} / {stats['total']}")
    lines += ["", "## Historical PyTorch context", "",
        "The existing T=1 QIF dense PyTorch FP32 result is 1.329127400 gross J/image,",
        "0.543892035 net J/image, and 106.440745 images/s. It is context only.", "",
        "## Scope", "",
        "Fixed-shape batch-1 dense ANN-form TensorRT inference; engine build, preprocessing,",
        "transfers, output copies, CUDA Graph, and accuracy evaluation are excluded. This does not",
        "measure T=8 spike execution or the synchronous chip's temporal-reuse mechanism.", ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-engine", type=Path, default=ENGINE_DIR / "SpikingLETNet_shallow_max_fp32.plan")
    parser.add_argument("--int8-engine", type=Path, default=ENGINE_DIR / "SpikingLETNet_shallow_max_int8.plan")
    parser.add_argument("--precision-coverage", type=Path, default=ENGINE_DIR / "precision_coverage.json")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nvml-index", type=int, default=None)
    parser.add_argument("--warmup-images", type=int, default=200)
    parser.add_argument("--measure-images", type=int, default=5000)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--sample-interval-ms", type=float, default=100.0)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args()
    if min(args.warmup_images, args.measure_images, args.pairs) <= 0:
        parser.error("warmup-images, measure-images, and pairs must be positive")
    if args.idle_seconds <= 0 or args.sample_interval_ms <= 0:
        parser.error("idle-seconds and sample-interval-ms must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    coverage = json.loads(args.precision_coverage.read_text())
    engine_paths = {"fp32": args.fp32_engine, "int8": args.int8_engine}
    pynvml.nvmlInit()
    try:
        handle, binding = protocol.resolve_nvml_handle(args.gpu, args.nvml_index)
        foreign = protocol.foreign_compute_pids(handle)
        if foreign and not args.allow_shared_gpu:
            raise RuntimeError(f"foreign GPU compute processes detected: {foreign}")
        runners = {p: TensorRTEngine(path, device) for p, path in engine_paths.items()}
        images, paths = preload_images(args.split_file, max(args.measure_images, args.warmup_images), device)
        print(f"preloaded {len(images)} images on {device}", flush=True)
        for p in ("fp32", "int8"):
            wall, cuda = runners[p].run_loop(images, args.warmup_images)
            print(f"warmed {p}: wall={wall:.3f}s cuda={cuda:.3f}s", flush=True)
        interval = args.sample_interval_ms / 1000.0
        trials: list[dict[str, Any]] = []
        trace: list[protocol.PowerSample] = []
        schedule = 0
        for pair in range(1, args.pairs + 1):
            for precision in ("fp32", "int8"):
                schedule += 1
                pre_s, pre = protocol.measure_idle(handle, device, schedule, f"{precision}_idle_pre", args.idle_seconds, interval)
                active, active_samples = active_measurement(handle, runners[precision], images, args.measure_images, schedule, precision, interval)
                post_s, post = protocol.measure_idle(handle, device, schedule, f"{precision}_idle_post", args.idle_seconds, interval)
                baseline = statistics.fmean([protocol.stable_mean_power(pre, pre_s), protocol.stable_mean_power(post, post_s)])
                gross = float(active["gross_energy_j"])
                net = gross - baseline * float(active["active_wall_s"])
                row = {"schedule_index": schedule, "pair": pair, "precision": precision,
                    "precision_trial": pair, "measure_images": args.measure_images, **active,
                    "idle_pre_s": pre_s, "idle_post_s": post_s, "idle_baseline_w": baseline,
                    "gross_j_per_image": gross / args.measure_images, "net_energy_j": net,
                    "net_j_per_image": net / args.measure_images,
                    "images_per_second": args.measure_images / float(active["active_wall_s"]),
                    "active_sample_count": len(active_samples), "idle_pre_sample_count": len(pre),
                    "idle_post_sample_count": len(post)}
                if not all(math.isfinite(float(row[k])) for k in ("gross_j_per_image", "net_j_per_image", "images_per_second")):
                    raise RuntimeError(f"non-finite trial result: {row}")
                trials.append(row); trace.extend(pre + active_samples + post)
                write_csv(run_dir / "trials.partial.csv", trials)
                write_csv(run_dir / "power_trace.partial.csv", [asdict(x) for x in trace])
                print(f"pair {pair} {precision}: gross={row['gross_j_per_image']:.9f} net={row['net_j_per_image']:.9f} throughput={row['images_per_second']:.3f}", flush=True)
        grouped = {p: [row for row in trials if row["precision"] == p] for p in ("fp32", "int8")}
        summaries = {p: precision_summary(rows) for p, rows in grouped.items()}
        comparison = compare(grouped["fp32"], grouped["int8"], summaries)
        status = "ready" if all(x["status"] == "ready" for x in summaries.values()) else "unstable"
        result = {"schema_version": 1, "measurement": "pro6000_tensorrt_fp32_vs_int8",
            "status": status, "model": "SpikingLETNet_shallow_max",
            "mode": "T=1 ANN-form dense TensorRT inference", "primary_metric": "gross_j_per_image",
            "secondary_metric": "net_j_per_image", "thresholds": THRESHOLDS, "batch_size": 1,
            "input_size": [400, 400], "warmup_images": args.warmup_images,
            "measure_images": args.measure_images, "pairs": args.pairs,
            "idle_seconds": args.idle_seconds, "sample_interval_ms": args.sample_interval_ms,
            "tf32": "off", "cuda_graph": False, "accuracy_evaluated": False,
            "engines": {p: {"path": str(path.resolve()), "sha256": protocol.sha256_file(path)} for p, path in engine_paths.items()},
            "split_file": str(args.split_file.resolve()), "split_sha256": protocol.sha256_file(args.split_file),
            "sample_ids": sample_ids(paths), "gpu": protocol.gpu_info(handle, binding),
            "foreign_compute_pids": foreign, "precision_coverage": coverage,
            "precision_summaries": summaries, "comparison": comparison, "trial_results": trials}
        manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "command": list(sys.argv),
            "git_commit": protocol.git_commit(PROJECT_ROOT), "software": {"python": platform.python_version(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "tensorrt": trt.__version__,
            "numpy": np.__version__, "pynvml": getattr(pynvml, "__version__", None)}, "result_status": status}
        write_json(run_dir / "result.json", result); write_json(run_dir / "manifest.json", manifest)
        write_csv(run_dir / "trials.csv", trials); write_csv(run_dir / "power_trace.csv", [asdict(x) for x in trace])
        (run_dir / "result.md").write_text(report_markdown(result))
        for partial in (run_dir / "trials.partial.csv", run_dir / "power_trace.partial.csv"):
            partial.unlink(missing_ok=True)
        print(json.dumps({"run_dir": str(run_dir), "status": status, **comparison}, indent=2))
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
