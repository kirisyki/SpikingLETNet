#!/usr/bin/env python3
"""Thread tuning, latency benchmarking, and provider profiling."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    LOCAL_MODELS_DIR,
    PORTABLE_MODELS,
    RESULTS_DIR,
    affinity_profiles,
    create_session,
    environment_summary,
    load_inputs,
    measured_runs,
    model_path_for,
    preprocess_image,
    process_affinity,
    read_json,
    summarize_latencies,
    utc_now,
    validation_entries,
    write_json,
)


LABELS = ("fp32", "w8a8")


def thread_candidates(cpu_count: int) -> list[int]:
    return sorted({max(1, min(value, cpu_count)) for value in (1, 2, 4, 8, 12, 16, 24, cpu_count)})


def tune(args: argparse.Namespace) -> dict[str, Any]:
    profiles, topology = affinity_profiles()
    inputs = load_inputs(args.input_samples)
    candidates: list[dict[str, Any]] = []
    for profile_name, cpu_ids in profiles.items():
        for threads in thread_candidates(len(cpu_ids)):
            model_results = {}
            with process_affinity(cpu_ids):
                for label in LABELS:
                    model_path = model_path_for(args.provider, label)
                    cache = LOCAL_MODELS_DIR / "openvino_cache" / label
                    session = create_session(
                        model_path,
                        args.provider,
                        threads,
                        cache_dir=cache if args.provider == "openvino" else None,
                    )
                    model_results[label] = measured_runs(
                        session,
                        inputs,
                        warmup_seconds=args.tune_warmup_seconds,
                        min_iterations=args.tune_iterations,
                        min_measure_seconds=args.tune_measure_seconds,
                    )
            candidates.append(
                {
                    "affinity_profile": profile_name,
                    "cpu_ids": cpu_ids,
                    "threads": threads,
                    "models": model_results,
                }
            )

    best_by_model = {
        label: min(candidate["models"][label]["median_ms"] for candidate in candidates)
        for label in LABELS
    }
    for candidate in candidates:
        normalized = [
            candidate["models"][label]["median_ms"] / best_by_model[label] for label in LABELS
        ]
        candidate["shared_score_geometric_mean"] = math.sqrt(normalized[0] * normalized[1])
    selected = min(candidates, key=lambda item: item["shared_score_geometric_mean"])
    report = {
        "generated_at_utc": utc_now(),
        "provider": args.provider,
        "selection_policy": "lowest geometric mean of normalized FP32 and W8A8 median latency",
        "environment": environment_summary(),
        "topology": topology,
        "best_median_ms_by_model": best_by_model,
        "selected": {
            "affinity_profile": selected["affinity_profile"],
            "cpu_ids": selected["cpu_ids"],
            "threads": selected["threads"],
            "shared_score_geometric_mean": selected["shared_score_geometric_mean"],
        },
        "candidates": candidates,
    }
    output = RESULTS_DIR / f"tuning_{args.provider}.json"
    write_json(output, report)
    print(
        f"Selected {args.provider}: profile={selected['affinity_profile']} "
        f"threads={selected['threads']} score={selected['shared_score_geometric_mean']:.4f}"
    )
    return report


def selected_config(provider: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"tuning_{provider}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run benchmark.py tune --provider {provider}")
    return read_json(path)["selected"]


def latency(args: argparse.Namespace) -> dict[str, Any]:
    selected = selected_config(args.provider)
    inputs = load_inputs(args.input_samples)
    image_paths = [image for image, _ in validation_entries(args.input_samples)]
    model_results = {}
    with process_affinity(selected["cpu_ids"]):
        for label in LABELS:
            path = model_path_for(args.provider, label)
            cache = LOCAL_MODELS_DIR / "openvino_cache" / label
            session = create_session(
                path,
                args.provider,
                selected["threads"],
                cache_dir=cache if args.provider == "openvino" else None,
            )
            model_results[label] = measured_runs(
                session,
                inputs,
                warmup_seconds=args.warmup_seconds,
                min_iterations=args.iterations,
            )
            model_results[label]["model_path"] = str(path)
            model_results[label]["session_providers"] = session.get_providers()
            model_results[label]["provider_options"] = session.get_provider_options()

            # Auxiliary wall-clock view, deliberately separate from the formal
            # preloaded-input session.run distribution.
            e2e_seconds = []
            for index in range(args.end_to_end_iterations):
                started = time.perf_counter()
                array = preprocess_image(image_paths[index % len(image_paths)])
                session.run(["logits"], {"image": array})
                e2e_seconds.append(time.perf_counter() - started)
            model_results[label]["end_to_end_observational"] = summarize_latencies(e2e_seconds)

    fp32_ms = model_results["fp32"]["median_ms"]
    w8a8_ms = model_results["w8a8"]["median_ms"]
    report = {
        "generated_at_utc": utc_now(),
        "provider": args.provider,
        "timing_boundary": "session.run only; image I/O and preprocessing excluded",
        "auxiliary_timing_boundary": "PNG decode + RGB conversion + resize + normalization + session.run; argmax/output persistence excluded",
        "batch": 1,
        "input_shape": [1, 3, 400, 400],
        "selected_config": selected,
        "environment": environment_summary(),
        "models": model_results,
        "comparison": {
            "w8a8_speedup_over_fp32_median": fp32_ms / w8a8_ms,
            "w8a8_latency_reduction_fraction": 1.0 - w8a8_ms / fp32_ms,
        },
    }
    output = RESULTS_DIR / f"latency_{args.provider}.json"
    write_json(output, report)
    print(f"PASS: latency results written to {output}")
    return report


def profile(args: argparse.Namespace) -> dict[str, Any]:
    selected = selected_config(args.provider)
    inputs = load_inputs(min(args.input_samples, 3))
    reports = {}
    profile_dir = RESULTS_DIR / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    with process_affinity(selected["cpu_ids"]):
        for label in LABELS:
            path = model_path_for(args.provider, label)
            prefix = profile_dir / f"{args.provider}_{label}"
            cache = LOCAL_MODELS_DIR / "openvino_cache" / label
            session = create_session(
                path,
                args.provider,
                selected["threads"],
                cache_dir=cache if args.provider == "openvino" else None,
                profile_prefix=prefix,
            )
            for index in range(args.profile_iterations):
                session.run(["logits"], {"image": inputs[index % len(inputs)]})
            raw_profile = Path(session.end_profiling())
            target = profile_dir / f"{args.provider}_{label}_ort_profile.json"
            if raw_profile.resolve() != target.resolve():
                shutil.move(str(raw_profile), target)
            events = json.loads(target.read_text(encoding="utf-8"))
            provider_counts: Counter[str] = Counter()
            provider_duration_us: Counter[str] = Counter()
            op_counts: dict[str, Counter[str]] = defaultdict(Counter)
            for event in events:
                event_args = event.get("args") or {}
                provider = event_args.get("provider")
                if not provider:
                    continue
                provider_counts[provider] += 1
                provider_duration_us[provider] += int(event.get("dur") or 0)
                op_counts[provider][event_args.get("op_name") or "unknown"] += 1
            reports[label] = {
                "raw_profile": str(target.relative_to(RESULTS_DIR)),
                "session_providers": session.get_providers(),
                "provider_event_counts": dict(provider_counts),
                "provider_duration_us": dict(provider_duration_us),
                "provider_op_counts": {
                    name: dict(sorted(counts.items())) for name, counts in provider_counts_and_ops(op_counts)
                },
            }

    if args.provider == "cpu":
        w8a8_ops = reports["w8a8"]["provider_op_counts"].get("CPUExecutionProvider", {})
        integer_event_count = sum(
            count
            for name, count in w8a8_ops.items()
            if name.startswith("QLinear") or name in {"QGemm", "MatMulInteger", "ConvInteger"}
        )
        if integer_event_count == 0:
            raise RuntimeError("CPU EP W8A8 profile has no observable quantized operator events")
        reports["w8a8"]["observed_quantized_operator_events"] = integer_event_count
    else:
        for label, item in reports.items():
            counts = item["provider_event_counts"]
            if counts.get("OpenVINOExecutionProvider", 0) == 0:
                raise RuntimeError(f"OpenVINO EP did not execute any profiled nodes for {label}")
            if counts.get("CPUExecutionProvider", 0) != 0:
                raise RuntimeError(f"OpenVINO EP silently fell back to CPU EP for {label}")

    report = {
        "generated_at_utc": utc_now(),
        "provider": args.provider,
        "selected_config": selected,
        "models": reports,
        "note": "QDQ presence alone does not prove INT8 execution; use this provider assignment with the optimized graph/OpenVINO runtime report.",
    }
    output = RESULTS_DIR / f"provider_profile_{args.provider}.json"
    write_json(output, report)
    print(f"PASS: provider profile written to {output}")
    return report


def provider_counts_and_ops(op_counts: dict[str, Counter[str]]) -> list[tuple[str, Counter[str]]]:
    return sorted(op_counts.items(), key=lambda item: item[0])


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("cpu", "openvino"), default="cpu")
    parser.add_argument("--input-samples", type=int, default=100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune_parser = subparsers.add_parser("tune")
    add_shared_arguments(tune_parser)
    tune_parser.add_argument("--tune-warmup-seconds", type=float, default=2.0)
    tune_parser.add_argument("--tune-iterations", type=int, default=5)
    tune_parser.add_argument("--tune-measure-seconds", type=float, default=2.0)
    tune_parser.set_defaults(function=tune)

    latency_parser = subparsers.add_parser("latency")
    add_shared_arguments(latency_parser)
    latency_parser.add_argument("--warmup-seconds", type=float, default=30.0)
    latency_parser.add_argument("--iterations", type=int, default=200)
    latency_parser.add_argument("--end-to-end-iterations", type=int, default=20)
    latency_parser.set_defaults(function=latency)

    profile_parser = subparsers.add_parser("profile")
    add_shared_arguments(profile_parser)
    profile_parser.add_argument("--profile-iterations", type=int, default=5)
    profile_parser.set_defaults(function=profile)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("input_samples", "tune_iterations", "iterations", "end_to_end_iterations", "profile_iterations"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
