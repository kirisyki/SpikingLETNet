"""Run the pre-registered ternary experiment sequence with safe resume."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ternary_QAT_checkpoint"
TRAIN_SCRIPT = REPO_ROOT / "Network/QAT_snn_ternary.py"
EVAL_SCRIPT = REPO_ROOT / "Network/evaluate_precision_comparison.py"
PRIMARY_SEED = 1234
REPLICATION_SEEDS = [4321, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def run_dir(output_root: Path, activation_mode: str, seed: int) -> Path:
    activation = "a4" if activation_mode == "a4" else "a1p58"
    return (
        output_root
        / "udd"
        / (
            "SpikingLETNet_shallow_max_"
            f"w1p58_{activation}_bs64_seed{seed}"
        )
    )


def is_complete(directory: Path) -> bool:
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("status") == "complete"
        and (directory / "model_q_best.pth").is_file()
    )


def run_logged(command: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] command: {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{now()}] returncode: {completed.returncode}\n")
        log.flush()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def train_variant(
    args: argparse.Namespace,
    activation_mode: str,
    seed: int,
    logs_dir: Path,
) -> Path:
    directory = run_dir(args.output_root, activation_mode, seed)
    if is_complete(directory):
        return directory

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--activation-mode",
        activation_mode,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-epochs",
        str(args.max_epochs),
        "--seed",
        str(seed),
        "--gpu",
        args.gpu,
        "--log-interval",
        str(args.log_interval),
    ]
    last_checkpoint = directory / "checkpoint_last.pth"
    if last_checkpoint.is_file():
        command.extend(["--resume", str(last_checkpoint)])
    else:
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                "Incomplete non-empty run has no resumable checkpoint: "
                f"{directory}"
            )
        command.extend(["--output-dir", str(directory)])

    label = "a4" if activation_mode == "a4" else "a1p58"
    run_logged(command, logs_dir / f"train_{label}_seed{seed}.log")
    if not is_complete(directory):
        raise RuntimeError(f"Trainer exited without complete manifest: {directory}")
    return directory


def evaluate_seed(
    args: argparse.Namespace,
    ternary_a4_dir: Path,
    ternary_a1p58_dir: Path,
    seed: int,
    logs_dir: Path,
) -> Path:
    output_dir = args.output_root / f"comparison_seed{seed}"
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--ternary-a4-checkpoint",
        str(ternary_a4_dir / "model_q_best.pth"),
        "--ternary-a1p58-checkpoint",
        str(ternary_a1p58_dir / "model_q_best.pth"),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        "20",
        "--num-workers",
        str(args.num_workers),
        "--gpu",
        args.gpu,
        "--overwrite",
    ]
    run_logged(command, logs_dir / f"evaluate_seed{seed}.log")
    result = output_dir / "comparison.json"
    if not result.is_file():
        raise RuntimeError(f"Evaluator did not produce {result}")
    return result


def replicated_report(
    comparison_paths: List[Path], output_root: Path
) -> Dict[str, Any]:
    comparisons = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in comparison_paths
    ]
    w4_values = [
        item["results"]["W4/A4"]["miou"] for item in comparisons
    ]
    ternary_values = [
        item["results"]["W1.58/A4"]["miou"] for item in comparisons
    ]
    gaps = [
        (w4 - ternary) * 100.0
        for w4, ternary in zip(w4_values, ternary_values)
    ]
    mean_gap = statistics.mean(gaps)
    if mean_gap >= 2.0:
        verdict = "supports_w4_accuracy_value_after_replication"
    elif mean_gap <= 0.0:
        verdict = "does_not_support_w4_accuracy_value_after_replication"
    else:
        verdict = "inconclusive_after_replication"
    report = {
        "seeds": [PRIMARY_SEED, *REPLICATION_SEEDS],
        "w4_miou": w4_values,
        "ternary_a4_miou": ternary_values,
        "ternary_a4_mean_miou": statistics.mean(ternary_values),
        "ternary_a4_population_std_miou": statistics.pstdev(ternary_values),
        "gap_points_w4_minus_ternary": gaps,
        "mean_gap_points": mean_gap,
        "threshold_points": 2.0,
        "verdict": verdict,
    }
    atomic_json(report, output_root / "replicated_main_comparison.json")
    markdown = [
        "# Replicated W4/A4 vs W1.58/A4 comparison",
        "",
        "| Seed | W4/A4 mIoU | W1.58/A4 mIoU | Gap (points) |",
        "|---:|---:|---:|---:|",
    ]
    for seed, w4, ternary, gap in zip(
        report["seeds"], w4_values, ternary_values, gaps
    ):
        markdown.append(
            f"| {seed} | {w4:.6f} | {ternary:.6f} | {gap:.4f} |"
        )
    markdown.extend(
        [
            "",
            f"- Ternary mean mIoU: {report['ternary_a4_mean_miou']:.6f}",
            "- Ternary population standard deviation: "
            f"{report['ternary_a4_population_std_miou']:.6f}",
            f"- Mean W4 advantage: {mean_gap:.4f} points",
            f"- Verdict: `{verdict}`",
        ]
    )
    (output_root / "replicated_main_comparison.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return report


def run(args: argparse.Namespace) -> None:
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_root / "pipeline_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "pipeline_manifest.json"
    manifest: Dict[str, Any] = {
        "status": "running",
        "started_at": now(),
        "primary_seed": PRIMARY_SEED,
        "replication_seeds_if_ambiguous": REPLICATION_SEEDS,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "stages": [],
    }
    atomic_json(manifest, manifest_path)

    try:
        primary_a4 = train_variant(
            args, "a4", PRIMARY_SEED, logs_dir
        )
        manifest["stages"].append(
            {"stage": "train_w1p58_a4_seed1234", "status": "complete"}
        )
        atomic_json(manifest, manifest_path)

        primary_a1p58 = train_variant(
            args, "ternary", PRIMARY_SEED, logs_dir
        )
        manifest["stages"].append(
            {"stage": "train_w1p58_a1p58_seed1234", "status": "complete"}
        )
        atomic_json(manifest, manifest_path)

        primary_comparison = evaluate_seed(
            args,
            primary_a4,
            primary_a1p58,
            PRIMARY_SEED,
            logs_dir,
        )
        comparison_payload = json.loads(
            primary_comparison.read_text(encoding="utf-8")
        )
        decision = comparison_payload["decision"]
        manifest["primary_decision"] = decision
        manifest["stages"].append(
            {"stage": "unified_comparison_seed1234", "status": "complete"}
        )
        atomic_json(manifest, manifest_path)

        comparison_paths = [primary_comparison]
        if decision["verdict"] == "ambiguous_add_two_seeds":
            for seed in REPLICATION_SEEDS:
                replicated_a4 = train_variant(
                    args, "a4", seed, logs_dir
                )
                comparison_paths.append(
                    evaluate_seed(
                        args,
                        replicated_a4,
                        primary_a1p58,
                        seed,
                        logs_dir,
                    )
                )
                manifest["stages"].append(
                    {
                        "stage": f"train_and_compare_w1p58_a4_seed{seed}",
                        "status": "complete",
                    }
                )
                atomic_json(manifest, manifest_path)
            manifest["replicated_decision"] = replicated_report(
                comparison_paths, args.output_root
            )

        manifest.update({"status": "complete", "completed_at": now()})
        atomic_json(manifest, manifest_path)
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(manifest, manifest_path)
        raise


if __name__ == "__main__":
    run(parse_args())
