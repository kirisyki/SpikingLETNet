#!/usr/bin/env python3
"""Recompute QAD repair statistics from saved source-level aggregates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


NETWORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NETWORK_DIR.parent
os.chdir(NETWORK_DIR)

from qad_distortion_repair.statistics import summarize_repair  # noqa: E402
from run_qad_distortion_repair_probe import write_flat_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270806)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    result_dir = (
        args.result_dir if args.result_dir.is_absolute() else REPO_ROOT / args.result_dir
    ).resolve()
    summary, detail = summarize_repair(
        qif_rows=read_json(result_dir / "source_qif_aggregates.json"),
        feature_rows=read_json(result_dir / "source_feature_aggregates.json"),
        confusion_rows=read_json(result_dir / "source_confusion.json"),
        flip_rows=read_json(result_dir / "source_prediction_flips.json"),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(result_dir / "analysis_summary.json", summary)
    write_flat_csv(result_dir / "source_statistics.csv", detail["sources"])
    write_flat_csv(result_dir / "layer_statistics.csv", detail["layers"])
    write_flat_csv(result_dir / "feature_statistics.csv", detail["features"])
    write_flat_csv(result_dir / "stage_statistics.csv", detail["stages"])


if __name__ == "__main__":
    main()

