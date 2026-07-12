#!/usr/bin/env python3
"""Run the formal long-window protocol over 5,000 unique images.

The base measurement CLI receives ``--measure-images 50000`` while this wrapper
caps GPU preloading at the first 5,000 validation images. The active loop cycles
those tensors ten times, producing a roughly 40-second window in which the NVML
cumulative energy counter can be cross-checked reliably against power integration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tensorrt_energy_analysis import measure_energy as base


UNIQUE_IMAGES = 5000
_original_preload = base.preload_images


def capped_preload(split_file: Path, count: int, device):
    return _original_preload(split_file, min(count, UNIQUE_IMAGES), device)


def argument_value(flag: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except ValueError:
        return default


def main() -> None:
    base.preload_images = capped_preload
    base.main()
    run_id = argument_value("--run-id", "")
    if not run_id:
        return
    output_dir = Path(argument_value("--output-dir", str(base.DEFAULT_OUTPUT)))
    run_dir = output_dir / run_id
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text())
    result["unique_preloaded_images"] = UNIQUE_IMAGES
    result["active_image_cycles"] = result["measure_images"] / UNIQUE_IMAGES
    result["long_window_reason"] = (
        "5000 TensorRT inferences lasted about 4 seconds and caused a 10-15% discrepancy "
        "between the NVML cumulative energy counter and bounded power integration."
    )
    base.write_json(result_path, result)
    with (run_dir / "result.md").open("a") as handle:
        handle.write(
            "\n## Long-window protocol\n\n"
            f"Each active trial executes {result['measure_images']} batch-1 inferences by cycling "
            f"the first {UNIQUE_IMAGES} preloaded validation images "
            f"{result['active_image_cycles']:.0f} times.\n"
        )


if __name__ == "__main__":
    main()

