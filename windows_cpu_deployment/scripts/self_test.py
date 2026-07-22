#!/usr/bin/env python3
"""Fast local checks that do not require Intel PCM or OpenVINO."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from common import OPENVINO_PORTABLE_MODELS, PORTABLE_MODELS, load_inputs
from inspect_models import inspect
from pcm_tools import parse_pcm_csv


def main() -> None:
    for path in list(PORTABLE_MODELS.values()) + list(OPENVINO_PORTABLE_MODELS.values()):
        inspect(path)
    image = load_inputs(1)[0]
    assert image.shape == (1, 3, 400, 400) and image.dtype.name == "float32"

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pcm.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["System", "System", "System", "Socket 0"])
            writer.writerow(["Date", "Time", "Proc Energy (Joules)", "TEMP"])
            writer.writerow(["2026-01-01", "00:00:00", "123.5", "20"])
        parsed = parse_pcm_csv(path)
        assert parsed["package_energy_j"] == 123.5
        assert parsed["thermal_headroom_min_c"] == 20.0
    print("PASS: self test")


if __name__ == "__main__":
    main()
