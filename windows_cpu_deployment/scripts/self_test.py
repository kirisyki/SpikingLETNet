#!/usr/bin/env python3
"""Fast local checks that do not require Intel PCM, HWiNFO, or OpenVINO."""

from __future__ import annotations

import csv
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from common import OPENVINO_PORTABLE_MODELS, PORTABLE_MODELS, load_inputs
from inspect_models import inspect
from hwinfo_tools import inspect_report, integrate_window, read_hwinfo_csv
from pcm_tools import parse_pcm_csv
from run_hwinfo_energy import aggregate


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
        hwinfo_path = Path(directory) / "hwinfo.csv"
        started = datetime.now().astimezone().replace(microsecond=0)
        with hwinfo_path.open("w", encoding="utf-16", newline="") as handle:
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
            for index in range(241):
                current = started + timedelta(seconds=index)
                writer.writerow(
                    [
                        current.strftime("%m/%d/%Y"),
                        current.strftime("%H:%M:%S"),
                        "50.0",
                        "20.0",
                        "80.0",
                        "No",
                    ]
                )
        samples, metadata = read_hwinfo_csv(hwinfo_path, date_order="mdy")
        assert metadata["encoding"] == "utf-16"
        assert inspect_report(hwinfo_path, "mdy", None)["passed"]
        integrated = integrate_window(
            samples,
            samples[0].timestamp_s + 30.0,
            samples[0].timestamp_s + 210.0,
        )
        assert integrated["valid_quality"]
        assert not integrated["thermal_invalid_reasons"]
        assert integrated["sample_completeness"] == 1.0
        assert abs(integrated["trapezoidal_energy_j"] - 9000.0) < 1e-6

        derived = {
            "trial_id": "synthetic",
            "hwinfo_cpu_package_j_per_image": 9.0,
            "hwinfo_dynamic_cpu_package_j_per_image": 7.2,
            "mean_latency_ms": 5.0,
            "hwinfo_cpu_package_power_w": 50.0,
            "integration_method_relative_spread": 0.0,
        }
        insufficient = aggregate([{"derived": derived}])
        assert not insufficient["stable"]
        assert insufficient["hwinfo_cpu_package_j_per_image_cv"] is None
        assert insufficient["latency_cv"] is None
        json.dumps(insufficient, allow_nan=False)
        stable = aggregate(
            [{"derived": {**derived, "trial_id": str(index)}} for index in range(5)]
        )
        assert stable["stable"]
        assert stable["hwinfo_cpu_package_j_per_image_cv"] == 0.0

    print("PASS: self test")


if __name__ == "__main__":
    main()
