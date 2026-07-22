#!/usr/bin/env python3
"""Capture the target machine and runtime configuration."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil

from common import RESULTS_DIR, affinity_profiles, environment_summary, write_json


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    profiles, topology = affinity_profiles()
    info: dict[str, Any] = environment_summary()
    info.update(
        {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "memory_bytes": psutil.virtual_memory().total,
            "affinity_profiles": profiles,
            "topology": topology,
            "power_plan": command_output(["powercfg", "/GETACTIVESCHEME"]) if os.name == "nt" else None,
            "windows_version": platform.win32_ver() if os.name == "nt" else None,
        }
    )
    output = RESULTS_DIR / "system_info.json"
    write_json(output, info)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
