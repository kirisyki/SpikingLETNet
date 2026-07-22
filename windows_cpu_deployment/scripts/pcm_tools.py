#!/usr/bin/env python3
"""Intel PCM CSV parsing and exact external-program measurement helpers."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[list[str]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
        except UnicodeError as exc:
            last_error = exc
    raise RuntimeError(f"Cannot decode PCM CSV {path}: {last_error}")


def _number(value: str) -> float | None:
    text = value.strip().replace(" ", "")
    if not text or text.lower() in {"n/a", "nan", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_pcm_csv(path: Path) -> dict[str, Any]:
    """Parse the two-row header emitted by Intel PCM's pcm.exe."""

    rows = _read_rows(path)
    if len(rows) < 3:
        raise RuntimeError(f"PCM CSV {path} contains fewer than three non-empty rows")
    group_header, metric_header = rows[0], rows[1]
    data_rows = rows[2:]
    width = max(len(group_header), len(metric_header))
    group_header += [""] * (width - len(group_header))
    metric_header += [""] * (width - len(metric_header))

    def find_index(group_prefix: str, metric: str) -> int | None:
        for index, (group, name) in enumerate(zip(group_header, metric_header)):
            if group.strip().lower().startswith(group_prefix) and name.strip().lower() == metric:
                return index
        return None

    energy_index = find_index("system", "proc energy (joules)")
    if energy_index is None:
        for index, name in enumerate(metric_header):
            if name.strip().lower() == "proc energy (joules)":
                energy_index = index
                break
    if energy_index is None:
        raise RuntimeError(
            f"PCM CSV has no 'Proc Energy (Joules)' column; metrics={metric_header}"
        )

    thermal_index = None
    for index, (group, name) in enumerate(zip(group_header, metric_header)):
        if group.strip().lower().startswith("socket") and name.strip().upper() == "TEMP":
            thermal_index = index
            break

    parsed_rows = []
    for row in data_rows:
        if energy_index >= len(row):
            continue
        energy = _number(row[energy_index])
        if energy is None:
            continue
        thermal = _number(row[thermal_index]) if thermal_index is not None and thermal_index < len(row) else None
        parsed_rows.append(
            {
                "package_energy_j": energy,
                "thermal_headroom_c": thermal,
                "date": row[0].strip() if row else None,
                "time": row[1].strip() if len(row) > 1 else None,
            }
        )
    if not parsed_rows:
        raise RuntimeError(f"PCM CSV {path} contains no numeric package-energy samples")
    # delay=0 plus an external command produces exactly one sample. Summing also
    # makes this robust if a PCM build emits multiple samples.
    energies = [row["package_energy_j"] for row in parsed_rows]
    thermal = [row["thermal_headroom_c"] for row in parsed_rows if row["thermal_headroom_c"] is not None]
    return {
        "package_energy_j": sum(energies),
        "thermal_headroom_min_c": min(thermal) if thermal else None,
        "samples": parsed_rows,
        "header_groups": group_header,
        "header_metrics": metric_header,
    }


def run_pcm_external(
    pcm_exe: Path,
    external_command: list[str],
    csv_path: Path,
    log_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not pcm_exe.is_file():
        raise FileNotFoundError(pcm_exe)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.unlink(missing_ok=True)
    command = [
        str(pcm_exe),
        "0",
        f"-csv={csv_path.resolve()}",
        "-nc",
        "--no-color",
        "--",
        *external_command,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    log_path.write_text(
        "COMMAND\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"pcm.exe failed with exit code {completed.returncode}; see {log_path}")
    result = parse_pcm_csv(csv_path)
    result.update(
        {
            "pcm_command": subprocess.list2cmdline(command),
            "pcm_returncode": completed.returncode,
            "pcm_csv": str(csv_path),
            "pcm_log": str(log_path),
        }
    )
    return result
