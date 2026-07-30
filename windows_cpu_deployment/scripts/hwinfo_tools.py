#!/usr/bin/env python3
"""Parse HWiNFO sensor logs and integrate CPU package power over trial markers."""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import math
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as datetime_time
from pathlib import Path
from typing import Any, Sequence

from common import sha256_file, write_json


@dataclass(frozen=True)
class HwinfoSample:
    timestamp_s: float
    power_w: float
    thermal_headroom_c: float | None
    temperature_c: float | None
    thermal_throttling: bool | None


def _decode(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    encodings = (
        ("utf-16", "utf-8-sig", "gb18030", "cp1252")
        if raw.startswith((b"\xff\xfe", b"\xfe\xff"))
        else ("utf-8-sig", "utf-16", "gb18030", "cp1252")
    )
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return raw.decode(encoding), encoding
        except (UnicodeError, UnicodeDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"Cannot decode HWiNFO log {path}: {last_error}")


def _normalize(value: str) -> str:
    return " ".join(value.strip().strip("\ufeff").split()).lower()


def _number(value: str) -> float | None:
    text = value.strip().replace("\u00a0", "").replace(" ", "")
    if not text or text.lower() in {"n/a", "nan", "-", "unknown"}:
        return None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def _date_from_fields(value: str, order: str) -> date:
    fields = [int(item) for item in re.findall(r"\d+", value)]
    if len(fields) != 3:
        raise ValueError(f"Unsupported HWiNFO date: {value!r}")
    if fields[0] >= 1000:
        year, month, day = fields
    elif fields[2] >= 1000:
        if order == "ymd":
            raise ValueError(f"Date {value!r} is not year-first")
        if order == "dmy" or (order == "auto" and fields[0] > 12):
            day, month, year = fields
        elif order == "mdy" or (order == "auto" and fields[1] > 12):
            month, day, year = fields
        else:
            # HWiNFO normally follows the Windows locale. Ambiguous slash
            # dates default to m/d/y; users can force --date-order dmy.
            month, day, year = fields
    else:
        raise ValueError(f"HWiNFO date has no four-digit year: {value!r}")
    return date(year, month, day)


def _time_from_field(value: str) -> datetime_time:
    cleaned = value.strip()
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%I:%M:%S.%f %p", "%I:%M:%S %p"):
        try:
            return datetime.strptime(cleaned, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Unsupported HWiNFO time: {value!r}")


def _local_timestamp(date_value: str, time_value: str, date_order: str) -> float:
    naive = datetime.combine(
        _date_from_fields(date_value, date_order),
        _time_from_field(time_value),
    )
    # HWiNFO timestamps are local wall-clock values. Running analysis on the
    # measured PC makes astimezone() apply the same Windows time zone/DST rules.
    return naive.astimezone().timestamp()


def _column_candidates(headers: Sequence[str], predicate: Any) -> list[int]:
    return [index for index, value in enumerate(headers) if predicate(_normalize(value))]


def _choose_single(headers: Sequence[str], indices: list[int], description: str) -> int:
    if not indices:
        raise RuntimeError(f"HWiNFO log has no {description} column; headers={list(headers)}")
    if len(indices) > 1:
        names = [headers[index] for index in indices]
        raise RuntimeError(
            f"HWiNFO log has multiple {description} columns {names}; pass an exact --power-column"
            if description == "CPU Package Power"
            else f"HWiNFO log has multiple {description} columns {names}"
        )
    return indices[0]


def read_hwinfo_csv(
    path: Path,
    *,
    date_order: str = "auto",
    power_column: str | None = None,
) -> tuple[list[HwinfoSample], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    text, encoding = _decode(path)
    nonempty = [line for line in text.splitlines() if line.strip()]
    if not nonempty:
        raise RuntimeError(f"HWiNFO log is empty: {path}")
    first = nonempty[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    rows = [row for row in csv.reader(io.StringIO(text), delimiter=delimiter) if any(cell.strip() for cell in row)]

    header_index = None
    for index, row in enumerate(rows[:20]):
        normalized = [_normalize(value) for value in row]
        has_date = any(value == "date" or value.endswith("|date") for value in normalized)
        has_time = any(value == "time" or value.endswith("|time") for value in normalized)
        if power_column is not None:
            has_power = any(value.strip() == power_column for value in row)
        else:
            has_power = any(
                "cpu package power" in value and "[w]" in value
                for value in normalized
            )
        if has_date and has_time and has_power:
            header_index = index
            break
    if header_index is None:
        raise RuntimeError("Cannot locate HWiNFO Date/Time/CPU Package Power header")
    headers = rows[header_index]
    normalized = [_normalize(value) for value in headers]

    date_index = _choose_single(
        headers,
        [index for index, value in enumerate(normalized) if value == "date" or value.endswith("|date")],
        "Date",
    )
    time_index = _choose_single(
        headers,
        [index for index, value in enumerate(normalized) if value == "time" or value.endswith("|time")],
        "Time",
    )
    if power_column is not None:
        power_indices = [index for index, value in enumerate(headers) if value.strip() == power_column]
    else:
        power_indices = [
            index
            for index, value in enumerate(normalized)
            if "cpu package power" in value and "[w]" in value
        ]
    power_index = _choose_single(headers, power_indices, "CPU Package Power")

    headroom_indices = _column_candidates(
        headers, lambda value: "distance to tjmax" in value or "thermal headroom" in value
    )
    temperature_indices = _column_candidates(
        headers,
        lambda value: (
            ("cpu package" in value or "core max" in value)
            and ("[°c]" in value or "[c]" in value)
            and "power" not in value
        ),
    )
    throttling_indices = _column_candidates(headers, lambda value: "thermal throttling" in value)
    thermal_available = bool(headroom_indices or temperature_indices or throttling_indices)

    samples_by_time: dict[float, HwinfoSample] = {}
    skipped_rows = 0
    minimum_width = max(date_index, time_index, power_index) + 1
    for row in rows[header_index + 1 :]:
        if len(row) < minimum_width:
            skipped_rows += 1
            continue
        try:
            timestamp = _local_timestamp(row[date_index], row[time_index], date_order)
        except (ValueError, OSError):
            skipped_rows += 1
            continue
        power = _number(row[power_index])
        if power is None or not math.isfinite(power) or power < 0:
            skipped_rows += 1
            continue

        headroom_values = [
            value
            for index in headroom_indices
            if index < len(row)
            for value in [_number(row[index])]
            if value is not None and math.isfinite(value)
        ]
        temperature_values = [
            value
            for index in temperature_indices
            if index < len(row)
            for value in [_number(row[index])]
            if value is not None and math.isfinite(value)
        ]
        throttle_values = []
        for index in throttling_indices:
            if index >= len(row):
                continue
            value = _normalize(row[index])
            if not value or value in {"n/a", "unknown", "-"}:
                continue
            throttle_values.append(value in {"yes", "true", "1", "active", "on", "是"})
        samples_by_time[timestamp] = HwinfoSample(
            timestamp_s=timestamp,
            power_w=power,
            thermal_headroom_c=min(headroom_values) if headroom_values else None,
            temperature_c=max(temperature_values) if temperature_values else None,
            thermal_throttling=any(throttle_values) if throttle_values else None,
        )
    samples = [samples_by_time[key] for key in sorted(samples_by_time)]
    if len(samples) < 3:
        raise RuntimeError(f"HWiNFO log has fewer than three usable samples: {path}")
    intervals = [
        right.timestamp_s - left.timestamp_s for left, right in zip(samples, samples[1:])
        if right.timestamp_s > left.timestamp_s
    ]
    metadata = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "encoding": encoding,
        "delimiter": delimiter,
        "header_row_index": header_index,
        "headers": headers,
        "date_column": headers[date_index],
        "time_column": headers[time_index],
        "power_column": headers[power_index],
        "headroom_columns": [headers[index] for index in headroom_indices],
        "temperature_columns": [headers[index] for index in temperature_indices],
        "throttling_columns": [headers[index] for index in throttling_indices],
        "thermal_telemetry_available": thermal_available,
        "samples": len(samples),
        "skipped_rows": skipped_rows,
        "start_local": datetime.fromtimestamp(samples[0].timestamp_s).astimezone().isoformat(),
        "end_local": datetime.fromtimestamp(samples[-1].timestamp_s).astimezone().isoformat(),
        "median_interval_seconds": statistics.median(intervals),
        "maximum_interval_seconds": max(intervals),
        "power_w": {
            "minimum": min(item.power_w for item in samples),
            "median": statistics.median(item.power_w for item in samples),
            "maximum": max(item.power_w for item in samples),
        },
    }
    return samples, metadata


def _interpolate_power(left: HwinfoSample, right: HwinfoSample, timestamp: float) -> float:
    if right.timestamp_s <= left.timestamp_s:
        return left.power_w
    fraction = (timestamp - left.timestamp_s) / (right.timestamp_s - left.timestamp_s)
    return left.power_w + fraction * (right.power_w - left.power_w)


def integrate_window(
    samples: Sequence[HwinfoSample],
    start_epoch_s: float,
    end_epoch_s: float,
    *,
    minimum_completeness: float = 0.99,
    minimum_median_interval_s: float = 0.8,
    maximum_median_interval_s: float = 1.2,
    maximum_gap_s: float = 2.5,
    maximum_integration_spread: float = 0.01,
    minimum_thermal_headroom_c: float = 2.0,
    maximum_temperature_c: float = 98.0,
    require_thermal: bool = True,
) -> dict[str, Any]:
    if end_epoch_s <= start_epoch_s:
        raise ValueError("Energy window end must be after start")
    times = [item.timestamp_s for item in samples]
    left_index = bisect.bisect_right(times, start_epoch_s) - 1
    right_index = bisect.bisect_left(times, end_epoch_s)
    quality_reasons: list[str] = []
    if left_index < 0 or right_index >= len(samples):
        return {
            "valid_quality": False,
            "quality_reasons": ["window_not_bracketed"],
            "start_epoch_s": start_epoch_s,
            "end_epoch_s": end_epoch_s,
        }

    raw = list(samples[left_index : right_index + 1])
    raw_intervals = [
        right.timestamp_s - left.timestamp_s for left, right in zip(raw, raw[1:])
    ]
    if not raw_intervals or any(value <= 0 for value in raw_intervals):
        return {
            "valid_quality": False,
            "quality_reasons": ["invalid_sample_timestamps"],
            "start_epoch_s": start_epoch_s,
            "end_epoch_s": end_epoch_s,
        }
    median_interval = statistics.median(raw_intervals)
    max_gap = max(raw_intervals)
    inside = [item for item in raw if start_epoch_s <= item.timestamp_s <= end_epoch_s]
    expected = max(1, round((end_epoch_s - start_epoch_s) / median_interval))
    completeness = min(1.0, len(inside) / expected)

    if not minimum_median_interval_s <= median_interval <= maximum_median_interval_s:
        quality_reasons.append("median_sampling_interval")
    if max_gap > maximum_gap_s:
        quality_reasons.append("maximum_sampling_gap")
    if completeness < minimum_completeness:
        quality_reasons.append("sample_completeness")

    start_power = _interpolate_power(raw[0], raw[1], start_epoch_s)
    end_power = _interpolate_power(raw[-2], raw[-1], end_epoch_s)
    points: list[tuple[float, float]] = [(start_epoch_s, start_power)]
    points.extend(
        (item.timestamp_s, item.power_w)
        for item in raw
        if start_epoch_s < item.timestamp_s < end_epoch_s
    )
    points.append((end_epoch_s, end_power))

    left_j = 0.0
    right_j = 0.0
    trapezoid_j = 0.0
    for (left_time, left_power), (right_time, right_power) in zip(points, points[1:]):
        duration = right_time - left_time
        left_j += left_power * duration
        right_j += right_power * duration
        trapezoid_j += (left_power + right_power) * 0.5 * duration
    spread = None
    if trapezoid_j <= 0:
        quality_reasons.append("nonpositive_integrated_energy")
    else:
        spread = (
            max(left_j, right_j, trapezoid_j)
            - min(left_j, right_j, trapezoid_j)
        ) / trapezoid_j
    if spread is not None and spread > maximum_integration_spread:
        quality_reasons.append("integration_method_spread")

    headrooms = [
        item.thermal_headroom_c for item in inside if item.thermal_headroom_c is not None
    ]
    temperatures = [item.temperature_c for item in inside if item.temperature_c is not None]
    throttles = [
        item.thermal_throttling for item in inside if item.thermal_throttling is not None
    ]
    thermal_available = bool(headrooms and temperatures and throttles)
    if require_thermal:
        if not headrooms:
            quality_reasons.append("thermal_headroom_telemetry_missing")
        if not temperatures:
            quality_reasons.append("temperature_telemetry_missing")
        if not throttles:
            quality_reasons.append("thermal_throttling_telemetry_missing")
    thermal_reasons = []
    if headrooms and min(headrooms) <= minimum_thermal_headroom_c:
        thermal_reasons.append("thermal_headroom")
    if temperatures and max(temperatures) >= maximum_temperature_c:
        thermal_reasons.append("maximum_temperature")
    if any(throttles):
        thermal_reasons.append("thermal_throttling")

    return {
        "valid_quality": not quality_reasons,
        "quality_reasons": quality_reasons,
        "thermal_invalid_reasons": thermal_reasons,
        "start_epoch_s": start_epoch_s,
        "end_epoch_s": end_epoch_s,
        "duration_seconds": end_epoch_s - start_epoch_s,
        "raw_samples_inside": len(inside),
        "expected_samples": expected,
        "sample_completeness": completeness,
        "median_interval_seconds": median_interval,
        "maximum_gap_seconds": max_gap,
        "boundary_start_power_w": start_power,
        "boundary_end_power_w": end_power,
        "left_rectangle_energy_j": left_j,
        "right_rectangle_energy_j": right_j,
        "trapezoidal_energy_j": trapezoid_j,
        "integration_method_relative_spread": spread,
        "thermal_telemetry_available": thermal_available,
        "thermal_headroom_min_c": min(headrooms) if headrooms else None,
        "temperature_max_c": max(temperatures) if temperatures else None,
        "thermal_throttling_observed": any(throttles) if throttles else None,
    }


def inspect_report(path: Path, date_order: str, power_column: str | None) -> dict[str, Any]:
    samples, metadata = read_hwinfo_csv(
        path, date_order=date_order, power_column=power_column
    )
    issues = []
    if not metadata["headroom_columns"]:
        issues.append("thermal_headroom_column_missing")
    if not metadata["temperature_columns"]:
        issues.append("temperature_column_missing")
    if not metadata["throttling_columns"]:
        issues.append("thermal_throttling_column_missing")
    if not 0.8 <= metadata["median_interval_seconds"] <= 1.2:
        issues.append("median_sampling_interval")
    return {
        "passed": not issues,
        "issues": issues,
        "metadata": metadata,
        "first_sample": asdict(samples[0]),
        "last_sample": asdict(samples[-1]),
        "note": "Formal per-window quality is checked against trial markers.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--log", type=Path, required=True)
    inspect_parser.add_argument("--date-order", choices=("auto", "ymd", "mdy", "dmy"), default="auto")
    inspect_parser.add_argument("--power-column")
    inspect_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect_report(args.log, args.date_order, args.power_column)
    if args.output:
        write_json(args.output, report)
    status = "PASS" if report["passed"] else "FAIL"
    metadata = report["metadata"]
    sample_count = metadata["samples"]
    power_name = metadata["power_column"]
    median_interval = metadata["median_interval_seconds"]
    print(
        f"{status}: {sample_count} samples, power={power_name!r}, "
        f"median interval={median_interval:.3f}s"
    )
    if not report["passed"]:
        issues_text = ",".join(report["issues"])
        print(f"issues={issues_text}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
