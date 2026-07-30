#!/usr/bin/env python3
"""Keep warmed ORT sessions resident while an energy backend measures trials."""

from __future__ import annotations

import argparse
import json
import socketserver
import time
from pathlib import Path
from typing import Any

from common import (
    LOCAL_MODELS_DIR,
    create_session,
    load_inputs,
    model_path_for,
    process_affinity,
    utc_now,
    write_json,
)


class State:
    def __init__(self, args: argparse.Namespace) -> None:
        self.inputs = load_inputs(args.input_samples)
        self.sessions = {}
        for label in ("fp32", "w8a8"):
            path = model_path_for(args.provider, label)
            cache = LOCAL_MODELS_DIR / "openvino_cache" / label
            self.sessions[label] = create_session(
                path,
                args.provider,
                args.threads,
                cache_dir=cache if args.provider == "openvino" else None,
            )
        self._warmup(args.warmup_seconds)
        self.stop_requested = False

    def _warmup(self, seconds_per_model: float) -> None:
        for label, session in self.sessions.items():
            started = time.perf_counter()
            index = 0
            checksum = 0.0
            while time.perf_counter() - started < seconds_per_model:
                output = session.run(["logits"], {"image": self.inputs[index % len(self.inputs)]})[0]
                checksum += float(output.reshape(-1)[0])
                index += 1
            print(f"warmed {label}: {index} iterations checksum={checksum:.6f}", flush=True)

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action")
        duration = float(request.get("duration", 0.0))
        if action == "stop":
            self.stop_requested = True
            return {"action": "stop", "ok": True}
        if duration <= 0:
            raise ValueError("duration must be positive")
        if action == "idle":
            started_utc = utc_now()
            started_epoch_ns = time.time_ns()
            started = time.perf_counter()
            time.sleep(duration)
            elapsed = time.perf_counter() - started
            ended_epoch_ns = time.time_ns()
            return {
                "action": "idle",
                "started_at_utc": started_utc,
                "ended_at_utc": utc_now(),
                "started_epoch_ns": started_epoch_ns,
                "ended_epoch_ns": ended_epoch_ns,
                "elapsed_seconds": elapsed,
                "iterations": 0,
            }
        if action != "run":
            raise ValueError(f"Unsupported action: {action}")
        label = request.get("model")
        if label not in self.sessions:
            raise ValueError(f"Unsupported model: {label}")
        session = self.sessions[label]
        started_utc = utc_now()
        started_epoch_ns = time.time_ns()
        started = time.perf_counter()
        index = 0
        checksum = 0.0
        while time.perf_counter() - started < duration:
            output = session.run(["logits"], {"image": self.inputs[index % len(self.inputs)]})[0]
            checksum += float(output.reshape(-1)[0])
            index += 1
        elapsed = time.perf_counter() - started
        ended_epoch_ns = time.time_ns()
        return {
            "action": "run",
            "model": label,
            "started_at_utc": started_utc,
            "ended_at_utc": utc_now(),
            "started_epoch_ns": started_epoch_ns,
            "ended_epoch_ns": ended_epoch_ns,
            "elapsed_seconds": elapsed,
            "iterations": index,
            "mean_seconds_per_image": elapsed / index,
            "checksum": checksum,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("cpu", "openvino"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--cpu-ids", type=int, nargs="+", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--input-samples", type=int, default=100)
    args = parser.parse_args()

    with process_affinity(args.cpu_ids):
        state = State(args)

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    request = json.loads(self.rfile.readline().decode("utf-8"))
                    response = state.execute(request)
                    response["ok"] = True
                except Exception as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write((json.dumps(response, allow_nan=False) + "\n").encode("utf-8"))

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
            server.timeout = 1.0
            write_json(
                args.ready_file,
                {
                    "host": "127.0.0.1",
                    "port": server.server_address[1],
                    "provider": args.provider,
                    "threads": args.threads,
                    "cpu_ids": args.cpu_ids,
                    "ready_at_utc": utc_now(),
                },
            )
            while not state.stop_requested:
                server.handle_request()


if __name__ == "__main__":
    main()
