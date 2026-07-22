#!/usr/bin/env python3
"""Tiny PCM-wrapped client that triggers a warmed energy benchmark server."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--action", choices=("run", "idle", "stop"), required=True)
    parser.add_argument("--model", choices=("fp32", "w8a8"))
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = {"action": args.action, "duration": args.duration, "model": args.model}
    with socket.create_connection((args.host, args.port), timeout=args.duration + 60.0) as connection:
        connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
        chunks = []
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    response = json.loads(b"".join(chunks).decode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(response, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "Energy server returned an unknown error"))


if __name__ == "__main__":
    main()
