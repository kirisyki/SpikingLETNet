#!/usr/bin/env python3
"""Generate hashes for immutable files in the transfer bundle."""

from __future__ import annotations

from pathlib import Path

from common import PACKAGE_ROOT, sha256_file


EXCLUDED_PARTS = {".venv", "results", "local_optimized", "__pycache__"}


def main() -> None:
    output = PACKAGE_ROOT / "checksums.sha256"
    files = []
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(PACKAGE_ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        files.append(relative)
    lines = [f"{sha256_file(PACKAGE_ROOT / relative)}  {relative.as_posix()}" for relative in sorted(files)]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} hashes to {output}")


if __name__ == "__main__":
    main()
