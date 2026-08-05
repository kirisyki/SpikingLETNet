#!/usr/bin/env python3
"""Resume the amended FP32 stage while preserving CPU DataLoader RNG state."""

from __future__ import annotations

import os
import sys
from pathlib import Path

NETWORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NETWORK_DIR))
os.chdir(NETWORK_DIR)

from squat_comparison.resume_compat import install_cpu_checkpoint_restore  # noqa: E402

install_cpu_checkpoint_restore()

from squat_comparison.train_fp32_amended_100 import main  # noqa: E402


if __name__ == "__main__":
    main()

