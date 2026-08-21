#!/usr/bin/env python3
"""Add an explicit seed to the frozen W4A4 baseline trainer.

The historical baseline entrypoint remains unchanged.  This thin adapter
injects a seed-specific ``ExperimentProtocol`` and forwards every other
argument to ``train_quantization_baseline.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

import train_quantization_baseline as baseline
from quantization_comparison.protocol import ExperimentProtocol


APPROVED_FORMAL_SEEDS = (2345, 3456)


def protocol_factory(seed: int) -> Callable[[str], ExperimentProtocol]:
    """Return the exact 16-epoch baseline protocol for one training seed."""

    return lambda method: ExperimentProtocol(method, seed=seed)


def parse_wrapper_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    wrapper, forwarded = parse_wrapper_args(argv)
    smoke = "--smoke" in forwarded
    if not smoke and wrapper.seed not in APPROVED_FORMAL_SEEDS:
        raise ValueError(
            f"formal seed must be one of {APPROVED_FORMAL_SEEDS}, "
            f"got {wrapper.seed}"
        )
    if "--output-dir" not in forwarded and "--resume" not in forwarded:
        raise ValueError("multi-seed runs require --output-dir or --resume")

    baseline.ExperimentProtocol = protocol_factory(wrapper.seed)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *forwarded]
        baseline.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
