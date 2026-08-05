"""Restore the paper-grade STE-QAT budget without changing the short run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantization_comparison.protocol import ExperimentProtocol


LONG_PROTOCOL_VERSION = "w4a4-ste-comparison-v3-long127"


@dataclass(frozen=True)
class LongBudgetSTEProtocol(ExperimentProtocol):
    """Same W4A4 protocol, with 127 run epochs and a 150-epoch LR horizon."""

    run_epochs: int = 127
    schedule_epochs: int = 150

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.method != "ste":
            raise ValueError("the long-budget protocol is restricted to STE-QAT")
        if self.run_epochs != 127 or self.schedule_epochs != 150:
            raise ValueError("the long-budget run/schedule must remain 127/150")

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["protocol_version"] = LONG_PROTOCOL_VERSION
        return payload
