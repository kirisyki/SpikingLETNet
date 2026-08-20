from __future__ import annotations

import numpy as np

from qad_distortion_repair.statistics import paired_summary


def test_paired_summary_reports_repair_direction() -> None:
    result = paired_summary(
        np.asarray([0.10, 0.20, 0.30]),
        np.asarray([0.20, 0.30, 0.40]),
        rng=np.random.default_rng(7),
        replicates=200,
    )
    assert result["qad_minus_ste"] < 0
    assert result["relative_reduction_fraction"] > 0
    assert result["qad_lower_source_count"] == 3

