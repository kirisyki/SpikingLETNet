from __future__ import annotations

import torch

from squat_comparison.audit import OutputAudit
from squat_comparison.pipeline import PIPELINE_STAGES, assert_no_qad_compute_nodes
from squat_comparison.training import DeviceConfusionMatrix


def test_confusion_matrix_and_output_audit_match_hand_calculation() -> None:
    confusion = DeviceConfusionMatrix(2, torch.device("cpu"))
    labels = torch.tensor([[0, 0], [1, 1]])
    predictions = torch.tensor([[0, 1], [1, 1]])
    confusion.update(labels, predictions)
    result = confusion.result()
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["per_class_iou"] == [0.5, 2 / 3]
    assert result["miou"] == (0.5 + 2 / 3) / 2
    assert result["pixel_accuracy"] == 0.75

    audit = OutputAudit(2)
    audit.update(torch.tensor([[[[0.0, 1.0]], [[0.0, 1.0]]]]))
    summary = audit.result()
    assert summary["all_zero_pixel_fraction"] == 0.5
    assert summary["maximum_tie_fraction"] == 1.0


def test_pipeline_has_no_qad_compute_node() -> None:
    assert_no_qad_compute_nodes()
    joined = " ".join(PIPELINE_STAGES).lower()
    assert "read_frozen_qad_json" in joined
    assert "qad_train" not in joined

