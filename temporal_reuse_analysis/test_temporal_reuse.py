from __future__ import annotations

import unittest

import torch
from torch import nn

from temporal_reuse_analysis.analyze_temporal_reuse import compute_step_histogram


class TemporalReuseMathTests(unittest.TestCase):
    def test_one_by_one_codes_have_expected_compute_steps(self) -> None:
        layer = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        codes = torch.tensor([[[[0.0, 1.0, 3.0, 8.0]]]])
        histogram = compute_step_histogram(codes, layer, timestep=8)
        self.assertEqual(histogram.tolist(), [0, 2, 0, 0, 0, 0, 1, 0, 1])

    def test_window_uses_minimum_positive_code(self) -> None:
        layer = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        codes = torch.tensor([[[[0.0, 5.0, 0.0], [8.0, 3.0, 0.0], [0.0, 0.0, 0.0]]]])
        histogram = compute_step_histogram(codes, layer, timestep=8)
        self.assertEqual(histogram.tolist(), [0, 0, 0, 0, 0, 0, 1, 0, 0])

    def test_all_zero_padded_windows_still_compute_timestep_one(self) -> None:
        layer = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        codes = torch.zeros(1, 1, 2, 2)
        histogram = compute_step_histogram(codes, layer, timestep=8)
        self.assertEqual(histogram[1].item(), 4)
        self.assertEqual(histogram.sum().item(), 4)

    def test_padding_zeros_do_not_lower_positive_minimum(self) -> None:
        layer = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        codes = torch.tensor([[[[8.0]]]])
        histogram = compute_step_histogram(codes, layer, timestep=8)
        self.assertEqual(histogram[1].item(), 1)

    def test_dilated_window_uses_actual_receptive_field(self) -> None:
        layer = nn.Conv2d(1, 1, kernel_size=3, dilation=2, bias=False)
        codes = torch.full((1, 1, 5, 5), 8.0)
        codes[0, 0, 2, 2] = 1.0
        histogram = compute_step_histogram(codes, layer, timestep=8)
        self.assertEqual(histogram[8].item(), 1)

    def test_conv_transpose_uses_zero_inserted_windows(self) -> None:
        layer = nn.ConvTranspose2d(
            1, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        codes = torch.tensor([[[[1.0]]]])
        histogram = compute_step_histogram(codes, layer, timestep=8)
        self.assertEqual(histogram[8].item(), 4)
        self.assertEqual(histogram.sum().item(), 4)


if __name__ == "__main__":
    unittest.main()

