"""Isolated first-image-input quantization ablation utilities."""

from .variant import exempt_first_qlayer_input, qlayer_input_policy

__all__ = ["exempt_first_qlayer_input", "qlayer_input_policy"]
