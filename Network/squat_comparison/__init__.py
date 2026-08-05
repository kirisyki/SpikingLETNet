"""Independent direct-SNN QAT+SQUAT comparison implementation.

This package intentionally does not import the repository's historical QAD
quantizers.  The only shared code is the unquantized network topology, data
set, and task loss.
"""

from .lif_neuron import DirectLIF
from .state_quantizer import ThresholdCenteredStateQuantizer

__all__ = ["DirectLIF", "ThresholdCenteredStateQuantizer"]
