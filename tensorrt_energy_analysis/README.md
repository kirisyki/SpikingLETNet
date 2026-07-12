# TensorRT FP32/INT8 Energy Experiment

This directory contains the TensorRT deployment and PRO6000 board-energy comparison for the fixed
batch-1 ANN-form `SpikingLETNet_shallow_max` ONNX model. Original files under `onnx_models/` and
existing PyTorch measurements are read-only inputs.

The experiment uses `/root/autodl-tmp/conda-envs/QSNN/bin/python`, TensorRT 11.1, CUDA Python
bindings, and NVML. Generated models, engines, build provenance, raw power traces, and reports stay
inside this directory.

