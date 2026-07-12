"""Minimal batch-1 TensorRT runtime backed by PyTorch CUDA buffers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import tensorrt as trt
import torch


class TensorRTEngine:
    def __init__(self, engine_path: Path, device: torch.device) -> None:
        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)
        if device.type != "cuda":
            raise ValueError(f"TensorRT requires a CUDA device, got {device}")
        self.engine_path = engine_path
        self.device = device
        torch.cuda.set_device(device)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"failed to create execution context for {engine_path}")
        self.input_name = "image"
        self.output_name = "logits"
        if list(self.engine.get_tensor_shape(self.input_name)) != [1, 3, 400, 400]:
            raise RuntimeError(f"unexpected engine input shape: {self.engine.get_tensor_shape(self.input_name)}")
        if list(self.engine.get_tensor_shape(self.output_name)) != [1, 6, 400, 400]:
            raise RuntimeError(f"unexpected engine output shape: {self.engine.get_tensor_shape(self.output_name)}")
        if self.engine.get_tensor_dtype(self.input_name) != trt.float32:
            raise RuntimeError(f"unexpected input dtype: {self.engine.get_tensor_dtype(self.input_name)}")
        if self.engine.get_tensor_dtype(self.output_name) != trt.float32:
            raise RuntimeError(f"unexpected output dtype: {self.engine.get_tensor_dtype(self.output_name)}")
        self.stream = torch.cuda.Stream(device=device)
        self.output = torch.empty((1, 6, 400, 400), dtype=torch.float32, device=device)
        if not self.context.set_tensor_address(self.output_name, self.output.data_ptr()):
            raise RuntimeError("failed to set TensorRT output address")

    def enqueue(self, input_tensor: torch.Tensor) -> None:
        if input_tensor.device != self.device:
            raise ValueError(f"input is on {input_tensor.device}, expected {self.device}")
        if input_tensor.dtype != torch.float32 or tuple(input_tensor.shape) != (1, 3, 400, 400):
            raise ValueError(f"unexpected input tensor: {input_tensor.dtype} {tuple(input_tensor.shape)}")
        if not input_tensor.is_contiguous():
            raise ValueError("TensorRT input tensor must be contiguous")
        if not self.context.set_tensor_address(self.input_name, input_tensor.data_ptr()):
            raise RuntimeError("failed to set TensorRT input address")
        if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")

    def synchronize(self) -> None:
        self.stream.synchronize()

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.enqueue(input_tensor)
        self.synchronize()
        return self.output.detach().clone()

    def run_loop(self, images: Sequence[torch.Tensor], count: int) -> tuple[float, float]:
        if not images or count <= 0:
            raise ValueError("images and count must be non-empty/positive")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        self.synchronize()
        wall_start = torch.cuda._utils._get_device_index(self.device)  # validate device early
        del wall_start
        import time

        started = time.perf_counter()
        start_event.record(self.stream)
        for index in range(count):
            self.enqueue(images[index % len(images)])
        end_event.record(self.stream)
        self.synchronize()
        return time.perf_counter() - started, start_event.elapsed_time(end_event) / 1000.0

