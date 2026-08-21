# UDD6 FP32 与 W8A8 ONNX mIoU 对比

两个模型使用相同的完整 UDD6 validation split、相同图像预处理和同一个六分类全局混淆矩阵。
W8A8 是从 FP32 checkpoint 静态校准得到的 QDQ 混合精度 ONNX；仅适合量化的 Conv、ConvTranspose、MatMul 和 Gemm 路径量化为 8-bit，其余算子可保留 FP32。

| 模型 | mIoU | mIoU (%) | Pixel accuracy | 相对 FP32（pp） |
|---|---:|---:|---:|---:|
| FP32 ONNX | 0.678504 | 67.8504 | 0.842105 | 0 |
| W8A8 QDQ ONNX | 0.658349 | 65.8349 | 0.828458 | -2.0155 |

W8A8 相对 FP32 的全局 validation mIoU 变化为 **-2.0155 pp**；两者逐像素 argmax 一致率为 **89.4740%**。

## 逐类 IoU

| 模型 | background | facade | road | vegetation | vehicle | roof |
|---|---:|---:|---:|---:|---:|---:|
| FP32 ONNX | 0.544356 | 0.630382 | 0.639268 | 0.889507 | 0.544290 | 0.823223 |
| W8A8 QDQ ONNX | 0.522477 | 0.600569 | 0.613698 | 0.875686 | 0.526869 | 0.810794 |

## 口径限制

- 这是 ONNX Runtime CPU 对源 FP32/W8A8 QDQ ONNX 的精度评估，不是 TensorRT engine 的 deployed mIoU。
- W8A8 是训练后静态量化（PTQ），不是单独训练的 W8A8 QAT checkpoint。
- validation 同时用于历史 checkpoint 选择，因此结果应称为 validation mIoU。
