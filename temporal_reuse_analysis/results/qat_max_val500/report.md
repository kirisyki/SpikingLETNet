# SpikingLETNet QAT Temporal Reuse Result

- Images: `500` (UDD6 validation, fixed order, batch size 1)
- Included quantized spiking convolution layers: `59`
- Hardware timesteps: `8`
- **MAC-weighted mean computed timesteps: `3.059817`**
- Layer-equal mean computed timesteps: `2.282671`
- Window-weighted mean computed timesteps: `2.876372`
- MAC skip ratio versus fixed T=8: `61.7523%`
- MAC reduction factor versus fixed T=8: `2.6145x`

The main result is MAC-weighted. Timestep 1 is always computed. Each later timestep is
compared only with timestep 1 at per-input-channel convolution-window granularity.

## Per-layer result

| Layer | Op | Kernel | Mean steps | MAC skip vs T=8 |
|---|---:|---:|---:|---:|
| `upsample_3.conv` | conv_transpose2d | 3x3 | 5.099008 | 36.2624% |
| `DAB_Block_6.DAB_Module_6_0.conv3x1.conv` | conv2d | 3x1 | 3.975697 | 50.3038% |
| `DAB_Block_1.DAB_Module_1_0.ddconv3x1.conv` | conv2d | 3x1 | 3.873643 | 51.5795% |
| `upsample_2.conv` | conv_transpose2d | 3x3 | 3.671891 | 54.1014% |
| `DAB_Block_1.DAB_Module_1_0.dconv3x1.conv` | conv2d | 3x1 | 3.656501 | 54.2937% |
| `DAB_Block_1.DAB_Module_1_0.conv1x1.conv` | conv2d | 1x1 | 3.625657 | 54.6793% |
| `DAB_Block_6.DAB_Module_6_0.conv1x3.conv` | conv2d | 1x3 | 3.611032 | 54.8621% |
| `DAB_Block_3.DAB_Module_3_0.conv3x1.conv` | conv2d | 3x1 | 3.383897 | 57.7013% |
| `DAB_Block_6.DAB_Module_6_0.dconv3x1.conv` | conv2d | 3x1 | 3.319778 | 58.5028% |
| `final_conv` | conv2d | 3x3 | 3.311132 | 58.6108% |
| `DAB_Block_6.DAB_Module_6_0.conv1x1.conv` | conv2d | 1x1 | 3.187965 | 60.1504% |
| `DAB_Block_6.DAB_Module_6_0.ddconv3x1.conv` | conv2d | 3x1 | 3.158165 | 60.5229% |
| `upsample_1.conv` | conv_transpose2d | 3x3 | 3.115430 | 61.0571% |
| `downsample_2.conv3x3.conv` | conv2d | 3x3 | 3.086807 | 61.4149% |
| `init_conv.2.conv` | conv2d | 3x3 | 3.007159 | 62.4105% |
| `init_conv.1.conv` | conv2d | 3x3 | 3.000337 | 62.4958% |
| `DAB_Block_5.DAB_Module_5_0.conv3x1.conv` | conv2d | 3x1 | 2.983760 | 62.7030% |
| `DAB_Block_6.DAB_Module_6_0.conv1x1_in.conv` | conv2d | 1x1 | 2.980826 | 62.7397% |
| `DAB_Block_3.DAB_Module_3_0.conv1x3.conv` | conv2d | 1x3 | 2.810066 | 64.8742% |
| `downsample_1.conv3x3.conv` | conv2d | 3x3 | 2.773885 | 65.3264% |
| `DAB_Block_1.DAB_Module_1_0.conv1x3.conv` | conv2d | 1x3 | 2.636056 | 67.0493% |
| `DAB_Block_5.DAB_Module_5_0.conv1x3.conv` | conv2d | 1x3 | 2.587657 | 67.6543% |
| `LC1.dconv3x1` | conv2d | 3x1 | 2.507639 | 68.6545% |
| `DAB_Block_1.DAB_Module_1_0.conv3x1.conv` | conv2d | 3x1 | 2.498339 | 68.7708% |
| `DAB_Block_2.DAB_Module_2_0.conv3x1.conv` | conv2d | 3x1 | 2.313201 | 71.0850% |
| `DAB_Block_3.DAB_Module_3_0.dconv3x1.conv` | conv2d | 3x1 | 2.278110 | 71.5236% |
| `DAB_Block_3.DAB_Module_3_0.ddconv3x1.conv` | conv2d | 3x1 | 2.278110 | 71.5236% |
| `DAB_Block_5.DAB_Module_5_0.conv1x1_in.conv` | conv2d | 1x1 | 2.250826 | 71.8647% |
| `DAB_Block_2.DAB_Module_2_0.conv1x3.conv` | conv2d | 1x3 | 2.197858 | 72.5268% |
| `LC3.dconv3x1` | conv2d | 3x1 | 2.155966 | 73.0504% |
| `DAB_Block_6.DAB_Module_6_0.dconv1x3.conv` | conv2d | 1x3 | 2.154160 | 73.0730% |
| `downsample_3.conv3x3.conv` | conv2d | 3x3 | 2.125792 | 73.4276% |
| `DAB_Block_3.DAB_Module_3_0.conv1x1_in.conv` | conv2d | 1x1 | 2.055684 | 74.3039% |
| `DAB_Block_2.DAB_Module_2_0.dconv3x1.conv` | conv2d | 3x1 | 1.908712 | 76.1411% |
| `DAB_Block_2.DAB_Module_2_0.ddconv3x1.conv` | conv2d | 3x1 | 1.908712 | 76.1411% |
| `DAB_Block_1.DAB_Module_1_0.conv1x1_in.conv` | conv2d | 1x1 | 1.879951 | 76.5006% |
| `DAB_Block_3.DAB_Module_3_0.conv1x1.conv` | conv2d | 1x1 | 1.863515 | 76.7061% |
| `DAB_Block_5.DAB_Module_5_0.dconv3x1.conv` | conv2d | 3x1 | 1.823277 | 77.2090% |
| `LC2.dconv3x1` | conv2d | 3x1 | 1.790951 | 77.6131% |
| `DAB_Block_1.DAB_Module_1_0.ddconv1x3.conv` | conv2d | 1x3 | 1.775184 | 77.8102% |
| `DAB_Block_1.DAB_Module_1_0.dconv1x3.conv` | conv2d | 1x3 | 1.727785 | 78.4027% |
| `DAB_Block_5.DAB_Module_5_0.ddconv3x1.conv` | conv2d | 3x1 | 1.723518 | 78.4560% |
| `DAB_Block_5.DAB_Module_5_0.conv1x1.conv` | conv2d | 1x1 | 1.634664 | 79.5667% |
| `DAB_Block_2.DAB_Module_2_0.conv1x1.conv` | conv2d | 1x1 | 1.606548 | 79.9182% |
| `DAB_Block_5.DAB_Module_5_0.ddconv1x3.conv` | conv2d | 1x3 | 1.567677 | 80.4040% |
| `DAB_Block_2.DAB_Module_2_0.conv1x1_in.conv` | conv2d | 1x1 | 1.564066 | 80.4492% |
| `DAB_Block_6.DAB_Module_6_0.ddconv1x3.conv` | conv2d | 1x3 | 1.427438 | 82.1570% |
| `DAB_Block_3.DAB_Module_3_0.dconv1x3.conv` | conv2d | 1x3 | 1.318996 | 83.5125% |
| `DAB_Block_3.DAB_Module_3_0.ddconv1x3.conv` | conv2d | 1x3 | 1.230699 | 84.6163% |
| `DAB_Block_2.DAB_Module_2_0.dconv1x3.conv` | conv2d | 1x3 | 1.151637 | 85.6045% |
| `DAB_Block_2.DAB_Module_2_0.ddconv1x3.conv` | conv2d | 1x3 | 1.102216 | 86.2223% |
| `DAB_Block_4.DAB_Module_4_0.conv3x1.conv` | conv2d | 3x1 | 1.000000 | 87.5000% |
| `DAB_Block_4.DAB_Module_4_0.conv1x3.conv` | conv2d | 1x3 | 1.000000 | 87.5000% |
| `DAB_Block_4.DAB_Module_4_0.dconv3x1.conv` | conv2d | 3x1 | 1.000000 | 87.5000% |
| `DAB_Block_4.DAB_Module_4_0.dconv1x3.conv` | conv2d | 1x3 | 1.000000 | 87.5000% |
| `DAB_Block_4.DAB_Module_4_0.ddconv3x1.conv` | conv2d | 3x1 | 1.000000 | 87.5000% |
| `DAB_Block_4.DAB_Module_4_0.ddconv1x3.conv` | conv2d | 1x3 | 1.000000 | 87.5000% |
| `DAB_Block_4.DAB_Module_4_0.conv1x1.conv` | conv2d | 1x1 | 1.000000 | 87.5000% |
| `DAB_Block_5.DAB_Module_5_0.dconv1x3.conv` | conv2d | 1x3 | 1.000000 | 87.5000% |

Full integer counts, 1–8-step histograms, quantized-code ranges, layer geometry,
and MAC totals are available in `layer_statistics.csv`; provenance is in `summary.json`.
