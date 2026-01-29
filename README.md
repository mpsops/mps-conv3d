# MPS Conv3D

3D Convolution for Apple Silicon (M1/M2/M3/M4).

**Drop-in replacement** for `torch.nn.functional.conv3d` on MPS.

## Why?

3D convolutions are essential for video models:
- **Synchformer**: Audio-visual synchronization
- **I3D**: Video classification
- **SlowFast**: Action recognition
- **C3D**: Video feature extraction
- **MMAudio**: Audio generation from video

But PyTorch's MPS backend doesn't support 3D convolutions:
```
NotImplementedError: aten::slow_conv3d_forward is not implemented for MPS
```

This package provides a native Metal implementation.

## Installation

```bash
pip install mps-conv3d
```

Or from source:

```bash
git clone https://github.com/mpsops/mps-conv3d
cd mps-conv3d
pip install -e .
```

## Quick Start

### Patch All Conv3D Operations (Recommended)

```python
from mps_conv3d import patch_conv3d

# Patch at the start of your script
patch_conv3d()

# Now all conv3d operations use MPS!
import torch
import torch.nn.functional as F

x = torch.randn(1, 3, 16, 112, 112, device='mps')
w = torch.randn(64, 3, 3, 7, 7, device='mps')
out = F.conv3d(x, w, padding=(1, 3, 3))  # Uses MPS!
```

### Direct Usage

```python
import torch
from mps_conv3d import conv3d

x = torch.randn(1, 3, 16, 112, 112, device='mps')
w = torch.randn(64, 3, 3, 7, 7, device='mps')

out = conv3d(x, w, stride=1, padding=(1, 3, 3))
```

### Conv3d Module

```python
from mps_conv3d import Conv3d

conv = Conv3d(
    in_channels=3,
    out_channels=64,
    kernel_size=(3, 7, 7),
    stride=(1, 2, 2),
    padding=(1, 3, 3)
).to('mps')

x = torch.randn(1, 3, 16, 112, 112, device='mps')
out = conv(x)
```

## API Reference

### `conv3d(input, weight, bias, stride, padding, dilation, groups)`

Same signature as `torch.nn.functional.conv3d`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor | Input tensor (N, C_in, D, H, W) |
| `weight` | Tensor | Weight tensor (C_out, C_in/groups, kD, kH, kW) |
| `bias` | Tensor | Optional bias (C_out,) |
| `stride` | int/tuple | Stride of convolution |
| `padding` | int/tuple | Padding added to input |
| `dilation` | int/tuple | Dilation of kernel |
| `groups` | int | Number of groups |

### `patch_conv3d()`

Monkey-patches `torch.nn.functional.conv3d` to use MPS implementation for MPS tensors.

### `unpatch_conv3d()`

Restores original `torch.nn.functional.conv3d`.

## Compatibility

- **PyTorch**: 2.0+
- **macOS**: 12.0+ (Monterey)
- **Hardware**: Apple Silicon (M1/M2/M3/M4)

## Features

- Full forward and backward pass (training supported)
- fp32 and fp16 supported
- Groups and dilation supported
- Drop-in compatible with PyTorch API

## License

MIT
