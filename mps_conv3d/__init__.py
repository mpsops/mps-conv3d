"""
MPS Conv3D - 3D Convolution for Apple Silicon

Drop-in replacement for torch.nn.functional.conv3d on MPS.
Used in video models: Synchformer, I3D, SlowFast, C3D, etc.
"""

import torch
from torch import nn, Tensor
from torch.autograd import Function
from typing import Optional, Tuple, Union
import torch.nn.functional as F

__version__ = "0.1.0"


def _load_library():
    """Load the native extension."""
    try:
        from mps_conv3d import _C as _lib
        return _lib
    except ImportError:
        import os
        from torch.utils.cpp_extension import load

        src_dir = os.path.join(os.path.dirname(__file__), "csrc")
        _lib = load(
            name="mps_conv3d",
            sources=[os.path.join(src_dir, "conv3d_mps.mm")],
            extra_cflags=["-std=c++17"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=False,
        )
        return _lib


_lib_cache = None


def _get_lib():
    global _lib_cache
    if _lib_cache is None:
        _lib_cache = _load_library()
    return _lib_cache


class _Conv3DFunction(Function):
    """Autograd function for Conv3D."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dilation: Tuple[int, int, int],
        groups: int,
    ) -> Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        lib = _get_lib()
        output = lib.conv3d_forward(
            input, weight,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            groups
        )

        if bias is not None:
            output = output + bias.view(1, -1, 1, 1, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, weight, bias = ctx.saved_tensors
        lib = _get_lib()

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = lib.conv3d_backward_input(
                grad_output, weight, input.shape,
                ctx.stride[0], ctx.stride[1], ctx.stride[2],
                ctx.padding[0], ctx.padding[1], ctx.padding[2],
                ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                ctx.groups
            )

        if ctx.needs_input_grad[1]:
            grad_weight = lib.conv3d_backward_weight(
                grad_output, input, weight.shape,
                ctx.stride[0], ctx.stride[1], ctx.stride[2],
                ctx.padding[0], ctx.padding[1], ctx.padding[2],
                ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                ctx.groups
            )

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3, 4])

        return grad_input, grad_weight, grad_bias, None, None, None, None


def _normalize_tuple(value, n, name):
    """Convert int or tuple to n-tuple."""
    if isinstance(value, int):
        return (value,) * n
    if isinstance(value, (list, tuple)):
        if len(value) == n:
            return tuple(value)
        elif len(value) == 1:
            return (value[0],) * n
    raise ValueError(f"{name} must be an int or {n}-tuple")


def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
) -> Tensor:
    """
    3D convolution on MPS.

    Drop-in replacement for torch.nn.functional.conv3d.

    Args:
        input: Input tensor (N, C_in, D, H, W)
        weight: Weight tensor (C_out, C_in/groups, kD, kH, kW)
        bias: Optional bias tensor (C_out,)
        stride: Stride of convolution
        padding: Padding added to input
        dilation: Dilation of kernel
        groups: Number of groups

    Returns:
        Output tensor (N, C_out, D_out, H_out, W_out)
    """
    if input.device.type != "mps":
        # Fallback to PyTorch for non-MPS
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)

    stride = _normalize_tuple(stride, 3, "stride")
    padding = _normalize_tuple(padding, 3, "padding")
    dilation = _normalize_tuple(dilation, 3, "dilation")

    return _Conv3DFunction.apply(
        input.contiguous(), weight.contiguous(), bias,
        stride, padding, dilation, groups
    )


class Conv3d(nn.Module):
    """
    3D Convolution layer for MPS.

    Drop-in replacement for torch.nn.Conv3d.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to input
        dilation: Dilation of kernel elements
        groups: Number of blocked connections
        bias: If True, adds a learnable bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = _normalize_tuple(kernel_size, 3, "kernel_size")
        self.stride = _normalize_tuple(stride, 3, "stride")
        self.padding = _normalize_tuple(padding, 3, "padding")
        self.dilation = _normalize_tuple(dilation, 3, "dilation")
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return conv3d(
            input, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


_original_conv3d = None


def patch_conv3d():
    """
    Monkey-patch torch.nn.functional.conv3d to use MPS implementation.

    Call this at the start of your script to automatically use MPS conv3d
    for all 3D convolution operations.
    """
    global _original_conv3d

    if _original_conv3d is not None:
        return  # Already patched

    _original_conv3d = F.conv3d

    def patched_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.device.type == 'mps':
            return conv3d(input, weight, bias, stride, padding, dilation, groups)
        return _original_conv3d(input, weight, bias, stride, padding, dilation, groups)

    F.conv3d = patched_conv3d
    print("MPS Conv3D: Patched F.conv3d")


def unpatch_conv3d():
    """Restore original torch.nn.functional.conv3d."""
    global _original_conv3d
    if _original_conv3d is not None:
        F.conv3d = _original_conv3d
        _original_conv3d = None


def is_available() -> bool:
    """Check if MPS is available."""
    return torch.backends.mps.is_available()
