"""Test mps-conv3d: FP32, FP16, BF16 support"""

import torch
import time

# Build and load the extension
print("Loading mps_conv3d...")
from mps_conv3d import conv3d, Conv3d, is_available

print(f"MPS available: {is_available()}")

def test_forward(dtype, name):
    """Test forward pass"""
    torch.manual_seed(42)

    B, C_in, D, H, W = 2, 16, 8, 16, 16
    C_out = 16
    K = 3

    input = torch.randn(B, C_in, D, H, W, device='mps', dtype=dtype)
    weight = torch.randn(C_out, C_in, K, K, K, device='mps', dtype=dtype)

    output = conv3d(input, weight, stride=1, padding=1)

    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    ok = not has_nan and not has_inf

    expected_shape = (B, C_out, D, H, W)
    shape_ok = output.shape == expected_shape

    print(f"  {name} forward: shape={output.shape}, dtype={output.dtype}, ok={ok and shape_ok}")
    return ok and shape_ok

def test_forward_with_bias(dtype, name):
    """Test forward pass with bias"""
    torch.manual_seed(42)

    B, C_in, D, H, W = 2, 16, 8, 16, 16
    C_out = 16
    K = 3

    input = torch.randn(B, C_in, D, H, W, device='mps', dtype=dtype)
    weight = torch.randn(C_out, C_in, K, K, K, device='mps', dtype=dtype)
    bias = torch.randn(C_out, device='mps', dtype=dtype)

    output = conv3d(input, weight, bias=bias, stride=1, padding=1)

    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    ok = not has_nan and not has_inf

    print(f"  {name} forward+bias: shape={output.shape}, ok={ok}")
    return ok

def test_backward(dtype, name):
    """Test backward pass"""
    torch.manual_seed(42)

    B, C_in, D, H, W = 2, 8, 4, 8, 8
    C_out = 8
    K = 3

    input = torch.randn(B, C_in, D, H, W, device='mps', dtype=dtype, requires_grad=True)
    weight = torch.randn(C_out, C_in, K, K, K, device='mps', dtype=dtype, requires_grad=True)

    output = conv3d(input, weight, stride=1, padding=1)
    loss = output.sum()
    loss.backward()

    grad_input_ok = input.grad is not None and not torch.isnan(input.grad).any()
    grad_weight_ok = weight.grad is not None and not torch.isnan(weight.grad).any()
    ok = grad_input_ok and grad_weight_ok

    print(f"  {name} backward: input={grad_input_ok}, weight={grad_weight_ok}")
    return ok

def test_module(dtype, name):
    """Test Conv3d module"""
    torch.manual_seed(42)

    B, C_in, D, H, W = 2, 16, 8, 16, 16
    C_out = 16
    K = 3

    module = Conv3d(C_in, C_out, K, padding=1).to('mps').to(dtype)
    input = torch.randn(B, C_in, D, H, W, device='mps', dtype=dtype)

    output = module(input)

    ok = not torch.isnan(output).any() and not torch.isinf(output).any()
    expected_shape = (B, C_out, D, H, W)
    shape_ok = output.shape == expected_shape

    print(f"  {name} Conv3d module: shape={output.shape}, ok={ok and shape_ok}")
    return ok and shape_ok

def test_stride_dilation(dtype, name):
    """Test with stride and dilation"""
    torch.manual_seed(42)

    B, C_in, D, H, W = 2, 16, 16, 32, 32
    C_out = 16
    K = 3
    stride = 2
    dilation = 2
    padding = dilation

    input = torch.randn(B, C_in, D, H, W, device='mps', dtype=dtype)
    weight = torch.randn(C_out, C_in, K, K, K, device='mps', dtype=dtype)

    output = conv3d(input, weight, stride=stride, padding=padding, dilation=dilation)

    ok = not torch.isnan(output).any() and not torch.isinf(output).any()
    print(f"  {name} stride={stride}, dilation={dilation}: shape={output.shape}, ok={ok}")
    return ok

def test_gradient_correctness():
    """Test gradient correctness via numerical differentiation"""
    torch.manual_seed(42)

    B, C_in, D, H, W = 1, 2, 3, 4, 4
    C_out = 2
    K = 3
    eps = 1e-3

    input = torch.randn(B, C_in, D, H, W, device='mps', dtype=torch.float32, requires_grad=True)
    weight = torch.randn(C_out, C_in, K, K, K, device='mps', dtype=torch.float32, requires_grad=True)

    def forward():
        return conv3d(input, weight, stride=1, padding=1)

    def check_grad(name, param):
        # Analytical gradient
        input.grad = weight.grad = None
        output = forward()
        output.sum().backward()
        grad_analytical = param.grad.clone()

        # Numerical gradient
        grad_numerical = torch.zeros_like(param)
        param_data = param.data.clone()

        for i in range(min(param.numel(), 50)):  # Check first 50 elements
            param.data = param_data.clone()
            param.data.view(-1)[i] += eps
            out_plus = forward().sum().item()

            param.data = param_data.clone()
            param.data.view(-1)[i] -= eps
            out_minus = forward().sum().item()

            grad_numerical.view(-1)[i] = (out_plus - out_minus) / (2 * eps)

        param.data = param_data

        # Compare (only checked elements)
        n = min(param.numel(), 50)
        diff = (grad_analytical.view(-1)[:n] - grad_numerical.view(-1)[:n]).abs()
        rel_diff = (diff / (grad_analytical.view(-1)[:n].abs() + 1e-6)).mean().item() * 100
        return rel_diff < 10.0  # 10% tolerance for numerical precision

    ok_input = check_grad('input', input)
    ok_weight = check_grad('weight', weight)

    ok = ok_input and ok_weight
    print(f"  Gradient correctness: input={ok_input}, weight={ok_weight}")
    return ok

def compare_fp32_bf16():
    """Compare FP32 vs BF16 outputs"""
    torch.manual_seed(42)

    B, C_in, D, H, W = 1, 8, 4, 8, 8
    C_out = 8
    K = 3

    input_fp32 = torch.randn(B, C_in, D, H, W, device='mps', dtype=torch.float32)
    weight_fp32 = torch.randn(C_out, C_in, K, K, K, device='mps', dtype=torch.float32)

    output_fp32 = conv3d(input_fp32, weight_fp32, stride=1, padding=1)

    # BF16
    input_bf16 = input_fp32.to(torch.bfloat16)
    weight_bf16 = weight_fp32.to(torch.bfloat16)

    output_bf16 = conv3d(input_bf16, weight_bf16, stride=1, padding=1)

    diff = (output_fp32 - output_bf16.to(torch.float32)).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / (output_fp32.abs() + 1e-6)).mean().item() * 100

    ok = rel_diff < 5.0
    print(f"  FP32 vs BF16: max_diff={max_diff:.6f}, rel_diff={rel_diff:.2f}%, ok={ok}")
    return ok

def benchmark(dtype, name, warmup=5, runs=20):
    """Benchmark forward pass"""
    torch.manual_seed(42)

    B, C_in, D, H, W = 4, 64, 16, 32, 32
    C_out = 64
    K = 3

    input = torch.randn(B, C_in, D, H, W, device='mps', dtype=dtype)
    weight = torch.randn(C_out, C_in, K, K, K, device='mps', dtype=dtype)

    for _ in range(warmup):
        _ = conv3d(input, weight, stride=1, padding=1)
    torch.mps.synchronize()

    start = time.time()
    for _ in range(runs):
        _ = conv3d(input, weight, stride=1, padding=1)
    torch.mps.synchronize()
    elapsed = time.time() - start

    ms = (elapsed / runs) * 1000
    print(f"  {name}: {ms:.2f} ms")
    return ms

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Testing mps-conv3d")
    print("=" * 50)

    all_ok = True

    print("\n1. Forward pass:")
    all_ok &= test_forward(torch.float32, "FP32")
    all_ok &= test_forward(torch.float16, "FP16")
    all_ok &= test_forward(torch.bfloat16, "BF16")

    print("\n2. Forward with bias:")
    all_ok &= test_forward_with_bias(torch.float32, "FP32")
    all_ok &= test_forward_with_bias(torch.float16, "FP16")
    all_ok &= test_forward_with_bias(torch.bfloat16, "BF16")

    print("\n3. Backward pass:")
    all_ok &= test_backward(torch.float32, "FP32")
    all_ok &= test_backward(torch.float16, "FP16")
    all_ok &= test_backward(torch.bfloat16, "BF16")

    print("\n4. Conv3d module:")
    all_ok &= test_module(torch.float32, "FP32")
    all_ok &= test_module(torch.float16, "FP16")
    all_ok &= test_module(torch.bfloat16, "BF16")

    print("\n5. Stride and dilation:")
    all_ok &= test_stride_dilation(torch.float32, "FP32")
    all_ok &= test_stride_dilation(torch.float16, "FP16")
    all_ok &= test_stride_dilation(torch.bfloat16, "BF16")

    print("\n6. Gradient correctness:")
    all_ok &= test_gradient_correctness()

    print("\n7. FP32 vs BF16 comparison:")
    all_ok &= compare_fp32_bf16()

    print("\n8. Benchmarks:")
    benchmark(torch.float32, "FP32")
    benchmark(torch.float16, "FP16")
    benchmark(torch.bfloat16, "BF16")

    print("\n" + "=" * 50)
    if all_ok:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 50)
