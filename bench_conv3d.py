#!/usr/bin/env python3
"""Benchmark mps-conv3d vs PyTorch native conv3d"""

import torch
import torch.nn.functional as F
import time
import sys

def bench(fn, name, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
        torch.mps.synchronize()

    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.mps.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) / iters * 1000
    return ms

def main():
    print("=== Conv3D Benchmark: mps-conv3d vs PyTorch native ===\n")

    # Force rebuild
    if "--rebuild" in sys.argv:
        print("Rebuilding extension...")
        import subprocess
        subprocess.run(["pip", "install", "-e", ".", "--no-build-isolation"], check=True)

    # Test config
    x = torch.randn(1, 64, 16, 64, 64, device='mps', dtype=torch.float16)
    w = torch.randn(128, 64, 3, 3, 3, device='mps', dtype=torch.float16)
    pad = 1

    print(f"Input:  {x.shape} ({x.dtype})")
    print(f"Weight: {w.shape}")
    print(f"Output: (1, 128, 16, 64, 64)\n")

    # PyTorch native
    native_ms = bench(lambda: F.conv3d(x, w, padding=pad), 'PyTorch native')
    print(f"PyTorch native: {native_ms:.2f} ms")

    # mps-conv3d
    from mps_conv3d import conv3d as mps_conv3d
    mps_ms = bench(lambda: mps_conv3d(x, w, padding=pad), 'mps-conv3d')
    print(f"mps-conv3d:     {mps_ms:.2f} ms")

    # Verify correctness
    out_native = F.conv3d(x, w, padding=pad)
    out_mps = mps_conv3d(x, w, padding=pad)
    max_diff = (out_native - out_mps).abs().max().item()
    print(f"\nMax diff: {max_diff:.6f}")

    # Result
    speedup = native_ms / mps_ms
    if speedup > 1:
        print(f"\n>>> mps-conv3d is {speedup:.2f}x FASTER <<<")
    else:
        print(f"\n>>> mps-conv3d is {1/speedup:.2f}x slower <<<")

if __name__ == "__main__":
    main()
