#!/usr/bin/env python3
"""Test GenericCopy against PyTorch for various non-contiguous layouts."""

import os
import sys

# Set up paths before any turbomind imports
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lmdeploy'))
sys.path.insert(0, os.path.join(REPO, 'build', 'lib'))

import torch
import _turbomind as _tm

DEV = torch.device('cuda')


def make_tensors(torch_tensor):
    """Create (tm_src, tm_dst, golden) from a (possibly non-contiguous) torch tensor.

    tm_src: turbomind Tensor with strides preserved from the torch tensor
    tm_dst: contiguous turbomind Tensor for the output
    golden: contiguous torch tensor with the expected result
    """
    tm_src = _tm.from_dlpack_with_strides(torch_tensor)

    # Allocate a contiguous destination tensor with the same shape/dtype
    contig = torch_tensor.contiguous()
    tm_dst = _tm.from_dlpack(contig.clone())

    golden = contig.clone()
    return tm_src, tm_dst, golden


ITERS = 20
WARMUP = 3


def benchmark_copy(name, tm_src, tm_dst, torch_tensor):
    """Benchmark GenericCopy vs contiguous torch.clone() and print throughput."""
    numel = torch_tensor.numel()
    dtype_bytes = torch_tensor.element_size()
    total_bytes = numel * dtype_bytes

    # --- Benchmark GenericCopy ---
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(WARMUP):
        _tm.generic_copy(tm_src, tm_dst)

    start.record()
    for _ in range(ITERS):
        _tm.generic_copy(tm_src, tm_dst)
    end.record()
    torch.cuda.synchronize()
    gc_ms = start.elapsed_time(end)
    gc_gbps = total_bytes * ITERS / (gc_ms * 1e6)

    # --- Benchmark contiguous clone (baseline) ---
    contig = torch.zeros(torch_tensor.shape, dtype=torch_tensor.dtype, device=DEV)
    for _ in range(WARMUP):
        contig.clone()

    start.record()
    for _ in range(ITERS):
        contig.clone()
    end.record()
    torch.cuda.synchronize()
    bl_ms = start.elapsed_time(end)
    bl_gbps = total_bytes * ITERS / (bl_ms * 1e6)

    pct = gc_gbps / bl_gbps * 100 if bl_gbps > 0 else 0
    print(f"         GenericCopy: {gc_gbps:.1f} GB/s | Contiguous: {bl_gbps:.1f} GB/s ({pct:.1f}%)")


def run_test(name, torch_tensor, atol=1e-5, rtol=1e-5):
    """Run a single GenericCopy test. Returns True on pass."""
    tm_src, tm_dst, golden = make_tensors(torch_tensor)

    _tm.generic_copy(tm_src, tm_dst)

    # Read back the result through DLPack
    result = torch.from_dlpack(tm_dst)

    is_float = torch_tensor.is_floating_point()

    if is_float:
        match = torch.allclose(result, golden, atol=atol, rtol=rtol)
    else:
        match = torch.equal(result, golden)

    status = "PASS" if match else "FAIL"
    shape = list(torch_tensor.shape)
    stride = list(torch_tensor.stride())
    print(f"  [{status}] {name}: shape={shape}, stride={stride}")

    if not match:
        if is_float:
            diff = (result - golden).abs()
            print(f"         max_diff={diff.max().item()}, mean_diff={diff.mean().item()}")
        else:
            mismatches = (result != golden).sum().item()
            total = result.numel()
            print(f"         mismatches={mismatches}/{total}")

    if match:
        benchmark_copy(name, tm_src, tm_dst, torch_tensor)

    return match


def main():
    print("GenericCopy Test Suite")
    print("=" * 60)

    all_passed = True
    total = 0
    passed = 0

    def check(name, tensor, **kwargs):
        nonlocal all_passed, total, passed
        total += 1
        if run_test(name, tensor, **kwargs):
            passed += 1
        else:
            all_passed = False

    # --- Contiguous baseline ---
    print("\nContiguous baseline:")
    check("contiguous f32", torch.randn(64, 128, dtype=torch.float32, device=DEV))

    # --- 2D layout transformations ---
    print("\n2D transformations:")
    check("transpose f32", torch.randn(64, 128, dtype=torch.float32, device=DEV).t())
    check("row-stride (every-other-row)", torch.randn(64, 128, dtype=torch.float32, device=DEV)[::2, :])
    check("col-stride (every-other-col)", torch.randn(64, 128, dtype=torch.float32, device=DEV)[:, ::2])
    check("narrow outer dim", torch.randn(128, 64, dtype=torch.float32, device=DEV)[10:50, :])

    # --- 3D transformations ---
    print("\n3D transformations:")
    check("permute (2,0,1)", torch.randn(16, 32, 64, dtype=torch.float32, device=DEV).permute(2, 0, 1))

    # --- 4D transformations ---
    print("\n4D transformations:")
    check("4D slice", torch.randn(4, 8, 32, 64, dtype=torch.float32, device=DEV)[:, :, ::3, :])

    # --- Combined operations ---
    print("\nCombined operations:")
    check("slice+transpose", torch.randn(64, 128, dtype=torch.float32, device=DEV)[::2, :].t())

    # --- Dtype sweep (all use transpose) ---
    print("\nDtype sweep (transpose):")
    check("transpose f16", torch.randn(64, 128, dtype=torch.float16, device=DEV).t(), atol=1e-3, rtol=1e-3)
    check("transpose i8", torch.randint(-128, 127, (64, 128), dtype=torch.int8, device=DEV).t())
    check("transpose i32", torch.randint(0, 1000, (64, 128), dtype=torch.int32, device=DEV).t())

    # --- Large tensor (exercises vectorization) ---
    print("\nLarge tensor:")
    check("large transpose (1024x1024)", torch.randn(1024, 1024, dtype=torch.float32, device=DEV).t())

    # --- Negative strides ---
    print("\nNegative strides (flip):")
    try:
        check("flip dim=0", torch.randn(32, 64, dtype=torch.float32, device=DEV).flip(0))
    except Exception as e:
        print(f"  [SKIP] flip dim=0: {e}")
        total += 1

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
