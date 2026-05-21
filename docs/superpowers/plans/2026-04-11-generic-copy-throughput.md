# GenericCopy Throughput Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add memory throughput (GB/s) measurement to each GenericCopy test case, with a contiguous copy baseline for comparison.

**Architecture:** Add a `benchmark_copy` helper that uses CUDA events to time `_tm.generic_copy()` and `torch.clone()` over 20 iterations. Integrate into `run_test` so throughput prints inline after each passing correctness check.

**Tech Stack:** PyTorch CUDA events, existing `_turbomind.generic_copy` binding

---

## File Structure

| File | Responsibility |
|---|---|
| `test_generic_copy.py` | Add `benchmark_copy` helper, modify `run_test` to print throughput inline |

---

### Task 1: Add throughput measurement to `test_generic_copy.py`

**Files:**
- Modify: `test_generic_copy.py`

- [ ] **Step 1: Add the `benchmark_copy` helper function**

Add this function after `make_tensors` (after line 32) and before `run_test`:

```python
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
    contig = torch.randn(torch_tensor.shape, dtype=torch_tensor.dtype, device=DEV)
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
```

- [ ] **Step 2: Modify `run_test` to call `benchmark_copy` on pass**

In `run_test`, after the PASS/FAIL print block (after line 63), and before `return match`, add:

```python
    if match:
        benchmark_copy(name, tm_src, tm_dst, torch_tensor)
```

The full `run_test` function should look like this when done:

```python
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
```

- [ ] **Step 3: Run the test to verify correctness and throughput output**

Run: `cd /data/lmdeploy-copy && python test_generic_copy.py`

Expected: All 13 tests pass, each with an inline throughput line like:
```
  [PASS] transpose f32: shape=[128, 64], stride=[1, 64]
         GenericCopy: 245.2 GB/s | Contiguous: 512.1 GB/s (47.9%)
```

The "flip dim=0" test may SKIP (no throughput line for skipped tests).

- [ ] **Step 4: Commit**

```bash
cd /data/lmdeploy-copy
git add test_generic_copy.py
git commit -m "feat(test): add throughput benchmarking to GenericCopy test suite"
```
