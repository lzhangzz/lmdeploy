# GenericCopy Throughput Metric Design

## Problem

The GenericCopy test suite only checks correctness. We need to measure memory throughput (GB/s) for each test case and compare against a contiguous copy baseline, so we can quantify how much bandwidth is lost to non-contiguous access patterns.

## Solution

Add throughput measurement to the existing `test_generic_copy.py` using CUDA events from Python. After each correctness check passes, benchmark both GenericCopy and a contiguous `torch.clone()` baseline, then print inline throughput numbers with a percentage.

## Scope

- **Modify only:** `test_generic_copy.py`
- **No C++ changes** — use `torch.cuda.Event` for GPU timing from Python
- **No new files**

## Design

### Measurement protocol

For each test case, after the correctness check passes:

1. **Warmup**: Run `_tm.generic_copy(tm_src, tm_dst)` 3 times
2. **Time GenericCopy**: Record CUDA start/end events, run `_tm.generic_copy()` 20 iterations, compute elapsed ms
3. **Time contiguous baseline**: Create a contiguous tensor of the same byte size, run `torch.clone()` 20 iterations, compute elapsed ms
4. **Compute throughput**: `bytes = numel * dtype_size`, `throughput_GBps = (bytes * iterations) / (elapsed_ms * 1e6)`
5. **Print inline** after the correctness line

### Contiguous baseline

- Create a contiguous tensor with the same shape and dtype as the test tensor
- Measure `torch.clone()` throughput (equivalent to a device-to-device memcpy)
- This represents peak achievable bandwidth for that data size
- Report the ratio: `GenericCopy_throughput / baseline_throughput * 100%`

### Iteration count

20 iterations. Enough to amortize kernel launch overhead and Python overhead, while keeping total test runtime under a few seconds.

### Output format

```
  [PASS] transpose f32: shape=[64, 128], stride=[1, 64]
         GenericCopy: 245.2 GB/s | Contiguous: 512.1 GB/s (47.9%)
```

If a correctness test fails, skip the throughput measurement for that case.

### Implementation details

- Use `torch.cuda.Event(enable_timing=True)` for GPU-side timing
- `start_event.record(); for _ in range(N): op(); end_event.record(); torch.cuda.synchronize(); elapsed = start_event.elapsed_time(end_event)`
- Throughput formula: `GB/s = (numel * dtype_bytes * iterations) / (elapsed_ms * 1e6)`
- The contiguous baseline tensor uses `torch.randn()` with same shape/dtype, measured via `torch.clone()`

## Limitations

- `torch.clone()` includes some PyTorch overhead; the baseline is not a pure `cudaMemcpy`, but it's close enough for relative comparison
- Small tensors (< 1 MB) will show noisy throughput numbers due to fixed overhead per launch
- The percentage is relative to contiguous copy on the same data size, not the GPU's theoretical peak bandwidth
