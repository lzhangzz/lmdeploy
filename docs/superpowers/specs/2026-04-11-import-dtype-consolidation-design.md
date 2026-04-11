# Consolidate _turbomind imports and dtype mappings

**Date:** 2026-04-11
**Status:** Draft

## Problem

`import _turbomind` appears 10 times across 3 files in the deploy package — 5 wrapped in `try/except ImportError` with silent fallbacks, 5 unguarded. The guards provide no real safety: every caller of the guarded functions eventually does an unguarded import, so if `_turbomind` is missing the pipeline crashes regardless. The guards just make debugging harder by delaying the error.

Additionally, two near-identical `torch.dtype → _tm.DataType` mapping dicts exist in `load_context.py` (`_torch_dtype_to_cpp` with 7 entries, `_infer_compute_dtype` with a 3-entry subset).

## Design

### 1. Module-level `import _turbomind` in each file

Add `import _turbomind as _tm` at the top of each affected file. Remove all per-function imports and all `try/except ImportError` guards. If the C++ extension is not built, let Python raise `ImportError` immediately at import time — clear, obvious, and impossible to misdiagnose.

### 2. Consolidate dtype maps in `load_context.py`

Replace the two duplicated mapping dicts with two module-level constants:

- `_STR_TO_DTYPE`: maps config strings (`'float32'`, `'float16'`, `'bfloat16'`) to `_tm.DataType`
- `_TORCH_TO_CPP`: maps `torch.dtype` to `_tm.DataType` (7 entries: fp32, fp16, bf16, int32, int64, int8, uint8)

Simplify the helper functions to one-liners using these maps. `_infer_compute_dtype` reuses `_TORCH_TO_CPP` instead of maintaining its own 3-entry subset.

### 3. Remove `_noop` fallback context manager

The `_noop` class existed as a fallback when `_turbomind` wasn't available. With the guard removed, it's dead code.

## File changes

| File | Change |
|---|---|
| `lmdeploy/turbomind/deploy/load_context.py` | Module-level `import _turbomind`, consolidate dtype maps, remove 7 per-function imports, remove `_noop` |
| `lmdeploy/turbomind/deploy/linear.py` | Replace guarded `from _turbomind import DataFormat` with `import _turbomind as _tm` |
| `lmdeploy/turbomind/deploy/kind_map.py` | Add module-level `import _turbomind as _tm`, remove 2 guards in `to_data_format()` |

## What stays the same

- `lmdeploy/turbomind/turbomind.py` — already has unguarded module-level `import _turbomind`
- `lmdeploy/turbomind/deploy/configs.py` — already has unguarded module-level `import _turbomind`
- All function signatures and behavior — no API changes
- `DataFormat` usage in `Linear` dataclass — same, just referenced as `_tm.DataFormat`
