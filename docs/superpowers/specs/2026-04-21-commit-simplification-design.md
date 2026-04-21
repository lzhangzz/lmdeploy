# Commit-path simplification in `Builder`

Date: 2026-04-21
Scope: `lmdeploy/turbomind/deploy/builder/_base.py`, `lmdeploy/turbomind/deploy/load_context.py`, `lmdeploy/turbomind/deploy/kind_map.py`, `lmdeploy/turbomind/deploy/spec.py`, `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`, `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`.

## Motivation

The three commit-family surfaces on `Builder` — `_commit_linear`, `_commit_tensor`, and `_add_norm_child` — share a universal "shard → cuda → contiguous → alloc → cast → copy" dance but duplicate it four times (the fourth copy lives in the module-level `_commit_tensors`). `_commit_linear` additionally mixes GPU-invariant preparation with per-GPU work inside a single loop, computes `in_dim`/`out_dim` twice, carries a `max(1, 128)` expression that always evaluates to `128`, and references logic with a confusingly named `_commit_tensors` helper that has a dead `group_size` parameter.

The `group_size` local variable in `_commit_linear` is the same concept as `WeightFormat.block_in`: the per-group quantization block size along the K (input) axis. Having it as a separate threaded variable is redundant terminology. The `if group_size == 0: group_size = 128` default exists only because `fmt.block_in` for AWQ / GPTQ / compressed-tensors is declared as `0` (a "resolve at conversion time" sentinel) and the real value stored on `Spec._group_size` is never propagated into the WeightFormat that reaches `_commit_linear`. Today this happens to work for AWQ/GPTQ/FP8 because `_SUPPORTED_GROUP_SIZES` restricts them to the default value of 128 — but compressed-tensors officially supports `group_size=32` and would be silently clobbered to 128 by the default, producing wrong kernels. That's a latent bug.

The deferred `data_format` attach block in `_commit_linear` is dead: it writes a value into `linear.data_format` that no live code path ever reads (the only reader, `preprocess_linear`, is itself unreachable).

Adjacent to all this, `lmdeploy/turbomind/deploy/load_context.py` is a fossilized ~500-line copy of the same commit stack. No live Python code imports it; only historical design docs reference it.

The goal is a holistic cleanup: deduplicate the copy dance into one primitive, hoist everything GPU-invariant in `_commit_linear` above the per-GPU loop, broaden the TP-split validation to cover every kind and both split sides, remove the dead deferred-attach, drop padding from the commit layer (upstream's job), delete `load_context.py`, and thread the runtime block sizes symmetrically into `WeightFormat` via `build_linear` so `fmt.block_in` and `fmt.block_out` are authoritative by the time a `Linear` reaches `_commit_linear`. This last change removes the `group_size` concept as a separate thread, eliminates the default, and fixes the compressed-tensors bug.

External method signatures of `_commit_linear`, `_commit_tensor`, and `_add_norm_child` are unchanged. All six builder callers (`attention.py`, `ffn.py`, `moe.py`, `mla.py`, `deltanet.py`, `builder/linear.py`) continue to work without modification.

## Architecture

### Resolving `block_in` / `block_out` at `build_linear` time

`WeightFormat.block_in` and `WeightFormat.block_out` are currently declared statically per format. Some formats declare them as `0`, meaning "resolve at conversion time from the model's quant_config". Today nothing ever resolves them; the downstream consumer (`_commit_linear`) papers over the zero with a hardcoded `128` default.

The fix is to thread runtime block sizes from the `Spec` through `build_linear`:

```python
def build_linear(params, prefix, *, index=None,
                 block_in: int = 0, block_out: int = 0) -> Linear | None:
    ...
    fmt = next((f for f in FORMAT_PRIORITY if f.accepts(available)), None)
    if fmt is None:
        return None

    replacements = {}
    if fmt.block_in == 0 and block_in > 0:
        replacements['block_in'] = block_in
    if fmt.block_out == 0 and block_out > 0:
        replacements['block_out'] = block_out
    if replacements:
        fmt = dataclasses.replace(fmt, **replacements)

    tensors = {
        kind: fmt.normalizer(available[s], kind)
        for s, kind in fmt.suffix_map.items() if s in available
    }
    if not tensors:
        return None
    fmt.complete_tensors(tensors)
    return Linear(tensors=tensors, weight_format=fmt, data_format=None)
```

The hardcoded `data_format = fmt.to_data_format(0, group_size=0)` is dropped: `data_format` is dead on the live read side, so producing `None` here is equivalent to producing any other value.

`Spec` and source-model callers of `build_linear` pass `self._group_size` as both `block_in` and `block_out`. In practice today only `block_in` takes effect because no current format has `block_out == 0`; the symmetric call-site prepares for any future format that needs a runtime `block_out`.

### `_copy_shard_to_param`

```python
def _copy_shard_to_param(handle, param_name, shard, *,
                        alloc_shape=None, alloc_dtype=None):
    """Move shard to GPU, allocate the C++ param slot, cast, and copy.

    Invariant: ``dst.byte_size == shard.nbytes`` after the cast.  Upstream
    is responsible for any padding/reshape needed to satisfy this.  A
    mismatch is a bug and raises immediately.

    alloc_shape / alloc_dtype default to the shard's own shape / dtype.
    Override only to express shape/dtype *relabels* where byte size is
    preserved (e.g. quantized weight: physical int32 [in, out/8] stored
    in a logical UINT4 [in, out] C++ slot).
    """
```

Concrete behavior:

- Ensure `shard.is_cuda` and `shard.is_contiguous()`.
- `alloc_shape = alloc_shape or list(shard.shape)`; `alloc_dtype = alloc_dtype or _torch_dtype_to_cpp(shard.dtype)`.
- `dst = handle.param(param_name).alloc(alloc_shape, alloc_dtype)`.
- `shard = _cast_shard_for_tm(shard, dst)`.
- `assert dst.byte_size == shard.nbytes`.
- `dst.copy_from(shard)`.

No padding branch. `_commit_tensor`, `_add_norm_child`, and `_commit_linear` all funnel through this single helper.

### `_shard`

```python
def _shard(tensor, split_dim, tp, rank):
    """Return the ``rank``-th split along ``split_dim``, or the tensor unchanged."""
    if split_dim is None or tp <= 1:
        return tensor
    return tensor.split(tensor.shape[split_dim] // tp, dim=split_dim)[rank]
```

Used by `_commit_tensor` and by the per-kind loop inside `_commit_linear`.

### `_infer_cpp_linear_dtype` simplification

Returns a single value instead of a `(cpp_dtype, group_size)` tuple:

```python
def _infer_cpp_linear_dtype(linear: Linear):
    fmt = linear.weight_format
    if fmt is not None and fmt.cpp_dtype_name is not None:
        cpp_dtype = getattr(_tm.DataType, fmt.cpp_dtype_name, None)
        if cpp_dtype is not None:
            return cpp_dtype
    weight = linear.tensors.get("weight")
    if weight is not None:
        return _TORCH_TO_CPP.get(weight.dtype)
    return None
```

The previous second return (`fmt.block_in or 0`) is no longer needed anywhere — callers that want the block size read `fmt.block_in` directly.

### `_commit_tensor` rewrite

```python
def _commit_tensor(self, name, tensor, split_side=None):
    self._ensure_handles()
    if tensor is None:
        return
    tp = self._tp if split_side else 1
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None
    for i, handle in enumerate(self._handles):
        with self._contexts[i]:
            rank = self._rank_for(i) if tp > 1 else 0
            shard = _shard(tensor, split_dim, tp, rank)
            _copy_shard_to_param(handle, name, shard)
```

Shrinks from ~39 lines to ~12. Zero behavior change.

### `_commit_linear` rewrite

```python
def _commit_linear(self, name, linear, split_side=None, model_dtype=None):
    self._ensure_handles()
    w = linear.tensors.get('weight')
    if w is None:
        return

    # --- GPU-invariant preparation -------------------------------------
    cpp_dtype = _infer_cpp_linear_dtype(linear)
    fmt = linear.weight_format
    block_in = (fmt.block_in or 0) if fmt is not None else 0

    tp = self._tp if split_side else 1
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    in_dim, out_dim = w.shape[0], w.shape[-1]
    if split_side == SplitSide.OUTPUT:
        out_dim //= tp
    elif split_side == SplitSide.INPUT:
        in_dim //= tp

    compute_dtype = (model_dtype if model_dtype is not None
                     else _infer_compute_dtype(linear))
    lin_cfg = _tm.LinearConfig()
    lin_cfg.input_dim = in_dim
    lin_cfg.output_dim = out_dim
    lin_cfg.data_type = compute_dtype or _tm.DataType.TYPE_INVALID
    lin_cfg.has_bias = 'bias' in linear.tensors

    packer = fmt.packer if fmt else None
    if packer is not None:
        tensors = {k: packer(t, k) for k, t in linear.tensors.items()}
    else:
        tensors = linear.tensors
    is_quantized = fmt is not None and fmt.name != 'trivial'

    # Uniform TP-split validation: every kind split along some axis must
    # have that axis evenly divisible by tp.  Covers weight, scales,
    # zeros, bias; covers both INPUT and OUTPUT split_side; respects the
    # bias-on-INPUT no-split rule.  Runs after the packer hoist so it
    # sees the tensors that will actually be split.
    if tp > 1 and split_dim is not None:
        for kind, tensor in tensors.items():
            kind_split_dim = split_dim
            if kind == 'bias' and split_side == SplitSide.INPUT:
                kind_split_dim = None
            if kind_split_dim is not None:
                d = tensor.shape[kind_split_dim]
                assert d % tp == 0, (
                    f"TP split: {name}.{kind} dim {kind_split_dim} has "
                    f"size {d}, not divisible by tp={tp}.")

    # --- Per-GPU commit ------------------------------------------------
    for i, handle in enumerate(self._handles):
        with self._contexts[i]:
            rank = self._rank_for(i) if tp > 1 else 0

            linear_mod = handle.child(name) or handle.create_child(name, lin_cfg)
            linear_mod.set_weight_spec(cpp_dtype, block_in)

            for kind, tensor in tensors.items():
                kind_split_dim = split_dim
                if kind == 'bias' and split_side == SplitSide.INPUT:
                    kind_split_dim = None
                shard = _shard(tensor, kind_split_dim, tp, rank)

                if kind == 'weight' and is_quantized:
                    alloc_shape, alloc_dtype = [in_dim, out_dim], cpp_dtype
                elif kind == 'weight' and model_dtype is not None:
                    alloc_shape, alloc_dtype = None, model_dtype
                else:
                    alloc_shape, alloc_dtype = None, None

                _copy_shard_to_param(linear_mod, kind, shard,
                                     alloc_shape=alloc_shape,
                                     alloc_dtype=alloc_dtype)
```

Shrinks from ~105 lines to ~50. Changes relative to today:

1. `in_dim` / `out_dim` computed once (not twice).
2. Block-scale TP validation → uniform per-kind check covering every kind and both split sides, run once above the loop.
3. Packer applied once per kind (not per GPU × kind).
4. `LinearConfig` built once, reused on each GPU's "child doesn't exist" branch.
5. `compute_dtype` / `_infer_compute_dtype` run once.
6. `max(1, 128)` default removed. Safe because `build_linear` now resolves `fmt.block_in` from Spec at conversion time, so every quantized format arrives at `_commit_linear` with a live `block_in`. Trivial formats still pass `0`, which is ignored on the C++ side.
7. No `group_size` local. The concept is `fmt.block_in`; reading it directly eliminates the redundant thread.
8. `_infer_cpp_linear_dtype` returns a scalar (no tuple).
9. `handle.child(name) or handle.create_child(name, lin_cfg)` replaces the explicit if-None block.
10. Module-level `_commit_tensors` function is deleted; its loop body is inlined. Resolves the `_commit_tensor` / `_commit_tensors` naming collision and drops its dead `group_size` parameter.
11. No padding logic in the commit layer.
12. `weight`-absent guard promoted to top-of-function (early return). Previously deferred to first-call-only; all live callers always pass `weight`, so this is a no-op in practice.
13. Deferred `data_format` attach deleted. The old block wrote into `linear.data_format`, which the live tree never reads. `Linear.data_format` remains as a field (still written by `build_linear` and fusion helpers) — fully purging it is out of scope.

### `_add_norm_child` rewrite

```python
def _add_norm_child(self, name, tensor, data_type=None, *, norm_eps):
    self._ensure_handles()
    from .norm import make_norm_config
    if data_type is None:
        data_type = _tm.DataType.TYPE_FP32
    norm_cfg = make_norm_config(dim=tensor.shape[-1],
                                data_type=data_type,
                                norm_eps=norm_eps)
    for i, handle in enumerate(self._handles):
        with self._contexts[i]:
            child = handle.create_child(name, norm_cfg)
            _copy_shard_to_param(child, 'weight', tensor)
```

Shrinks from ~33 lines to ~12. `norm_cfg` is already GPU-invariant in today's code; the only change is replacing the inline shard-move-alloc-cast-copy with `_copy_shard_to_param`. Zero behavior change.

### `load_context.py` deletion

`lmdeploy/turbomind/deploy/load_context.py` is removed entirely. Grep of the live tree confirms no Python code imports from it; only historical design docs reference it.

## Behavior changes summary

Four observable changes relative to today:

1. **`byte_size != nbytes` now asserts** inside `_copy_shard_to_param` instead of silently zero-padding. If any existing model relied on silent padding, the fix is to pad upstream in the transform pipeline rather than re-introduce padding here.

2. **TP-divisibility is now validated for every kind, both split sides.** Today's check fires only on `SplitSide.OUTPUT` + `block_out` + `{scales, zeros}`. The new check fires on weight, scales, zeros, and bias for both `INPUT` and `OUTPUT`. Mismatches that previously failed later at `tensor.split` with a poor message now fail here with a descriptive assertion.

3. **Compressed-tensors models with `group_size=32` now work correctly.** Previously silently clobbered to 128 by the removed default; now `fmt.block_in` carries the real value through to `set_weight_spec`.

4. **Trivial formats pass `0` to `set_weight_spec`.** Today they get silently bumped to `128`. C++ `MakeLinearWeightFormat` short-circuits on `IsTrivialFloatType` before consulting the value, so this is a no-op at the C++ boundary, but it removes a lie from the Python side.

## Files changed

| File | Change |
| --- | --- |
| `lmdeploy/turbomind/deploy/builder/_base.py` | Add `_copy_shard_to_param`, `_shard`. Simplify `_infer_cpp_linear_dtype` (scalar return). Rewrite `_commit_tensor`, `_commit_linear`, `_add_norm_child`. Delete module-level `_commit_tensors`. Remove `group_size` local, default, and dead deferred `data_format` attach. |
| `lmdeploy/turbomind/deploy/load_context.py` | Delete file. |
| `lmdeploy/turbomind/deploy/kind_map.py` | `build_linear` accepts symmetric `block_in` / `block_out` parameters; clones format via `dataclasses.replace` when sentinels need resolving; drops hardcoded `to_data_format(0, 0)` call. |
| `lmdeploy/turbomind/deploy/spec.py` | `Spec.build_linear` (line 151) passes `self._group_size` as both `block_in` and `block_out`. |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Two `build_linear` call sites updated to pass `self._group_size`. |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | One `build_linear` call site updated to pass `self._group_size`. |

No changes to `linear.py`, `norm.py`, any builder file (`attention.py`, `ffn.py`, `moe.py`, `mla.py`, `deltanet.py`, `builder/linear.py`), or any C++ source.

## Commit plan

Four independent commits, each verifiable standalone:

1. **`refactor(deploy): delete dead load_context.py`** — pure Python file removal; no C++ rebuild needed. Verify with one smoke test on any trivial model at `tp=1` to confirm nothing silently imported it at runtime.

2. **`refactor(kind_map): thread block_in / block_out through build_linear; drop hardcoded data_format call`** — add `block_in` and `block_out` parameters to `build_linear`; clone via `dataclasses.replace` when sentinels resolve; update `spec.py` and the two source_model specs to pass `self._group_size` through. The `_commit_linear` code intentionally still carries the `if group_size == 0: group_size = 128` default — this makes the commit a no-op for AWQ/GPTQ/FP8 (where `fmt.block_in` resolves to exactly the same 128 that the default would have supplied) while delivering the compressed-tensors `group_size=32` fix on its own. Verify with AWQ (regression guard) and a compressed-tensors `group_size=32` model if locally available (the previously-broken case should now produce coherent output).

3. **`refactor(builder): _copy_shard_to_param / _shard helpers; simplify _commit_tensor and _add_norm_child`** — introduce the two primitives, rewrite `_commit_tensor` and `_add_norm_child` around them. No changes yet to `_commit_linear`. Verify on a trivial dense model at `tp=1` and `tp=2`.

4. **`refactor(builder): hoist invariants in _commit_linear, uniform TP validation, drop group_size default and dead attaches, delete _commit_tensors helper`** — the main change, enabled by commit 2 having resolved `fmt.block_in` upstream. Verify across the full model matrix (see below).

## Verification plan

Per `AGENTS.md`:

- Test with `scripts/test_turbomind_model.py`.
- Require at least 128 tokens of coherent, prompt-relevant output per run. Gibberish indicates a silent commit-path bug.
- Check `get_gpu_usage` for empty GPUs before each run.
- Select models via `list_models` / `get_model_cache_path` from the model-server MCP.
- No pip install; no `setup.py`.

Commit-4 model matrix:

| Format axis | Purpose | TP |
| --- | --- | --- |
| Trivial dense | control; exercises the simple path end-to-end | 1, 2 |
| AWQ or GPTQ | exercises `block_in==0` resolution via `build_linear`, quantized `alloc_shape`/`alloc_dtype` relabel, uniform TP check on scales/zeros | 1, 2 |
| FP8 | exercises static `block_in=128` / `block_out=128` unchanged path, block-scale TP validation | 1, 2 |
| Compressed-tensors with `group_size=32`, if locally available | verifies the latent bug is fixed — `fmt.block_in` now carries 32 through to `set_weight_spec` instead of being silently clobbered to 128 | 1 |
| MLA (DeepSeek-style), if locally available | exercises `mla.py`'s loop of `_commit_linear` calls through fold+pad pipeline | 2 |
| DeltaNet, if locally available | exercises `deltanet.py`'s `in_proj_all` split plus `A_log` / `dt_bias` / `conv1d` via `_commit_tensor` | 1 |

Pick the smallest per category via `list_models` at implementation time to keep the test loop fast.

Assertion regressions: if the new `byte_size == nbytes` assert or the new uniform TP-divisibility assert fires on a previously-working model, the fix is upstream in the transform pipeline, not a loosening of the commit layer. Such a case should be investigated, upstream-fixed if trivial, or split into a separate follow-up PR if not.
