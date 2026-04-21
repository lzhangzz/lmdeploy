# Commit-path simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deduplicate and simplify `Builder._commit_linear`, `Builder._commit_tensor`, and `Builder._add_norm_child` in `lmdeploy/turbomind/deploy/builder/_base.py`; thread `block_in` / `block_out` through `build_linear` so `WeightFormat` is authoritative; delete the fossilized `load_context.py`.

**Architecture:** Introduce one shard-copy primitive (`_copy_shard_to_param`) and one TP-split helper (`_shard`) that replace the four copies of the shard/alloc/cast/copy dance. Hoist all GPU-invariant work in `_commit_linear` above the per-GPU loop. Plumb runtime quantization block sizes from `Spec._group_size` into the `WeightFormat` carried by each `Linear` via `dataclasses.replace`, so the downstream `_commit_linear` never needs a hardcoded default. Broaden the TP-split validation to cover every tensor kind and both split sides.

**Tech Stack:** Python 3.10+, `dataclasses`, `torch`, `_turbomind` pybind module, `scripts/test_turbomind_model.py` for end-to-end verification.

**Spec:** `docs/superpowers/specs/2026-04-21-commit-simplification-design.md`

---

## Preliminaries

This refactor is behavior-preserving except for four documented tightenings (see spec "Behavior changes summary"). There are no Python unit tests covering the commit path today; verification is end-to-end: load a model with TurboMind, run inference, confirm the response is coherent human language of at least 128 tokens. Per `AGENTS.md`:

- Before any run: call the `get_gpu_usage` MCP tool and pick empty GPU(s).
- Discover test models: call the model-server MCP's `list_models` tool and pick the smallest model per format family.
- For each chosen model, call `get_model_cache_path` to get the cache dir.
- Test command template (from repo root): `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py <model_path> <cache_dir> <tp> <gpus>`.
- A good run prints a `--- response begin ---` / `--- response end ---` block containing coherent prose answering "Write a short paragraph about the importance of reading books." and reports `generated: N` where N >= 100 (hitting the 128-token cap or near it). Gibberish or truncation at a few tokens indicates a silent bug.
- Record which model paths you used so the same models can be rerun across tasks.

File structure — all edits, no new files:

| File | Responsibility after the refactor |
| --- | --- |
| `lmdeploy/turbomind/deploy/builder/_base.py` | Builder base class + shard-copy primitive + `_commit_linear` / `_commit_tensor` / `_add_norm_child` |
| `lmdeploy/turbomind/deploy/kind_map.py` | `WeightFormat` registry; `build_linear` resolves runtime block sizes into the format |
| `lmdeploy/turbomind/deploy/spec.py` | `TextModelSpec._linear` forwards `self._group_size` into `build_linear` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Direct `build_linear` call sites pass `self._group_size` |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Direct `build_linear` call site passes `self._group_size` |
| `lmdeploy/turbomind/deploy/load_context.py` | DELETED |

---

### Task 1: Delete dead `load_context.py`

**Files:**
- Delete: `lmdeploy/turbomind/deploy/load_context.py`

- [ ] **Step 1.1: Verify no live imports reference the file**

Run: `cd /data/lmdeploy-modeling && rg "from .*load_context|import load_context" -g '!*.md' -g '!*.txt' .`

Expected output: no matches. If any match is found (in a `.py` file), STOP and report — the file is not actually dead.

- [ ] **Step 1.2: Delete the file**

Run: `cd /data/lmdeploy-modeling && git rm lmdeploy/turbomind/deploy/load_context.py`

- [ ] **Step 1.3: Smoke-test a trivial dense model at tp=1**

First check an empty GPU via the `get_gpu_usage` MCP tool; pick one free GPU id, call it `$GPU`. Use the model-server MCP to find the smallest trivial dense model and get its `cache_dir` (e.g. a small Qwen or Llama dense checkpoint). Call the path `$MODEL_PATH` and the cache dir `$CACHE_DIR`.

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $MODEL_PATH $CACHE_DIR 1 $GPU`

Expected: exit code 0; `--- response begin ---` block contains coherent prose; `generated:` count at or near 128.

- [ ] **Step 1.4: Commit**

```bash
cd /data/lmdeploy-modeling
git commit -m "refactor(deploy): delete dead load_context.py

The file was a fossilized copy of the commit stack that lived in
builder/_base.py. No live Python code imports from it; only historical
design docs referenced it."
```

---

### Task 2: Thread `block_in` / `block_out` through `build_linear`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/kind_map.py` (import, `build_linear` at line 536)
- Modify: `lmdeploy/turbomind/deploy/spec.py` (`_linear` at line 149-151)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (lines 299-302)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (line 228)

- [ ] **Step 2.1: Add `replace` to the dataclasses import in `kind_map.py`**

At the top of `lmdeploy/turbomind/deploy/kind_map.py`, change:

```python
from dataclasses import dataclass
```

to:

```python
from dataclasses import dataclass, replace
```

- [ ] **Step 2.2: Update `build_linear` signature and body**

In `lmdeploy/turbomind/deploy/kind_map.py`, replace the `build_linear` function (starting at line 536) with:

```python
def build_linear(
    params: dict[str, torch.Tensor],
    prefix: str,
    *,
    index: int | None = None,
    block_in: int = 0,
    block_out: int = 0,
) -> Linear | None:
    """Build a ``Linear`` bundle from checkpoint tensors at *prefix*.

    Probes every known checkpoint suffix (union of all format suffix maps),
    classifies the format by running each ``WeightFormat.accepts`` predicate
    in ``FORMAT_PRIORITY`` order, then normalises the collected tensors with
    the winning format's normalizer.

    When *index* is given, each collected tensor is sliced by ``[index]``
    before classification and normalisation (used for packed expert tensors
    where the expert dimension is the leading axis).

    ``block_in`` and ``block_out`` resolve the format's quantization block
    sizes at conversion time.  A format with ``block_in == 0`` (AWQ, GPTQ,
    compressed-tensors) declares "use the runtime group_size".  The caller
    passes that value; we clone the format with ``dataclasses.replace`` so
    the returned ``Linear`` carries an authoritative ``WeightFormat``.  The
    same mechanism applies to ``block_out == 0``, reserved for future
    formats.  Passing ``0`` means "no runtime value available" and leaves
    the format's declared sentinel in place.

    The returned ``Linear`` is in TM layout ``[in, out]`` and carries the
    detected ``WeightFormat`` for downstream use in ``_commit_linear``.
    Returns ``None`` if no tensors are found at *prefix*.
    """
    from .linear import Linear

    available: dict[str, torch.Tensor] = {
        s: params[prefix + s] for s in ALL_SUFFIXES if (prefix + s) in params
    }
    if index is not None:
        available = {s: t[index] for s, t in available.items()}

    fmt = next((f for f in FORMAT_PRIORITY if f.accepts(available)), None)
    if fmt is None:
        return None

    replacements: dict[str, int] = {}
    if fmt.block_in == 0 and block_in > 0:
        replacements['block_in'] = block_in
    if fmt.block_out == 0 and block_out > 0:
        replacements['block_out'] = block_out
    if replacements:
        fmt = replace(fmt, **replacements)

    tensors: dict[str, torch.Tensor] = {
        kind: fmt.normalizer(available[s], kind)
        for s, kind in fmt.suffix_map.items()
        if s in available
    }
    if not tensors:
        return None

    fmt.complete_tensors(tensors)
    return Linear(tensors=tensors, weight_format=fmt, data_format=None)
```

Key changes from today: added `block_in` / `block_out` kw-only parameters with default 0; added the `replace` block that clones the format when a runtime value resolves a sentinel; dropped the dead `fmt.to_data_format(0, group_size=0)` call (the result was never read — see spec).

- [ ] **Step 2.3: Update `TextModelSpec._linear` in `spec.py`**

In `lmdeploy/turbomind/deploy/spec.py`, replace the `_linear` method (lines 149-151):

```python
    def _linear(self, pfx: str):
        from .kind_map import build_linear
        return build_linear(self.params, pfx,
                            block_in=self._group_size,
                            block_out=self._group_size)
```

The symmetric `block_out=self._group_size` is forward-looking: no current format has `block_out == 0`, so it's a no-op today, but future formats that do will pick it up without changing call sites.

- [ ] **Step 2.4: Update direct `build_linear` calls in `qwen3_5_spec.py`**

In `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`, replace the two direct calls (lines 299-302):

```python
        gate_up_lin = build_linear(self.params, f'{pfx}.gate_up_proj',
                                   index=expert_idx,
                                   block_in=self._group_size,
                                   block_out=self._group_size)
        down_lin = build_linear(self.params, f'{pfx}.down_proj',
                                index=expert_idx,
                                block_in=self._group_size,
                                block_out=self._group_size)
```

- [ ] **Step 2.5: Update the direct `build_linear` call in `gpt_oss_spec.py`**

In `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`, replace the call at line 228:

```python
        lin = build_linear(self.params, prefix, index=expert,
                           block_in=self._group_size,
                           block_out=self._group_size)
```

- [ ] **Step 2.6: Regression-verify with an AWQ model at tp=1**

Pick a small AWQ model via the `list_models` MCP (group_size will be 128 for AWQ per `_SUPPORTED_GROUP_SIZES`). Get its `cache_dir` via `get_model_cache_path`. Check an empty GPU via `get_gpu_usage`.

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $AWQ_MODEL_PATH $AWQ_CACHE 1 $GPU`

Expected: coherent response of ~128 tokens. This verifies that AWQ (where `fmt.block_in` is now resolved to 128 at `build_linear` time instead of by the downstream default) behaves identically.

- [ ] **Step 2.7: If available, verify the compressed-tensors `group_size=32` fix**

Use `list_models` to check if any compressed-tensors model with `group_size=32` is cached locally. If yes, run the smoke test against it at tp=1. Expected: coherent response — this path was producing garbage today because the `max(1, 128)` default silently clobbered the real group_size. If no such model is locally available, record "compressed-tensors g=32 coverage: none locally available" and move on.

- [ ] **Step 2.8: Commit**

```bash
cd /data/lmdeploy-modeling
git add lmdeploy/turbomind/deploy/kind_map.py \
        lmdeploy/turbomind/deploy/spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(kind_map): thread block_in / block_out through build_linear

build_linear now accepts block_in / block_out kw-only parameters and
clones the detected WeightFormat via dataclasses.replace when a format
sentinel (block_in=0 or block_out=0) is resolved from the runtime
quant_config. TextModelSpec._linear and the two direct build_linear
call sites in qwen3_5_spec and gpt_oss_spec forward self._group_size.

Fixes a latent bug where compressed-tensors models with group_size=32
were silently run with the AWQ/GPTQ default of 128. The dead
to_data_format(0, group_size=0) call in build_linear is removed — the
result was never read on the consumer side."
```

---

### Task 3: Add `_copy_shard_to_param` + `_shard` helpers; rewrite `_commit_tensor` and `_add_norm_child`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py` (add helpers; rewrite `_commit_tensor` at lines 494-532 and `_add_norm_child` at lines 534-566)

- [ ] **Step 3.1: Add `_copy_shard_to_param` helper in `_base.py`**

Add this function in `lmdeploy/turbomind/deploy/builder/_base.py` immediately before the `_commit_tensors` module-level function (around line 210, between the "Core tensor commit" section header and the existing `_commit_tensors` definition):

```python
def _copy_shard_to_param(handle, param_name: str, shard: torch.Tensor, *,
                         alloc_shape: list[int] | None = None,
                         alloc_dtype=None) -> None:
    """Move shard to GPU, allocate the C++ param slot, cast, and copy.

    Invariant: ``dst.byte_size == shard.nbytes`` after the cast.  Upstream
    is responsible for any padding/reshape needed to satisfy this.  A
    mismatch raises immediately.

    ``alloc_shape`` / ``alloc_dtype`` default to the shard's own shape /
    dtype.  Override only to express shape/dtype *relabels* where byte
    size is preserved (e.g. quantized weight: physical int32
    [in, out/8] stored in a logical UINT4 [in, out] C++ slot).
    """
    if not shard.is_cuda:
        shard = shard.cuda(0).contiguous()
    elif not shard.is_contiguous():
        shard = shard.contiguous()

    if alloc_shape is None:
        alloc_shape = list(shard.shape)
    if alloc_dtype is None:
        alloc_dtype = _torch_dtype_to_cpp(shard.dtype)

    dst = handle.param(param_name).alloc(alloc_shape, alloc_dtype)
    shard = _cast_shard_for_tm(shard, dst)
    assert dst.byte_size == shard.nbytes, (
        f"{param_name}: alloc byte_size={dst.byte_size} != "
        f"shard.nbytes={shard.nbytes}")
    dst.copy_from(shard)
```

- [ ] **Step 3.2: Add `_shard` helper in `_base.py`**

Add this function immediately after `_copy_shard_to_param`:

```python
def _shard(tensor: torch.Tensor, split_dim: int | None, tp: int,
           rank: int) -> torch.Tensor:
    """Return the ``rank``-th split along ``split_dim``, or the tensor unchanged.

    Used wherever a TP shard is selected from a broadcast-by-default
    tensor.  A ``split_dim`` of ``None`` or ``tp <= 1`` returns the tensor
    untouched.
    """
    if split_dim is None or tp <= 1:
        return tensor
    return tensor.split(tensor.shape[split_dim] // tp, dim=split_dim)[rank]
```

- [ ] **Step 3.3: Rewrite `_commit_tensor`**

In `lmdeploy/turbomind/deploy/builder/_base.py`, replace the `_commit_tensor` method (lines 494-532) with:

```python
    def _commit_tensor(self, name: str, tensor: torch.Tensor | None,
                       split_side: SplitSide | None = None):
        """Commit a raw tensor to a named parameter on all GPUs.

        Parameters
        ----------
        name : str
            Parameter name within the module.
        tensor : torch.Tensor | None
            The tensor data.  ``None`` is a no-op.
        split_side : SplitSide | None
            TP split semantics.  ``None`` means broadcast.
        """
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

- [ ] **Step 3.4: Rewrite `_add_norm_child`**

In `lmdeploy/turbomind/deploy/builder/_base.py`, replace the `_add_norm_child` method (lines 534-566) with:

```python
    def _add_norm_child(self, name: str, tensor: torch.Tensor,
                        data_type=None, *, norm_eps):
        """Create a NormConfig child and commit weight tensor.

        Parameters
        ----------
        name : str
            Child module name (e.g. ``"attention_norm"``).
        tensor : torch.Tensor
            The norm weight tensor.
        data_type : C++ DataType value | None
            Compute dtype for the norm.  Defaults to FP32 if not set.
        norm_eps : float
            RMS norm epsilon.  Required.
        """
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

- [ ] **Step 3.5: Smoke-test trivial dense at tp=1**

Using the same trivial model and cache as Task 1, pick an empty GPU via `get_gpu_usage`:

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $MODEL_PATH $CACHE_DIR 1 $GPU`

Expected: coherent response near 128 tokens. This exercises `_add_norm_child` (RMS norms), `_commit_tensor` (LinearBuilder embeddings / lm_head via `set_weight`), and `_commit_tensor` for any present scalar params.

- [ ] **Step 3.6: Smoke-test trivial dense at tp=2**

Via `get_gpu_usage`, pick two empty GPUs (call them `$GPU0,$GPU1`).

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $MODEL_PATH $CACHE_DIR 2 $GPU0,$GPU1`

Expected: coherent response near 128 tokens. This exercises the `_shard` helper in `_commit_tensor` (embeddings split along output dim) and the per-GPU loop in `_add_norm_child`.

- [ ] **Step 3.7: Commit**

```bash
cd /data/lmdeploy-modeling
git add lmdeploy/turbomind/deploy/builder/_base.py
git commit -m "refactor(builder): add _copy_shard_to_param / _shard helpers; simplify _commit_tensor and _add_norm_child

_copy_shard_to_param encapsulates the universal move-to-GPU, allocate,
cast, byte-size check, copy sequence. _shard encapsulates the TP shard
selection. _commit_tensor and _add_norm_child shrink to thin loops
around these helpers.

The new dst.byte_size == shard.nbytes assertion replaces today's
silent zero-padding behavior for the module-level _commit_tensors
function (still present but untouched in this commit)."
```

---

### Task 4: Rewrite `_commit_linear`; delete module-level `_commit_tensors`; simplify `_infer_cpp_linear_dtype`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py`
  - Simplify `_infer_cpp_linear_dtype` at lines 92-104.
  - Delete module-level `_commit_tensors` at lines 214-282.
  - Rewrite `_commit_linear` at lines 387-492.

- [ ] **Step 4.1: Simplify `_infer_cpp_linear_dtype` to a scalar return**

In `lmdeploy/turbomind/deploy/builder/_base.py`, replace the `_infer_cpp_linear_dtype` function (lines 92-104) with:

```python
def _infer_cpp_linear_dtype(linear: Linear):
    """Determine the C++ ``DataType`` for a ``Linear`` bundle.

    Returns the ``_tm.DataType`` value corresponding to the declared
    ``weight_format.cpp_dtype_name`` when set, else the C++ equivalent of
    the weight tensor's torch dtype, else ``None``.  The quantization
    block size is no longer returned here — callers read it directly from
    ``linear.weight_format.block_in``.
    """
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

- [ ] **Step 4.2: Delete the module-level `_commit_tensors` function**

In `lmdeploy/turbomind/deploy/builder/_base.py`, delete the entire `_commit_tensors` function spanning today's lines 210-283 (the function header block comment `# Core tensor commit (moved from commit.py)` plus `def _commit_tensors(...)` through its body). Its logic will be inlined into `_commit_linear` in the next step. Leave the `_copy_shard_to_param` and `_shard` helpers (added in Task 3) in place.

- [ ] **Step 4.3: Rewrite `_commit_linear`**

In `lmdeploy/turbomind/deploy/builder/_base.py`, replace the `_commit_linear` method (today at lines 387-492, after deletions the line numbers will shift — locate it by the method signature `def _commit_linear(self, name: str, linear: Linear,`):

```python
    def _commit_linear(self, name: str, linear: Linear,
                       split_side: SplitSide | None = None,
                       model_dtype=None):
        """Commit a ``Linear`` bundle to a named child on all GPUs.

        On first call for a given ``name`` the child ``LinearWeight`` is
        created via ``handle.create_child`` using a ``LinearConfig``
        derived from the linear's dimensions and compute dtype; on
        subsequent calls the existing child is reused.  Tensor data is
        then sharded per rank (for TP) and copied to the C++ slots via
        ``_copy_shard_to_param``.

        Parameters
        ----------
        name : str
            Child module name (e.g. ``"w_qkv"``).
        linear : Linear
            The linear bundle to commit.
        split_side : SplitSide | None
            TP split semantics.  ``None`` means broadcast (no split).
        model_dtype : C++ DataType value | None
            The model's configured compute dtype.  When set, trivial
            (non-quantized) weights use this dtype instead of the weight
            tensor's dtype, preventing mismatches when the checkpoint
            stores weights in a different precision than the model
            config (e.g. BF16 weights in an FP16 model).
        """
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

        # Uniform TP-split validation: every kind split along some axis
        # must have that axis evenly divisible by tp.  Covers weight,
        # scales, zeros, bias; covers both INPUT and OUTPUT split_side;
        # respects the bias-on-INPUT no-split rule.  Runs after the
        # packer hoist so it sees the tensors that will actually be
        # split.
        if tp > 1 and split_dim is not None:
            for kind, tensor in tensors.items():
                kind_split_dim = split_dim
                if kind == 'bias' and split_side == SplitSide.INPUT:
                    kind_split_dim = None
                if kind_split_dim is not None:
                    d = tensor.shape[kind_split_dim]
                    assert d % tp == 0, (
                        f"TP split: {name}.{kind} dim {kind_split_dim} "
                        f"has size {d}, not divisible by tp={tp}.")

        # --- Per-GPU commit ------------------------------------------------
        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0

                linear_mod = (handle.child(name)
                              or handle.create_child(name, lin_cfg))
                linear_mod.set_weight_spec(cpp_dtype, block_in)

                for kind, tensor in tensors.items():
                    kind_split_dim = split_dim
                    if kind == 'bias' and split_side == SplitSide.INPUT:
                        kind_split_dim = None
                    shard = _shard(tensor, kind_split_dim, tp, rank)

                    if kind == 'weight' and is_quantized:
                        alloc_shape, alloc_dtype = ([in_dim, out_dim],
                                                    cpp_dtype)
                    elif kind == 'weight' and model_dtype is not None:
                        alloc_shape, alloc_dtype = None, model_dtype
                    else:
                        alloc_shape, alloc_dtype = None, None

                    _copy_shard_to_param(linear_mod, kind, shard,
                                         alloc_shape=alloc_shape,
                                         alloc_dtype=alloc_dtype)
```

- [ ] **Step 4.4: Import check**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python -c "from lmdeploy.turbomind.deploy.builder._base import Builder, _copy_shard_to_param, _shard; print('OK')"`

Expected: `OK`. This catches import-level errors (typos, missing references) before launching a full model test.

- [ ] **Step 4.5: Verify trivial dense at tp=1 and tp=2**

Using the same trivial model from Task 3 steps 3.5 and 3.6, re-run at both tp settings via `get_gpu_usage` for empty GPUs.

tp=1: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $MODEL_PATH $CACHE_DIR 1 $GPU`

tp=2: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $MODEL_PATH $CACHE_DIR 2 $GPU0,$GPU1`

Expected: coherent response at ~128 tokens in both runs.

- [ ] **Step 4.6: Verify AWQ (or GPTQ) at tp=1 and tp=2**

Using the AWQ model from Task 2 step 2.6:

tp=1: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $AWQ_MODEL_PATH $AWQ_CACHE 1 $GPU`

tp=2: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $AWQ_MODEL_PATH $AWQ_CACHE 2 $GPU0,$GPU1`

Expected: coherent responses. Exercises the quantized `alloc_shape=[in_dim, out_dim]` / `alloc_dtype=UINT4` relabel path in the inlined loop, and the uniform TP check at tp=2.

- [ ] **Step 4.7: Verify FP8 at tp=1 and tp=2**

Use `list_models` to find a small FP8 model (group_size=128 statically). Get its cache via `get_model_cache_path`. Pick empty GPUs.

tp=1: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $FP8_MODEL_PATH $FP8_CACHE 1 $GPU`

tp=2: `cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $FP8_MODEL_PATH $FP8_CACHE 2 $GPU0,$GPU1`

Expected: coherent responses. Exercises the `block_out=128` static block-scale TP validation path.

- [ ] **Step 4.8: Verify MLA (DeepSeek-style) if locally available**

Use `list_models` to check for a DeepSeek-style MLA model. If one is cached, run at tp=2:

`cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $MLA_MODEL_PATH $MLA_CACHE 2 $GPU0,$GPU1`

Expected: coherent response. Exercises `mla.py`'s loop of `_commit_linear` calls through the fold+pad pipeline. If no MLA model is cached, record "MLA coverage: none locally available" and proceed.

- [ ] **Step 4.9: Verify DeltaNet if locally available**

Use `list_models` to check for a DeltaNet / Gated Delta Net model. If one is cached, run at tp=1:

`cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib python scripts/test_turbomind_model.py $DN_MODEL_PATH $DN_CACHE 1 $GPU`

Expected: coherent response. Exercises `deltanet.py`'s `in_proj_all` via `_commit_linear` plus `A_log` / `dt_bias` / `conv1d` via `_commit_tensor`. If no DeltaNet model is cached, record "DeltaNet coverage: none locally available" and proceed.

- [ ] **Step 4.10: Handle any new assertion failures**

If `_copy_shard_to_param`'s `dst.byte_size == shard.nbytes` assert or the uniform TP-divisibility assert fires on a previously-working model, the fix is upstream in the transform pipeline — NOT to loosen the commit-layer check. Investigate which kind/shape triggered it. If the upstream fix is trivial, make it as an additional step in this task. If non-trivial, revert step 4.3 back to the old `_commit_linear` on that branch and open a follow-up issue; do not hide the assert.

- [ ] **Step 4.11: Commit**

```bash
cd /data/lmdeploy-modeling
git add lmdeploy/turbomind/deploy/builder/_base.py
git commit -m "refactor(builder): hoist invariants in _commit_linear, uniform TP validation, drop group_size default

_commit_linear now computes GPU-invariant state once above the
per-GPU loop: cpp_dtype, block_in from fmt.block_in (authoritative
after the earlier build_linear refactor), in_dim/out_dim, compute_dtype,
LinearConfig, packer-applied tensors, and uniform TP-divisibility
validation across every kind and both split sides. The per-GPU loop
only resolves rank, get-or-creates the child, and iterates kinds
through _copy_shard_to_param.

The module-level _commit_tensors helper is deleted (its body is
inlined; its dead group_size parameter goes with it). The
_commit_tensor / _commit_tensors naming collision is resolved. The
max(1, 128) default for group_size is gone — block_in arrives
resolved from build_linear. The dead deferred data_format attach is
gone; data_format has no live readers. _infer_cpp_linear_dtype now
returns a scalar (callers read block_in directly from fmt).

No behavior change for any correct caller; the new assertions tighten
invariants that today's code satisfies implicitly."
```

---

## Self-Review

**1. Spec coverage:**

| Spec section / requirement | Implementing task / step |
| --- | --- |
| `_copy_shard_to_param` helper | 3.1 |
| `_shard` helper | 3.2 |
| `_infer_cpp_linear_dtype` scalar return | 4.1 |
| `_commit_tensor` rewrite | 3.3 |
| `_commit_linear` rewrite (hoist, uniform TP, inline, no default, no deferred attach, no padding, early return on missing weight, `handle.child() or ...`) | 4.3 (with 4.1 for dtype helper, 4.2 for deletion of `_commit_tensors`) |
| `_add_norm_child` rewrite | 3.4 |
| Delete `load_context.py` | 1.2 |
| `build_linear` accepts `block_in` / `block_out`; clones via `dataclasses.replace`; drops `to_data_format(0, 0)` | 2.1, 2.2 |
| `TextModelSpec._linear` forwards `self._group_size` | 2.3 |
| `qwen3_5_spec.py` direct calls forward | 2.4 |
| `gpt_oss_spec.py` direct call forwards | 2.5 |
| Behavior change: byte_size assert | 3.1 (helper), 4.3 (consumed by inlined loop) |
| Behavior change: uniform TP validation | 4.3 |
| Behavior change: compressed-tensors g=32 fix | 2.2 |
| Behavior change: trivial passes 0 to set_weight_spec | 4.3 |
| Verification matrix: trivial × {1, 2} | 3.5, 3.6, 4.5 |
| Verification matrix: AWQ × {1, 2} | 2.6, 4.6 |
| Verification matrix: FP8 × {1, 2} | 4.7 |
| Verification matrix: compressed-tensors(32) | 2.7 |
| Verification matrix: MLA | 4.8 |
| Verification matrix: DeltaNet | 4.9 |
| Assertion regression policy | 4.10 |

No gaps.

**2. Placeholder scan:**

- No "TBD" / "TODO" / "implement later".
- Test-model identities are discovered at implementation time via MCP (`list_models` + `get_model_cache_path`) rather than hardcoded. This is deliberate: the cached model catalog is dynamic. Each verification step specifies the category and the expected behavior, and the discovery commands are explicit.
- "If available locally" branches (2.7, 4.8, 4.9) instruct the engineer to record absence and proceed — not skip silently.

**3. Type consistency:**

- `_copy_shard_to_param(handle, param_name, shard, *, alloc_shape=None, alloc_dtype=None)` signature in 3.1 matches all three call sites: `_commit_tensor` (3.3, no overrides), `_add_norm_child` (3.4, no overrides), and `_commit_linear` (4.3, passes `alloc_shape`/`alloc_dtype` as kwargs).
- `_shard(tensor, split_dim, tp, rank)` signature in 3.2 matches call sites in 3.3 and 4.3.
- `_infer_cpp_linear_dtype(linear)` simplified to scalar in 4.1 matches the single caller in 4.3 (`cpp_dtype = _infer_cpp_linear_dtype(linear)`).
- `build_linear(params, prefix, *, index=None, block_in=0, block_out=0)` signature in 2.2 matches call sites in 2.3, 2.4, 2.5 (all use kw-only `block_in=` / `block_out=`).
