# add_input_projections Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `fuse_gdn_in_proj` (80-line monolith) into a pipeline of named standalone functions, mirroring the `add_qkv_proj` pattern.

**Architecture:** Split `in_proj_qkv` into separate Q, K, V linears early via `split_qkv`, reuse `_ensure_compatible_formats` for format compat, create `fuse_gdn` for TP interleaving + concat. Delete the `fuse_gdn_in_proj` monolith and the defensive `qkv_split=None` fallback.

**Tech Stack:** Python, PyTorch.

---

### Task 1: Remove defensive `qkv_split=None` from spec

The spec's `_linear_qkv_split` has an `else: None` fallback that is never reached in practice (linear attention layers always have these config fields). Remove it — let the code crash if fields are missing. Also update the type annotation in the base class.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:97-108`
- Modify: `lmdeploy/turbomind/deploy/spec.py:24`

- [ ] **Step 1: Remove the `else: None` branch in `qwen3_5_spec.py`**

Change lines 97-108 from:
```python
        # QKV dimensions for GDN layers: Q/K share key heads, V uses value heads
        ln_key_heads = model_cfg.get("linear_num_key_heads", 0)
        ln_val_heads = model_cfg.get("linear_num_value_heads", 0)
        ln_key_dim = model_cfg.get("linear_key_head_dim", 0)
        ln_val_dim = model_cfg.get("linear_value_head_dim", 0)
        if ln_key_heads and ln_val_heads:
            q_dim = ln_key_heads * ln_key_dim
            k_dim = ln_key_heads * ln_key_dim
            v_dim = ln_val_heads * ln_val_dim
            self._linear_qkv_split = (q_dim, k_dim, v_dim)
        else:
            self._linear_qkv_split = None
```
to:
```python
        # QKV dimensions for GDN layers: Q/K share key heads, V uses value heads
        ln_key_heads = model_cfg["linear_num_key_heads"]
        ln_val_heads = model_cfg["linear_num_value_heads"]
        ln_key_dim = model_cfg["linear_key_head_dim"]
        ln_val_dim = model_cfg["linear_value_head_dim"]
        q_dim = ln_key_heads * ln_key_dim
        k_dim = ln_key_heads * ln_key_dim
        v_dim = ln_val_heads * ln_val_dim
        self._linear_qkv_split = (q_dim, k_dim, v_dim)
```

- [ ] **Step 2: No change to `spec.py`**

The base class `_linear_qkv_split: tuple[int, int, int] | None = None` stays as-is. The `| None = None` default is correct because the base class defines it for all specs, and only Qwen3.5 sets it. Specs without linear attention (qwen3, gpt_oss) never read it.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(spec): remove defensive qkv_split=None fallback"
```

---

### Task 2: Rewrite `add_input_projections` pipeline

Create `split_qkv` and `fuse_gdn` functions. Rewrite `add_input_projections` to use the new pipeline. Delete `fuse_gdn_in_proj` and `_GDN_IN_PROJ_KEYS`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/deltanet.py`

- [ ] **Step 1: Add `split_qkv` function (before the `DeltaNetBuilder` class)**

Add this function after the `_tp_interleave_tensor` function (after line 39):

```python
def split_qkv(linear: Linear,
              qkv_split: tuple[int, int, int]) -> tuple[Linear, Linear, Linear]:
    """Split combined QKV linear into Q, K, V linears along output dim."""
    q_dim, k_dim, v_dim = qkv_split
    new_linears = []
    offset = 0
    for dim in qkv_split:
        tensors = {}
        for kind, t in linear.tensors.items():
            out_dim = t.dim() - 1
            tensors[kind] = t.narrow(out_dim, offset, dim).contiguous()
        new_linears.append(Linear(tensors=tensors,
                                  weight_format=linear.weight_format,
                                  data_format=linear.data_format))
        offset += dim
    return tuple(new_linears)
```

- [ ] **Step 2: Add `fuse_gdn` function (after `split_qkv`, before the `DeltaNetBuilder` class)**

```python
def fuse_gdn(q: Linear, k: Linear, v: Linear,
             z: Linear, b: Linear, a: Linear, *,
             tp: int) -> Linear:
    """Fuse GDN input projections with TP interleaving.

    Layout per tp-shard: [Q | K | V | Z | B | A].
    For tp=1 reduces to simple concat along output dim.
    """
    components = [q, k, v, z, b, a]

    if tp <= 1:
        return Linear.concat_out_dim(components)

    first = components[0]
    fused_tensors: dict[str, torch.Tensor] = {}
    for kind in first.tensors:
        parts = []
        all_1d = True
        d = -1
        for lin in components:
            t = lin.tensors.get(kind)
            if t is None:
                continue
            if t.dim() > 1:
                all_1d = False
                d = t.dim() - 1
                parts.append(_tp_interleave_tensor(t, tp, d))
            else:
                # 1-D tensors (bias): simple concat
                parts.append(t)
        if not parts:
            continue
        if all_1d:
            fused_tensors[kind] = torch.cat(parts, dim=0)
        else:
            fused = torch.cat(parts, dim=d + 1)
            shape = list(fused.shape)
            final = shape[:d] + [shape[d] * shape[d + 1]] + shape[d + 2:]
            fused_tensors[kind] = fused.reshape(final)

    return Linear(tensors=fused_tensors, weight_format=first.weight_format,
                  data_format=first.data_format)
```

- [ ] **Step 3: Delete `_GDN_IN_PROJ_KEYS` constant**

Delete lines 32-33:
```python
_GDN_IN_PROJ_KEYS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
```

- [ ] **Step 4: Delete `fuse_gdn_in_proj` function**

Delete the entire `fuse_gdn_in_proj` function (lines 42-146).

- [ ] **Step 5: Rewrite `add_input_projections` method**

Add the `_ensure_compatible_formats` import at the top of the file. Change the existing imports from:
```python
from ._base import Builder, SplitSide
```
to:
```python
from ._base import Builder, SplitSide, _ensure_compatible_formats
```

Replace the `add_input_projections` method (lines 157-185) with:
```python
    def add_input_projections(self, *, in_proj_qkv=None, in_proj_z=None,
                              in_proj_b=None, in_proj_a=None, out_proj=None,
                              qkv_split=None):
        """Fuse GDN input projections via pipeline, commit all linears.

        Pipeline: split_qkv -> ensure_compatible_formats -> fuse_gdn -> commit.
        """
        q, k, v = split_qkv(in_proj_qkv, qkv_split)
        group = _ensure_compatible_formats(
            {"q": q, "k": k, "v": v, "z": in_proj_z, "b": in_proj_b, "a": in_proj_a})
        fused = fuse_gdn(group["q"], group["k"], group["v"],
                         group["z"], group["b"], group["a"],
                         tp=self._tp)
        self._commit_linear("in_proj_all", fused, SplitSide.OUTPUT,
                            model_dtype=self.config.data_type)
        if out_proj is not None:
            self._commit_linear("out_proj", out_proj, SplitSide.INPUT,
                                model_dtype=self.config.data_type)
```

- [ ] **Step 6: Verify no remaining references to deleted code**

Run:
```bash
cd /data/lmdeploy-modeling && grep -rn "fuse_gdn_in_proj\|_GDN_IN_PROJ_KEYS" lmdeploy/
```
Expected: no output (all references deleted).

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/deltanet.py
git commit -m "refactor(builder): decompose fuse_gdn_in_proj into split_qkv + fuse_gdn pipeline"
```

---

### Task 3: Verify with model test

- [ ] **Step 1: Check GPU availability**

Run: check `get_gpu_usage` MCP tool for empty GPUs.

- [ ] **Step 2: Test Qwen3.5-27B TP=1**

Run:
```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3.5-27B --tp 1
```
Expected: Model responds with meaningful text (at least 128 tokens).

- [ ] **Step 3: Test Qwen3.5-27B TP=2**

Run:
```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3.5-27B --tp 2
```
Expected: Model responds with meaningful text (at least 128 tokens).
