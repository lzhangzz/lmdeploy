# add_qkv_proj Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `merge_qkv_linear` monolith into a pipeline of standalone functions in `AttentionBuilder.add_qkv_proj`, and move RoPE permutation from builder to specs.

**Architecture:** Five new standalone functions (`dequant_mixed`, `pad_for_tp`, `split_output_gate`, `fuse_qkv`, `reorder_rotary_emb_linear`) replace the monolithic `merge_qkv_linear`. RoPE moves from builder to specs. Each function is added first, then wired up atomically, then old code is deleted.

**Tech Stack:** Python, PyTorch, _turbomind C++ bindings

**Design spec:** `docs/superpowers/specs/2026-04-16-add-qkv-proj-refactor-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `lmdeploy/turbomind/deploy/source_model/utils.py` | Modify | Add `reorder_rotary_emb_linear` |
| `lmdeploy/turbomind/deploy/builder/attention.py` | Modify | Add 4 pipeline functions, rewrite `add_qkv_proj`, delete old code |
| `lmdeploy/turbomind/deploy/builder/_base.py` | Modify | Delete `_block_ops_need_dequant` |
| `lmdeploy/turbomind/deploy/builder/__init__.py` | Modify | Update exports |
| `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Modify | Add RoPE before `add_qkv_proj` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Modify | Add RoPE before `add_qkv_proj` |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Modify | Add RoPE before `add_qkv_proj` |

---

### Task 1: Add `reorder_rotary_emb_linear` to `utils.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py:126-144`

This function applies RoPE permutation to all tensors in a `Linear` object. It is quantization-aware: if `block_out % head_dim != 0`, it dequantizes first. For scales/zeros when blocks align with heads, it shuffles at block granularity.

- [ ] **Step 1: Add the import and function**

Add to `lmdeploy/turbomind/deploy/source_model/utils.py` after the existing `reorder_rotary_emb` function (after line 144). First add the import for `Linear` and `TRIVIAL_FORMAT` at the top of the file (after the existing imports around line 9):

```python
from ..kind_map import TRIVIAL_FORMAT
```

Then add the function after `reorder_rotary_emb`:

```python
def _dequant_linear(linear):
    """Dequantize a quantized Linear to trivial when the format provides dequant."""
    from ..kind_map import TRIVIAL_FORMAT
    fmt = linear.weight_format
    if fmt is None or fmt.dequant is None:
        return linear
    new_tensors = fmt.dequant(linear.tensors)
    from ..linear import Linear
    return Linear(tensors=new_tensors, weight_format=TRIVIAL_FORMAT, data_format=None)


def reorder_rotary_emb_linear(linear, head_dim: int, rope_dim: int):
    """Apply RoPE permutation to all tensors in a Linear.

    Quantization-aware:
    - If quantized and block_out % head_dim != 0, dequantizes first
      (permuting within a head would cross block boundaries).
    - For weight/bias: element-level RoPE permutation.
    - For scales/zeros when block_out % head_dim == 0: block-level channel
      shuffling. Each head maps to (block_out / head_dim) complete blocks,
      so we apply the same interleave pattern at block granularity.
    - For scales/zeros when dequantized: skipped (trivial format has none).
    """
    from ..linear import Linear

    wfmt = linear.weight_format
    block_out = (wfmt.block_out or 0) if wfmt is not None else 0

    # If blocks don't align with heads, dequant first
    if block_out and block_out % head_dim != 0:
        linear = _dequant_linear(linear)
        block_out = 0

    new_tensors = {}
    for kind, tensor in linear.tensors.items():
        if kind in ("scales", "zeros") and block_out > 0:
            # Block-level shuffle: each head = (block_out / head_dim) blocks
            blocks_per_head = block_out // head_dim
            n_heads = tensor.size(-1) // blocks_per_head
            t = tensor.view(*tensor.shape[:-1], n_heads, blocks_per_head)
            t = reorder_rotary_emb(t, blocks_per_head, rope_dim * blocks_per_head // head_dim)
            new_tensors[kind] = t.reshape(tensor.shape)
        elif tensor.size(-1) % head_dim == 0:
            new_tensors[kind] = reorder_rotary_emb(tensor, head_dim, rope_dim)
        else:
            new_tensors[kind] = tensor

    return Linear(tensors=new_tensors, weight_format=linear.weight_format,
                  data_format=linear.data_format)
```

Note: The `_dequant_linear` here is a local duplicate to avoid circular imports (`utils.py` lives in `source_model/` and importing from `builder/_base.py` would create a circular dependency). The builder already has its own `_dequant_linear` in `_base.py`.

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/utils.py
git commit -m "feat(source_model): add reorder_rotary_emb_linear for Linear-level RoPE"
```

---

### Task 2: Add pipeline functions to `attention.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py`

Add the four new standalone functions (`dequant_mixed`, `pad_for_tp`, `split_output_gate`, `fuse_qkv`) before the `AttentionBuilder` class. These are not wired up yet — just adding them alongside existing code.

- [ ] **Step 1: Add `dequant_mixed`**

Insert after the `_ATTN_TP_RULES` dict (after line 31), before the old helper section. This replaces the `_ensure_compatible_formats` + `_block_ops_need_dequant` + inline dequant logic from `merge_qkv_linear`.

```python
# ---------------------------------------------------------------------------
# New pipeline functions (replacing merge_qkv_linear)
# ---------------------------------------------------------------------------


def dequant_mixed(q: Linear, k: Linear, v: Linear):
    """Dequantize to trivial if formats are mixed.

    Two cases:
    1. q, k, v have different weight formats -> dequant all to trivial
    2. Some are already trivial (e.g. from reorder_rotary_emb_linear)
       -> dequant the rest so all match for fusion
    """
    names = {lin.weight_format.name for lin in (q, k, v) if lin.weight_format}
    if len(names) <= 1:
        # All same format (or all None) -- check if any are trivial while others aren't
        trivial = {lin.weight_format.name == 'trivial' for lin in (q, k, v)
                   if lin.weight_format}
        if len(trivial) <= 1:
            return q, k, v
    return _dequant_linear(q), _dequant_linear(k), _dequant_linear(v)
```

- [ ] **Step 2: Add `pad_for_tp`**

Insert after `dequant_mixed`:

```python
def pad_for_tp(q: Linear, k: Linear, v: Linear, *,
               tp: int, head_dim: int,
               q_heads: int, kv_heads: int):
    """Make head counts tp-divisible.

    q: pad with zero heads to reach tp-divisible count.
    kv: repeat heads to reach tp-divisible count (preserves real data).
    Also handles quantization block alignment.
    """
    def _adjust_linear(linear, heads, is_kv: bool):
        """Adjust one linear's head count. Pad for q, repeat for kv."""
        wfmt = linear.weight_format
        block_out = (wfmt.block_out or 0) if wfmt is not None else 0

        if heads % tp == 0:
            return linear

        target_heads = ((heads + tp - 1) // tp) * tp
        new_tensors = {}

        for kind, tensor in linear.tensors.items():
            is_block_kind = kind in ("scales", "zeros") and block_out > 0

            if is_block_kind:
                # Block-scale: pad or repeat at block granularity
                blocks_per_head = block_out // head_dim
                head_blocks = tensor.size(-1) // blocks_per_head
                target_blocks = target_heads * blocks_per_head
                deficit = target_blocks - head_blocks
                if deficit > 0:
                    if is_kv:
                        # Repeat: each head's blocks get repeated
                        n_repeat = target_heads // heads
                        new_tensors[kind] = tensor.repeat_interleave(n_repeat, dim=-1)
                    else:
                        # Pad with identity scale=1, zero=0
                        pad_val = 1.0 if kind == "scales" else 0.0
                        padding = torch.full(
                            [*tensor.shape[:-1], deficit],
                            pad_val, dtype=tensor.dtype, device=tensor.device)
                        new_tensors[kind] = torch.cat([tensor, padding], dim=-1)
                else:
                    new_tensors[kind] = tensor
            else:
                # Per-element tensor (weight, bias, qweight)
                out_dim = tensor.dim() - 1
                per_head = tensor.size(out_dim) // heads
                target_size = target_heads * per_head
                deficit = target_size - tensor.size(out_dim)

                if deficit > 0:
                    if is_kv:
                        # Repeat: reshape to [batch, heads, head_dim] then repeat
                        if tensor.dim() == 2:
                            reshaped = tensor.view(tensor.size(0), heads, per_head)
                            n_repeat = target_heads // heads
                            reshaped = reshaped.repeat(1, 1, n_repeat)
                            new_tensors[kind] = reshaped.reshape(
                                tensor.size(0), target_heads * per_head)
                        else:
                            reshaped = tensor.view(heads, per_head)
                            n_repeat = target_heads // heads
                            reshaped = reshaped.repeat(1, n_repeat)
                            new_tensors[kind] = reshaped.reshape(target_heads * per_head)
                    else:
                        # Pad with zeros
                        new_tensors[kind] = pad_out_dim(tensor, target_size, out_dim)
                else:
                    new_tensors[kind] = tensor

        return Linear(tensors=new_tensors, weight_format=linear.weight_format,
                      data_format=linear.data_format)

    q = _adjust_linear(q, q_heads, is_kv=False)
    k = _adjust_linear(k, kv_heads, is_kv=True)
    v = _adjust_linear(v, kv_heads, is_kv=True)
    return q, k, v
```

- [ ] **Step 3: Add `split_output_gate`**

Insert after `pad_for_tp`:

```python
def split_output_gate(q: Linear, *, head_dim: int):
    """Split output gate from Q projection (Qwen3.5).

    Q's output dim is 2 * head_num * head_dim. Reshape to
    [batch, head_num, 2, head_dim], split into q_real and gate.
    """
    new_q_tensors = {}
    gate_tensors = {}

    for kind, tensor in q.tensors.items():
        head_num = tensor.size(-1) // (head_dim * 2)
        orig_shape = list(tensor.shape)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.view(tensor.size(0), head_num, 2, head_dim)
        q_real = tensor[:, :, 0, :].contiguous().reshape(-1, head_num * head_dim)
        gate = tensor[:, :, 1, :].contiguous().reshape(-1, head_num * head_dim)
        if len(orig_shape) == 1:
            q_real = q_real.squeeze(0)
            gate = gate.squeeze(0)
        new_q_tensors[kind] = q_real
        gate_tensors[kind] = gate

    return (Linear(tensors=new_q_tensors, weight_format=q.weight_format,
                   data_format=q.data_format),
            Linear(tensors=gate_tensors, weight_format=q.weight_format,
                   data_format=q.data_format))
```

- [ ] **Step 4: Add `fuse_qkv`**

Insert after `split_output_gate`:

```python
def fuse_qkv(q: Linear, k: Linear, v: Linear, *,
             tp: int, gate: Linear | None = None):
    """Fuse Q, K, V (and optionally gate) into a single w_qkv Linear.

    Concatenates output channels with TP interleaving.
    Layout per tp-shard: [Q | K | V] or [Q | K | V | Gate].
    """
    merged_tensors: dict[str, torch.Tensor] = {}
    all_kinds = sorted(set(q.tensors) | set(k.tensors) | set(v.tensors))

    for kind in all_kinds:
        qt = q.tensors.get(kind)
        kt = k.tensors.get(kind)
        vt = v.tensors.get(kind)
        if qt is None or kt is None or vt is None:
            continue

        is_2d = qt.dim() == 2

        def reshape(x):
            return x.view(x.size(0), tp, -1) if is_2d else x.view(tp, -1)

        components = [reshape(qt), reshape(kt), reshape(vt)]
        if gate is not None:
            gt = gate.tensors.get(kind)
            if gt is not None:
                components.append(reshape(gt))

        merged = torch.cat(components, dim=-1)
        merged = merged.view(-1, merged.size(-1) * tp)
        if not is_2d:
            merged.squeeze_()
        merged_tensors[kind] = merged

    return Linear(tensors=merged_tensors, weight_format=q.weight_format,
                  data_format=q.data_format)
```

Also add `pad_out_dim` import at the top of the file. The existing import from `..linear` needs to include `pad_out_dim`:

At line 13, change:
```python
from ..linear import Linear
```
to:
```python
from ..linear import Linear, pad_out_dim
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "feat(builder): add pipeline functions for add_qkv_proj refactor"
```

---

### Task 3: Wire up the new pipeline in `add_qkv_proj` and update specs

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:232-245` (rewrite `add_qkv_proj`)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:56-80`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:157-181`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:74-98`

This is one atomic change: switch `add_qkv_proj` to use the new pipeline (which doesn't do RoPE) AND add RoPE calls in all 3 specs. Both must happen together to avoid double-permuting or missing RoPE.

- [ ] **Step 1: Rewrite `add_qkv_proj`**

Replace the `add_qkv_proj` method in `AttentionBuilder` (lines 232-245):

```python
    def add_qkv_proj(self, q, k, v):
        """Fuse Q/K/V into a single w_qkv with TP interleave, commit.

        Pipeline: dequant_mixed -> pad_for_tp -> [split_output_gate] -> fuse_qkv -> commit.
        RoPE permutation is done by the spec before calling this method.
        """
        q, k, v = dequant_mixed(q, k, v)
        q, k, v = pad_for_tp(q, k, v, tp=self._tp,
                              head_dim=self.config.head_dim,
                              q_heads=self.config.head_num,
                              kv_heads=self.config.kv_head_num)
        gate = None
        if self.config.attn_output_gate:
            q, gate = split_output_gate(q, head_dim=self.config.head_dim)
        merged = fuse_qkv(q, k, v, tp=self._tp, gate=gate)
        self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                            model_dtype=self.config.data_type)
```

- [ ] **Step 2: Update `qwen3_spec.py`**

Add `reorder_rotary_emb_linear` to the import at line 23:
```python
from .utils import parse_rope_param, reorder_rotary_emb, reorder_rotary_emb_linear
```

In the `attn` method (around line 56), add RoPE calls before `add_qkv_proj`. Change:

```python
        q = self._linear(f"{pfx}.q_proj")
        k = self._linear(f"{pfx}.k_proj")
        v = self._linear(f"{pfx}.v_proj")
        o = self._linear(f"{pfx}.o_proj")

        mc = self._mc
        tp = self._attn_tp
```

to:

```python
        q = self._linear(f"{pfx}.q_proj")
        k = self._linear(f"{pfx}.k_proj")
        v = self._linear(f"{pfx}.v_proj")
        o = self._linear(f"{pfx}.o_proj")

        q = reorder_rotary_emb_linear(q, self._mc.size_per_head, self._rope_dim)
        k = reorder_rotary_emb_linear(k, self._mc.size_per_head, self._rope_dim)

        mc = self._mc
        tp = self._attn_tp
```

- [ ] **Step 3: Update `qwen3_5_spec.py`**

Add `reorder_rotary_emb_linear` to the import at line 31:
```python
from .utils import parse_rope_param, reorder_rotary_emb, reorder_rotary_emb_linear
```

In the `attn` method (around line 159), add RoPE calls before `add_qkv_proj`. Change:

```python
        q = self._linear(f"{pfx}.q_proj")
        k = self._linear(f"{pfx}.k_proj")
        v = self._linear(f"{pfx}.v_proj")
        o = self._linear(f"{pfx}.o_proj")

        mc = self._mc
        tp = self._attn_tp
```

to:

```python
        q = self._linear(f"{pfx}.q_proj")
        k = self._linear(f"{pfx}.k_proj")
        v = self._linear(f"{pfx}.v_proj")
        o = self._linear(f"{pfx}.o_proj")

        q = reorder_rotary_emb_linear(q, self._mc.size_per_head, self._rope_dim)
        k = reorder_rotary_emb_linear(k, self._mc.size_per_head, self._rope_dim)

        mc = self._mc
        tp = self._attn_tp
```

- [ ] **Step 4: Update `gpt_oss_spec.py`**

Add `reorder_rotary_emb_linear` to the import at line 34:
```python
from .utils import parse_rope_param, reorder_rotary_emb_linear
```

In the `attn` method (around line 76), add RoPE calls before `add_qkv_proj`. Change:

```python
        q = self._linear(f"{pfx}.q_proj")
        k = self._linear(f"{pfx}.k_proj")
        v = self._linear(f"{pfx}.v_proj")
        o = self._linear(f"{pfx}.o_proj")

        mc = self._mc
        tp = self._attn_tp
```

to:

```python
        q = self._linear(f"{pfx}.q_proj")
        k = self._linear(f"{pfx}.k_proj")
        v = self._linear(f"{pfx}.v_proj")
        o = self._linear(f"{pfx}.o_proj")

        q = reorder_rotary_emb_linear(q, self._mc.size_per_head, self._rope_dim)
        k = reorder_rotary_emb_linear(k, self._mc.size_per_head, self._rope_dim)

        mc = self._mc
        tp = self._attn_tp
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(builder): wire up add_qkv_proj pipeline, move RoPE to specs"
```

---

### Task 4: Integration test — Qwen3

**Files:** None (testing only)

Verify Qwen3 (dense and MoE variants) still produces correct output after the refactor.

- [ ] **Step 1: Check GPU availability**

Run: `nvidia-smi` to verify an empty GPU.

- [ ] **Step 2: Run Qwen3 dense test**

Run from repo root:
```bash
python scripts/test_turbomind_model.py <qwen3-model-path> --tp 1 --max-new-tokens 128 --prompt "Hello, how are you today?"
```

Expected: Model responds with coherent English text. The response must contain meaningful human words relevant to the prompt. Gibberish = bug.

- [ ] **Step 3: If test fails, debug and fix**

If the model produces gibberish, the most likely causes are:
1. RoPE permutation applied twice (spec + builder) — check that `permute_qk` logic was fully removed
2. Wrong fusion order in `fuse_qkv` — check TP interleaving matches old `_merge_qkv`
3. Head padding/repeat error in `pad_for_tp` — check output dimensions match original

Fix and re-test until the response is correct.

---

### Task 5: Integration test — Qwen3.5 and GPT-OSS

**Files:** None (testing only)

Verify Qwen3.5 (with output gate) and GPT-OSS (MoE) still produce correct output.

- [ ] **Step 1: Check GPU availability**

Run: `nvidia-smi` to verify an empty GPU.

- [ ] **Step 2: Run Qwen3.5 test**

Run from repo root:
```bash
python scripts/test_turbomind_model.py <qwen3.5-model-path> --tp 1 --max-new-tokens 128 --prompt "Hello, how are you today?"
```

Expected: Coherent English response. Qwen3.5 uses `attn_output_gate=True` so `split_output_gate` must work correctly.

- [ ] **Step 3: Run GPT-OSS test**

Run from repo root:
```bash
python scripts/test_turbomind_model.py <gpt-oss-model-path> --tp 1 --max-new-tokens 128 --prompt "Hello, how are you today?"
```

Expected: Coherent English response.

- [ ] **Step 4: If any test fails, debug and fix**

Same debugging approach as Task 4. For Qwen3.5, pay special attention to `split_output_gate` and the gate path in `fuse_qkv`. For GPT-OSS, check `pad_for_tp` KV repeat behavior with its GQA ratio.

---

### Task 6: Delete old code and update exports

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py`
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:150-175`
- Modify: `lmdeploy/turbomind/deploy/builder/__init__.py`

Only do this after all integration tests pass (Tasks 4-5).

- [ ] **Step 1: Delete old functions from `attention.py`**

Delete the following from `attention.py`:
- `_reorder_rotary_emb` function (lines 38-79)
- `_merge_qkv` function (lines 82-94)
- `_merge_qkvg` function (lines 97-110)
- `merge_qkv_linear` function (lines 113-217)

Keep: `_ATTN_TP_RULES`, all new pipeline functions, `AttentionBuilder` class.

Update the import line at the top of the file. Remove unused imports. Change:
```python
from ._base import Builder, SplitSide, _dequant_linear, _ensure_compatible_formats, _block_ops_need_dequant
```
to:
```python
from ._base import Builder, SplitSide, _dequant_linear, _ensure_compatible_formats
```

(`_ensure_compatible_formats` is still used by `dequant_mixed` indirectly — actually no, `dequant_mixed` has its own logic. Check if anything else in attention.py still uses `_ensure_compatible_formats`. If not, remove it from the import.)

Actually, `dequant_mixed` does its own format comparison. Remove `_ensure_compatible_formats` from the import too:
```python
from ._base import Builder, SplitSide, _dequant_linear
```

- [ ] **Step 2: Delete `_block_ops_need_dequant` from `_base.py`**

Delete lines 150-175 from `lmdeploy/turbomind/deploy/builder/_base.py` (the `_block_ops_need_dequant` function).

Check that nothing else imports it:
```bash
grep -r '_block_ops_need_dequant' --include='*.py' lmdeploy/
```
Expected: no matches (we removed the import from attention.py in Step 1).

- [ ] **Step 3: Update `__init__.py` exports**

Change `lmdeploy/turbomind/deploy/builder/__init__.py` line 12 from:
```python
from .attention import AttentionBuilder, merge_qkv_linear
```
to:
```python
from .attention import AttentionBuilder
```

And remove `'merge_qkv_linear'` from `__all__`.

Also check that nothing else imports `merge_qkv_linear`:
```bash
grep -r 'merge_qkv_linear' --include='*.py' lmdeploy/
```
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py \
        lmdeploy/turbomind/deploy/builder/_base.py \
        lmdeploy/turbomind/deploy/builder/__init__.py
git commit -m "refactor(builder): delete old merge_qkv_linear and helpers"
```

---

### Task 7: Final integration test with TP=2

**Files:** None (testing only)

Verify the refactor works with tensor parallelism = 2, which exercises the `pad_for_tp` and `fuse_qkv` TP interleaving paths more thoroughly.

- [ ] **Step 1: Check GPU availability (2 GPUs)**

- [ ] **Step 2: Run Qwen3 with TP=2**

```bash
python scripts/test_turbomind_model.py <qwen3-model-path> --tp 2 --max-new-tokens 128 --prompt "Hello, how are you today?"
```

Expected: Coherent English response.

- [ ] **Step 3: Run one more model with TP=2**

```bash
python scripts/test_turbomind_model.py <model-path> --tp 2 --max-new-tokens 128 --prompt "Explain quantum computing in simple terms."
```

Expected: Coherent response.

- [ ] **Step 4: If tests fail, debug and fix**

For TP=2 issues, focus on:
1. `pad_for_tp` — are heads being padded correctly for the tp=2 case?
2. `fuse_qkv` — is the TP interleaving producing the correct shard layout?
3. Compare the fused weight shape against the old code's output.
