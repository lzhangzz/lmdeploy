# Unify reorder_rotary_emb Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `reorder_rotary_emb` (tensor) and `reorder_rotary_emb_linear` (Linear) into a single public `reorder_rotary_emb` function.

**Architecture:** The current tensor-level shuffle becomes a private `_reorder_rotary_emb` helper. The public `reorder_rotary_emb` dispatches on `isinstance(x, Linear)` — Linear inputs get the quantization-aware path (dequant fallback, block-level shuffle), everything else goes to the tensor helper.

**Tech Stack:** Python, PyTorch, TurboMind deploy pipeline

---

### Task 1: Refactor utils.py — rename and unify

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py:158-221`

- [ ] **Step 1: Rename tensor function to `_reorder_rotary_emb`**

Replace line 158 function definition and update all internal calls within the Linear path:

```python
def _reorder_rotary_emb(x: torch.Tensor, head_dim: int, rope_dim: int):
    """Reorder rotary embedding layout for TurboMind's RoPE kernel."""
    if rope_dim < head_dim:
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), head_num, head_dim)
        rotary = x[:, :, :rope_dim]
        passthrough = x[:, :, rope_dim:]
        rotary = rotary.view(x.size(0), head_num, 2, rope_dim // 2).transpose(2, 3).contiguous()
        rotary = rotary.view(x.size(0), head_num, rope_dim)
        x = torch.cat([rotary, passthrough], dim=-1)
        return x.reshape(orig_shape)
    else:
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        return x.view(-1, head_num, 2, head_dim // 2).transpose(2, 3).reshape(x.shape)
```

- [ ] **Step 2: Replace `reorder_rotary_emb_linear` with the unified `reorder_rotary_emb`**

Delete lines 179-221 (the old `reorder_rotary_emb_linear`) and replace with:

```python
def reorder_rotary_emb(x, head_dim: int, rope_dim: int, *, data_type=None):
    """Apply RoPE layout permutation.

    Accepts either a ``Linear`` or a raw ``torch.Tensor``.

    For ``Linear`` inputs the permutation is applied to every tensor in the
    bundle with quantization awareness (block-alignment check, dequant
    fallback, block-level shuffling for scales/zeros).  ``data_type`` is
    required and must not be ``None``.

    For ``torch.Tensor`` inputs the element-level interleave-transpose is
    applied directly.  ``data_type`` is ignored.
    """
    from ..linear import Linear

    if isinstance(x, Linear):
        if data_type is None:
            raise TypeError(
                "data_type is required when passing a Linear to reorder_rotary_emb"
            )
        wfmt = x.weight_format
        block_out = wfmt.block_out or 0

        # If blocks don't align with heads, dequant first
        if block_out and block_out % head_dim != 0:
            x = _dequant_linear(x, data_type=data_type)
            block_out = 0

        new_tensors = {}
        for kind, tensor in x.tensors.items():
            if kind in ("scales", "zeros") and block_out > 0:
                blocks_per_head = block_out // head_dim
                if blocks_per_head <= 1:
                    new_tensors[kind] = tensor
                else:
                    rope_dim_blocks = rope_dim * blocks_per_head // head_dim
                    new_tensors[kind] = _reorder_rotary_emb(tensor, blocks_per_head, rope_dim_blocks)
            elif tensor.size(-1) % head_dim == 0:
                new_tensors[kind] = _reorder_rotary_emb(tensor, head_dim, rope_dim)
            else:
                new_tensors[kind] = tensor

        return Linear(tensors=new_tensors, weight_format=x.weight_format,
                      data_format=x.data_format)

    return _reorder_rotary_emb(x, head_dim, rope_dim)
```

Key differences from the original code:
- Function name is `reorder_rotary_emb` (was `reorder_rotary_emb_linear`)
- First param renamed from `linear` to `x` (accepts both types)
- `data_type` is now `Optional` with `None` default, guarded by `TypeError` for Linear inputs
- Internal calls go to `_reorder_rotary_emb` instead of the old public name

- [ ] **Step 3: Verify no references to deleted names remain in utils.py**

Search `utils.py` for `reorder_rotary_emb_linear` — there should be zero matches. The file should export only `reorder_rotary_emb` (public) and `_reorder_rotary_emb` (private).

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/utils.py
git commit -m "deploy: unify reorder_rotary_emb for Linear and Tensor inputs"
```

---

### Task 2: Update call sites in spec files

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:20,141-144,157-158`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:20,189-192,205-206`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:19,147-150`

- [ ] **Step 1: Update qwen3_spec.py**

Import line 20 — remove `reorder_rotary_emb_linear`:
```python
from .utils import layer_progress, reorder_rotary_emb
```

Lines 141-144 — replace `reorder_rotary_emb_linear` with `reorder_rotary_emb`:
```python
        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
```

Lines 157-158 — already use `reorder_rotary_emb`, no change needed (these are tensor calls, which still work).

- [ ] **Step 2: Update qwen3_5_spec.py**

Import line 20 — remove `reorder_rotary_emb_linear`:
```python
from .utils import layer_progress, reorder_rotary_emb
```

Lines 189-192 — replace `reorder_rotary_emb_linear` with `reorder_rotary_emb`:
```python
        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
```

Lines 205-206 — already use `reorder_rotary_emb`, no change needed.

- [ ] **Step 3: Update gpt_oss_spec.py**

Import line 19 — replace `reorder_rotary_emb_linear` with `reorder_rotary_emb`:
```python
from .utils import layer_progress, reorder_rotary_emb
```

Lines 147-150 — replace `reorder_rotary_emb_linear` with `reorder_rotary_emb`:
```python
        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
```

- [ ] **Step 4: Verify no references to the old name remain**

```bash
grep -rn "reorder_rotary_emb_linear" lmdeploy/
```

Expected: zero matches across all Python files.

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "deploy: migrate spec files to unified reorder_rotary_emb"
```

---

### Task 3: Smoke-test with a model conversion

**Files:** None (verification only)

- [ ] **Step 1: Check GPU availability**

Use `get_gpu_usage` MCP tool to find an empty GPU.

- [ ] **Step 2: Run model test**

```bash
python scripts/test_turbomind_model.py <model_id> --tp 1
```

Use any Qwen3 or GPT-OSS model from the model registry. Verify:
- The conversion completes without errors
- The model responds with meaningful text (not gibberish) to a test prompt
- Response length is at least 128 tokens

- [ ] **Step 3: Commit (only if any fixes were needed)**

Only commit if bugs were found and fixed during testing. Otherwise no commit needed for this task.
