# Distributor Extract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `LayerWriter` into `distributor.py`, rename to `Distributor`, and unify `_load_global` to use the same abstraction.

**Architecture:** Move the GPU fan-out class to its own module with a new name, then rewrite `_load_global` to use it instead of raw GPU iteration. The `_process_*` methods only need a rename of the type they receive.

**Tech Stack:** Python, pybind11 C++ handles, TurboMind model loading pipeline.

---

### Task 1: Create `distributor.py` with the `Distributor` class

**Files:**
- Create: `lmdeploy/turbomind/deploy/distributor.py`

- [ ] **Step 1: Create the new file**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Distributor: distributes module creation and weight commits across all GPUs."""
from __future__ import annotations

from .load_context import commit_linear, commit_tensor


class Distributor:
    """Wraps N GPU handles for a single logical module.

    Distributes create_child / commit_linear / commit_tensor
    across all GPUs with bound TP configuration.
    """

    def __init__(self, handles, tp=1, ranks=None):
        self._handles = handles
        self._tp = tp
        self._ranks = ranks

    @property
    def tp_size(self):
        return self._tp

    def _rank_for(self, gpu_idx):
        if self._ranks and self._tp > 1:
            return self._ranks[gpu_idx]
        return 0

    def create_child(self, name, config, tp=None, ranks=None):
        """Create a typed module child on ALL GPUs.

        Calls ``config.for_rank(rank).to_cpp()`` per GPU.
        Returns a new Distributor scoped to the created children,
        with tp/ranks rebound if provided (otherwise inherited).
        """
        new_tp = tp if tp is not None else self._tp
        new_ranks = ranks if ranks is not None else self._ranks
        children = []
        for i, handle in enumerate(self._handles):
            rank = new_ranks[i] if new_ranks and new_tp > 1 else 0
            child = handle.create_child(name, config.for_rank(rank).to_cpp())
            children.append(child)
        return Distributor(children, tp=new_tp, ranks=new_ranks)

    def commit_linear(self, name, linear, split_side=None, model_dtype=None):
        """Commit a Linear bundle to all GPUs.

        If split_side is given, uses bound tp/ranks for sharding.
        If split_side is None, broadcasts (tp=1).
        """
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            rank = self._rank_for(i) if tp > 1 else 0
            commit_linear(handle, linear, name,
                          split_side=split_side, split_num=tp,
                          rank=rank, model_dtype=model_dtype)

    def commit_tensor(self, name, tensor, split_side=None):
        """Commit a raw tensor to all GPUs."""
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            rank = self._rank_for(i) if tp > 1 else 0
            commit_tensor(handle, tensor, name,
                          split_side=split_side, split_num=tp,
                          rank=rank)
```

- [ ] **Step 2: Verify the file parses**

Run: `python -c "from lmdeploy.turbomind.deploy.distributor import Distributor; print('OK')"`

Set `PYTHONPATH` as described in CLAUDE.md first:
```
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/distributor.py
git commit -m "feat(deploy): add Distributor class extracted from LayerWriter"
```

---

### Task 2: Update `text_model_loader.py` — imports and rename

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

- [ ] **Step 1: Replace import and remove `LayerWriter` class definition**

In `text_model_loader.py`, make these changes:

**a)** Add the new import after the existing imports (after line 17, before `if TYPE_CHECKING`):

```python
from .distributor import Distributor
```

**b)** Delete the entire `LayerWriter` class (lines 24-81).

**c)** Rename all remaining `LayerWriter` references to `Distributor`:

- Line 105: `_layer_writer(self, layer: int) -> LayerWriter:` → `_layer_writer(self, layer: int) -> Distributor:`
- Line 117: `return LayerWriter(handles)` → `return Distributor(handles)`
- Line 132: `def _process_norms(self, writer: LayerWriter,` → `def _process_norms(self, writer: Distributor,`
- Line 150: `def _process_attention(self, writer: LayerWriter,` → `def _process_attention(self, writer: Distributor,`
- Line 187: `def _process_ffn(self, writer: LayerWriter,` → `def _process_ffn(self, writer: Distributor,`
- Line 235: `def _process_moe(self, writer: LayerWriter,` → `def _process_moe(self, writer: Distributor,`
- Line 277: `parent = LayerWriter(children)` → `parent = Distributor(children)`
- Line 323: `def _process_linear_attn(self, writer: LayerWriter,` → `def _process_linear_attn(self, writer: Distributor,`

**d)** Remove unused imports. After the rename, `commit_linear` and `commit_tensor` are no longer used directly in this file (they are used only inside `Distributor`). Remove them from the `load_context` import on lines 11-14:

Change:
```python
from .load_context import (
    _cpp_dtype, _act_type_id,
    commit_linear, commit_tensor,
    _ATTN_TP_RULES, _FFN_TP_RULES, _LINEAR_ATTN_TP_RULES,
)
```
To:
```python
from .load_context import (
    _cpp_dtype, _act_type_id,
    _ATTN_TP_RULES, _FFN_TP_RULES, _LINEAR_ATTN_TP_RULES,
)
```

Note: `_FFN_TP_RULES` is imported but unused — leave it for now, it's out of scope.

**e)** Also remove the now-unused `LinearConfig` import from the configs import on lines 7-10 if `_load_global` will be rewritten in Task 3 to not import it (but it still uses `LinearConfig` via `Distributor.create_child` — actually `_load_global` uses `LinearConfig` directly, so keep it). Keep the configs import as-is.

Wait — after Task 3, `_load_global` still uses `LinearConfig` and `NormConfig`, so keep both.

- [ ] **Step 2: Verify the file parses**

Run: `python -c "from lmdeploy.turbomind.deploy.text_model_loader import TextModelLoader; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): replace LayerWriter with imported Distributor"
```

---

### Task 3: Unify `_load_global` to use `Distributor`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py` (the `_load_global` method and surrounding code)

- [ ] **Step 1: Add `_root_distributor` factory method**

Add this method to `TextModelLoader`, right after `_layer_writer` (which is around the old line 117):

```python
    def _root_distributor(self) -> Distributor:
        """Create a Distributor wrapping the root handles from all GPUs."""
        handles = []
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            handles.append(root)
        return Distributor(handles)
```

- [ ] **Step 2: Rewrite `_load_global` to use Distributor**

Replace the entire `_load_global` method with:

```python
    def _load_global(self, spec: 'TextModelSpec'):
        from .linear import pad_out_dim

        mc = self.model.model_config
        tp = self.attn_tp * self.model.attn_cp_size
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        dtype = _cpp_dtype(mc.data_type)
        hidden = mc.hidden_units

        self._ensure_ranks()
        root = self._root_distributor()

        # Token embeddings (column-parallel)
        emb = spec.tok_embeddings()
        if emb is not None:
            emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
            tok_cfg = LinearConfig(
                input_dim=padded_vocab,
                output_dim=hidden // tp,
                data_type=dtype)
            tok_emb = root.create_child('tok_embeddings', tok_cfg,
                                        tp=tp, ranks=self._attn_ranks)
            tok_emb.commit_tensor('weight', emb_padded,
                                  split_side=SplitSide.OUTPUT)

        # Final norm (broadcast)
        norm = spec.norm_weight()
        if norm is not None:
            norm_cfg = NormConfig(dim=hidden, data_type=dtype)
            norm_mod = root.create_child('norm', norm_cfg)
            norm_mod.commit_tensor('weight', norm)

        # Output head (column-parallel, transposed)
        output = spec.output_weight()
        if output is not None:
            output_padded = pad_out_dim(output, padded_vocab, dim=0)
            output_t = output_padded.t()
            out_cfg = LinearConfig(
                input_dim=hidden,
                output_dim=padded_vocab // tp,
                data_type=dtype)
            output_mod = root.create_child('output', out_cfg,
                                           tp=tp, ranks=self._attn_ranks)
            output_mod.commit_tensor('weight', output_t,
                                     split_side=SplitSide.OUTPUT)
```

**Key behavioral differences from the old code:**

1. `self._ensure_ranks()` is called — the old code used `self.model.tp_ranks(gpu)[0]` directly per-GPU, but since `_root_distributor` will be combined with `tp=tp, ranks=self._attn_ranks`, the ranks must be pre-computed.

2. No GPU loop — `root.create_child(tp=tp, ranks=self._attn_ranks)` distributes across all GPUs with per-GPU rank binding.

3. Typed `NormConfig(dim=hidden, data_type=dtype)` replaces raw `_tm.NormConfig()` — identical conversion via `to_cpp()`.

4. The `import _turbomind as _tm` inside the norm block is eliminated.

5. `dtype` and `hidden` are hoisted outside the GPU loop — they're model-level constants.

- [ ] **Step 3: Verify the file parses**

Run: `python -c "from lmdeploy.turbomind.deploy.text_model_loader import TextModelLoader; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): unify _load_global to use Distributor"
```

---

### Task 4: Update stale `LayerWriter` references

**Files:**
- Modify: `lmdeploy/turbomind/deploy/load_context.py`

- [ ] **Step 1: Update docstring in `commit_ffn`**

In `load_context.py`, the `commit_ffn` function at line 357 has a docstring referencing `LayerWriter`:

Change:
```python
    """DEPRECATED: Use LayerWriter + fuse_ffn_linears directly."""
```
To:
```python
    """DEPRECATED: Use Distributor + fuse_ffn_linears directly."""
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/load_context.py
git commit -m "docs(load_context): update LayerWriter reference to Distributor"
```

---

### Task 5: Verify with model tests

**Files:** None (testing only)

- [ ] **Step 1: Check GPU availability**

Use `get_gpu_usage` MCP tool to confirm GPUs are available and not occupied.

- [ ] **Step 2: Test a model with TP=1**

Use the turbomind-tester agent to test a model (e.g., a small Llama or Qwen model) with TP=1.

Prompt should request at least 128 tokens and the response must contain meaningful human words.

This exercises both `_load_global` (tok_embeddings, norm, output) and `_load_layer` (all `_process_*` methods).

- [ ] **Step 3: Test a model with TP=2 (if GPUs available)**

Use the turbomind-tester agent to test with TP=2.

This exercises the TP sharding path in `Distributor.create_child` and `commit_tensor` — the critical path that changed in `_load_global`.

- [ ] **Step 4: Verify responses are meaningful**

Every response must contain coherent, relevant text. Gibberish indicates a bug in the weight loading pipeline.
