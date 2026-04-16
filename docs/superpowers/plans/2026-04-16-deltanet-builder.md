# DeltaNetBuilder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `DeltaNetBuilder(Builder)` subclass that encapsulates DeltaNet weight loading, then simplify `qwen3_5_spec.py` to use it.

**Architecture:** New subclass in `builder.py` alongside `AttentionBuilder`, `FfnBuilder`, `MoeBuilder`. Four convenience methods absorb fusing, transposition, TP interleaving, and commit logic currently inline in `qwen3_5_spec.py:linear_attn()`. The spec still owns checkpoint reading via `self._get()` / `self._linear()`.

**Tech Stack:** Python, PyTorch, TurboMind C++ extension (`_turbomind`)

---

### Task 1: Add DeltaNetBuilder class to builder.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder.py` (insert after `MoeBuilder`, line ~1167)

- [ ] **Step 1: Add DeltaNetBuilder class after MoeBuilder**

Insert the following class after the `MoeBuilder` class (after line 1167 in `builder.py`):

```python


# ---------------------------------------------------------------------------
# DeltaNetBuilder -- Gated Delta Net input projections, scalar params, conv1d
# ---------------------------------------------------------------------------


class DeltaNetBuilder(Builder):
    """DeltaNet (Gated Delta Net) weight loading builder."""

    def add_input_projections(self, *, in_proj_qkv=None, in_proj_z=None,
                              in_proj_b=None, in_proj_a=None, out_proj=None,
                              qkv_split=None):
        """Fuse GDN input projections, commit all linears with TP rules.

        Internally calls ``fuse_gdn_in_proj`` to merge qkv/z/b/a into a
        single ``in_proj_all`` with TP interleaving.  Commits each resulting
        linear using ``_LINEAR_ATTN_TP_RULES`` for split-side lookup.
        """
        linears = {}
        if in_proj_qkv is not None:
            linears["in_proj_qkv"] = in_proj_qkv
        if in_proj_z is not None:
            linears["in_proj_z"] = in_proj_z
        if in_proj_b is not None:
            linears["in_proj_b"] = in_proj_b
        if in_proj_a is not None:
            linears["in_proj_a"] = in_proj_a
        if out_proj is not None:
            linears["out_proj"] = out_proj

        linears = fuse_gdn_in_proj(linears, self._tp, qkv_split)

        model_dtype = self.config.data_type
        for name, lin in linears.items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            split_side = rule.get('split_side')
            self._commit_linear(name, lin, split_side=split_side,
                                model_dtype=model_dtype)

    def add_scalar_params(self, a_log=None, dt_bias=None):
        """Commit A_log and dt_bias as OUTPUT-split tensors."""
        if a_log is not None:
            self._commit_tensor("A_log", a_log, split_side=SplitSide.OUTPUT)
        if dt_bias is not None:
            self._commit_tensor("dt_bias", dt_bias, split_side=SplitSide.OUTPUT)

    def add_conv1d(self, conv1d, qkv_split=None):
        """Transpose HF layout to TM layout, TP-reshape if needed, commit.

        HF stores conv1d as [conv_dim, d_conv]; TM kernel expects
        [d_conv, conv_dim].  When tp > 1 and *qkv_split* is provided,
        the Q/K/V sub-dims are TP-interleaved.
        """
        if conv1d is None:
            return
        # Squeeze leading singleton dim if present
        if conv1d.ndim == 3 and conv1d.shape[1] == 1:
            conv1d = conv1d.squeeze(1)
        # Transpose: HF [conv_dim, d_conv] -> TM [d_conv, conv_dim]
        conv1d = conv1d.t().contiguous()
        # TP Q/K/V interleaving
        if self._tp > 1 and qkv_split is not None:
            q_dim, k_dim, v_dim = qkv_split
            d_conv = conv1d.shape[0]
            tp = self._tp
            q_part = conv1d[:, :q_dim]
            k_part = conv1d[:, q_dim:q_dim + k_dim]
            v_part = conv1d[:, q_dim + k_dim:]
            conv1d = torch.cat([
                q_part.reshape(d_conv, tp, q_dim // tp),
                k_part.reshape(d_conv, tp, k_dim // tp),
                v_part.reshape(d_conv, tp, v_dim // tp),
            ], dim=2).reshape(d_conv, -1).contiguous()
        self._commit_tensor("conv1d", conv1d, split_side=SplitSide.OUTPUT)

    def add_norm(self, norm_weight, data_type):
        """Add inline norm child."""
        self._add_norm_child("norm", norm_weight, data_type=data_type)
```

- [ ] **Step 2: Build to verify no syntax errors**

Run: `cd /data/lmdeploy-modeling/build && ninja _turbomind`
Expected: Build succeeds (DeltaNetBuilder is Python-only, no C++ changes)

- [ ] **Step 3: Verify import**

Run: `PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "from lmdeploy.turbomind.deploy.builder import DeltaNetBuilder; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder.py
git commit -m "feat(deploy): add DeltaNetBuilder with convenience methods"
```

---

### Task 2: Simplify qwen3_5_spec.py to use DeltaNetBuilder

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`

This task replaces the inline `Builder` + `_commit_*` calls in `linear_attn()` with `DeltaNetBuilder` methods, removes the dead D parameter code, and cleans up imports.

- [ ] **Step 1: Update imports**

In `qwen3_5_spec.py`, replace the import block at lines 17-22:

```python
from ..builder import (
    AttentionBuilder, Builder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _LINEAR_ATTN_TP_RULES, _act_type_id,
    fuse_gdn_in_proj,
)
```

with:

```python
from ..builder import (
    AttentionBuilder, DeltaNetBuilder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, TextModelBuilder,
    _act_type_id,
)
```

Changes: `Builder` removed (no longer used directly), `DeltaNetBuilder` added, `SplitSide` removed (now internal to builder), `_LINEAR_ATTN_TP_RULES` removed (now internal to builder), `fuse_gdn_in_proj` removed (now internal to builder).

- [ ] **Step 2: Replace the `linear_attn` method body**

Replace the entire `linear_attn` method (lines 195-256):

```python
    def linear_attn(self, pfx, layer):
        """Return Builder for linear-attention (Gated Delta Net)."""
        # Read GDN input projection linears
        la_linears: dict[str, Linear] = {}
        for key in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]:
            lin = self._linear(f"{pfx}.{key}")
            if lin is not None:
                la_linears[key] = lin

        mc = self._mc
        tp = self._attn_tp
        dtype = self._cpp_dtype()

        dn_cfg = DeltaNetConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype)
        linear_attn = Builder(dn_cfg, self._contexts,
                              tp=tp, ranks=self._attn_ranks)

        # Fuse GDN input projections
        la_linears = fuse_gdn_in_proj(la_linears, tp, self._linear_qkv_split)

        # Commit linear bundles with TP rules from the rule table
        for name, lin in la_linears.items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            split_side = rule.get('split_side')
            linear_attn._commit_linear(name, lin, split_side=split_side,
                                       model_dtype=dtype)

        # Inline params: A_log, dt_bias
        for key in ["A_log", "dt_bias"]:
            t = self._get(f"{pfx}.{key}")
            linear_attn._commit_tensor(key, t, split_side=SplitSide.OUTPUT)

        # Inline param: conv1d
        conv1d = self._get(f"{pfx}.conv1d.weight")
        if conv1d.ndim == 3 and conv1d.shape[1] == 1:
            conv1d = conv1d.squeeze(1)
        # C++ kernel expects [d_conv, conv_dim]; HF stores [conv_dim, d_conv].
        conv1d = conv1d.t().contiguous()
        if self._attn_tp > 1 and self._linear_qkv_split is not None:
            q_dim, k_dim, v_dim = self._linear_qkv_split
            d_conv = conv1d.shape[0]
            tp = self._attn_tp
            q_part = conv1d[:, :q_dim]
            k_part = conv1d[:, q_dim:q_dim + k_dim]
            v_part = conv1d[:, q_dim + k_dim:]
            conv1d = torch.cat([
                q_part.reshape(d_conv, tp, q_dim // tp),
                k_part.reshape(d_conv, tp, k_dim // tp),
                v_part.reshape(d_conv, tp, v_dim // tp),
            ], dim=2).reshape(d_conv, -1).contiguous()
        linear_attn._commit_tensor("conv1d", conv1d, split_side=SplitSide.OUTPUT)

        # Inline param: D
        d_param = self._get(f"{pfx}.D")
        linear_attn._commit_tensor("D", d_param, split_side=SplitSide.OUTPUT)

        # Inline norm children
        norm = self._get(f"{pfx}.norm.weight")
        linear_attn._add_norm_child("norm", norm, data_type=dtype)

        return linear_attn
```

with:

```python
    def linear_attn(self, pfx, layer):
        """Return DeltaNetBuilder for linear-attention (Gated Delta Net)."""
        mc = self._mc
        tp = self._attn_tp
        dtype = self._cpp_dtype()

        dn_cfg = DeltaNetConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype)
        builder = DeltaNetBuilder(dn_cfg, self._contexts,
                                  tp=tp, ranks=self._attn_ranks)

        builder.add_input_projections(
            in_proj_qkv=self._linear(f"{pfx}.in_proj_qkv"),
            in_proj_z=self._linear(f"{pfx}.in_proj_z"),
            in_proj_b=self._linear(f"{pfx}.in_proj_b"),
            in_proj_a=self._linear(f"{pfx}.in_proj_a"),
            out_proj=self._linear(f"{pfx}.out_proj"),
            qkv_split=self._linear_qkv_split)
        builder.add_scalar_params(
            a_log=self._get(f"{pfx}.A_log"),
            dt_bias=self._get(f"{pfx}.dt_bias"))
        builder.add_conv1d(
            self._get(f"{pfx}.conv1d.weight"),
            qkv_split=self._linear_qkv_split)
        builder.add_norm(
            self._get(f"{pfx}.norm.weight"), data_type=dtype)
        return builder
```

- [ ] **Step 3: Verify import**

Run: `PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "from lmdeploy.turbomind.deploy.source_model.qwen3_5_spec import Qwen3_5Spec; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(deploy): use DeltaNetBuilder in qwen3_5_spec, remove dead D code"
```

---

### Task 3: Verify model correctness with turbomind-tester

**Files:** None (verification only)

- [ ] **Step 1: Check GPU is free**

Use `get_gpu_usage` MCP tool. Expected: GPU with < 1000 MiB used.

- [ ] **Step 2: Run Qwen3.5-35B-A3B model test**

Use the turbomind-tester agent with model `Qwen/Qwen3.5-35B-A3B`, prompt "Tell me about the history of artificial intelligence in 200 words", max_new_tokens >= 128.

Expected: Model produces coherent, meaningful text about AI history. No gibberish, no errors, no OOM.

- [ ] **Step 3: If test passes, final commit (if any unstaged changes)**

If everything works, no additional commit needed — all changes were committed in Tasks 1 and 2.
