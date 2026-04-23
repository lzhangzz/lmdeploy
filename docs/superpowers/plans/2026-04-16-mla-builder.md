# MLABuilder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract MLA attention logic into a dedicated `MLABuilder` class with its own `MLAConfig`, following the same pattern as `DeltaNetBuilder`.

**Architecture:** New `MLAConfig` dataclass in `module_configs.py` and new `MLABuilder` class in `builder.py`. The spec's `attn()` simplifies to a read-and-delegate. The MLA fold+pad transform moves from the spec into the builder. The C++ side is unchanged — MLA still uses `AttentionWeight`.

**Tech Stack:** Python, PyTorch, TurboMind C++ pybind11 bindings.

---

### Task 1: Add MLAConfig to module_configs.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/module_configs.py:110` (after `AttentionConfig` class)

- [ ] **Step 1: Add MLAConfig dataclass**

Insert the following after the `AttentionConfig` class (after line 110, before `class FfnConfig`):

```python
@dataclass
class MLAConfig:
    """Config for MLA (Multi-head Latent Attention) weight modules."""
    hidden_dim: int = 0
    head_num: int = 0
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_nope_dim: int = 0       # key dim without RoPE
    qk_rope_dim: int = 0       # RoPE dim within each head
    v_head_dim: int = 0
    size_per_head: int = 0     # effective head dim
    kv_head_num: int = 0       # for MQA kv_a_proj
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0
    window_size: int = -1

    k_type_name: str = 'AttentionWeight'

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype, window_size,
                          qk_nope_dim=0):
        """Build from ModelConfig. qk_nope_dim must be passed explicitly
        since ModelConfig does not carry it."""
        qk_rope_dim = mc.qk_rope_dim or 0
        kv_lora_rank = mc.kv_lora_rank or 0
        v_head_dim = mc.v_head_dim or 0
        size_per_head = qk_nope_dim + qk_rope_dim
        if kv_lora_rank and kv_lora_rank != qk_nope_dim:
            size_per_head = kv_lora_rank + qk_rope_dim
            v_head_dim = kv_lora_rank
        return cls(
            hidden_dim=mc.hidden_units,
            head_num=mc.head_num,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=mc.q_lora_rank or 0,
            qk_nope_dim=qk_nope_dim,
            qk_rope_dim=qk_rope_dim,
            v_head_dim=v_head_dim,
            size_per_head=size_per_head,
            kv_head_num=mc.kv_head_num,
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
            window_size=window_size,
        )

    def for_rank(self, rank):
        return replace(self, tp_rank=rank)

    def to_cpp(self):
        cfg = _tm.AttentionConfig()
        cfg.hidden_dim = self.hidden_dim
        cfg.head_dim = self.size_per_head
        cfg.head_num = self.head_num
        cfg.kv_head_num = self.kv_head_num
        cfg.kv_lora_rank = self.kv_lora_rank
        cfg.q_lora_rank = self.q_lora_rank
        cfg.qk_rope_dim = self.qk_rope_dim
        cfg.v_head_dim = self.v_head_dim
        cfg.tp_size = self.tp_size
        cfg.tp_rank = self.tp_rank
        cfg.data_type = _tm.DataType(self.data_type)
        cfg.window_size = self.window_size
        cfg.has_bias = False
        cfg.qk_norm = False
        cfg.attn_sink = False
        cfg.attn_output_gate = False
        return cfg
```

- [ ] **Step 2: Verify syntax**

Run: `cd /data/lmdeploy-modeling && python -c "from lmdeploy.turbomind.deploy.module_configs import MLAConfig; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/module_configs.py
git commit -m "feat(deploy): add MLAConfig dataclass for MLABuilder"
```

---

### Task 2: Add MLABuilder to builder.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder.py` — add TP rules table (after `_LINEAR_ATTN_TP_RULES` at line 161), add class (after `DeltaNetBuilder` at line 1246)

- [ ] **Step 1: Add _MLA_TP_RULES table**

Insert after `_LINEAR_ATTN_TP_RULES` (after line 161):

```python
_MLA_TP_RULES: dict[str, dict] = {
    "q_a_proj":  dict(split_side=SplitSide.OUTPUT),
    "q_b_proj":  dict(split_side=SplitSide.OUTPUT),
    "kv_a_proj": dict(split_side=SplitSide.OUTPUT),
    "wo":        dict(split_side=SplitSide.INPUT),
}
```

- [ ] **Step 2: Add MLABuilder class**

Insert after `DeltaNetBuilder` (after line 1246):

```python
# ---------------------------------------------------------------------------
# MLABuilder -- MLA projections, fold+pad, norms
# ---------------------------------------------------------------------------


class MLABuilder(Builder):
    """MLA (Multi-head Latent Attention) weight loading builder."""

    def add_projections(self, *, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj,
                        wo):
        """Apply MLA fold+pad, then commit each projection.

        The fold consumes kv_b_proj — its information is absorbed into
        q_b_proj and wo.  After the fold, kv_b_proj is not committed.
        """
        linears = {
            "q_a_proj": q_a_proj,
            "q_b_proj": q_b_proj,
            "kv_a_proj": kv_a_proj,
            "wo": wo,
        }
        if kv_b_proj is not None:
            linears["kv_b_proj"] = kv_b_proj

        self._fold_and_pad(linears)

        model_dtype = self.config.data_type
        for name, lin in linears.items():
            if lin is None:
                continue
            rule = _MLA_TP_RULES.get(name, {})
            split_side = rule.get('split_side')
            self._commit_linear(name, lin, split_side=split_side,
                                model_dtype=model_dtype)

    def add_norms(self, *, q_a_norm, kv_a_norm, data_type=None):
        """Create norm children for q_a_layernorm and kv_a_layernorm."""
        if q_a_norm is not None:
            self._add_norm_child('q_a_layernorm', q_a_norm,
                                 data_type=data_type)
        if kv_a_norm is not None:
            self._add_norm_child('kv_a_layernorm', kv_a_norm,
                                 data_type=data_type)

    # ------------------------------------------------------------------
    # MLA fold+pad (moved from glm4_moe_lite_spec)
    # ------------------------------------------------------------------

    def _fold_and_pad(self, linears: dict[str, Linear]):
        """Fold kv_b_proj into q_b_proj and wo, then pad wo.

        Weight tensors are temporarily transposed to HF layout [out, in]
        for the fold arithmetic, then transposed back to TM layout [in, out].
        """
        # Temporarily convert weight tensors from TM [in, out] to HF [out, in].
        for lin in linears.values():
            for k in list(lin.tensors.keys()):
                t = lin.tensors[k]
                if t.dim() >= 2:
                    lin.tensors[k] = t.t().contiguous()
        try:
            self._fold_and_pad_hf(linears)
        finally:
            # Convert weight tensors back from HF [out, in] to TM [in, out].
            for lin in linears.values():
                for k in list(lin.tensors.keys()):
                    t = lin.tensors[k]
                    if t.dim() >= 2:
                        lin.tensors[k] = t.t().contiguous()

    def _fold_and_pad_hf(self, linears: dict[str, Linear]):
        """Inner fold logic; expects all weight tensors in HF layout [out, in]."""
        cfg = self.config
        head_num = cfg.head_num
        qk_rope_dim = cfg.qk_rope_dim
        qk_nope_dim = cfg.qk_nope_dim
        kv_lora_rank = cfg.kv_lora_rank
        v_head_dim = cfg.v_head_dim
        size_per_head = cfg.size_per_head

        q_b_lin = linears.get("q_b_proj")
        kv_b_lin = linears.pop("kv_b_proj", None)
        o_lin = linears.get("wo")

        if q_b_lin is not None and kv_b_lin is not None and o_lin is not None:
            q_b = q_b_lin.tensors.get("weight")
            kv_b = kv_b_lin.tensors.get("weight")
            o = o_lin.tensors.get("weight")

            if (q_b is not None and kv_b is not None and o is not None
                    and torch.is_floating_point(q_b)
                    and torch.is_floating_point(kv_b)):
                orig_q_head_dim = q_b.size(0) // head_num
                orig_qk_nope_dim = orig_q_head_dim - qk_rope_dim
                orig_v_head_dim = o.size(1) // head_num
                target_nope_dim = size_per_head - qk_rope_dim

                if (orig_qk_nope_dim != target_nope_dim
                        or orig_v_head_dim != v_head_dim):
                    # Split kv_b into kc and vc
                    kv_b_per_head = kv_b.reshape(
                        head_num, orig_qk_nope_dim + orig_v_head_dim,
                        kv_lora_rank)
                    kc_w = kv_b_per_head[:, :orig_qk_nope_dim, :]
                    vc_w = kv_b_per_head[:, orig_qk_nope_dim:, :]

                    # Fold kc into q_b_proj
                    q_b_per_head = q_b.reshape(
                        head_num, orig_q_head_dim, q_b.size(1))
                    q_nope_w = q_b_per_head[:, :orig_qk_nope_dim, :]
                    q_rope_w = q_b_per_head[:, orig_qk_nope_dim:, :]
                    q_nope_expanded = torch.bmm(
                        kc_w.transpose(1, 2), q_nope_w)
                    q_b_folded = torch.cat(
                        [q_nope_expanded, q_rope_w], dim=1)
                    q_b_lin.tensors["weight"] = q_b_folded.reshape(
                        head_num * size_per_head, q_b.size(1))

                    # Fold vc into o_proj
                    o_per_head = o.reshape(
                        o.size(0), head_num, orig_v_head_dim)
                    o_folded = torch.bmm(
                        o_per_head.permute(1, 0, 2), vc_w)
                    o_lin.tensors["weight"] = o_folded.permute(
                        1, 0, 2).reshape(
                            o.size(0), head_num * kv_lora_rank)

        # Pad wo from [hidden, head_num*v_head_dim]
        #           to [hidden, head_num*size_per_head]
        if o_lin is not None:
            o_w = o_lin.tensors["weight"]
            cur_v = o_w.size(1) // head_num
            if cur_v < size_per_head:
                o_w = o_w.reshape(o_w.size(0), head_num, cur_v)
                o_w = torch.nn.functional.pad(
                    o_w, (size_per_head - cur_v, 0, 0, 0, 0, 0))
                o_lin.tensors["weight"] = o_w.reshape(
                    o_w.size(0), head_num * size_per_head)
```

- [ ] **Step 3: Verify import**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.builder import MLABuilder; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder.py
git commit -m "feat(deploy): add MLABuilder with fold+pad and projection commits"
```

---

### Task 3: Update glm4_moe_lite_spec.py to use MLABuilder

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

- [ ] **Step 1: Update imports**

Replace the import block at lines 16-25:

Old:
```python
from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, TextModelBuilder,
    _act_type_id,
)
from ..linear import Linear
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig,
    ModuleListConfig, MoeConfig,
)
```

New:
```python
from ..builder import (
    DecoderLayerBuilder, FfnBuilder,
    MLABuilder, MoeBuilder, ModuleListBuilder, TextModelBuilder,
    _act_type_id,
)
from ..linear import Linear
from ..module_configs import (
    DecoderLayerConfig, FfnConfig,
    MLAConfig, ModuleListConfig, MoeConfig,
)
```

- [ ] **Step 2: Replace attn() method**

Replace the `attn()` method at lines 65-107 with:

```python
    def attn(self, pfx, layer):
        """Return MLABuilder for MLA attention."""
        mc = self._mc
        tp = self._attn_tp
        dtype = self._cpp_dtype()
        cfg = self.cfg

        mla_cfg = MLAConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype, window_size=-1,
            qk_nope_dim=cfg['qk_nope_head_dim'])
        builder = MLABuilder(mla_cfg, self._contexts,
                             tp=tp, ranks=self._attn_ranks)

        q_b = self._linear(f"{pfx}.q_b_proj") or self._linear(f"{pfx}.q_proj")
        builder.add_projections(
            q_a_proj=self._linear(f"{pfx}.q_a_proj"),
            q_b_proj=q_b,
            kv_a_proj=self._linear(f"{pfx}.kv_a_proj_with_mqa"),
            kv_b_proj=self._linear(f"{pfx}.kv_b_proj"),
            wo=self._linear(f"{pfx}.o_proj"),
        )
        builder.add_norms(
            q_a_norm=self._get(f"{pfx}.q_a_layernorm.weight"),
            kv_a_norm=self._get(f"{pfx}.kv_a_layernorm.weight"),
            data_type=dtype,
        )
        return builder
```

- [ ] **Step 3: Remove _mla_fold_and_pad and _mla_fold_and_pad_hf methods**

Delete lines 203-289 (the `_mla_fold_and_pad` and `_mla_fold_and_pad_hf` methods, plus the section comment). These are now in `MLABuilder`.

- [ ] **Step 4: Verify import**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.source_model.glm4_moe_lite_spec import Glm4MoeLiteSpec; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(deploy): use MLABuilder in glm4_moe_lite_spec, remove inline MLA logic"
```

---

### Task 4: Test MLA model end-to-end

**Files:**
- No file changes; verification only

- [ ] **Step 1: Check GPU availability**

Run: use `get_gpu_usage` MCP tool to find an empty GPU.

- [ ] **Step 2: Find the cached MLA model**

Run: use `list_models` MCP tool to find a GLM-4 MoE Lite model (or other MLA model) in the cache.

- [ ] **Step 3: Run the model test**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python build/toy_example.py <model_path>`
(Adjust command based on the actual test script and model path.)

Expected: Model loads and produces coherent text with 128+ tokens. No errors during weight loading.

- [ ] **Step 4: Verify output quality**

Read the model output. It must contain meaningful human words relevant to the test prompt. Gibberish indicates a regression in the fold+pad or weight commit logic.
