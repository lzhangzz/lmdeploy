# MLABuilder Design

## Summary

Extract MLA (Multi-head Latent Attention) logic from `glm4_moe_lite_spec.py` into a dedicated `MLABuilder` class in `builder.py`, with its own `MLAConfig` in `module_configs.py`. This follows the same pattern as `DeltaNetBuilder` — a purpose-specific builder that encapsulates weight transforms, TP sharding, and C++ module commit behind a clean API.

## Motivation

Currently, MLA attention in `glm4_moe_lite_spec.py` uses `AttentionBuilder` directly with the generic `add_linear()` escape hatch for each projection. The MLA-specific fold+pad transform and norm child binding are done inline in the spec. This works but means:

1. The spec is doing builder-level work (fold+pad is a weight transform, not a spec concern)
2. MLA logic can't be reused across specs without duplication
3. No clear API boundary between "what to read" (spec) and "how to transform and commit" (builder)

## Design

### MLAConfig (module_configs.py)

A new dataclass holding MLA-specific geometry. Produces the same `_tm.AttentionConfig` on the C++ side (MLA already uses `AttentionWeight` as the C++ module type).

```python
@dataclass
class MLAConfig:
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
        # MLA-specific fields defaulted
        cfg.has_bias = False
        cfg.qk_norm = False
        cfg.attn_sink = False
        cfg.attn_output_gate = False
        return cfg
```

### MLABuilder (builder.py)

Extends `Builder` directly. Owns the MLA fold+pad transform, projection commits, and norm children.

**TP rules:**
```python
_MLA_TP_RULES: dict[str, dict] = {
    "q_a_proj":  dict(split_side=SplitSide.OUTPUT),
    "q_b_proj":  dict(split_side=SplitSide.OUTPUT),
    "kv_a_proj": dict(split_side=SplitSide.OUTPUT),
    "wo":        dict(split_side=SplitSide.INPUT),
}
```

`kv_b_proj` is NOT in the table — it is consumed entirely by the fold logic and never committed as a standalone weight.

**Methods:**

#### `add_projections(self, *, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, wo)`

Accepts the five MLA Linear objects as keyword arguments:

1. Applies fold+pad transform (the logic currently in `_mla_fold_and_pad` / `_mla_fold_and_pad_hf`)
2. Commits each resulting projection via `_commit_linear()` using `_MLA_TP_RULES`

After the fold, `kv_b_proj` is consumed (its information is absorbed into `q_b_proj` and `wo`).

#### `add_norms(self, *, q_a_norm, kv_a_norm, data_type=None)`

Creates norm children for `q_a_layernorm` and `kv_a_layernorm`. Skips `None` tensors.

### Spec Usage (glm4_moe_lite_spec.py)

The spec's `attn()` method becomes a simple read-and-delegate:

```python
def attn(self, pfx, layer):
    mc = self._mc
    tp = self._attn_tp
    dtype = self._cpp_dtype()
    cfg = self.cfg  # HF config dict

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

The `_mla_fold_and_pad()` and `_mla_fold_and_pad_hf()` methods are removed from the spec entirely.

## Files Changed

| File | Change |
|------|--------|
| `lmdeploy/turbomind/deploy/module_configs.py` | Add `MLAConfig` dataclass |
| `lmdeploy/turbomind/deploy/builder.py` | Add `_MLA_TP_RULES` + `MLABuilder` class |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Simplify `attn()` to use `MLABuilder`, remove `_mla_fold_and_pad*` methods |

No C++ changes — MLA already uses `AttentionWeight` on the C++ side.

## Testing

Existing GLM-4 MoE Lite model test covers MLA end-to-end. This is a pure refactoring with no behavioral change. Verify by testing the MLA model and confirming output matches expectations.
