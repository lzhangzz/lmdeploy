# Merge Qwen3.5 Input Model Classes

Date: 2026-04-17

## Problem

`lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` defines two near-identical
`BaseInputModel` subclasses — `Qwen3_5InputModel` (registered as `qwen3_5`) and
`Qwen3_5MoeInputModel` (registered as `qwen3_5-moe`). They share the same
`_spec_class` (`Qwen3_5Spec`) and the same `_layer_pattern`, and both build
`model_info()` from the same `_qwen35_model_info_base(cfg)` helper.

The sibling `qwen3_spec.py` already uses the pattern we want: a single
`Qwen3InputModel` class with stacked registration decorators for both `qwen3`
and `qwen3-moe`, and a `model_info()` that conditionally adds MoE fields when
`num_experts > 0`. We want Qwen3.5 to follow the same pattern.

## Goals

- Collapse the two `qwen3_5_spec.py` input-model classes into one.
- Mirror the structure of `Qwen3InputModel` (stacked decorators,
  conditional MoE block inside `model_info()`).
- Preserve all behavior required by the TurboMind deploy pipeline for both
  dense and MoE Qwen3.5 checkpoints.

## Non-goals

- No changes to `Qwen3_5Spec`, `_qwen35_model_info_base`, or
  `map_packed_qwen35_experts`.
- No changes to `qwen3_spec.py` or other source models.
- No refactor of `BaseInputModel` or the `_loader_mappings` mechanism.

## Design

Replace lines 366-411 of `qwen3_5_spec.py` with a single class:

```python
@INPUT_MODELS.register_module(name='qwen3_5-moe')
@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5InputModel(BaseInputModel):
    """Input model for Qwen3.5 (dense and MoE)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Qwen3_5Spec
    _loader_mappings = [map_packed_qwen35_experts]

    def model_info(self) -> dict:
        cfg = self.model_config
        info = _qwen35_model_info_base(cfg)
        n_experts = cfg.get('num_experts', 0)
        if n_experts:
            info.update(
                expert_num=n_experts,
                expert_inter_size=cfg['moe_intermediate_size'],
                experts_per_token=cfg['num_experts_per_tok'],
                inter_size=cfg.get('shared_expert_intermediate_size', 0),
                moe_shared_gate=True,
                scoring_func='softmax',
                norm_topk_prob=True,
            )
        return info
```

### Key decisions

1. **`_loader_mappings` applied unconditionally.**
   `map_packed_qwen35_experts` is a regex substitution matching
   `mlp\.experts\.(?:gate_up|down)_proj$`. Dense Qwen3.5 checkpoints contain
   no such keys, so applying the mapping is a no-op there. Keeping it on the
   single class keeps one source of truth for both registrations.

2. **MoE fields emitted only when `num_experts > 0`.**
   The previous dense `model_info()` unconditionally emitted
   `expert_num=0`, `expert_inter_size=0`, `experts_per_token=0`,
   `moe_shared_gate=True`, `scoring_func='softmax'`, and
   `norm_topk_prob=True`. Downstream code that cares about these keys is
   guarded by `expert_num > 0` anyway — this matches `Qwen3InputModel` exactly
   and removes misleading always-true MoE flags from dense output.

3. **Direct lookups for required MoE config keys.**
   When `num_experts` is set in a Qwen3.5 config, `moe_intermediate_size` and
   `num_experts_per_tok` are always present. Use `cfg[...]` instead of
   `cfg.get(..., 0)` so a malformed config fails loudly rather than silently
   producing zeros. `shared_expert_intermediate_size` keeps its
   `.get(..., 0)` default since we haven't verified it's mandatory.

4. **Drop the dense variant's `shared_expert_intermediate_size` branch.**
   It was dead code: a dense Qwen3.5 config doesn't set that key, so the
   branch never fired in practice. With the `num_experts > 0` guard in place
   the logic moves cleanly into the MoE branch.

5. **Decorator order matches `qwen3_spec.py`** — `-moe` outer, base inner.
   Purely stylistic.

## Files touched

- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — replace
  lines 366-411 with the merged class above.

## Behavioral diff

For `qwen3_5-moe` checkpoints: identical output, except that
`expert_inter_size` and `experts_per_token` become required keys in the
config (they already were in practice).

For `qwen3_5` dense checkpoints: `model_info()` no longer contains the five
zero/constant MoE keys (`expert_num`, `expert_inter_size`,
`experts_per_token`, `moe_shared_gate`, `scoring_func`, `norm_topk_prob`).
Any downstream consumer reading these unconditionally on a dense model would
break — but the sibling `Qwen3InputModel` has shipped with the same
conditional pattern, so the pipeline already tolerates their absence.

## Testing

No unit tests exist for this module. Verification is a full-pipeline run via
`scripts/test_turbomind_model.py` (handled by the `turbomind-tester` agent)
on:

- A dense Qwen3.5 checkpoint.
- A Qwen3.5-MoE checkpoint (exercises both the MoE branch and
  `map_packed_qwen35_experts`).

Pass criterion: both runs produce outputs matching pre-change behavior.
