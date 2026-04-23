# Unify MoE Gate Loading via `self._linear`

Date: 2026-04-23

## Summary

Replace the manual `Linear(...)` construction used to load MoE gate/router weights across all 4 MoE specs with the existing `self._linear(pfx)` helper on `TextModelSpec`. Net removal of ~40 lines and 5 copy-paste sites, with no behavioral change.

## Motivation

Each MoE spec's `moe()` method contains a ~9-line block that fetches the gate weight (and optional bias) from `self.params`, transposes the 2D weight, and wraps it in a `Linear` with `TRIVIAL_FORMAT`:

```python
dtype = self._cpp_dtype()
gate_w = self._get(f'{pfx}.gate.weight')          # or .router.weight
gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
tensors = {'weight': gate_w}
gate_bias = self._get(f'{pfx}.gate.bias')
if gate_bias is not None:
    tensors['bias'] = gate_bias
m.add_gate('gate', Linear(
    tensors,
    weight_format=TRIVIAL_FORMAT,
    data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype()),
), model_dtype=dtype)
```

This block appears 5 times (once per spec, twice in `qwen3_5_spec.py` for `gate` + `shared_gate`). It is a reimplementation of `self._linear(pfx)` / `build_linear(...)`, which already handles every one of these steps — and `build_linear`'s docstring even cites this exact case ("fallback to `TRIVIAL_FORMAT` when the expected quantized tensors are absent, e.g. a gate/router linear in a quantized model").

## Equivalence

| Step | Manual block | `self._linear(pfx)` |
|---|---|---|
| Fetch `.weight` + optional `.bias` | `self._get(...)` twice + if-branch | `build_linear` probes trivial suffix map `{.weight, .bias}` |
| Transpose 2D weight | `w.t() if w.dim() > 1 else w` | `_normalize_trivial` does `x.t()` when `x.dim() >= 2` |
| Pick `TRIVIAL_FORMAT` for the gate in a quantized model | hard-coded | `build_linear` falls back when the model's `weight_format.accepts()` rejects (gates never have `.qweight`/`.scales`/etc.) |
| `data_format` | `TRIVIAL_FORMAT.make_data_format(self._cpp_dtype())` | `fmt.make_data_format(data_type)` — same once fallback selects trivial |

Notes:
- The `if w.dim() > 1` guard in the manual code is dead: `nn.Linear.weight` is always 2D.
- Manual code keeps the weight tensor on CPU; `_linear` moves it to CUDA via `_normalize_trivial`. `_commit_linear` handles both — no correctness change, just earlier transfer.
- `model_dtype=self._cpp_dtype()` passed to `m.add_gate(...)` is preserved on the call — still needed for the BF16-weights-in-FP16-model case documented on `_commit_linear.model_dtype`.

## Changes

All 4 files in `lmdeploy/turbomind/deploy/source_model/`. Each gate-wrapping block collapses to a single `m.add_gate(...)` call.

### `qwen3_spec.py` — `moe()` (lines 193-198)

Before:

```python
gate_w = self._get(f'{pfx}.gate.weight')
gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
m.add_gate('gate', Linear({'weight': gate_w},
                          weight_format=TRIVIAL_FORMAT,
                          data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype())),
           model_dtype=self._cpp_dtype())
```

After:

```python
m.add_gate('gate', self._linear(f'{pfx}.gate'),
           model_dtype=self._cpp_dtype())
```

### `glm4_moe_lite_spec.py` — `moe()` (lines 236-247)

Before (same shape as `qwen3_spec` block, `{pfx}.gate` prefix, plus a defensive bias branch that never fires for GLM):

```python
dtype = self._cpp_dtype()
gate_w = self._get(f'{pfx}.gate.weight')
gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
tensors = {'weight': gate_w}
gate_bias = self._get(f'{pfx}.gate.bias')
if gate_bias is not None:
    tensors['bias'] = gate_bias
m.add_gate('gate', Linear(
    tensors,
    weight_format=TRIVIAL_FORMAT,
    data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype()),
), model_dtype=dtype)
```

After:

```python
m.add_gate('gate', self._linear(f'{pfx}.gate'),
           model_dtype=self._cpp_dtype())
```

The subsequent `m.add_param('score_correction_bias', ...)` line (GLM's `e_score_correction_bias`) is **unchanged** — it is a separate MoE parameter, not a gate input.

### `gpt_oss_spec.py` — `moe()` (lines 198-209)

Before: same 9-line shape, `{pfx}.router` prefix, real bias present.

After:

```python
m.add_gate('gate', self._linear(f'{pfx}.router'),
           model_dtype=self._cpp_dtype())
```

The `.bias` is picked up automatically by the trivial suffix map.

### `qwen3_5_spec.py` — `moe()` (lines 269-282)

Two gates. Before: two back-to-back blocks (6-7 lines each) for `{pfx}.gate.weight` and `{pfx}.shared_expert_gate.weight`, neither with bias.

After:

```python
m.add_gate('gate', self._linear(f'{pfx}.gate'),
           model_dtype=self._cpp_dtype())
m.add_gate('shared_gate', self._linear(f'{pfx}.shared_expert_gate'),
           model_dtype=self._cpp_dtype())
```

### Import cleanup

After the change, `TRIVIAL_FORMAT` is unused in all 4 spec files. `Linear` is still used in `gpt_oss_spec.py._deinterleave` and `qwen3_5_spec.py._packed_moe_expert_indexed` (packed-expert helpers) — keep those imports; drop the unused `TRIVIAL_FORMAT` import from all 4 files, and drop `Linear` from `qwen3_spec.py` and `glm4_moe_lite_spec.py` (where it is no longer referenced).

## Files affected

- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` — drop gate block; drop `TRIVIAL_FORMAT`, `Linear` imports
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — drop two gate blocks; drop `TRIVIAL_FORMAT` import (keep `Linear`)
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` — drop gate block; drop `TRIVIAL_FORMAT`, `Linear` imports
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` — drop gate block; drop `TRIVIAL_FORMAT` import (keep `Linear`)

No changes to `TextModelSpec`, `MoeBuilder`, `build_linear`, or any builder code.

## Verification

- Smoke-test with `scripts/test_turbomind_model.py` on one model per spec:
  - `qwen3-moe` (exercises `qwen3_spec` MoE path)
  - `qwen3_5-moe` (exercises both `gate` and `shared_gate`)
  - `glm4-moe-lite` (GLM-4.7-Flash)
  - `gpt-oss` (router with bias)
- Each model must respond with meaningful text ≥128 tokens to confirm no weight-loading regression.
- For at least one model, also verify in a **quantized** format (e.g. AWQ/FP8/MXFP4 variant) to confirm the trivial-fallback path works as expected for gates in quantized checkpoints.

## Non-goals

- Not touching the GLM `score_correction_bias` `m.add_param(...)` call — scope confirmed in brainstorming.
- Not refactoring the broader `moe()` method skeleton (cfg.clone → MoeBuilder → expert loop) — scope confirmed in brainstorming.
- No new helper method on `TextModelSpec` or `MoeBuilder` — the existing `self._linear` is sufficient.
