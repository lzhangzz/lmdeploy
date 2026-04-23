# Unify Packed MoE Expert Handling

Date: 2026-04-23

## Summary

Replace the two ad-hoc packed-MoE-expert helper chains in `qwen3_5_spec.py`
and `gpt_oss_spec.py` with a single free function `read_packed_moe_expert`
in `source_model/utils.py`. The function parameterizes the two variation
axes that distinguish the two specs — gate_up split scheme (contiguous vs.
stride-2 interleaved) and trivial-format layout quirk (`[n_experts, out,
in]` vs. `[n_experts, in, out]`) — via two bool kwargs. Net ~22 lines
removed and, more importantly, one authoritative implementation of the
split + layout logic shared by both specs.

## Motivation

Packed MoE experts are stored in two tensors per layer:

- `experts.gate_up_proj` — shape `[n_experts, *, *]`, contains a fused
  gate+up projection for every expert.
- `experts.down_proj`   — shape `[n_experts, *, *]`, contains the down
  projection for every expert.

Both `qwen3_5_spec.py` and `gpt_oss_spec.py` need to:

1. Read one expert's slice via `build_linear(index=expert_idx, ...)`.
2. Split the fused gate_up into two separate w1 / w3 Linears.
3. Build an `FfnBuilder` with `fused_moe=True` and `fuse_silu=False`.

The two specs differ on exactly two axes:

| Axis                 | qwen3.5 packed             | gpt-oss packed                                    |
|----------------------|----------------------------|---------------------------------------------------|
| gate_up split scheme | contiguous `[:half]`/`[half:]` | stride-2 interleaved `[::2]`/`[1::2]`         |
| trivial layout       | `[n_experts, out, in]` (HF standard) | `[n_experts, in, out]` (needs extra `.t()`) |

Today these differences are expressed as two separate multi-method code
paths:

- **qwen3_5_spec.py**: `_moe_expert_ffn` (unpacked→packed fallback, 8
  lines) + `_packed_moe_expert_indexed` (read + contiguous split + build,
  33 lines).
- **gpt_oss_spec.py**: `_read_packed_expert` (read + layout fixup, 11
  lines) + `_deinterleave` (interleaved split, 11 lines) +
  `_packed_expert_ffn` (orchestrate + build, 22 lines).

~77 lines of near-duplicate logic (33 in qwen3_5's `_packed_moe_expert_indexed`
+ 44 across gpt-oss's three helpers), and adding any third packed-expert
spec would mean pasting another full copy.

## Design

### New helper in `source_model/utils.py`

```python
def read_packed_moe_expert(
    params: dict,
    gate_up_pfx: str,
    down_pfx: str,
    expert_idx: int,
    *,
    data_type,
    weight_format,
    interleaved: bool = False,
    trans: bool = False,
) -> tuple[Linear, Linear, Linear]:
    """Read one packed MoE expert's fused gate_up + down and split into
    (w1, w2, w3) Linears in TM layout.

    ``gate_up_pfx`` and ``down_pfx`` are the full prefixes to the two
    packed tensors (e.g. ``'model.layers.5.mlp.experts.gate_up_proj'``).
    The caller — not this helper — composes these strings.

    Parameters
    ----------
    interleaved : bool
        Split scheme for the fused gate_up output dim.
        ``False`` → contiguous ``[..., :half]`` / ``[..., half:]`` (qwen3.5).
        ``True``  → stride-2 interleaved ``[..., ::2]`` / ``[..., 1::2]`` (gpt-oss).
    trans : bool
        For trivial-format checkpoints that store the packed tensor in
        ``[n_experts, in, out]`` layout (gpt-oss), transposes the 2D weight
        to undo the HF-to-TM transpose applied by ``_normalize_trivial``.
        Only applies to the ``weight`` kind (bias is 1-D; quantized kinds
        use their own format-specific normalizer).
    """
    gate_up = build_linear(params, gate_up_pfx, index=expert_idx,
                           data_type=data_type, weight_format=weight_format)
    down    = build_linear(params, down_pfx,    index=expert_idx,
                           data_type=data_type, weight_format=weight_format)

    if trans:
        for lin in (gate_up, down):
            if lin.weight_format.name == 'trivial':
                w = lin.tensors.get('weight')
                if w is not None and w.dim() == 2:
                    lin.tensors['weight'] = w.t().contiguous()

    w1_t: dict[str, torch.Tensor] = {}
    w3_t: dict[str, torch.Tensor] = {}
    for kind, t in gate_up.tensors.items():
        if interleaved:
            w1_t[kind] = t[..., ::2].contiguous()
            w3_t[kind] = t[..., 1::2].contiguous()
        else:
            half = t.shape[-1] // 2
            w1_t[kind] = t[..., :half].contiguous()
            w3_t[kind] = t[..., half:].contiguous()
    w1 = Linear(tensors=w1_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)
    w3 = Linear(tensors=w3_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)
    return w1, down, w3
```

Notes:

- Return type is `tuple[Linear, Linear, Linear]` — no `Optional`. If
  `build_linear` returns `None` (no tensors at all at the given prefix)
  the next line raises `AttributeError` on `.tensors`. In a packed-expert
  code path this is a checkpoint bug, not a condition, so a crash is
  preferable to a defensive `None`-return that would only push the
  failure further away.
- Two prefix arguments (`gate_up_pfx`, `down_pfx`), each a full prefix
  into the checkpoint namespace. The helper concatenates nothing.
- `trans` applies symmetrically to both reads (`gate_up` and `down`)
  because gpt-oss stores both in the same non-standard layout. It only
  affects the trivial format — quantized formats have their own
  normalizers that already know the right layout.

### `qwen3_5_spec.py` call sites

Delete `_moe_expert_ffn` (lines 283-290) and `_packed_moe_expert_indexed`
(lines 292-324). Replace with:

```python
def _packed_moe_ffn(self, mlp_pfx, expert_idx, inter_size):
    w1, w2, w3 = read_packed_moe_expert(
        self.params,
        f'{mlp_pfx}.experts.gate_up_proj',
        f'{mlp_pfx}.experts.down_proj',
        expert_idx,
        data_type=self._cpp_dtype(),
        weight_format=self._weight_format,
    )
    cfg = self._ffn_cfg.clone()
    cfg.inter_size = inter_size
    cfg.fuse_silu  = False
    cfg.fused_moe  = True
    m = FfnBuilder(cfg, self._contexts,
                   tp=self.engine_cfg.mlp_tp_size,
                   ranks=self._mlp_ranks)
    m.add_ffn(w1, w2, w3)
    return m

def _moe_expert_ffn(self, mlp_pfx, layer, expert_idx, inter_size):
    expert_pfx = f'{mlp_pfx}.experts.{expert_idx}'
    return (self.ffn(expert_pfx, layer, inter_size=inter_size, fused_moe=True)
            or self._packed_moe_ffn(mlp_pfx, expert_idx, inter_size))
```

The `or`-fallback replaces the explicit `if result is not None: return
result` ladder. Semantics are identical: `self.ffn(...)` returns a
truthy `FfnBuilder` when any of the per-expert `gate_proj` / `up_proj` /
`down_proj` keys exist, else returns `None` and the packed fallback
fires. `moe()` itself is unchanged — the expert loop still calls
`self._moe_expert_ffn(pfx, layer, e, self._expert_inter_size)`.

Drop the `Linear` import — no longer referenced.

### `gpt_oss_spec.py` call sites

Delete `_read_packed_expert` (lines 224-234), `_deinterleave` (lines
236-246), `_packed_expert_ffn` (lines 248-269). Replace with:

```python
def _packed_moe_ffn(self, mlp_pfx, expert_idx, inter_size):
    w1, w2, w3 = read_packed_moe_expert(
        self.params,
        f'{mlp_pfx}.experts.gate_up_proj',
        f'{mlp_pfx}.experts.down_proj',
        expert_idx,
        data_type=self._cpp_dtype(),
        weight_format=self._weight_format,
        interleaved=True,
        trans=True,
    )
    cfg = self._ffn_cfg.clone()
    cfg.inter_size = inter_size
    cfg.fuse_silu  = False
    cfg.fused_moe  = True
    m = FfnBuilder(cfg, self._contexts,
                   tp=self.engine_cfg.mlp_tp_size,
                   ranks=self._mlp_ranks)
    m.add_ffn(w1, w2, w3)
    return m
```

`moe()`'s expert loop was:

```python
for e in range(self.num_experts(layer)):
    experts[str(e)] = self._packed_expert_ffn(
        f'{pfx}.experts.{e}', self._expert_inter_size)
```

becomes:

```python
for e in range(self.num_experts(layer)):
    experts[str(e)] = self._packed_moe_ffn(
        pfx, e, self._expert_inter_size)
```

The old call passed `f'{pfx}.experts.{e}'` as a single string and the
helper re-parsed the expert id via `rsplit('.', 1)`. The new call
passes the expert id explicitly — no string parsing.

Drop the `Linear` import (no longer referenced) and the `torch` import
(the remaining body uses no `torch` APIs directly).

## What disappears vs. what appears

**Disappears** (~83 lines):

- `qwen3_5_spec._moe_expert_ffn` (old form, 8 lines)
- `qwen3_5_spec._packed_moe_expert_indexed` (33 lines)
- `qwen3_5_spec` `Linear` import
- `gpt_oss_spec._read_packed_expert` (11 lines)
- `gpt_oss_spec._deinterleave` (11 lines, including `@staticmethod` and
  class-level dict-returning helper shape)
- `gpt_oss_spec._packed_expert_ffn` (22 lines)
- `gpt_oss_spec` `Linear` + `torch` imports

**Appears** (~61 lines):

- `source_model/utils.py`: `read_packed_moe_expert` (~30 lines).
- `qwen3_5_spec._packed_moe_ffn` (~14 lines) + `_moe_expert_ffn`
  rewritten as a 3-line `or`-fallback.
- `gpt_oss_spec._packed_moe_ffn` (~14 lines).

Net ~22 lines removed. More important than line count: one
authoritative implementation of the split-and-layout logic, and adding
a third packed-expert spec now requires only a 14-line `_packed_moe_ffn`
plus (if needed) a declaration of the two bool kwargs.

## Equivalence

The new code path produces bit-identical outputs to the current one.
Tracing both specs:

### qwen3.5 (contiguous split, trivial layout)

Today's `_packed_moe_expert_indexed`:
1. `build_linear(gate_up_proj, index=e, data_type, weight_format)` → `gate_up_lin`.
2. `build_linear(down_proj,   index=e, data_type, weight_format)` → `down_lin`.
3. For each kind, `t[..., :half]` → `gate`, `t[..., half:]` → `up`.
4. Wrap `gate` / `up` tensors into new `Linear` objects reusing
   `gate_up_lin.weight_format` and `gate_up_lin.data_format`.
5. Build `FfnBuilder` with `fuse_silu=False, fused_moe=True,
   inter_size=expert_inter`.

New `_packed_moe_ffn` via `read_packed_moe_expert(interleaved=False,
trans=False)`:
1-4 identical. 5 identical (stays in the spec).

### gpt-oss (interleaved split, `[in, out]` trivial layout)

Today's `_read_packed_expert` + `_deinterleave` + `_packed_expert_ffn`:
1. `build_linear(gate_up_proj, index=e, data_type, weight_format)` → lin.
2. If `lin.weight_format.name == 'trivial'` and `weight.dim() == 2`,
   apply `lin.tensors['weight'] = w.t().contiguous()`.
3. Same for `down_proj`.
4. For each kind in `gate_up.tensors`, `t[..., ::2]` → `gate`,
   `t[..., 1::2]` → `up`. Wrap into `Linear`.
5. Build `FfnBuilder` with `fuse_silu=False, fused_moe=True,
   inter_size=expert_inter`.

New `_packed_moe_ffn` via `read_packed_moe_expert(interleaved=True,
trans=True)`:
1-4 identical (the helper applies the same trans fixup loop over both
linears and the same stride-2 split when `interleaved=True`). 5
identical.

No change to `build_linear`, `_normalize_trivial`, `FfnBuilder`,
`add_ffn`, or any C++ path.

## Files affected

- `lmdeploy/turbomind/deploy/source_model/utils.py` — add
  `read_packed_moe_expert` (~30 lines). Also needs `Linear` and
  `build_linear` imports.
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — replace the
  two deleted methods with `_packed_moe_ffn` + the slim `_moe_expert_ffn`
  fallback; drop the `Linear` import.
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` — replace the
  three deleted methods with `_packed_moe_ffn`; update the expert loop
  in `moe()`; drop the `Linear` and `torch` imports.

## Non-goals

- Not changing `build_linear`, `WeightFormat`, or per-format
  normalizers. The `trans` kwarg is a spec-side compensation for the
  non-standard gpt-oss layout; moving that into the normalizer is a
  larger design (would need a per-call layout hint) that this change
  deliberately avoids.
- Not refactoring `self.ffn()` to accept pre-built Linears. The
  `FfnBuilder` ceremony stays duplicated across the two
  `_packed_moe_ffn` methods (5 lines each) because pulling it out would
  either (a) require spec state access via extra kwargs — reverts to
  the "fat util" shape explicitly rejected during brainstorming — or
  (b) require a base-class helper on `TextModelSpec`, also rejected.
- Not touching the gate / shared_gate loading. That was unified in a
  separate change (`2026-04-23-unify-moe-gate-loading-design.md`).
- Not touching qwen3_spec or glm4_moe_lite_spec. Neither uses packed
  experts; their per-expert paths already go through `self.ffn()`.
- Not changing qwen3.5's unpacked→packed fallback semantics. Callers
  today rely on the packed path being the active one for all shipping
  qwen3.5-MoE checkpoints; the fallback is kept in case an unpacked
  variant exists or is added.

## Verification

Per `AGENTS.md`, run `scripts/test_turbomind_model.py` with one model
per affected code path and confirm each produces ≥128 tokens of coherent
human text:

- **Qwen3.5-MoE** — exercises `interleaved=False, trans=False`
  (contiguous split, trivial BF16 layout).
- **gpt-oss with mxfp4 native** — exercises `interleaved=True` on the
  mxfp4 normalizer path (no `trans` fixup applies — `trans` only
  affects `trivial`).
- **gpt-oss with BF16 weights** (if a dequantized checkpoint is
  available) — exercises `interleaved=True, trans=True` on the trivial
  path.

Bit-for-bit equivalence by construction: the sequence `build_linear(index=e)
→ optional trans fixup → split by scheme → Linear wrap → FfnBuilder`
is exactly what each spec did before, extracted and parameterized.
