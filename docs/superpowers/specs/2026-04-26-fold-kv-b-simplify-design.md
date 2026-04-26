# fold_kv_b simplification

## Summary

Replace shape-derived dimensions with config fields, and replace permute+bmm with einsum that keeps the natural `[R, H, ...]` tensor layout, eliminating all permutes.

## Changes

1. **Config fields replace shape arithmetic** — `orig_q_head_dim`, `orig_qk_nope_dim`, `orig_v_head_dim` are replaced by `cfg.qk_nope_dim`, `cfg.v_head_dim`, `cfg.kv_lora_rank`, which already carry these values.

2. **einsum replaces permute+bmm** — The per-head matmuls `q_nope @ kc^T` and `vc @ wo` use einsum subscripts that preserve the natural `[R, H, ...]` layout:
   - `"ihp,jhp->ihj"` on `q_nope [R,H,P]` and `kc [R,H,P]` gives `[R,H,R]`, allowing direct concatenation with `q_rope [R,H,S]` — zero permutes.
   - `"rhv,hvn->hrn"` on `vc [R,H,V]` and `o_h [H,V,N]` gives `[H,R,N]`, matching the original head-major output order.

3. **torch.split replaces manual slicing** — `q_nope, q_rope = q_b_h.split([P, S], dim=-1)` and `kc, vc = kv_b_h.split([P, V], dim=-1)`.

## Invariants preserved

- Output tensor shapes identical to original
- Head-major ordering in both q_b and wo outputs (required for correct TP splitting)
- Linear weight_format and data_format passthrough unchanged
- No change to function signature or call site

## Risk

`torch.einsum` has slightly more overhead than `bmm`, but for MLA dimensions (kv_lora_rank ≤ 512, head_num ≤ 128, qk_nope_dim ≤ 128) this is a one-time weight transform during model loading — negligible.
