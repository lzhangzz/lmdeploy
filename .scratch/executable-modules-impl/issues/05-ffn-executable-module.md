# 05: Ffn executable module

**What to build:** The FFN weight module becomes the executable Ffn: one forward covering the whole feed-forward — fused and separate projections, the FP8 fused-activation path, activation, and down-projection — dense and grouped (routing-table) variants — with the branching internal. The orchestration block that exists three times today collapses into this single body. The dense FFN layer class retires; the decoder loop and the shared-expert call site flip to module forwards.

**Blocked by:** 03 (Linear executable module).

**Status:** resolved

- [x] Ffn forward covers fused/separate projections, FP8 fused-activation, activation, and down-projection for dense and grouped inputs, branching internal
- [x] The copy-pasted orchestration exists in exactly one place
- [x] Dense FFN layer class retired; decoder and shared-expert call sites use module forwards
- [x] Build green; dense-model gate portion passes (meaningful responses, perf within gate)

## Answer

**Shape landed** (commit "Make Ffn the executable FFN module (ticket 05)"):

- `FfnWeight` → `Ffn` (`models/ffn_weight.{h,cc}` renamed to `ffn.{h,cc}`); registry string stays
  `"FfnWeight"` via `TM_MODULE_REGISTER_AS`; config, children (w1/w3/w2/w1w3), params unchanged.
- `Ffn::Forward(const Tensor&, Ref<Tensor>)` — dense; `Ffn::Forward(const Tensor&, const
  GroupedRoute&, Ref<Tensor>)` — grouped via the new `GroupedRoute{indices, offsets, f2E,
  num_valid_tokens}`. Both bodies hold the fused-vs-separate, FP8-fused-SiLU, activation-variant,
  and w2-scales branching — the single home of the orchestration that previously lived verbatim
  in `LlamaFfnLayer.cc` and twice in `moe_ffn_layer.cc` (both impls' blocks collapsed to one
  `block.Forward(input, GroupedRoute{...}, temp)` each). `bind` is the default recursive walk:
  Ffn holds no engine state; its GEMMs ride the Linear children.
- `LlamaFfnLayer.{h,cc}` deleted; `unified_decoder` calls `feed_forward->Forward(...)`
  directly (member + ffn_weights vector removed); both shared-expert sites call
  `moe.shared->Forward(...)` (the `MoeFfnLayerImpl::ffn_` member and its ctx plumbing removed;
  the now write-only `layer_id` dropped from `MoeFfnLayer::ForwardParam`).
- Reviewer-verified faithful transcription of all three retired blocks (slice arithmetic,
  activation variants, scale handling identical); only drift is debug labels losing the
  layer-id suffix.

**Gate:** build green; all four gate models end-to-end with meaningful ≥250-token responses
(Qwen3-8B dense, Qwen3-8B-AWQ, Qwen3-30B-A3B grouped, Qwen3.5-35B-A3B-FP8 grouped FP8
fused-SiLU path); bs=32 perf 2597.6 (Qwen3-8B, 99.0% of recorded baseline 2625.1) and 1559.5
(Qwen3-30B-A3B, 101.4% of 1537.6). Builders untouched (no lmdeploy/ changes).
