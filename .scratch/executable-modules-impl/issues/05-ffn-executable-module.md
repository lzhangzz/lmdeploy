# 05: Ffn executable module

**What to build:** The FFN weight module becomes the executable Ffn: one forward covering the whole feed-forward — fused and separate projections, the FP8 fused-activation path, dense and grouped (routing-table) variants — with the branching internal. The orchestration block that exists three times today collapses into this single body. The dense FFN layer class retires; the decoder loop and the shared-expert call site flip to module forwards.

**Blocked by:** 03 (Linear executable module).

**Status:** ready-for-agent

- [ ] Ffn forward covers fused/separate projections, FP8 fused-activation, activation, and down-projection for dense and grouped inputs, branching internal
- [ ] The copy-pasted orchestration exists in exactly one place
- [ ] Dense FFN layer class retired; decoder and shared-expert call sites use module forwards
- [ ] Build green; dense-model gate portion passes (meaningful responses, perf within gate)
