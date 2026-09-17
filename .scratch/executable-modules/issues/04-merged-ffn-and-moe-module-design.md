# Merged FFN and MoE module design

Type: grilling
Status: resolved

## Question

Design the merged FFN and MoE modules: the `Ffn` module (w1w3 fused versus w1/w3 separate, the FP8 fused-SiLU `inter_scales` path) and the `Moe` module (gate, shared expert, `LinkLinearExperts` block ownership, grouped GEMM indices/offsets). Eliminate the GEMM orchestration block copy-pasted three times (`LlamaFfnLayer.cc:51-90`, `MoeFfnDefaultImpl::Forward`, `MoeFfnA2AImpl::Forward`). Decide where routing (`invokeMoeGate_V2`/`NoAuxTC`) lives. Respect the hard constraints: checkpoints load as-is, no perf regression, engine contract meanings hold.

## Answer

Design drafted at [`drafts/ffn-moe-design.md`](../drafts/ffn-moe-design.md). Decisions:

- **Ffn owns the whole FFN**: `Ffn::Forward(input, output)` = w1w3 → activation → w2 with the FP8-fused-SiLU and w1/w3-separate branches internal; a grouped overload takes a `GroupedRoute` (indices/offsets/f2E/num_valid_tokens). The three verbatim copies of the orchestration block collapse into this one body. `LlamaFfnLayer` retires.
- **Moe owns routing and transport code**: `Moe::Forward(input, output, token_mask)` = gate → route (`invokeMoeGate_V2`/`NoAuxTC`; `SampleUniform` under `env.warm_up()`) → gather → `block_->Forward(route)` → scatter/combine → shared fold (`shared->Forward`, `shared_gate->Forward`). `MoeFfnLayer` and both impls retire; default-vs-DeepEP is an internal branch.
- **Transport/routing state owned per module instance** (explicit user decision, deviating from the context contract's per-kind pools): each per-layer Moe owns its `TokenDispatcher`, routing buffers, and factory-made scales-copy stream. Recorded consequences: dispatcher memory and JIT compile multiply by layer count; the context contract draft carries the amendment; Migration order and verification gates gains a MoE-stage memory-footprint and startup-cost gate.
- **Grouped block mechanism unchanged**: `experts` ModuleList stays the Python surface; `prepare()` builds the internal `block_` via `LinkLinearExperts`; `AliasRouted`/`meta_pack_` aliasing unchanged.
- **Naming**: classes `Ffn`/`Moe`, `type()` frozen to `"FfnWeight"`/`"MoeWeight"` (Linear pattern).
