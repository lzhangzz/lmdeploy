# Merged Attention module design

Type: grilling
Status: resolved

## Question

Design the merged Attention module: per-layer module instances versus today's all-layers `UnifiedAttentionLayer` plus a single `CacheRegistry`; where KV cache category registration and offsets live (cite `contracts.category-registration`, `ownership.module-cache`); the MLA branch; where rope parameters live (`init_rope_kernel_param` currently sits in the weight file but serves the executor). Decide GatedDeltaNet's place: its own module following the Attention pattern, with its recurrent/conv state cache contracts (`contracts.gated-deltanet`) preserved.

## Answer

Design drafted at [`drafts/attention-design.md`](../drafts/attention-design.md). Decisions:

- **Per-layer `Attention` modules.** Per-layer state = weights + registered KV byte offset + rope params (`RopeKernelParam` computed in the ctor, absorbing `init_rope_kernel_param`). The per-phase `AttentionData` is batch state, not layer state — `block_ptrs` are per-request cache-object bases shared by all layers; layers apply their own `prefix_offset_` (addressing identical to today's `cache_block_offset`).
- **Registration is a dedicated `register_cache(CacheRegistry&)` pass** after the Python build, before category finalization — rejecting the registry-on-ExecContext stretch (ExecContext stays pure execution services per the context contract). Each module registers its own per-layer bytes; accumulated layout equals today's single L-sized registration, so cache layout and checkpoints are unchanged. Canonical pass order: prepare → register_cache → finalize → bind → first Run.
- **Per-phase batch data in the attention kind pool**: one `AttentionData`-shaped `AttentionBatch` per phase per rank, filled at kSetup, read by every layer; the fill flow is mapped in BatchOp and decoder mapping.
- **Workspaces pooled** (contract default): `partial_O_`/`partial_ML_`/`split_cnt_` and reusable pinned staging from the attention kind pool; aux streams/events factory-made per module; CP post-context from `xc_->comm()`; warm-up noise-fill under `env.warm_up()`.
- **MLA is an internal branch** of `Attention::Forward` (`is_mla()`).
- **GDN is its own `DeltaNet` module** on the same pattern: conv/recurrent registration in `register_cache` (checkpoint category), `contracts.gated-deltanet` semantics preserved (per-layer conv offsets, recurrent composite parts, TP×CP folding).
- **Naming**: classes `Attention`/`DeltaNet`, `type()` frozen to `"AttentionWeight"`/`"DeltaNetWeight"` (Linear/Ffn/Moe pattern).
