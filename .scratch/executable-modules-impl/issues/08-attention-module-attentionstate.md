# 08: Attention executable module with AttentionState, and contract-doc sync

**What to build:** Per-layer attention modules run projections, q/k norms, core attention (including the MLA branch and CP), with the batch-level per-phase data living in one shared AttentionState per rank — filled at setup under the version-stamp guard so the fill is order-independent — and each layer applying its own registered KV offset. The unified attention layer class retires. The engine contract document is updated in the same change per its own sync rule: class renames in the cache-content, unified-attention, and gated-deltanet sections; registration timing reworded from "during construction" to "during the registration pass"; the lifecycle paragraph added.

**Blocked by:** 07 (Make the cache-registration pass real).

**Status:** resolved

- [x] Per-layer attention modules registered with the prefix category; accumulated layout byte-identical to today's, so existing checkpoints and cache blocks are unchanged
- [x] One AttentionState per rank: per-phase batch slots with version-stamped fill, fp32 workspace, pinned staging, CP context, aux stream
- [x] Unified attention layer class retired and deleted
- [x] Contract-doc amendments land in the same change (renames, registration timing, lifecycle paragraph)
- [x] Cache-behavior gate variants pass: prefix caching enabled, checkpointing enabled, both with meaningful responses

## Answer

**Shape landed:**

- `AttentionWeight` → `Attention` (`models/attention_weight.{h,cc}` → `attention.{h,cc}`);
  registry string stays `"AttentionWeight"` via `TM_MODULE_REGISTER_AS`; macros
  `ATTENTION_CHILDREN`/`ATTENTION_PARAMS`. ViT users (internvit, qwenvit) renamed
  mechanically; `VisionModelWeight::register_cache` is a no-op override so vision
  sub-trees never register into the text engine's categories.
- `Attention::register_cache`: each layer registers its own KV layout (prefix
  category, alignment 1) — reviewers verified the BlockConfig math
  (dtype/quant bits, kv/tp shard, share_kv) and walk order reproduce the retired
  single L-sized registration byte-for-byte. Engine layout params
  (`cache_block_seq_len`, `quant_policy`) ride the registry
  (`CacheRegistry::set_layout`, set in `Impl::RegisterCache`) since module
  configs are frozen. `prefix_offset_` is private, stamped per module;
  `prefix_registered_` gates the fold into `state->prefix_base` (vision modules
  never fold); `BlockIteratorParams` uses the region-relative
  `prefix_offset_ - prefix_base` so a non-zero base cannot double-count.
- `AttentionState` (attention.h): per-phase `AttentionBatch` slots, fp32
  split-KV workspace, pinned staging + mrope defaults, CP post-context, aux
  stream/events, warm-up RNG (shared, was per-rank in the retired layer),
  engine geometry. Held by `Context`; created once per rank in
  `Impl::Bind` under CreateEngine's guard shape (stream + device + pinned
  allocator — the pinned one is load-bearing for `kCPUpinned` staging).
- Version-stamp fill: `wave_` is `std::atomic<uint64_t>` starting at 1 (slots
  stamp from 0, so the first wave always fills); every non-kSetup op advances
  it (the walk runs on both scheduler and executor threads — hence atomic);
  `Setup` fills only when the slot stamp is stale — exactly one fill per
  kSetup wave, order-independent, matching the old per-op refill cadence.
- `Attention::Run` implements kAdd (dynamic NTK), kSetup (guarded fill),
  kPrepare (batch borrows, idempotent per module); `Forward(phase, input,
  output)` keeps the full transcription (MLA branch, qk-norm, output gate,
  CP offset math, aux-stream overlap, warm-up fill). Reviewers verified all
  kernel params argument-by-argument against the deleted class.
- Decoder: `Run` is the generic null-guarded walk over layer children
  (`for_each_child` yields null for absent optional children — caught by
  smoke, fixed); kForward keeps the orchestrated loop calling
  `attention->Forward` directly. GDN executor (`GatedDeltaNetLayer`) stays as
  bounded coexistence until ticket 09. `UnifiedAttentionLayer` deleted.
- README (same change): `module-lifecycle` section added stating the
  implemented pass order — prepare → register_cache → bind → category
  finalization at engine creation (finalization necessarily follows bind:
  warm-up inside engine creation already runs the model, and GDN's
  construction-time registration must precede finalization) — plus the
  default no-op `Run` dispatch rule; `unified-attention` rewritten (per-layer
  registration, shared state, version stamp, region-relative offsets);
  `cache-content`/`cache-geometry` renamed to Attention; `module-cache` and
  `category-registration` reworded to the pass. The gated-deltanet section
  rename rides with ticket 09 (its class rename) so the doc stays truthful
  per commit.

**Gate (perf/stage3.md):** four models meaningful ≥254 tokens; prefix-caching
and checkpoint variants (incl. GDN checkpoint category on the FP8 model) all
pass. Throughput: AWQ 101.1–101.9%, 30B 100.1–102.5%, 8B bs1/bs128
101.9%/99.5% of recorded baselines; 8B bs32 97.6% on the single matrix run →
three re-runs 2556–2610 (mean 2575.4) — same-day A/B vs stage-2's old
attention path is −0.3%, inside the ±1.1% run band (inter-day drift, per the
stage-1 fp8 precedent). FP8 101.1% of the fresh baseline-commit reference.
Loads within noise; footprints match stage-2 within 0.1%.
