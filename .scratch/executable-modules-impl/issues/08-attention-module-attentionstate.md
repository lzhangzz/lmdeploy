# 08: Attention executable module with AttentionState, and contract-doc sync

**What to build:** Per-layer attention modules run projections, q/k norms, core attention (including the MLA branch and CP), with the batch-level per-phase data living in one shared AttentionState per rank — filled at setup under the version-stamp guard so the fill is order-independent — and each layer applying its own registered KV offset. The unified attention layer class retires. The engine contract document is updated in the same change per its own sync rule: class renames in the cache-content, unified-attention, and gated-deltanet sections; registration timing reworded from "during construction" to "during the registration pass"; the lifecycle paragraph added.

**Blocked by:** 07 (Make the cache-registration pass real).

**Status:** ready-for-agent

- [ ] Per-layer attention modules registered with the prefix category; accumulated layout byte-identical to today's, so existing checkpoints and cache blocks are unchanged
- [ ] One AttentionState per rank: per-phase batch slots with version-stamped fill, fp32 workspace, pinned staging, CP context, aux stream
- [ ] Unified attention layer class retired and deleted
- [ ] Contract-doc amendments land in the same change (renames, registration timing, lifecycle paragraph)
- [ ] Cache-behavior gate variants pass: prefix caching enabled, checkpointing enabled, both with meaningful responses
