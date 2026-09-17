# 10: Coexistence cleanup and final gate

**What to build:** The contract the spec's landing strategy promises: every retired class and temporary coexistence shim is gone from the tree. Then the final gate across the full matrix — including the large TP model that exercises the comm and allreduce paths — plus the cache-behavior variants and the perf comparison against the recorded baselines.

**Blocked by:** 04 (Python delta and stage-1 gate), 06 (Moe executable module with shared MoeState, and stage-2 gate), 08 (Attention executable module with AttentionState, and contract-doc sync), 09 (DeltaNet executable module with DeltaNetState).

**Status:** ready-for-agent

- [ ] No retired executor, wrapper, or delegate class remains anywhere in the tree
- [ ] Full four-model matrix plus the large TP≥2 model pass end-to-end with meaningful responses of at least 128 tokens
- [ ] Cache-behavior variants (prefix caching, checkpointing) pass
- [ ] Decode throughput at or above 98% of baseline across the matrix; load times within noise
- [ ] Frozen-surface check passes one final time
