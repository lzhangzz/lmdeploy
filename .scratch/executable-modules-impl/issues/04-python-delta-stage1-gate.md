# 04: Python delta and stage-1 gate

**What to build:** The effort's entire Python-side change, and the checkpoint that closes stage 1. The standalone torch-linear path constructs an engine-root Context and binds it to its module after prepare (replacing the retired executor object); the two new driver entry points appear — the registration walk (still an empty pass at this stage) and the bind walk — and the model-load driver calls them in the spec's sequence. Then run the stage-1 gate: the frozen-surface checks and the four-model matrix with the perf comparison.

**Blocked by:** 01 (Record perf baselines), 03 (Linear executable module).

**Status:** ready-for-agent

- [ ] Standalone torch-linear path works via Context construction + bind; public API names unchanged
- [ ] Registration and bind entry points exist and the load driver calls them in the spec's order
- [ ] Frozen-list check passes (no new or renamed Python public names) and builders are byte-identical
- [ ] Four core models pass end-to-end with meaningful responses of at least 128 tokens, never batched
- [ ] Decode throughput at or above 98% of baseline with load time within noise
