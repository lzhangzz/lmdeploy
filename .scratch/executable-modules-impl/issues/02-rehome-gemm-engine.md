# 02: Re-home the GEMM engine into the engine root (prefactor)

**What to build:** Make the change easy, then make the easy change. The per-rank GEMM engine — engine object, dispatch cache, workspace, measure policy — moves out of the linear executor's private implementation into the engine-root Context, where the spec's engine-services section puts it. The old linear executor stays alive as a thin delegate over the re-homed engine so every existing call site keeps working unchanged. No module behavior changes in this ticket.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] GEMM engine, dispatch cache, workspace, and measure policy owned by the engine-root Context (one per rank)
- [ ] The linear executor becomes a pure delegate: same public surface, zero owned engine state
- [ ] Warm-up/tuning flow (measure mode, export/import) works through the re-homed engine
- [ ] Build green; one core-model smoke run shows unchanged behavior
