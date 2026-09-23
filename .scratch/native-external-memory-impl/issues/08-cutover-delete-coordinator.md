# 08: Cutover — delete the coordinator, rewire the loop

**What to build:** The old integration is removed and the native model becomes the only path: the engine loop calls the scheduler's reconcile and schedule entries with all non-retiring sequences and contains no external-memory-specific logic; the four loop hooks and the three scheduler probes of the PR 4983 shape are deleted along with the coordinator class and its dead public surface; the remaining contract amendments land (commit and admission leaf updates, the required/optional admission renaming, the external-eligibility leaf, the checklist leaf). One code path, no flags. Per spec §4 (entries), §11 (amendments), §12 (audit rows marked dropped).

**Blocked by:** 05 (retirement/cancel/shutdown), 06 (rank symmetry), 07 (pin ownership).

**Status:** ready-for-agent

- [ ] No trace of the coordinator, its session machine, or its hook call sites remains in the engine loop
- [ ] The scheduler's public surface contains the reconcile/schedule entries and no probe methods
- [ ] Every contract amendment from the design spec's §11 is landed in the engine contract README, and the README passes its own contract-sync rule against the code
- [ ] Full test suite and model tests pass with external memory disabled and enabled; single code path, no feature flag
