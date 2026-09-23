# 05: Retirement, cancel, shutdown

**What to build:** Sequence release and shutdown expressed entirely by the transaction: releasability (retiring, no in-flight batches, no pending moves, record terminal) consumed by the loop's retire step; the five-step cancel order (retiring flips; unsubmitted intents die the same iteration; submitted moves drain with results discarded; release follows the last fold); a quiesce entry for shutdown that drops unsubmitted intents, blocking-drains submitted ones, and forces records terminal without collectives. Per spec §9 (cancel order) and §7 (releasability, quiesce).

**Blocked by:** 03 (retrieve), 04 (store).

**Status:** ready-for-agent

- [ ] Canceling a sequence with a pending store drains and releases with no leaked pins and no cross-pass cleanup state
- [ ] A canceled in-flight retrieve discards its result at fold and never installs
- [ ] Shutdown quiesces all move state; destructor checks verify quiescence; no collective runs at shutdown
- [ ] The releasability contract leaf is landed in the engine contract README
