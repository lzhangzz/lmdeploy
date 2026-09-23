# 01: Skeleton — coverage state, intents, seam, granularity fact

**What to build:** The inert skeleton of the native external-memory model, landing beside the existing coordinator with zero behavior change: the per-sequence external-coverage record and lifecycle state, the inbound/outbound move-intent types, the seam interface (intents down, terminal outcomes up, poll-driven) with a no-op implementation, and the single external-granularity fact replacing the chunk-size constructor argument. All records stay in the "none" state; the old integration keeps working exactly as before. Per spec §3.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Sequence carries the coverage record and intent types; no coordinator field is removed yet
- [ ] Seam interface exists with a no-op implementation; nothing address-like or lease-like crosses it
- [ ] The external-granularity fact replaces the chunk-size constructor argument with identical observable behavior
- [ ] Engine builds; full test suite passes; model test passes with external memory disabled and enabled (old path), responses meaningful
