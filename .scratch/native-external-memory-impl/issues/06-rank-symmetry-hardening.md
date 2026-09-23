# 06: Rank-symmetry hardening

**What to build:** The rank-symmetry invariant enforced and audited: every collective's participation set is a pure function of cross-rank-identical transaction state, and all rank divergence enters through collectives inside the reconcile entry (query results broadcast; transfer outcomes reduced before folding; fold-time install-or-drop decisions identical across ranks). The legacy collectives that the design kills are removed — the admission-status reductions and the rank-0 store-range broadcast (native planning is deterministic) — while the checkpoint-availability reduction remains. Per spec §9 (invariant, collectives table).

**Blocked by:** 03 (retrieve), 04 (store).

**Status:** ready-for-agent

- [ ] A tensor-parallel run with external memory enabled shows no divergence, deadlock, or asymmetric folding across a workload exercising lookup, retrieve, and store
- [ ] The rank-0 store-range broadcast is gone; planning produces identical intents on every rank without it
- [ ] A retrieve succeeding on one rank but failing on another folds as failure on all ranks (all-or-nothing fallback)
- [ ] The participation invariant is stated in the engine contract README and holds under an adversarial workload (cancel-heavy, unhealthy-server)
