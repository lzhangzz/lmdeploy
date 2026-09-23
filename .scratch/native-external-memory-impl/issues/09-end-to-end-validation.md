# 09: End-to-end validation — the spec's four scenarios

**What to build:** Execute the design spec's four scenario walkthroughs against a real model with a live external-memory server, verifying each behaves as the spec's §13 predicts: retrieve-in-flight under admission pressure (waiting is skipped-not-stopped, liveness suppresses while healthy); cancel with pending store (five-step order, no leaks); shared-prefix double-retrieve (emergent sharing, fold-time subsumption, no cross-request coordination); warm-up (ineligible sequences fully inert, no chunk splitting). Spot-check the coverage audit's "expressed" claims against observed behavior.

**Blocked by:** 08 (cutover).

**Status:** ready-for-agent

- [ ] Scenario one walked under real memory pressure with scheduling-state observations matching the spec's prediction
- [ ] Scenarios two through four walked with the outcomes the spec states (no leaks, subsumption observed, warm-up inert)
- [ ] Model responses are meaningful and relevant throughout, at the required minimum length, per the repo's testing rules
- [ ] Any divergence between observed behavior and the spec is reported and either fixed or recorded as a spec amendment
