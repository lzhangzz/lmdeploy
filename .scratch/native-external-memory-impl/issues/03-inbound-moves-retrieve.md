# 03: Inbound moves end-to-end — retrieve

**What to build:** The first full tracer bullet: a prompt whose prefix exists externally resumes from retrieved content instead of recomputing. Planning applies the retrieve-decision rule (gap-fill above local coverage, preserve rule, last-prompt-token cap) and stages the intent's target slots into `alloc_blocks`; required admission allocates them atomically through the same `try_allocate_required` lambda as forwards — on failure the pass stops as any required failure and the deferred intent persists, retried next pass (no forward is planned for it while planned or in flight); the pass commits and submits the intent; the sequence waits (skipped, inactive, no re-planning churn) while the transfer runs; the fold installs on success (no foreign producer mark, targets not already valid) making the content ordinary local content the next planning resumes from — or abandons to local recompute on failure. Retrieve outcomes are reduced across ranks before folding. Per spec §4, §5, §6.

**Blocked by:** 02 (coverage acquisition).

**Status:** ready-for-agent

- [ ] A prompt with an external hit resumes from installed content end-to-end (verified with a real model: meaningful response, tokens skipped observable in scheduling state)
- [ ] Targets admitted atomically through the shared required-allocation path; admission failure stops the pass and the deferred intent persists, retried next pass
- [ ] Waiting sequences are skipped inside planning, never hidden from the transaction by loop logic
- [ ] Failed or install-refused retrieves abandon coverage and recompute locally
- [ ] The fold consumes only rank-reduced outcomes for retrieves
- [ ] The fold and scheduler-commit contract leaves it touches are landed in the engine contract README
