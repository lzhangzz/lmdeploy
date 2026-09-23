# 02: Coverage acquisition — eligibility, query, fold

**What to build:** A sequence that is external-eligible learns its external coverage natively: one scheduler-owned eligibility predicate (each exclusion carrying its documented reason) decides participation at admission, creating the coverage record and issuing the coverage query through the seam (rank 0 submits, results reach every rank identically); a Reconcile skeleton folds query outcomes into the record's extent and store cursor; planning abandons coverage when local coverage already subsumes the extent. No moves yet — sequences without retrieved content recompute exactly as a miss does today. Per spec §9 (eligibility, query broadcast) and §5 (subsumption rule).

**Blocked by:** 01 (skeleton).

**Status:** ready-for-agent

- [ ] Eligibility predicate in the scheduler; the record's existence is the eligibility fact
- [ ] Query issued at admission for eligible sequences; terminal outcome sets extent and store cursor (success → `kKnown` with cursor = extent; failure → `kDone`, 0/0 — stores still possible)
- [ ] Local-subsumption abandonment observed when local coverage reaches the extent
- [ ] Unit tests cover the record lifecycle and fold outcomes; no behavioral regression with external memory off and on (old path)
