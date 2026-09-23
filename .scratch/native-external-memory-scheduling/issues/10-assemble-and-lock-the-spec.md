# Assemble and lock the design spec

Type: task
Status: resolved
Blocked by: 03, 04, 05, 06, 07, 08, 09

## Question

Assemble the locked design spec from the resolved decisions — the terminal artifact of this map (per its Notes, this effort carries the spec into the map as its deliverable).

The spec consists of:

1. The design document under `.scratch/native-external-memory-scheduling/` stating the native External Memory scheduling model: external-memory representation, transaction semantics for External Moves, resume/admission/store behavior, lifecycle edges — each section citing the decision ticket it lands.
2. The specified `src/turbomind/engine/README.md` amendments: the exact contract leaves added or changed (code sketches of the normative wording), satisfying `checklist.contract-sync`.
3. The PR 4983 coverage audit: every item of the ticket-01 inventory mapped to its native expression or an explicit drop with reason.

Done means the map's four-part acceptance bar passes: inventory covered, contract leaves specified, ad-hoc surfaces gone by design, and the scenario walkthroughs (retrieve-in-flight under admission pressure, cancel with pending store, shared-prefix double-retrieve, warm-up) written out and survived.

Input (settled post-tickets, 2026-09-22): `CacheBlock::transfer_refs` is **necessary and survives** — the spec must state the pin mechanics with rationale: the daemon accesses device memory by raw IPC address (no stale detection can protect it), so hard exclusion is the only correctness mechanism for in-flight move sources/destinations; soft stamping cannot reach waiting sequences' targets (skipped by planning), releasability covers only request-owned slots, and a scheduler-side pinned-set is a strictly weaker duplicate of the per-slot counter whose fatal checks in `Deallocate`/`Invalidate` guard every free path. What changes natively is the driver: the seam acquires at submit and releases at fold (intent lifecycle), replacing coordinator choreography; the pins are the move's own lifetime state.

## Answer

Resolved 2026-09-22. The spec is assembled and locked at **`spec.md`** in this directory (the tracker-conventional spec path), drawing every section from the resolved decision tickets (03–09), both research assets, the contract, and the `transfer_refs` verdict.

Contents: §1 framing and scope locks; §2 domain model; §3 representation (planning state, intents, seam); §4 transaction semantics (Reconcile/Schedule, external-wait skip, fold, health-gated liveness); §5 resume selection and clamping; §6 admission (shared required-allocation path, stall table); §7 store path and retirement; §8 sharing and identity; §9 lifecycle edges (cancel order, eligibility, rank symmetry with the collectives table); §10 transfer pinning (`transfer_refs` mechanics); §11 the specified `engine/README.md` amendments — eight new/changed contract leaves and one checklist leaf, exact wording, landing with the implementation per `checklist.contract-sync`; §12 the PR 4983 coverage audit — every inventory category (C/P/E/S/G/L/F/R/M/T/H/V/X) mapped to Expressed / Changed / Dropped / Unchanged with home or reason; §13 the four scenario walkthroughs (retrieve-in-flight under admission pressure, cancel with pending store, shared-prefix double-retrieve, warm-up), each walked through the decided mechanisms and surviving; §14 out-of-scope recap.

Acceptance bar: (1) inventory covered — §12; (2) contract leaves specified — §11; (3) ad-hoc surfaces gone by design — §4 and the audit's E/S rows; (4) scenarios written and survived — §13. The map's destination — a locked design spec for scheduler-transaction-native External Memory support — is reached.
