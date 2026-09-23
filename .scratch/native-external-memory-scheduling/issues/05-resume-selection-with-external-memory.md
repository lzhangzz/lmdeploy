# Resume selection and clamping with external memory

Type: grilling
Status: resolved
Blocked by: 03, 04

## Question

How does `PlanResume` integrate External Coverage as a resume source, and how does forward-end clamping respect external-move granularity?

`contracts.resume-selection` fixes the local rules: resume only from content proven produced (`is_valid`, `filled_len`), sources ranked frontier/checkpoint/fork with fork-extension dominant. Decide where external content sits in this proof and precedence system:

- What counts as proof for external content (stored content was proven when stored — does the proof survive the round trip?), and how revalidation works on every pass per `contracts.cache-reuse`.
- Precedence between local hits (trie, checkpoints, fork) and external coverage: when both cover a prefix, who wins, and when does a local hit *subsume* an in-flight retrieval (today's `proof.push_back(local.resume_end >= end)` logic in lmcache.cc is the ad-hoc version).
- Interaction with `readonly_block_num` and `invariants.executable-context`.
- How `ClampForwardEnd` incorporates external-move granularity: today `transfer_chunk_size` leaks the provider's chunk size into the Scheduler constructor; decide whether clamping at external granularity is a scheduling-model concept or a provider fact consumed through the seam, and how chunk/block alignment composes with the boundary candidates (`contracts.scheduler-commit`).

Check every proposal against `principles.resume-proof` and `checklist.cache-validity`.

## Answer

Resolved 2026-09-21 through grilling (Q1–Q4 confirmed, with two sharpenings forced by challenge: the descriptor sketch collapsed to a single granularity fact, and "tier" eliminated from the vocabulary). The headline: **external coverage never enters resume precedence — `resume_len` stays a purely local computation, and external knowledge only decides whether to fetch.**

**The decision.**

1. **Proof travels with the store; install is the proof event.** Content was `is_valid` when stored; the extent is *advisory knowledge* — acquired once at lookup, lease-bounded, never per-pass revalidatable (the service cannot be introspected, G2). `resume_len` rises only from installed local validity; the record motivates a retrieve, and the retrieve's terminal success + install (ticket 04) is what makes content resumable. `principles.resume-proof` holds unchanged: neither generic validity nor external knowledge raises `resume_len` — only its installed result does.
2. **The retrieve-decision rule** replaces today's P4/P5 proof logic. Planning compares the local resume candidate `L` (from `FindResume`, exactly as today) against the extent `E`:
   - `L >= E` → **subsume**: `AbandonCoverage` through the seam (lease released service-side), record → `kDone`, no move. The native form of `proof.push_back(local.resume_end >= end)`.
   - `L < E` and the seam is healthy → plan one retrieve intent: `end = min(E, Align(prompt_len − 1))` (last prompt token always recomputed — `invariants.executable-context`), `preserve_end = min(local prefix end, end − external_chunk_size_ when checkpoints move)` (G3's trailing-chunk refresh rule, kept), `start = Align(preserve_end)` with the service skipping the overlap.
   - In-flight subsumption (a local publish covers the range while a retrieve runs): only while pre-submission (cancellable, F31); once submitted it drains and installs — redundant but correct.
   External never competes in resume precedence; it gap-fills above local coverage. The ticket's "who wins" question dissolves: local wins, external fills.
3. **Chunk-boundary clamping is scoped per sequence, fixing X2.** The `transfer_chunk_size` global side effect becomes a three-part condition:
   ```cpp
   external_chunk_size_ > 0                        // external memory present (0 = absent)
   && registry_.has_checkpoint()                   // chunk-positioned categories move
   && s.external.state != CoverageState::kNone     // this sequence participates in external coverage
   ```
   All cross-rank-identical (config, registry, static request attributes — `principles.boundary-policy` holds). Under the condition, the chunk boundary joins block boundaries and `B` as a `ClampForwardEnd` boundary candidate; S5a splitting and the S5d interval-bypass follow it, and ticket 07's chunk-boundary snapshot trigger rides the same participation (S5b's planning-stage arming later died — ticket 07's amendment). Ineligible sequences get pre-external behavior — a deliberate behavior change from PR 4983, audit-noted "changed, reason: X2". `ExternalDesc` is **dropped** (amendment to the representation sketch): `enabled` is redundant (granularity 0 is absence; health never gated the clamp) and `moves_checkpoints` derives from the scheduler's own `registry_`.
4. **Derived interactions, no change:** installed nodes are ordinary `is_valid` nodes, counted into `readonly_block_num` by the next `PlanResume` (`invariants.readonly-block-num` untouched); an external-waiting sequence is inactive with `input_len == 0`, so executable-context and async-progress bookkeeping are untouched; the `prompt_len − 1` cap preserves "at least one token executes".

**Vocabulary ruling (recorded, glossary updated):** "tier" is eliminated from the domain vocabulary — the design has two placements, the local pool and External Memory, both already named, and ticket 03 ruled external placement is not structural anyway. `Tier Move` → **External Move** (retrieve = inbound, store = outbound); code names use `external_` prefixes. Ticket titles and answers have since been scrubbed to the new vocabulary; the only remaining mentions of the retired word are inside vocabulary-ruling statements like this one and quotes of contract text that still uses it.

**Code sketch** (planning rule; the planned intent lives on the sequence — the PR's `restore_plan` placement — and commit moves it to `pending_retrieve`):

```cpp
// Inside PlanResume (inactive sequences), after local candidate selection:
// local: the FindResume result PlanResume already holds — no second call; it gains
// prefix_end, a value FindResume already computes internally
const int L = local.candidate.pos;  // frontier/checkpoint/fork, as today
auto&    ext = s.external;

if (ext.state == CoverageState::kKnown && !s.planned_retrieve && !s.pending_retrieve) {
    // a planned intent persists until admitted — the guard prevents overwrite, no re-derive
    if (L >= ext.extent) {
        service_.AbandonCoverage(s);          // lease released service-side
        ext.state = CoverageState::kDone;  // local subsumes external
    } else if (service_.healthy()) {        // single-rank view; TP aggregation is ticket 09's
        const int end      = std::min(ext.extent, Align(s.prompt_len - 1));
        const int preserve = std::min(local.prefix_end,
                                      registry_.has_checkpoint() ? end - external_chunk_size_ : end);
        auto intent = std::make_unique<RetrieveIntent>();
        intent->start = Align(preserve); intent->preserve_end = preserve; intent->end = end;
        // The PR's PrepareRestore walk, relocated into planning. Its two remaining parts are
        // owned elsewhere and not re-derived here: out-of-range nodes' existing prefixes (the
        // ordinary involved set) and the frontier (PlanResume).
        for (auto& node : s.block_ids) {
            if (preserve <= node->offset && node->offset < end) {
                intent->targets.push_back(cache_.Create(registry_.prefix().object_id()));
            }
        }
        for (int pos = intent->start + external_chunk_size_; pos <= end; pos += external_chunk_size_) {
            auto& node = *s.block_ids[pos / logical_.block_size() - 1];
            if (!node.checkpoint) {
                node.checkpoint = cache_.Create(registry_.checkpoint().object_id(), &node);
            }
            intent->targets.push_back(cache_.Create(registry_.checkpoint().object_id()));
        }
        for (auto* b : intent->targets) {  // fresh slots are never valid: unconditional staging
            s.involved_blocks.push_back(b);
            s.alloc_blocks.push_back(b);   // the required set §6's lambda allocates
        }
        s.planned_retrieve = std::move(intent);
    }
}
// resume_len itself is computed exactly as today — from local validity only.
```

**What dies or changes** (coverage-audit hooks): the P4a/P4b proof skips (→ the planning comparison), `ProbeResume` entirely (S1 — the coordinator peek; native planning runs `FindResume` in-process), the engine-side proof AllReduce T3 (→ the planning comparison, TP form decided in ticket 09), `RetrieveEnd` as coordinator math (→ intent-planning cap), and X2's global coupling (→ scoped, changed-with-reason).

**Handed downstream:** S5c's required-allocation-on-exact-chunk-landing guarantee — today a cross-file comment (X8) — must become a stated contract leaf → ticket 07 (ticket 07's 2026-09-22 amendment later re-expressed the leaf: decided at the publication decision point, allocated in optional admission). The clamp condition's third term depends on eligibility being carried by the coverage record (`kNone` = ineligible), whose predicate home is ticket 09's to settle. The pre-submission-only in-flight subsumption bounds what cross-request dedup can cancel → ticket 08.
