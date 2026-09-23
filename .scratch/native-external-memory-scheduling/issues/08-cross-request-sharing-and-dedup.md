# Cross-request sharing and retrieval dedup

Type: grilling
Status: resolved
Blocked by: 03, 05

## Question

How does the scheduling model handle multiple requests whose prefixes share external coverage?

Scenarios to decide against:

- Two requests miss locally, both hit externally on a shared prefix: does the transaction dedupe into one retrieval with the second waiting on the first's completion, retrieve twice, or retrieve once and publish locally so the second becomes a trie hit?
- A local publish lands while a retrieval for the same range is in flight (the subsumption question from ticket 05, seen from the sharing side): what cancels or ignores the redundant move, and who owns the inbound slots.
- Identity: does external content participate in `contracts.prefix-identity` (token + fingerprint + parent) so an external hit and a trie hit can be compared as the same node, or does external coverage stay per-sequence with sharing emerging from provider-side identity? Input from the representation decision: external identity is token-keyed only — `IPCCacheServerKey` carries `token_ids` with no fingerprint (service fact F17), while local trie identity folds image fingerprints. Same tokens / different image content is a potential wrong-KV collision remotely; `Request::cache_salt` exists in the protocol and is never set by the engine (gap G7). Decide whether the engine must fold content fingerprints into the remote key (e.g. via `cache_salt`) as part of the native identity model.
- Whether a retrieved range should be indexed into the `PrefixTrie` on arrival (making it cross-request reusable) or stays private to the retrieving sequence — and what that implies for `ownership.prefix` refcounts and eviction.

Today sharing is accidental (per-sequence sessions; `lmcache::Lookup` queries the whole prefix including local hits). Decide the native model explicitly. Check against `contracts.prefix-conflict` (block-level exclusion) and `contracts.cache-eviction`.

## Answer

Resolved 2026-09-22 through grilling (all four confirmed; Q2 was revised after a fact check killed the proposed mechanism). **Sharing is emergent, not built** — and the map's flagged identity hazard turned out to be already solved in the tree.

**The decision.**

1. **No cross-sequence dedup.** Two sequences missing locally and hitting externally on a shared prefix each plan and retrieve their own copy; no wait-on-another-sequence state, no lease coordination. Sharing emerges: the first install lands in trie-matched nodes (`is_valid`), so the second sequence's next planning finds `L >= E` and subsumes (ticket 05's rule). Duplicate bandwidth is paid only inside the in-flight window. New in this ticket: **fold-time subsumption** — at reconcile, a completed retrieve whose target nodes are already valid (another sequence installed first) drops its fresh slots instead of installing. No swap, no waste, one check at a fold that already exists. (Amends ticket 04's install semantics: install requires terminal success ∧ install-safety ∧ targets-not-already-valid; also refines ticket 05's "drains and installs — redundant but correct" to "drains and installs-or-drops".)
2. **No `cache_salt` folding; remote identity is token-level and content-true.** Fact established during grilling: `_replace_multimodal_token_ids` (lmdeploy/turbomind/turbomind.py:70, applied at :759 whenever `lmcache_addr` is set) rewrites multimodal placeholder spans to per-image cache-key ids — `int(fingerprint.hex(), 16) & 0xffff`, deliberately the vLLM/LMCache projection — *before* the engine sees the tokens. So the token ids that lookup/store key on already encode image content identity; the same-tokens/different-image collision flagged into this ticket does not exist on this path. G7 (`cache_salt` never set) closes as by-design, redundant. This also explains two existing behaviors: eligibility requires non-empty fingerprints (no fingerprint, no projection), and the replacement being lmcache-path-only. Recorded caveat: the projection is 16 bits, so distinct images collide with probability ~2⁻¹⁶ per pair (birthday-bound across many distinct images); the local trie retains full 256-bit fingerprint identity on top, so only the remote side carries the exposure — ecosystem-standard, service-side, out of scope.
3. **Install-into-trie is the sharing mechanism, confirmed explicitly.** A retrieved range is never separately "indexed on arrival": it installs into the nodes the sequence matched at admission, and cross-request reuse, refcounting (`ownership.prefix`), and eviction (`contracts.cache-eviction`) behave exactly as for locally produced content. Ticket 03's rulings, restated because this ticket asked.
4. **Leases stay per-sequence, service-side, implied by intent lifecycle.** A lease is the remote grant issued on lookup success — read-locks on matched chunks plus remote session bookkeeping (F20/F21) — guaranteeing the content you were told about survives until you fetch it; delegated on retrieve, released on abandon, TTL-covered on loss. Concurrent sequences may hold concurrent leases on the same content; no cross-request lease coordination exists in the scheduling model, and none is added.

**What dies or closes** (coverage-audit hooks): the dedup question (→ emergent sharing, no machinery); the salt question (→ closed by fact, no mechanism); G7 (→ by-design); the same-tokens/different-image hazard (→ non-existent on the lmcache path, recorded with the 16-bit birthday caveat); ticket 04's unconditional install (→ amended: fold-time subsumption).

**Handed downstream:** TP-rank symmetry of fold-time subsumption (all ranks must reach the same install-or-drop decision from identical node state) folds into ticket 09's existing `Reconcile` participation input. The eligibility rationale (empty-fingerprint exclusion = unprojectable spans) is input to ticket 09's eligibility home decision.
