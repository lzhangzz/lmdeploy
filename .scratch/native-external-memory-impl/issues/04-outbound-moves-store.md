# 04: Outbound moves end-to-end — store

**What to build:** Produced content lands in external memory as a transaction-planned outbound move: the publication stage plans store intents for coverage above the store cursor (sources are the just-published slots, ranges chunk-aligned, split around missing checkpoint slots); the intent carries the producing batch's completion event and the seam submits on it; the cursor advances at commit to the chunk-aligned produced end (`Align(produced_end)`), failures leaving permanent gaps (no retry); the fold clamps intents to produced coverage before submission. This ticket also lands the chunk-boundary clamping scoping — splitting applies only to participating sequences under the three-part condition — and the chunk-boundary snapshot leaf: on an exact chunk landing the snapshot slot is planned at the publication decision point (an additional trigger for the same `pending_publish` plan, independent of the publication gates) and allocated in optional admission; a failed allocation is a permanent gap and never stalls the forward. The PR's planning-stage arming (`PlanStoreCheckpoint`, `Sequence::store_checkpoint`) and its required-admission push die. Per spec §7 and §5 (clamp condition).

**Blocked by:** 02 (coverage acquisition).

**Status:** ready-for-agent

- [ ] Stored ranges appear remotely, chunk-aligned, only for participating sequences; ineligible sequences show pre-external clamping behavior (the X2 fix, observable)
- [ ] Submission follows the producing batch's completion event; no loop-turn deferral hook remains on the native path
- [ ] A forward finishing early stores only produced coverage (fold clamp); failed or skipped chunks leave permanent gaps, later stores start above them
- [ ] Chunk-boundary snapshots are planned at the publication decision point and allocated in optional admission when a forward lands exactly on the boundary; a failed allocation leaves a permanent gap and never stalls the forward (the stated leaf)
- [ ] Store-side contract leaves landed in the engine contract README
