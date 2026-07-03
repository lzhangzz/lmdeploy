# Checkpoint-interval-aware forward truncation

Date: 2026-07-02
Status: approved

## Problem

With `--enable-prefix-caching --cache-prompt auto --cache-generation none --cache-prompt-boundary-skip 2` and `cache_checkpoint_interval = 4096`, no
recurrent checkpoints are published past the prompt boundary of the first
request. Every later request resumes from the same old checkpoint
(`source=checkpoint` at 10579 in the observed log) and recomputes an
ever-growing suffix (362 → 5670 tokens), even though single passes computed
more than one full interval (req 4: 4899 tokens).

## Root cause

Recurrent checkpoints are only planned by
`Scheduler::PlanFullBlockPublication`, which requires the forward end to land
exactly on a block boundary:

```cpp
LogicalBlock& x = *s.block_ids[(end - 1) / logical_.block_size()];
if (x.offset + x.capacity != end) {
    return;  // partial block — nothing to publish
}
```

The admission loop in `RunRequiredAdmission` block-aligns the forward end only
for *intermediate* prefill chunks (`desired < ctx_end`). A pass that finishes
the prompt ends at `prompt_len`, which is generally mid-block, so the
min-interval condition is satisfied but the geometry gate rejects it — and the
checkpoint cannot be taken retroactively at `end/bs*bs` because the frontier's
recurrent state is only correct *at* the pass end.

A secondary defect: a request resuming from a restored checkpoint keeps
`last_ckpt_pos == 0`, so the interval test `end - last_ckpt_pos >= interval`
believes a checkpoint is immediately due, skewing spacing decisions.

## Design

Contract restated: `cache_checkpoint_interval` is a minimum spacing
(`checkpoint_min_interval`); whenever a prompt-region pass crosses the due
position (`last_ckpt_pos + interval`, measured from the latest checkpoint the
request restored or published), a checkpoint is taken at that pass's end. Two coordinated changes, both scheduler-only:

### 1. Checkpoint-aware truncation in `RunRequiredAdmission`

When a prompt-region pass would run past the next checkpoint-due position,
truncate the forward end to the last block boundary at or past that position,
even when the pass could otherwise finish the prompt. The remainder becomes
the next pass. In `Scheduler::RunRequiredAdmission`, replacing the current
two-way clamp:

```cpp
if (publish_prompt) {
    desired = prompt_boundary_pos;  // land exactly on B
}
else {
    // A recurrent checkpoint becomes due at last_ckpt_pos + interval. The
    // frontier state is checkpointable only at the pass end, so if this
    // prompt-region pass would run past the due position, end it on a block
    // boundary; PlanFullBlockPublication then takes the checkpoint there.
    // The remaining tokens run in the next pass. `aligned > begin` keeps
    // progress guaranteed (a due position inside the current partial block
    // cannot be honored and falls through untruncated).
    const int aligned = desired / bs * bs;
    const int due     = s.last_ckpt_pos + registry_.checkpoint_min_interval();
    if (registry_.has_checkpoint() && desired <= s.prompt_len && desired > due
        && aligned >= due && aligned > begin) {
        desired = aligned;
    }
    else if (desired < ctx_end) {  // partial chunk: truncate to a block boundary
        desired = aligned;
    }
}
```

Notes:

- `desired <= s.prompt_len` keeps the truncation prompt-region-only. The
  generation region is unaffected: decode passes advance one token and
  `PlanFullBlockPublication` already gates generation-region checkpoints on
  `cache_generation != none`.
- The truncation lands on the *last* block boundary within the admitted
  range (`aligned`), not the first boundary past `due`: min-interval
  semantics allow wider spacing, and this costs at most one extra pass per
  prompt regardless of how many intervals the pass crossed. The
  one-`pending_publish`-slot-per-request limit is respected without change.
- The `publish_prompt` branch keeps precedence, unchanged: when the pass can
  reach `B`, the boundary clamp wins and the boundary checkpoint bypasses the
  interval (existing `contracts.checkpoint-publish` behavior).
- Cost: one extra scheduling pass per crossed interval — the same cost a
  chunked prefill already pays today.

### 2. Initialize `last_ckpt_pos` from the restored checkpoint in `Resume`

When `Scheduler::Resume` selects a checkpoint restore (`restore_ckpt != 0`,
i.e. `source` is `kCheckpoint` or a fork carrying a checkpoint), record the
restored position as the last checkpoint position:

```cpp
if (ckpt && step > 0 && restore_ckpt) {
    s.restore_copies.push_back({restore_ckpt, s.frontier_cache_id});
    s.last_ckpt_pos = std::max(s.last_ckpt_pos, step);
}
```

Without this, a fresh request resuming at 10579 believes a checkpoint is due
immediately and change 1 would truncate its first tiny pass to publish a
checkpoint adjacent to the one it just restored. With it, spacing is measured
from the restored checkpoint. Frontier and prefix resumes leave
`last_ckpt_pos` untouched (it persists on the `Sequence` across passes until
`Release`, which already resets it to 0).

## Expected behavior on the observed log

req 1 (10579 → 10941): no checkpoint (362 \< 4096 past the restored
checkpoint). req 4 (10579 → 15478): due at 14675, pass truncates to the last
block boundary 15424, publishes `ckpt@15424`, second pass finishes
\[15424, 15478). Subsequent requests resume from 15424 instead of 10579.

## Contract impact (`src/turbomind/engine/README.md`)

- `contracts.scheduler-commit`: the forward-end clamp description gains the
  checkpoint-due block-boundary truncation as a third boundary candidate.
- `contracts.checkpoint-publish`: unchanged semantics (full-block group stays
  coverage-driven); add that the admission clamp guarantees a block-aligned
  pass end when the interval is due in the prompt region.
- `ownership.resume-len` / `contracts.resume-selection`: unchanged; `Resume`
  additionally records `last_ckpt_pos` for checkpoint-sourced resumes, which
  is publication bookkeeping, not resume selection.
- Checklist items touched: `checklist.scheduler-commit` (commit point
  unchanged), `checklist.cache-prepare` (`Resume` still emits intent only —
  `last_ckpt_pos` is plan-side bookkeeping, no allocation or device effect).

## Testing

- Rebuild (`ninja` in `build`) and rerun the multi-request scenario from the
  log (long shared prefix, `cache_checkpoint_interval=4096`,
  `cache-prompt-boundary-skip 2`) via `scripts/test_turbomind_model.py`;
  verify meaningful ≥128-token responses and that the cache log shows
  `ckpt@` positions spaced ≤ interval, with later requests resuming
  `source=checkpoint` near their divergence point.
- Verify decode-only behavior unchanged under `cache_generation=none` (no
  generation-region checkpoints, no truncation of decode passes).
