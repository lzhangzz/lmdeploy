# Checkpoint-Interval-Aware Forward Truncation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `cache_checkpoint_interval` honored: a prompt-region forward pass that crosses the checkpoint-due position ends on a block boundary so a recurrent checkpoint is published there, and a checkpoint-sourced resume measures spacing from the restored position.

**Architecture:** Two scheduler-only changes in `src/turbomind/engine/scheduler.cc` (spec: `docs/superpowers/specs/2026-07-02-checkpoint-interval-truncation-design.md`): (1) `Resume` records `last_ckpt_pos` when it restores a checkpoint; (2) the forward-end clamp in `RunRequiredAdmission` gains a checkpoint-due block-boundary truncation. Plus normative-contract updates in `src/turbomind/engine/README.md` (`contracts.scheduler-commit`, `contracts.checkpoint-publish`).

**Tech Stack:** C++ (TurboMind engine), ninja build in `build/`, GPU verification via `scripts/test_turbomind_model.py` (script must be used AS IS).

**Constraints (from AGENTS.md):** never install lmdeploy or run `setup.py`; GPU commands run outside the sandbox; check `nvidia-smi` for idle GPUs before any GPU run (MCP model-server tools are NOT available in this environment); every model test must produce meaningful text (≥128 requested tokens).

**Model (fixed):** `Qwen/Qwen3.5-27B`, HF cache dir `/mnt_cfs/huggingface_hub/hub/` (from the index at `/data/models.json`). In every command below: `<MODEL_ID>` = `Qwen/Qwen3.5-27B`, `<CACHE_DIR>` = `/mnt_cfs/huggingface_hub/hub/`.

There is no C++ unit-test harness for the scheduler in this repo; verification is behavioral, via the cache WARN logs (`scheduler.cc` `LogResume`/`LogPublished`) produced by the smoke-test script. Task 1 therefore captures a failing baseline first, and Task 4 re-runs the same scenario as the passing check.

---

### Task 1: Baseline reproduction (failing test)

**Files:** none modified.

- [ ] **Step 1: Ensure the tree is built**

From `build/`: run `ninja`. If the folder is not configured, first run `sh ../my_generate.sh` from `build/`. Expected: exit 0.

- [ ] **Step 2: Pick GPUs**

Run `nvidia-smi` (outside the sandbox) and pick idle GPUs (no processes, ~0 MiB used). Qwen3.5-27B in bf16 needs ~54 GB of weights plus KV/state cache (`cache_max_entry_count=0.5` in the script): use `--tp 2 --gpus <a>,<b>` on two idle GPUs (or `--tp 1` on a single GPU with ≥ 80 GB free).

- [ ] **Step 3: Create the shared-prefix prompt file**

Write `/tmp/ckpt_prompts.json`: a JSON array of 2 strings. Prompt 0 is a long base document (~10k tokens, e.g. a paragraph repeated many times) ending with a question. Prompt 1 is the same base document plus a distinct extra ~5000-token section and a different question (so the recompute region past the shared prefix exceeds `cache_checkpoint_interval=4096` in one pass). Generate it with a short Python snippet, e.g.:

```python
import json
para = ("The quick brown fox jumps over the lazy dog near the quiet river bank "
        "while the morning sun rises over the distant hills. ") * 1
base  = para * 400   # ~10k tokens
extra = para * 200   # ~5k tokens
json.dump([base + "\nSummarize the text above.",
           base + extra + "\nList three facts from the text above."],
          open("/tmp/ckpt_prompts.json", "w"))
```

- [ ] **Step 4: Run the baseline scenario (outside the sandbox, on the chosen GPU)**

```bash
python scripts/test_turbomind_model.py \
    --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
    --tp 2 --gpus <a>,<b> \
    --enable-prefix-caching --cache-prompt auto --cache-generation none \
    --cache-prompt-boundary-skip 2 --max-prefill-token-num 8192 \
    --session-len 32768 --max-new-tokens 256 \
    --prompt-file /tmp/ckpt_prompts.json --prompt-ids 0 1 1 2>&1 | tee /tmp/ckpt_baseline.log
```

(`--prompt-ids 0 1 1` runs the base prompt, then the extended prompt twice; the second extended run resumes from whatever checkpoints the first one published.)

- [ ] **Step 5: Confirm the failure signature**

In `/tmp/ckpt_baseline.log`, find the `[TM][WARN]` scheduler lines. Expected failure: the pass for prompt-id 1 that computes > 4096 tokens past its resume point publishes **no** `ckpt@`, and the repeated prompt-id 1 resumes (`resume ... source=...`) from the old prompt-0-era checkpoint, recomputing thousands of tokens. Also verify all three responses are meaningful English (per AGENTS.md). Record the resume positions for comparison in Task 4.

### Task 2: `Resume` records `last_ckpt_pos` for checkpoint-sourced resumes

**Files:**
- Modify: `src/turbomind/engine/scheduler.cc` (in `Scheduler::Resume`, restore-copy section, currently lines 629–631)

- [ ] **Step 1: Apply the edit**

Replace:

```cpp
    if (ckpt && step > 0 && restore_ckpt) {
        s.restore_copies.push_back({restore_ckpt, s.frontier_cache_id});
    }
```

with:

```cpp
    if (ckpt && step > 0 && restore_ckpt) {
        s.restore_copies.push_back({restore_ckpt, s.frontier_cache_id});
        // Measure recurrent-checkpoint spacing from the restored position, not
        // from 0: without this a fresh request resuming deep into a shared
        // prefix believes a checkpoint is immediately due.
        s.last_ckpt_pos = std::max(s.last_ckpt_pos, step);
    }
```

- [ ] **Step 2: Build**

From `build/`: `ninja`. Expected: exit 0, no warnings about `scheduler.cc`.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/engine/scheduler.cc
git commit -m "fix(scheduler): measure checkpoint spacing from the restored checkpoint"
```

### Task 3: Checkpoint-due truncation in `RunRequiredAdmission`

**Files:**
- Modify: `src/turbomind/engine/scheduler.cc` (forward-end clamp in `RunRequiredAdmission`, currently lines 1162–1167)
- Modify: `src/turbomind/engine/README.md` (`contracts.scheduler-commit` line 266, `contracts.checkpoint-publish` line 384)

- [ ] **Step 1: Apply the scheduler edit**

Replace:

```cpp
        if (publish_prompt) {
            desired = prompt_boundary_pos;  // land exactly on B
        }
        else if (desired < ctx_end) {  // partial chunk: truncate to a block boundary
            desired = desired / bs * bs;
        }
```

with:

```cpp
        if (publish_prompt) {
            desired = prompt_boundary_pos;  // land exactly on B
        }
        else {
            // A recurrent checkpoint becomes due at last_ckpt_pos + interval.
            // The frontier state is checkpointable only at the pass end, so a
            // prompt-region pass that would run past the due position ends on
            // a block boundary and PlanFullBlockPublication checkpoints there;
            // the remaining tokens run in the next pass. `aligned > begin`
            // guarantees progress (a due position inside the current partial
            // block cannot be honored and falls through untruncated).
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

(`begin`, `bs`, `ctx_end`, `registry_` are all already in scope at this point in the function.)

- [ ] **Step 2: Build**

From `build/`: `ninja`. Expected: exit 0.

- [ ] **Step 3: Update the normative contract (README.md)**

Per AGENTS.md: edit content only, do not re-wrap existing lines.

In `contracts.scheduler-commit` (line 266), change the clamp clause

> clamps each forward's end to a boundary candidate (a block boundary, or exactly B = prompt_len - cache_prompt_boundary_skip when `prompt_boundary_node` is set (the publish decision is finalized in `SetupForks`; the clamp fires on the pass that can reach `B`)),

to

> clamps each forward's end to a boundary candidate (a block boundary, or exactly B = prompt_len - cache_prompt_boundary_skip when `prompt_boundary_node` is set (the publish decision is finalized in `SetupForks`; the clamp fires on the pass that can reach `B`); when checkpoint bytes are registered and a prompt-region forward would run past the checkpoint-due position `last_ckpt_pos + checkpoint_min_interval`, its end is truncated to the last block boundary in the admitted range — at or past the due position and strictly past the forward begin — so the full-block checkpoint can be taken there, with the remainder running in the next pass),

In `contracts.checkpoint-publish` (line 384), after the sentence ending "with no knowledge of prompt-boundary mode.", insert:

> The admission clamp (`contracts.scheduler-commit`) guarantees the full-block group a block-aligned pass end whenever the minimum interval is due in the prompt region, and `Resume` seeds `last_ckpt_pos` from a restored checkpoint's position so spacing is measured from it.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/engine/scheduler.cc src/turbomind/engine/README.md
git commit -m "fix(scheduler): end prompt-region passes on a block boundary when a checkpoint is due"
```

### Task 4: GPU verification (passing test)

**Files:** none modified.

- [ ] **Step 1: Check GPU availability**

Run `nvidia-smi` (outside the sandbox); use the same (or other idle) GPUs as Task 1.

- [ ] **Step 2: Re-run the exact Task 1 scenario (outside the sandbox)**

Same command as Task 1 Step 4, with `tee /tmp/ckpt_fixed.log`.

- [ ] **Step 3: Verify the fix against the baseline**

In `/tmp/ckpt_fixed.log` check all of:

1. The prompt-id 1 pass that computes > 4096 tokens past its resume point is now split: its first pass ends block-aligned and logs `published ... ckpt@<pos>` with `<pos>` a multiple of 64, followed by a short second pass finishing the prompt tail.
2. Consecutive `ckpt@` positions along one request's trajectory are ≥ 4096 apart (no redundant checkpoint right after a `source=checkpoint` resume — this exercises the Task 2 fix).
3. The repeated prompt-id 1 run resumes with `source=checkpoint` (or fork) at the new, higher checkpoint position; its `computed` span is < 4096 + block_size tokens (versus thousands in `/tmp/ckpt_baseline.log`).
4. All three responses are meaningful English relevant to the prompts, ≥ 128 generated tokens requested (`--max-new-tokens 256`). Gibberish means the truncation broke resume state — stop and debug, do not proceed.

- [ ] **Step 4: Decode-region sanity check (outside the sandbox)**

Run the default single-prompt smoke test to confirm decode behavior is untouched under `cache_generation none`:

```bash
python scripts/test_turbomind_model.py \
    --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
    --tp 2 --gpus <a>,<b> \
    --enable-prefix-caching --cache-generation none --max-new-tokens 256
```

Expected: exit 0, meaningful response, no `ckpt@` positions past the prompt length in the WARN log.

- [ ] **Step 5: Commit the plan checkboxes / any log notes**

```bash
git add docs/superpowers/plans/2026-07-02-checkpoint-interval-truncation.md
git commit -m "docs: record checkpoint-interval verification results"
```
