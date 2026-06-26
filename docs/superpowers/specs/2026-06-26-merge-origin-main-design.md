# Design: Merge `origin/main` into `memory-1a`

Date: 2026-06-26
Status: Approved (design); implementation plan pending
Branch: `memory-1a` (HEAD `b189745a`)

## 1. Context & goal

`memory-1a` is the merge-base (`a4025b91`) plus a single large commit
`b189745a` — "feat(turbomind): memory allocator, object cache, and scheduler
integration" (+7471 / −3101 across 83 files). That commit replaces the old
TurboMind sequence/cache machinery: it **deletes** `BlockManager`,
`BlockTrie`, `SequenceManager`, `test_cache_manager.cc` and **adds**
`src/turbomind/memory/*`, `engine/scheduler.*`, `engine/block.*`,
`engine/cache_*`, `engine/prefix_*`, `models/llama/context_token_resource.h`.

`origin/main` has advanced **70 commits** past the merge-base.

**Goal:** merge `origin/main` into `memory-1a` and **fully integrate** main's
new features so everything works end-to-end on the new architecture — not a
textual-only resolution. The end state must build and pass verification.

## 2. Merge landscape (from a trial merge, then aborted)

12 conflicting paths:

- **10 content conflicts:** `engine/engine.cc` (8 hunks), `models/output_processor.cc` (4),
  `engine/request.h` (3), and 1 hunk each in `lmdeploy/messages.py`, `engine/engine.h`,
  `engine/model_executor.cc`, `kernels/attention/attention_params.h`,
  `models/input_processor.cc`, `models/llama/unified_attention_layer.cc`,
  `turbomind.cc`.
- **2 modify/delete conflicts:** `models/llama/SequenceManager.cc/.h` — deleted on our
  side, edited on main.

Only ~5 of main's 70 commits land inside the rewritten region. Their disposition:

- `support qwen3.5(vit) inference in turbomind backend` (#4602) — +4129 lines, but
  ~95% is **new files** (`models/qwen3_5vit/*`, `vision_model.*`, `layer_norm_weight.*`)
  that merge cleanly. Real work is at the wiring points. → workstream **W1**.
- `Add /get_ppl endpoint` (#4679) + CE-loss — threads through `request.h`,
  `output_processor.cc`, `engine.cc::Validate`, `messages.py`. → workstream **W2**.
- `fix cp inference` (#4619) — `cp_utils.*` clean add + 3 lines in
  `unified_attention_layer.cc` + attention `causal`. → workstream **W3**.
- `fix scheduler for ssm` (#4691) — **PyTorch-only** (`lmdeploy/pytorch/*`); merges
  clean, no TurboMind work.
- health-endpoint / `scheduler_tick` metrics — clean adds or flagged-deferred. → **W4**.

## 3. Strategy: two-phase merge

No worktree (per repo rule). Tag a safety ref first: `git tag premerge-memory-1a`.

- **Phase 1 — structural merge commit.** `git merge origin/main`; resolve all 12
  conflicts to **preserve the new architecture**; fold in additive/independent main
  changes; **exclude the qwen3.5-vit-introduced sources from CMake** (`qwen3_5vit/`,
  `vision_model.*`, and any vit-only support units such as `layer_norm_weight.*`) so
  the tree builds — the exact exclusion set is finalized by what the Phase-1 build
  requires; verify the text/SSM path; commit the merge.
- **Phase 2+ — feature integration commits** (each its own commit): W1 (qwen3.5-vit),
  W2 (get_ppl), W3 (cp fix), W4 (misc), then VL verification.

Rationale: the merge commit stays reviewable and bisectable ("merge builds + text
path works" is a clean checkpoint); the risky feature integration is isolated; if VL
verification is blocked by model/GPU/tooling, Phase 1 still stands.

Rollback: `git merge --abort` (Phase 1) or `git reset --hard premerge-memory-1a`.

## 4. Global merge rule: the stateful-session subsystem stays removed

`b189745a` deliberately removed the stateful-session / kill subsystem — the original
`Sequence` instance stored in `SequenceManager` that persisted **after a request
ended**. Verified: our branch has **zero** occurrences of `Request::session`,
`start_flag`/`end_flag`/`kill_flag`, `kill_reqs`, or `seq_mgr_`.

Rule: **never reintroduce it.** Main references `r->session.*` pervasively
(`engine.cc`, `gateway.cc/.h`, `model_request.cc`, `request_queue.h`, `request.h`,
and inside vit code). During the merge:

- Drop main's session/kill code rather than porting it.
- Where main hangs a **new feature** off `r->session.*`, re-express it against our
  flattened request model:
  - **W1:** `qwen3_5vit.cc:211` and `input_processor.cc:71` gate the vision encoder on
    `r.session.start_flag/end_flag`. Translate to our "first/prefill pass" notion
    (e.g. `step == 0` / the new arch's prefill detection) — **not** session flags.
  - **Phase 1 (Task 1.8):** the auto-merged `bind.cpp` `SessionParam` `start`/`end`
    bindings and the `End(session_id)` kill binding, plus any `End`/`start_flag`/`end_flag`
    in `model_request.{cc,h}`, are **stripped** (not adapted). `param.session.id/step`
    (our request-id/step carrier) stays.

Files where only our branch changed (main: 0 commits) — git takes our version cleanly,
so session/kill stay gone: `gateway.cc`, `gateway.h`, `request_queue.h`.

## 5. How the merge resolves files (beyond the 12 conflicts)

The 12 textual conflicts are the *small* part. The qwen3.5-vit PR (#4602) is ~90 files
and the merge resolves most of them silently — verified with
`git rev-list --count a4025b91..{origin/main,HEAD} -- <file>` and a throwaway
`git merge --no-ff` (inspected, then aborted). `origin/main` is at **`4778480b`
(77 commits ahead)**; the 7 newest commits touch **no `src/turbomind/` file** (Python
serve / pytorch / ascend / version bump), so this analysis is current.

- **Taken wholesale from main** (`ours:0` since base, so the merge just adopts main's
  version): the entire `kernels/attention/*` suite (a `causal` flag threaded through
  every `attention_sm*.cu`, `kernel_impl.h`, `attention_template.h`, `attention.cu`,
  `desc.h`, `registry.cu`, `mainloop_*.h`), **`rotary_embedding.h` + `llama_rope.h`
  (which DROP `RopeType::kMrope` for an orthogonal `MropeMode`)**, `attention_weight.cc`,
  `model_root.h`, `input_processor.h`, `model_executor.h`, `core/module.h`; plus all new
  files (`qwen3_5vit/*`, `vision_model.*`, `kernels/norm/*`, `layer_norm_weight.*`,
  get_ppl `cross_entropy_kernels.*`).
- **Auto-merged, both sides changed, no conflict:** `attention_universal.h` (our
  readonly-KV-store change and main's `first_K/last_K` causal change are in **disjoint
  regions** — safe), `attention_weight.h`, `language_model.cc` (its `PatchEmbedding`
  call line was changed only by main → becomes the **4-arg** form), `bind.cpp`,
  `model_request.cc/.h`. Verified outcomes: `Request::mm_inputs` auto-merges cleanly
  into `Request` (keep the `multimodal_input.h` include; inert in Phase 1 — this is what
  lets the **header-only** vit bindings in `bind.cpp` and `r->mm_inputs = param.mm_inputs`
  in `model_request.cc` compile untouched), but main's **`SessionParam::start_flag/
  end_flag` bindings and the `End(session_id)` kill method auto-merge in and must be
  stripped** (our branch has `Cancel`, not `End`; `SessionParam` itself resolves to ours
  = `{id, step}`).

**Consequence:** our code must *compile against main's refactored rope/attention*. So
Phase 1 **adopts** `MropeMode` / `causal` rather than "keeping ours", excludes only the
vit-*specific* `.cc`/`.cu` compile units (headers + header-only bindings stay), and
strips the auto-merged session/kill leaks. "keep ours" is valid only where our side's
callers/headers stay self-consistent.

### Conflict-resolution map (Phase 1)

> **Phase 1 execution update (commit `911c745b`).** Three rows below changed in
> practice — full detail in the plan's "Phase 1 execution notes". In short:
> (1) `model_root.h` was **restored to main** (vision child) and the two
> standalone vit **weight** units (`qwen3_5vit_weight.cc`,
> `qwen3_5vit_block_weight.cc`) are **compiled** in Phase 1, because Qwen3.5 is a
> VLM whose loader builds the vision weight sub-tree; the vision **encoder +
> kernels + `vision_model.cc`** stay excluded. (2) `bind.cpp`/`model_request`
> had **no** session/kill leaks to strip (auto-merge already produced our shape).
> (3) `input_processor.cc` kept the auto-merged multimodal embed body (inert),
> resolving only the include block (keep `vision_model.h`, drop `SequenceManager.h`).

| File | Phase-1 resolution |
|---|---|
| `engine.h` / `engine.cc` | Take **ours** (text-only ctor; drops vision threading, `seq_mgr_` metrics == the #4615 health change, and main's `Validate` lambda/`kill_reqs`). Vision re-added W1; `return_ppl` clause W2. |
| `request.h` (3 hunks) | Hunk1 includes: **keep BOTH** (`block.h` + `multimodal_input.h`). Hunk2: add `return_ppl` in `GenerationConfig`. Hunk3: keep our `Sequence` exec-state + append get_ppl `input_ce_loss`/`ce_loss`. `Request::mm_inputs` **auto-merges** (keep, inert). Only `Sequence::multimodal_inputs` → W1. No `end_flag`/`start_flag`. |
| `attention_params.h` | Take **theirs** (`causal{true}`, `layer_id`). |
| `unified_attention_layer.cc` (mrope) | Keep **ours**, changing only the guard `rope_param_.type == RopeType::kMrope` → `rope_param_.mrope_mode != MropeMode::kNone` (compile blocker — `kMrope` removed from the enum; the `mrope` sub-struct is kept by main, so the rest compiles unchanged). Vision env-source branch is W1. **Phase 1, not W1.** |
| `input_processor.cc` + `input_processor.h` | Take **main's** `.h` (4-arg `PatchEmbedding(..., env)`, since `language_model.cc` auto-merges to the 4-arg call) and adapt our `.cc` to the 4-arg signature (`env` unused until W1). |
| `model_executor.{h,cc}` | Take **ours** for both (only `engine.cc`, resolved-to-ours, constructs `ModelExecutor`). Force `model_executor.h` to ours. Vision param + `vision_model_->Run` added W1. |
| `model_root.h` | Force **ours** (no `vision_model_ptr()`/vision child needed in Phase 1). Restore main's in W1. |
| `output_processor.cc` | Take **ours** (`OutputRange`); CE-loss W2. |
| `turbomind.cc` | Take **ours** (text-only Engine construction); vision construction W1. |
| `bind.cpp` (auto-merged) | **Keep** the vit bindings (header-only — compile as-is once `Request::mm_inputs` exists) and our `Cancel` binding. **Strip** main's `SessionParam` `start`/`end` bindings and the `End(session_id)` kill binding (reference removed members). Keep the health `scheduler_tick` binding. |
| `model_request.{cc,h}` (auto-merged) | Keep ours + the auto-merged `r->mm_inputs = param.mm_inputs` and get_ppl `ce_loss` alloc. **Drop** any leaked `End(...)` decl/def and `session.start_flag/end_flag`. `param.session.id/step` stay (our id/step carrier). |
| `messages.py` | **Union** doc; keep main's `quant_policy` wording. |
| `SequenceManager.cc/.h` | **Keep deleted**; port `multimodal_inputs` *intent* onto `engine/request.h::Sequence` in W1. |
| CMake (`models/`, `kernels/norm/`, `python/`) | Comment out vit-specific **`.cc`/`.cu`** units (`qwen3_5vit/*`, `vision_model.cc`, `layer_norm_weight.cc`, `kernels/norm/layer_norm.cu`); keep core rope/attention **and all headers**. Restore W1. |

## 6. Feature integration workstreams

### W1 — qwen3.5-vit (largest)

Add `qwen3_5vit/`, `vision_model.*`, `layer_norm_weight.*` to CMake. Re-home the
multimodal carrier onto the new per-rank `Sequence` (`engine/request.h:154`), beside
`input_embeds`:

```cpp
// src/turbomind/engine/request.h
struct MultiModalData;  // defined in models/vision_model.h

struct Sequence {
    ...
    std::vector<Tensor> input_embeds;
    std::vector<int>    input_embeds_offsets;

    // persistent vision features (e.g. qwen3.5-vit), per-sequence
    std::vector<std::shared_ptr<MultiModalData>> multimodal_inputs;
    ...
};
```

Then:

- **Restore main's vision-aware headers** force-kept-ours in Phase 1
  (`model_executor.h`, `model_root.h`, `input_processor.h`) and uncomment the CMake +
  `bind.cpp` vit units.
- Thread `std::unique_ptr<VisionModel>` + `const ModelWeight&` through
  `Engine`/`Engine::Impl` ctors and `turbomind.cc` (re-add the params dropped in
  Phase 1).
- Run the vision encoder before restore copies in `model_executor.cc`:

```cpp
if (vision_model_) {
    vision_model_->Run(BatchOp::kPrepare, d.phase, env);
}
RunCopies(d.restore_copies);
```

- Drop main's interactive-only guards (stateless model): in `qwen3_5vit.cc::Add`,
  remove `if ((not r.session.start_flag) or (not r.session.end_flag)) return kInvalid;`
  and map main's `RequestCache` to our `Sequence`; in `input_processor.cc`, drop the
  `if (!r.session.end_flag) clone` branch (§4).
- mrope: the `MropeMode` plumbing already landed in Phase 1; W1 only feeds the
  `env.try_("mrope_length")` source from the C++ vision encoder and consumes `env` in
  the 4-arg `PatchEmbedding`.

### W2 — get_ppl / CE-loss

Re-apply main's `output_processor.cc` CE-loss onto our `OutputRange` structure; the
`request.h` fields (`return_ppl`, `input_ce_loss`, `ce_loss`) land in Phase 1; graft
the `return_ppl` clause into our `Validate`; wire the `/get_ppl` endpoint.

### W3 — cp inference fix

`cp_utils.*` land clean; keep the 3-line `unified_attention_layer.cc` addition;
`causal` already taken in Phase 1.

### W4 — misc

Health endpoint (clean add); `scheduler_tick` metrics revival onto the new scheduler
(deferred / optional).

## 7. Verification (tiered)

GPU runs must execute **outside the sandbox** (no driver in sandbox). The
`model-server` MCP is not enabled this session, so model paths are provided manually.

- **Hardware:** H200 (141 GB VRAM) — `Qwen/Qwen3.5-27B` runs safely at `--tp 1` on a
  single H200; still confirm a free GPU before launching.
- **Phase 1:** `ninja` green; `scripts/test_turbomind_model.py` (≥128 tokens, response
  verified coherent — gibberish = bug) on **`Qwen/Qwen3.5-27B`**
  (cache dir `/mnt_cfs/huggingface_hub/hub/`; set `HF_HUB_OFFLINE=1` and
  `HF_HUB_CACHE=/mnt_cfs/huggingface_hub/hub/` before loading per repo rule). Qwen3.5
  is a hybrid linear-attention model, so this one model covers both the text and the
  gated-deltanet/SSM path; confirm it exercises `GatedDeltaNetLayer`.
- **W1:** VL image inference on the tiger image
  `https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg`
  with a qwen3.5-vl model if model + GPU are available, else smoke-verify and flag.
  (No qwen3.5-vl model path has been provided yet — `/data/models.json` lists none —
  so this check is best-effort pending a model.)
- Always check `get_gpu_usage` / an empty GPU before running; on OOM check for other
  processes.
- Do **not** modify `scripts/test_turbomind_model.py`; do **not** batch tests in a
  bash loop.

## 8. Risk register & semantic-drift audit

- **Rope incompatibility (compile blocker):** main removed `RopeType::kMrope` and
  `llama_rope.h`/`rotary_embedding.h`/`attention_weight.cc` are taken from main, so our
  `unified_attention_layer.cc` must adopt `MropeMode` in Phase 1 (handled in §5 map).
- **Attention causal threading:** ~30 `kernels/attention/*` files taken from main thread
  a `causal` flag; `attention_universal.h` auto-merges (disjoint regions, verified safe),
  but the merged tree compiles *our* cache code against *main's* kernels — the Phase-1
  build is the real check.
- **Header/impl mismatch:** main changed `input_processor.h` (4-arg `PatchEmbedding`),
  `model_executor.h` (`VisionModel*` ctor), `model_root.h` (vision child); paired `.cc`
  resolved to ours. §5 map fixes each (take-main-`.h`+adapt `.cc`, or force-ours `.h`).
- **`bind.cpp` always compiles:** verified the auto-merged vit bindings are **header-only**
  (compile as-is with `Request::mm_inputs` present) — *not* the problem. The real leak is
  main's `SessionParam` `start`/`end` bindings + the `End(session_id)` kill binding, which
  reference removed members → strip in Phase 1 (Task 1.8).
- **Session/kill leak via auto-merge:** `bind.cpp`, `model_request.cc/.h` auto-merged with
  main's stateful code. Remove `start_flag`/`end_flag`/`End`; keep `Cancel` and the
  `param.session.id/step` carrier. Grep precisely (`start_flag|end_flag|kill_flag|kill_reqs|seq_mgr_|End(`),
  not bare `session` (which legitimately matches `SessionParam`/`session_id_`/`session_len_`).
- **vit hooks in auto-merged regions:** main's vit changes to `language_model.cc`
  (4-arg `PatchEmbedding`) and `input_processor.cc` — audit. (`unified_decoder.cc` is
  ours-only — main did not touch it — so no audit needed there.)
- **SSM / linear-attention contract:** changes to scheduler / cache / `BatchOp` must
  preserve the normative contract in `src/turbomind/engine/README.md` (reference items
  as `<section>.<leaf>`, e.g. `contracts.batchop-schedule`, `checklist.cache-memory`).
- **CMake correctness:** vit sources auto-merge into `models/CMakeLists.txt` /
  `kernels/norm/CMakeLists.txt` / `python/CMakeLists.txt`; Phase 1 excludes the
  vit-specific units (not the core rope/attention), W1 re-includes them.

> **W1 execution update.** qwen3.5-vit is integrated and verified. The Engine
> ctor threads only `std::unique_ptr<VisionModel>` (main's `weights` param was
> dead in our tree — its only consumers, `SequenceManager` and the
> `has_linear_attention` guard, were removed). The vit `.cc` was translated from
> main's `RequestCache`/`r.session` model to our `Sequence` (`env.at("requests")`
> → `Buffer_<Sequence*>`, `alpha` → `inflight_input_len`, interactive guard
> dropped); the attention layer borrows the encoder's mrope tensors via an
> `env.try_("mrope_length")` source branch. Verified on H200/tp1: text/SSM
> regression (`Qwen3.5-27B`) PASS, and VL runs of the same model passed
> end-to-end — an accurate description of `tiger.jpeg` and correct reading of the
> `resources/batch_memory.png` chart axis labels (full encoder + embed-merge +
> mrope path). See the plan's "W1 execution notes" for details.

> **W2 execution update.** get_ppl / CE-loss is integrated and verified. The whole
> Python `/get_ppl` surface (`messages.py`, `turbomind.py` `_get_ce_loss` +
> `c.return_ppl`, `pipeline.get_ppl`, `async_engine.async_get_ppl`, `api_server`
> `/get_ppl`, `bind.cpp` `return_ppl`) and `model_request.cc`'s `outputs["ce_loss"]`
> alloc already landed in the structural merge, so only the C++ compute and the
> admission guard were missing. The CE compute was ported onto our `OutputRange`
> arch as a self-contained `CeLossSegment { request, ce_loss, range, last }` captured
> at Setup — the executor side never touches `Sequence` (main's `b.rc` is gone). The
> per-request `ce_loss` accumulator persists on the `Sequence` across chunked-prefill
> forwards; `last = !c.input_ce_loss` (post-erosion) gates the single emit; the type-2
> trigger became `d.full_logits || d.full_ce_loss` (a no-op for non-ppl). `engine.cc`
> `Validate` now also rejects `return_ppl` under prefix caching. Verified on H200/tp1:
> text regression PASS, ppl smoke (coherent mean-NLL `2.34` < garbled `4.25`), VL
> regression still PASS. See the plan's "W2 execution notes" for details.

> **W3 execution update.** The cp inference fix (#4619 / `6276b3bd`) required no
> code changes — three of its four hunks landed byte-for-byte in the structural
> merge (`cp_utils.cu` `invokeFillNegInfML`, `cp_utils.h` decl, and the 3-line
> `unified_attention_layer.cc` `attn_cp_size > 1` init in the Clear block), and
> `cp_utils.cu` is already in the attention CMake target. The fourth hunk (a
> WARN→INFO `#victim` log in the deleted `SequenceManager.cc`) has no target in the
> refactored scheduler. cp was verified end-to-end (not deferred): a `tp=2`/`cp=2`
> run of `Qwen3.5-27B` on 2×H200 (cp is a sub-division of tp) produced coherent
> output for an async, varying-length batch including an early-finishing sequence —
> the exact finished-sequence/stale-`partial_ML` path the fix targets. See the
> plan's "W3 execution notes" for details.

> **W4 execution update.** Final phase done. All clean-add commits from
> `origin/main` are in the tree (`origin/main` is the merge-base — i.e. fully an
> ancestor of `HEAD`). The health-endpoint surface (`async_engine.health_probe`,
> `EngineHealthMonitor`, turbomind `get_health_status`/`ScheduleMetrics` binding)
> is present. TurboMind `UpdateScheduleMetrics` is still stubbed, so
> `GetScheduleMetrics` returns null → Python `None`; turbomind.py
> `get_schedule_metrics()` previously dereferenced that `None` and broke `/health`.
> Added two minimal None-guards (graceful degradation, **not** metrics revival):
> `get_schedule_metrics()` returns `None` when the backend has no metrics, and
> `metrics_processor.update_schedule_stats()` skips on `None`. With these, the probe
> reports healthy-when-idle (matching the documented "empty metrics" behavior).
> Clean rebuild (`ninja -t clean && ninja`, 482 targets) green; final text/SSM
> verification on `Qwen3.5-27B` PASS. See the plan's "W4 execution notes" for details.

## 9. Out of scope / deferred

- **Reviving `ScheduleMetrics` onto the new scheduler** (pre-existing gap from
  `b189745a`; `Engine::Impl::UpdateScheduleMetrics` is a stub). Consumers degrade
  gracefully (health probe = healthy-when-idle; metrics logger skips schedule stats),
  but the `/health` scheduler-stall detection and the Prometheus schedule gauges are
  inert until metrics are wired onto the new scheduler. Not required by any merged
  feature.
- Any revival of stateful sessions / interactive mode (§4 — permanently removed).
- **Verified, not deferred:** VL (W1, `tiger.jpeg` + `batch_memory.png`), get_ppl/CE
  (W2, ppl smoke), and cp multi-GPU (W3, `tp=2`/`cp=2` on 2×H200) were all confirmed
  end-to-end on `Qwen3.5-27B`.
