# 07: Make the cache-registration pass real

**What to build:** The registration walk of the spec's lifecycle becomes real: modules register their per-layer cache byte requirements with the cache registry during the dedicated pass, and the engine enforces the ordering — registration completes before category finalization, bind before the first run. At this ticket's close the walk is wired and exercised by the registry's own accounting; the attention and GDN modules that actually register arrive in the next two tickets.

**Blocked by:** 03 (Linear executable module).

**Status:** resolved

- [x] Registration walk runs after prepare and before category finalization, driven engine-side (not by Python loops)
- [x] Modules that register ranges keep only their offsets; the registry maps categories to object ids after all registrations
- [x] Models whose modules register nothing load and run unchanged
- [x] Build green; core-model smoke unchanged

## Answer

**Shape landed** (commit "Make the cache-registration pass real (ticket 07)"):

- `core::Module` grows `virtual void register_cache(::turbomind::CacheRegistry&)` — default
  recurses into children via `for_each_child`, exactly mirroring `bind`; `CacheRegistry` is
  forward-declared beside `Context`/`BatchOp` in module.h. No module overrides it yet
  (Attention/DeltaNet arrive in tickets 08/09), so today's trees walk as a no-op.
- Engine side: `TurboMind::Impl` owns `cache_registries_` (one per rank). `RegisterCache`
  creates the rank's registry (checkpoint interval from the engine config), runs
  `weights_[index]->register_cache(...)` over the tree, and is **idempotent** — `update_params`
  reloads re-run the pass after the previous registry moved into the live engine.
  `CreateEngine` uses the pass-created registry (TM_CHECK_NOTNULL enforces the pass ran;
  the pointer resets after the registry moves into the engine). Category finalization
  (`RegisterObjectIds`) stays inside CreateEngine, after the pass and after the bounded-
  coexistence construction-time registrations from the engine-side layer classes, which keep
  registering into the same registry until 08/09 retire them. `CacheCategory`'s
  close-on-finalization check (`object_id_ < 0`) already rejects any late registration.
- Normative order end to end: process_weight (prepare) → register_cache → category finalize
  (internal, in CreateEngine) → bind → engine start; the Python driver sequence from ticket 04
  drives it unchanged (only the `_register_cache` docstring was stale and is updated).

**Gate:** build green; all four gate models (Qwen3-8B, Qwen3-8B-AWQ, Qwen3-30B-A3B,
Qwen3.5-35B-A3B-FP8) load and run unchanged with meaningful 250+ token responses. Reviewer
blocker (double-run guard breaking repeated `update_params`) fixed by making the pass
idempotent; reviewer-verified that `param`/`engine_param_` were equivalent sources for the
checkpoint interval and that no other CacheRegistry construction site exists.
