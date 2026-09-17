# 04: Python delta and stage-1 gate

**What to build:** The effort's entire Python-side change, and the checkpoint that closes stage 1. The standalone torch-linear path constructs an engine-root Context and binds it to its module after prepare (replacing the retired executor object); the two new driver entry points appear — the registration walk (still an empty pass at this stage) and the bind walk — and the model-load driver calls them in the spec's sequence. Then run the stage-1 gate: the frozen-surface checks and the four-model matrix with the perf comparison.

**Blocked by:** 01 (Record perf baselines), 03 (Linear executable module).

**Status:** resolved

- [x] Standalone torch-linear path works via Context construction + bind; public API names unchanged
- [x] Registration and bind entry points exist and the load driver calls them in the spec's order
- [x] Frozen-list check passes (no new or renamed Python public names) and builders are byte-identical
- [x] Four core models pass end-to-end with meaningful responses of at least 128 tokens, never batched
- [x] Decode throughput at or above 98% of baseline with load time within noise

## Answer

**Shape landed** (commit "Retire LlamaLinear, land the Python delta (ticket 04)"):

- C++: `_tm.Context` binds the engine root (`py::init<int>` + `gemm()` accessor returning the
  `gemm::Gemm` weight-plan surface + `import_records`/`export_records` on the engine);
  `core::Module` gains a Python `bind(ctx)`; `TurboMind::RegisterCache`/`Bind` are separate
  bound entry points. The bind walk moved out of `Impl::ProcessWeights` into `Impl::Bind` so
  the Python driver sequence is normative: create_root → export → process_weight (prepare) →
  register_cache (empty pass until stage 3) → bind → create_engine. `LlamaLinear.{h,cu}`
  deleted; removed from models/CMakeLists.txt; friend/forward-decls in kernels/gemm/{gemm.h,
  exec_plan.h} cleaned (Linear friend kept); stale API comments reworded.
- Python (the effort's entire delta): `linear.py` constructs `_tm.Context(device)` per device,
  binds each module after `prepare()` (dense in `_prepare_one`, grouped destination after
  `LinkLinearExperts`), `get_weight_plan` via `self._engine.gemm()`, and
  `get_exec_plan`/`tune`/`forward_dense`/`forward_moe` rebind onto the module
  (`_tm.LinearWeight` methods, weight argument dropped). `turbomind.py` gains
  `_register_cache()`/`_bind()` called in the spec order in both `__init__` and
  `update_params`. Builders and `weight_format.py` byte-identical (gate-checked, zero diff).

**Gate results:**

- Frozen-list: no new/renamed Python public names; deliberate additions exactly
  `_tm.Context`, `TurboMind.register_cache`, `TurboMind.bind`, `Module.bind`
  (sanctioned by python-binding-rules.md); `_tm.LlamaLinear` gone. Builder import smoke OK;
  standalone path exercised by `tests/turbomind/linear/test_linear.py` (8 passed).
- Four models end-to-end (never batched, 256 generated tokens each, meaningful):
  Qwen3-8B, Qwen3-8B-AWQ, Qwen3-30B-A3B-Thinking-2507, Qwen3.5-35B-A3B-FP8.
- Perf (record: `perf/stage1.md`): dense-bf16 99.3–101.8%, dense-awq 100.4–102.7%,
  moe-bf16 99.5–102.0% of recorded baseline — pass outright. moe-fp8 measured 83–88% of
  its recorded baseline; bisect proved this is **environmental drift of the recorded fp8
  baseline, not a code regression**: the unmodified baseline commit re-measured this
  afternoon lands at 2036.6 (GPU 4) / 2062.6 (GPU 2) vs the recorded 2451.5, and every
  tree measured same-session clusters at 2023–2065 (ticket-04 tree at 100.7% of the
  baseline commit on the same GPU/session). Gate passed on the same-conditions comparison.
  Later gates should A/B against a freshly re-measured fp8 baseline; the recorded fp8
  numbers in `perf/baseline.md` are stale relative to the current machine.
- Loads: 6.08/7.00/16.07/20.70 s vs baseline 6.99/6.24/16.25/20.05 (same order as above) —
  within the session's drift noise given the fp8 finding.
