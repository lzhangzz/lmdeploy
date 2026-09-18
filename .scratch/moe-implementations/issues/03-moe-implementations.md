# 03: MoE implementations as module subclasses

**What to build:** The two MoE implementations become executable module
subclasses under the single frozen `"MoeWeight"` registry string — selected at
construction by a custom factory reading the ambient EngineParam — each holding
its layer's weights, each owning its per-rank state on the bind-pass map. The
merged state struct dies. Behavior is verbatim; the DeepEP path remains compiled
but unexecuted.

**Blocked by:** 01 (bind pass redesign), 02 (ambient EngineParam).

**Status:** resolved

- [ ] The MoE base keeps the frozen type string, children/params surface,
      config-field ctor, `prepare`/`link_block`, shared `Gate`, and declares a
      pure-virtual `Forward` with today's four arguments; the decoder's call
      surface is unchanged
- [ ] `MoeDefaultImpl` and `MoeDeepEPImpl` subclass the base, each in its own
      file pair together with its state; bodies are verbatim moves of the
      current internal branches; the DeepEP in-place output contract is
      documented on the declaration
- [ ] One custom factory under the frozen `"MoeWeight"` string selects from the
      ambient EngineParam: `"default"` → default subclass, `"deepep"` → DeepEP
      subclass, anything else fatal; the bfloat16 invariant lives on the DeepEP
      side; no expert-parallel-size conditioning
- [ ] `MoeDefaultState` (routing scratch) and `MoeDeepEPState` (dispatcher,
      top-k buffers, scales-copy stream/events) rendezvous on the map under
      `"moe.default"` / `"moe.deepep"`; buffers allocate on the engine build
      stream/allocator; the merged state struct and its slot are gone
- [ ] The MoE config class and the entire Python surface are untouched
- [ ] `ninja` clean (the DeepEP path compiles); `Qwen/Qwen3.5-35B-A3B`
      default-path run via `scripts/test_turbomind_model.py` answers with a
      meaningful ≥128-token response; the DeepEP-unexecuted caveat holds

## Answer

Implemented as specced (2026-09-18), landed in commit d0d5f6213.

- `Moe` base (`models/moe.{h,cc}`): frozen `type() == "MoeWeight"`, the
  X-macro children/params surface, the config-field ctor, `prepare` /
  `link_block`, the shared static `Gate`, and a pure-virtual `Forward`
  with today's four arguments; the base bind descends into the linked
  block. `MoeDefaultImpl` and `MoeDeepEPImpl` subclass it, each in its own
  file pair together with its state. Move-verification was mechanical: the
  five moved bodies (`ForwardDefault`, `ForwardA2A`, `InitDefault`,
  `InitA2A`, `CompileKernels`) diff against the pre-split file with zero
  changed lines except the added `TM_CHECK_NOTNULL(state_)` in the two
  forwards. The DeepEP in-place output contract is documented on the
  subclass declaration.
- One custom factory under the frozen `"MoeWeight"` string
  (`ModuleRegistry::register_type(name, callable)`) selects from
  `EngineParamGuard::current()`: `"default"` → default subclass,
  `"deepep"` → DeepEP subclass (USE_NCCL builds), anything else
  `TM_LOG_FATAL`; null ambient params fatal with a clear message. The
  bfloat16 invariant lives in `MoeDeepEPState`'s constructor. No
  expert-parallel-size conditioning anywhere in selection.
- `MoeDefaultState` (routing scratch) and `MoeDeepEPState` (dispatcher,
  top-k buffers, scales-copy stream/events) rendezvous on the bind-pass
  map under `"moe.default"` / `"moe.deepep"`; buffers allocate on the
  engine build stream/allocator (the walk runs under the engine guard).
  The merged `MoeState` struct, its Context slot, and its accessors are
  gone.
- `moe_a2a_backend = "auto"` never reaches the factory: the resolution
  moved to Python config normalization (`update_parallel_config`), now
  unconditional (`deepep` only when `ep > 1 and nnodes > 1`, else
  `default`) — user decision after rejecting an engine-side resolution in
  `CreateContext`. Same selection table as the old merged code for every
  previously working config.
- The MoE config class and the Python surface are untouched; the decoder
  calls the same `Forward(input, output, token_mask, local_token_nums)`
  through the base pointer.
- Gate: `ninja` clean (DeepEP compiles under USE_NCCL);
  `Qwen/Qwen3.5-35B-A3B` default-path run via
  `scripts/test_turbomind_model.py` AS IS answers 256 meaningful tokens,
  output indistinguishable from the pre-split run. DeepEP remains
  compiled-but-unexecuted by gates (per ADR 0001).

### DeepEP verification addendum (2026-09-18, post-landing)

On request, the DeepEP implementation was executed for the first time:
`Qwen/Qwen3.5-35B-A3B`, bf16, tp=2/ep=2 on one node (2 GPUs, nccl), driven
by a throwaway script outside the repo mirroring the normative smoke flow
(the normative script cannot express `moe_a2a_backend='deepep'` on a single
node — `auto` resolves to `default`). The factory selected
`MoeDeepEPImpl` from the ambient EngineParam; `MoeDeepEPState` initialized
the token dispatcher over NCCL symmetric memory (per-rank `symm_ctx`
warnings are its footprint) and ran the bind-time a2a JIT; forward
dispatch/combine produced a meaningful 256-token response (load 22.6 s,
inference 2.8 s). Two driver corrections along the way are environment
facts, not code: the ep spelling is `tp=N, ep=N` (enforced by
`derive_parallel_config`), and the one-off driver must not live in `/tmp`
directly (an unrelated `/tmp/h2.py` shadows the `h2` package).

### DeepEP verification: ep=4 and ep=8 (2026-09-18, same day)

Same driver and model, `tp=4/ep=4` on 4 GPUs and `tp=8/ep=8` on 8 GPUs:
every rank initialized the dispatcher's NCCL symmetric memory, and forward
dispatch/combine produced a meaningful 256-token response at both sizes
(load 70.3 s / 67.7 s — the bind-time a2a JIT dominates — inference 2.7 s /
4.5 s). DeepEP is now verified end-to-end at ep=2, 4, and 8.
