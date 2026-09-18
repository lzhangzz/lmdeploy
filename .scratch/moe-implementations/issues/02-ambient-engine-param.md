# 02: Ambient EngineParam in the build context

**What to build:** Context creation parks the rank's EngineParam in the
thread-local build context, beside the stream and allocator, so that module
construction can read engine policy and sizing under the parameter's own name.
No consumer yet; no behavior change. This is the prefactor that lets the MoE
factory select an implementation at construction without touching any config
class.

**Blocked by:** None (can start immediately; parallel to 01).

**Status:** resolved

- [ ] After context creation, the rank's EngineParam is readable ambiently from
      the build context at module-construction time
- [ ] Correct per rank under the Python side's multi-rank thread pool (each
      rank's loading thread sees its own engine params)
- [ ] No behavior change: `ninja` clean and a default-path model run via
      `scripts/test_turbomind_model.py` still answers meaningfully

## Answer

Implemented as specced (2026-09-18), with one mechanics correction discovered
during recon: `CreateContext` runs on the Python ThreadPoolExecutor loading
threads, but module construction (`_tm.create_module`) runs on the Python main
thread — so parking at context creation would never be visible at
construction. The parking point is the per-rank weight-context guard, the
only per-rank object alive on the building thread at `create_module` time:

- `EngineParamGuard` (declared beside `BindEnv` in `models/llama/context.h`):
  RAII parks the rank's `EngineParam` in a thread-local; `current()` returns
  it (nullptr outside a guard — standalone paths).
- `TurboMind::weight_context(index)` now returns the stream, allocator, and
  `const EngineParam*`; the pybind `context(g)` guard enters the
  `EngineParamGuard` in `__enter__` beside the `core::ContextGuard` (parking
  happens at guard enter, not guard creation — guards are all created
  up-front per rank, so creation-time parking would clobber ranks).
- Per-rank correctness under the thread pool follows automatically: each
  rank's building thread enters its own rank's guard before constructing that
  rank's modules.
- No consumer yet (ticket 03's MoE factory is the first). Gate: `ninja`
  clean; default-path `Qwen/Qwen3.5-35B-A3B` run answers meaningfully.
