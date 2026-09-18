# 01: Bind pass redesign — BindEnv and the transient state map

**What to build:** The bind pass carries its own environment and nothing else.
Every kind's per-rank state (Attention, DeltaNet, and MoE's current single state)
is created by the first module of the kind during the walk and shared with its
siblings through a transient, type-erased map that dies when the walk ends; the
engine's bind step only builds the environment and walks the tree; the Context
holds no module state of any kind afterwards. Model behavior is identical.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [ ] `bind(BindEnv&)` replaces `bind(Context&)`, with exactly: `const
      EngineParam& engine`, `turbomind::Context& context`, `CacheRegistry&
      registry`, `int& is_warm_up`, the shared-state map
      (`std::unordered_map<std::string, std::shared_ptr<void>>`), and `int&
      phases` — defined beside the Context, forward-declared from core as the
      Context is
- [ ] Attention, DeltaNet, and MoE states are created by the first module of
      the kind during the walk (sized from its own config fields, the engine
      params, and the context's stream/allocator); the rest fetch and cast
- [ ] The Context's kind-state slots and accessors are deleted; the kind-module
      scan and pre-walk state creation are deleted; Linear captures the GEMM
      engine via the env's context, unchanged
- [ ] DeltaNet's recurrent grid still registers while the cache category is
      open; per-layer registration and layer-order enforcement intact
- [ ] The standalone `Module.bind(context)` Python surface is unchanged (the
      pybind shim assembles a degenerate env internally)
- [ ] `ninja` clean in the build folder; `Qwen/Qwen3.5-35B-A3B` via
      `scripts/test_turbomind_model.py` (AS IS, never batched) answers with a
      meaningful ≥128-token response; standalone linear path smoke passes

## Answer

Implemented as specced (2026-09-18).

- `Module::bind(::turbomind::BindEnv&)` replaces `bind(Context&)`; `BindEnv`
  is defined beside the Context in `models/llama/context.h` with exactly the
  six approved members (`engine`, `context`, `registry`, `is_warm_up`,
  `shared`, `phases`), forward-declared from `core/module.h` as the Context
  is. Aggregate-initialized in `Impl::Bind` from
  `engine_params_[index] / ctx / *cache_registries_[index] / *ctx.is_warm_up /
  shared / phases_`.
- Attention ("attention"), DeltaNet ("deltanet"), and MoE ("moe" — merged
  state until ticket 03 splits it) states are created by the first module of
  the kind inside the walk; siblings fetch and `static_pointer_cast`. All
  modules co-own via `shared_ptr`; the map dies when `Impl::Bind` returns.
  DeltaNet's state keeps its exact ctor: the first binder collects the rank's
  DeltaNet modules by climbing to the decoder-layer list (the retired
  `Impl::Bind` scan, now module-side), so `RegisterLayer`'s prefilled-vector
  bind-order check and the bind-time recurrent-grid registration are
  verbatim.
- The Context's three kind-state slots, their accessors/setters, and
  `Impl::Bind`'s kind-module scan are deleted. The whole bind walk now runs
  under the engine guard (`ctx.core_stream, ctx.allocator,
  Allocator{kCPUpinned}`) — identical allocation view for the state ctors,
  which are the only allocators in the walk. Linear captures the GEMM engine
  via `env.context.linear_engine()`, unchanged.
- The pybind shim keeps the Python `Module.bind(context)` surface and
  assembles the degenerate env internally (default EngineParam, empty
  CacheRegistry, local flags, fresh map).
- Gate: `ninja` clean; `Qwen/Qwen3.5-35B-A3B` (MoE + GDN: covers default MoE,
  Attention, and DeltaNet state migrations in one run) via
  `scripts/test_turbomind_model.py` AS IS answered 256 meaningful tokens
  (load 21.8 s, inference 2.3 s); standalone linear smoke (dense
  `prepare_weight` → `impl.bind(engine)`, and grouped `LinkLinearExperts` →
  `bind`) passed on the shim.

Note: update_params is no longer supported (user, 2026-09-18); reload
semantics are moot. Doc references to update_params re-binds were dropped.
