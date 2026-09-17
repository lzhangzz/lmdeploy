# Engine services contract: Context and per-kind states

Draft spec section, rewritten by [Rework the ExecContext design](../issues/10-rework-execcontext-design.md); supersedes the v1 ExecContext design recorded in [Context shared-state contract](../issues/02-context-shared-state-contract.md). Amendment history at the end.

## Shape

- **`turbomind::Context` survives** (today's `models/llama/context.h` object) as the engine-side composition root: `core_stream` + allocator (build-time), the per-rank **GEMM engine** (`gemm::Gemm` + dispatch cache + workspace + measure policy — the contents of today's `LlamaLinear::Impl`), `device_prop`, comm groups. `is_warm_up` moves to the run environment (engine-owned, `env.warm_up()`).
- **Per-kind shared state objects exist only for kinds with cross-layer state**: `AttentionState`, `MoeState`, `DeltaNetState`. There is no `LinearState` and no Ffn state — Linear modules capture the GEMM engine itself; Ffn's GEMMs ride its Linear children.
- **Kind states are the retired executors' state, once per rank**: `AttentionState` = per-phase `AttentionBatch` slots, the fp32 workspace, pinned staging, CP context, aux stream + events. `MoeState` = `TokenDispatcher`, routing buffers, scales-copy stream — shared per rank, reversing the earlier per-module amendment. `DeltaNetState` = per-kind workspaces.
- **`bind(Context&)` runs after initialization**: the engine walks the module tree once; each module type captures typed references to what it needs. Stored references, no per-call service passing.

```cpp
// core::Module grows (default recurses into children):
virtual void bind(turbomind::Context& ctx);

// capture examples:
void Linear::bind(Context& ctx)    { engine_ = &ctx.linear_engine(); }
void Attention::bind(Context& ctx) { state_ = &ctx.attention_state(); }
void Moe::bind(Context& ctx)       { state_ = &ctx.moe_state(); }
```

## Lifecycle (normative pass order)

```text
Python builds tree -> prepare() walk -> register_cache(CacheRegistry&) walk
  -> category finalization -> bind(Context&) walk -> first Run
```

## What modules hold

- Captured references from `bind` (engine or kind state) — the sanctioned aliases.
- Weights and packed forms, per-layer cache offsets, and whatever their kind design places on the module.
- Never: per-module copies of the GEMM engine, per-kind workspaces, or per-layer duplicates of kind state — the kind state holds them once per rank.

## Warm-up and tuning

`env.warm_up()` per run. `set_measure`/`Export`/`Import`/`GetTuningSeq` stay on the GEMM engine object in `Context` (the old `LlamaLinear` surface, state-owned); `Linear::Tune` remains a thin module method for the `linear.py` pin. Dispatch-cache mutation stays executor-thread-only; measure mode only during warm-up.

## Standalone path

`linear.py` constructs a `turbomind::Context` (engine-lite instance) and calls `mod.bind(ctx)` after `prepare` — the module captures the engine reference. Identical story to the engine path; no separate standalone object.

## Thread placement

Carried over from v1 where still applicable: GEMM engine + dispatch cache on the model executor thread only; kind-state buffer contents executor-thread; comm/props immutable after build; engine-thread ops stay metadata-only per `contracts.object-address` / `contracts.executor-only`.

## Amendment history

1. v1 (`ExecContext` per-rank services object) — registry access through it rejected; fixed by the separate `register_cache` pass (Merged Attention module design).
2. Standalone construction of `ExecContext` rejected; per-call `Services` passing (minimal and bundle) rejected as the wrong direction.
3. Final shape (this document): surviving `Context` + per-kind states + `bind` capture; MoE transport reversed to shared `MoeState`; the `ExecContext` name is dead.
