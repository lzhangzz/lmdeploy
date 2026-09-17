# Rework the ExecContext design

Type: grilling
Status: resolved

## Question

The ExecContext shape decided in [Context shared-state contract](02-context-shared-state-contract.md) needs rework: in practice it has drawn two rejections — cache-registry access stretched through it (already fixed by the separate `register_cache` pass in [Merged Attention module design](05-merged-attention-module-design.md)), and the standalone Torch Linear path, where constructing an engine-shaped per-rank services object (comm handles, warm-up state, per-kind pools) for a single unbound module is the wrong shape. Decide the replacement: keep one services object but slim or split it (e.g. GEMM service, comm views, workspace pools as separately injected dependencies), or drop the central object for typed-parameter passing at the call boundary (TVM/MLC precedent from the prior-art research); settle the standalone/unbound GEMM backing story (self-hosted private engine as the module's unbound mode vs a slim standalone object vs a facade binding); and compile the amendment trail across the drafted assets — `drafts/context-contract.md`, the `bind` pass in `drafts/attention-design.md` and `drafts/batchop-decoder-mapping.md`, and the Linear prototype's `xc_` member. The resolution amends the recorded decision of Context shared-state contract where needed and unblocks the deferred sub-question of [Python binding-surface rules](07-python-binding-surface-rules.md).

## Answer

Contract rewritten at [`drafts/context-contract.md`](../drafts/context-contract.md); amendment notes stamped on the Linear prototype and the attention, BatchOp, and FFN/MoE drafts. Decisions:

- **`turbomind::Context` survives** as the engine-side composition root: the per-rank GEMM engine (`gemm::Gemm` + dispatch cache + workspace + measure policy, today's `LlamaLinear::Impl` contents), build-time stream/allocator, device props, comm. The `ExecContext` name is dead; the glossary entry reverts to `Context`.
- **`bind(Context&)` after initialization**: one pass over the tree; each module type captures typed references to what it needs. Linear captures the GEMM engine (`engine_ = &ctx.linear_engine()`); there is **no `LinearState`** and no Ffn state.
- **Per-kind shared state objects only for kinds with cross-layer state**: `AttentionState` (per-phase batch slots, fp32 workspace, pinned staging, CP context, aux stream), `MoeState`, `DeltaNetState` — the retired executors' state, once per rank, bound to every module of the kind.
- **MoE transport reversed to shared**: `MoeState` holds the `TokenDispatcher`, routing buffers, and scales-copy stream per rank — reversing the per-module amendment from [Merged FFN and MoE module design](04-merged-ffn-and-moe-module-design.md); the memory/JIT multiplication concern and its migration-gate rationale dissolve.
- **Standalone path**: `linear.py` constructs a `turbomind::Context` and calls `mod.bind(ctx)` after `prepare` — same story as the engine path; this answers the deferred sub-question of Python binding-surface rules, which is now unblocked.
- **Pass order stands**: prepare → `register_cache` → finalize → `bind(Context&)` → first Run. Warm-up via `env.warm_up()`; tuning ops on the engine object in `Context`; thread rules carried over from v1 where applicable.
- Rejected along the way (recorded so the spec review sees the space explored): registry-through-ExecContext, standalone ExecContext construction, per-call `Services` passing (minimal and bundle variants).
