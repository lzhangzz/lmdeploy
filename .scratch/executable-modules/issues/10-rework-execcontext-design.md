# Rework the ExecContext design

Type: grilling
Status: open

## Question

The ExecContext shape decided in [Context shared-state contract](02-context-shared-state-contract.md) needs rework: in practice it has drawn two rejections — cache-registry access stretched through it (already fixed by the separate `register_cache` pass in [Merged Attention module design](05-merged-attention-module-design.md)), and the standalone Torch Linear path, where constructing an engine-shaped per-rank services object (comm handles, warm-up state, per-kind pools) for a single unbound module is the wrong shape. Decide the replacement: keep one services object but slim or split it (e.g. GEMM service, comm views, workspace pools as separately injected dependencies), or drop the central object for typed-parameter passing at the call boundary (TVM/MLC precedent from the prior-art research); settle the standalone/unbound GEMM backing story (self-hosted private engine as the module's unbound mode vs a slim standalone object vs a facade binding); and compile the amendment trail across the drafted assets — `drafts/context-contract.md`, the `bind` pass in `drafts/attention-design.md` and `drafts/batchop-decoder-mapping.md`, and the Linear prototype's `xc_` member. The resolution amends the recorded decision of Context shared-state contract where needed and unblocks the deferred sub-question of [Python binding-surface rules](07-python-binding-surface-rules.md).
