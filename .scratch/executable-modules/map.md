# Executable modules: merge weight storage and forward execution

## Destination

A reviewed design spec at `.scratch/executable-modules/spec.md` for merging weight storage and forward execution into one executable `core::Module` per module type (Linear, FFN/MoE, Attention/GatedDeltaNet): concrete interfaces with code snippets, the Context shared-state contract, the Python binding-surface rules, and a staged migration order (Linear → FFN/MoE → Attention) with per-stage verification gates. Implementation is a separate follow-up effort; this map ends when the spec passes user review.

## Notes

- Domain: TurboMind C++ (`src/turbomind/`) and its Python loader (`lmdeploy/turbomind/`). Current split: data-only weight classes in `src/turbomind/models/` (`LinearWeight`, `AttentionWeight`, `FfnWeight`, `MoeWeight`, …, the `core::Module` tree) versus per-kind executors in `src/turbomind/models/llama/` (`LlamaLinear`, `UnifiedAttentionLayer`, `LlamaFfnLayer`, `MoeFfnLayer` × 2, `GatedDeltaNetLayer`) owned by `UnifiedDecoder`. Commit bc9a55488 already unified the linear *execution* path (`gemm::Family`, `WeightPlan`/`ExecPlan`).
- Design sessions: call the skills `grilling` and `domain-modeling` for decision tickets; `codebase-design` vocabulary (deep modules, seam placement) applies. Prototype tickets call `prototype`. Research tickets call `research`.
- Hard constraints (user-set): existing checkpoints load as-is; `src/turbomind/engine/README.md` BatchOp/cache contract meanings hold (cite as `<section>.<leaf>`); no perf regression; the Python binding surface (`lmdeploy/turbomind` builders + `linear.py` public API) stays stable. Python internals may adapt mechanically.
- Repo rules: prefer the right design shape over minimal changes; planning artifacts carry concrete code snippets, never placeholders; verify models with `scripts/test_turbomind_model.py` (meaningful response, at least 128 tokens, never batched in a loop); build with `ninja` in `build/`.

## Decisions so far

<!-- one line per closed ticket: [name](issues/NN-slug.md): gist of the answer -->

- [Prior-art research: module unification in sibling engines](issues/01-prior-art-module-unification-research.md): 10 patterns from torch/vLLM/SGLang/TRT-LLM/TVM-MLC — adopt module-owns-weights+forward with a per-quant method object (vLLM two-axis) and a staged per-kind migration gate; adapt post-load prepare, platform-dispatch factory, and Context-as-typed-forward-argument (TVM/MLC); avoid duck-typed layer-attribute seams and executor-only weight-less ownership (TRT-LLM C++).
- [Context shared-state contract](issues/02-context-shared-state-contract.md): `ExecContext` is the per-rank services object (GEMM engine+cache+workspace, comm, per-kind workspace pools shared across layer instances, stream/event factory, symm), injected once via a post-build bind pass; modules never hold raw streams/comms/engines or per-kind workspaces; warm-up rides engine-owned `env.warm_up()`; thread rules per accessor with GEMM executor-thread-only. Draft: `drafts/context-contract.md`.
- [Merged Linear module interface](issues/03-merged-linear-module-interface.md): validated canonical shape — one `Linear : core::Module` owning params + Forward (weight-is-executor call sites), `gemm::Family` stays the quant axis, `LlamaLinear` retires into `ExecContext`-backed methods (`set_measure`/`Export`/`Import` on the module), registry string frozen `"LinearWeight"`, `linear.py` names unchanged. Asset: `prototype/linear-module-interface.md`.
- [Merged FFN and MoE module design](issues/04-merged-ffn-and-moe-module-design.md): `Ffn::Forward` is the whole FFN (activation internal, dense + grouped overloads; the 3× copy-paste collapses); `Moe::Forward` owns routing + transport code with default-vs-DeepEP as an internal branch; transport/routing state owned per module instance (explicit amendment to the context contract's per-kind pools — memory/JIT multiply by layers, gate added to Migration tickets); `block_` mechanism and Python surface unchanged; classes `Ffn`/`Moe` with frozen strings. Asset: `drafts/ffn-moe-design.md`.
- [Merged Attention module design](issues/05-merged-attention-module-design.md): per-layer `Attention` modules (per-layer state = weights + registered offset + rope params; per-phase `AttentionData` is batch state in the attention kind pool); registration via a dedicated `register_cache(CacheRegistry&)` pass (ExecContext stays pure; canonical order prepare → register_cache → finalize → bind → Run); workspaces pooled; MLA internal branch; GDN is its own `DeltaNet` module on the same pattern with `contracts.gated-deltanet` preserved. Asset: `drafts/attention-design.md`.
- [BatchOp and decoder mapping](issues/06-batchop-and-decoder-mapping.md): glue (fused AR-residual-norm, ffn_group mapping, norm staging) stays in the shrunk decoder — only executor calls become module forwards; default no-op `Run(BatchOp, phase, env)` on `core::Module` with kinds implementing subsets; kForward keeps the orchestrated loop; pooled `AttentionBatch` filled via a versioned stamp guard in `Attention::kSetup`; README contract-sync amendment list compiled (class renames, registration timing, lifecycle paragraph). Asset: `drafts/batchop-decoder-mapping.md`.
- [Rework the ExecContext design](issues/10-rework-execcontext-design.md): `turbomind::Context` survives as the engine root (GEMM engine lives there); `bind(Context&)` after init, modules capture typed refs (Linear captures the engine — no `LinearState`); per-kind states only for cross-layer kinds (`AttentionState`/`MoeState`/`DeltaNetState`); MoE transport reversed to shared `MoeState`; standalone `linear.py` constructs a `Context` and binds; `ExecContext` name dead. Rewrites `drafts/context-contract.md`; unblocks Python binding-surface rules.
- [Python binding-surface rules](issues/07-python-binding-surface-rules.md): frozen list ratified (registry strings, configs, child/param names, module methods, `WeightQuery`/`WeightPlan`, `linear.py` public names, driver names); `_tm.LlamaLinear` retires into `_tm.Context()` + `mod.bind(ctx)` standalone construction; new surface = `_tm.Context` + separate `TurboMind.register_cache()`/`bind()` entry points; per-builder recipe is zero changes; builders byte-identical across all stages. Asset: `drafts/python-binding-rules.md`.

## Not yet specified

- Per-stage gate test matrix (quant variants × architecture × dense/MoE): sharpens once Migration order and verification gates starts.

## Out of scope

- Vision/VL encoder module redesign: those modules keep working via mechanical adaptation only, no redesign in this spec.
- Python↔C++ quant dispatch unification (`weight_format.py` vs `gemm::Family` twin tables): separate future effort.
- Decoder/engine-loop redesign beyond the mechanical shrink of `UnifiedDecoder`: BatchOp contract meanings hold, orchestration is not redesigned.
- Landing the implementation: the spec is the deliverable; implementation is the follow-up effort.
