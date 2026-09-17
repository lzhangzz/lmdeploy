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

## Not yet specified

- GatedDeltaNet cache-contract interplay with per-layer executable modules: sharpens once Merged Attention module design lands.
- Per-stage gate test matrix (quant variants × architecture × dense/MoE): sharpens once Migration order and verification gates starts.

## Out of scope

- Vision/VL encoder module redesign: those modules keep working via mechanical adaptation only, no redesign in this spec.
- Python↔C++ quant dispatch unification (`weight_format.py` vs `gemm::Family` twin tables): separate future effort.
- Decoder/engine-loop redesign beyond the mechanical shrink of `UnifiedDecoder`: BatchOp contract meanings hold, orchestration is not redesigned.
- Landing the implementation: the spec is the deliverable; implementation is the follow-up effort.
