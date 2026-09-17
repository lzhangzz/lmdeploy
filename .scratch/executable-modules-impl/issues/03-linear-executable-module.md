# 03: Linear executable module

**What to build:** The linear weight module becomes the executable Linear of the spec: it owns its forwards — dense, grouped (expert routing tables), act-scale companions, and plan replay — and the call sites flip to weight-is-executor (`weight->Forward(...)`, no executor argument, no executor member). The module base gains the default no-op BatchOp Run and the engine-capture bind from the spec's lifecycle section. Every existing linear call site — attention projections, FFN and MoE GEMMs inside the still-existing layers, and the model driver's embedding/output GEMMs — swaps to the module form. The shape, param names, and registry string are frozen per the spec's Linear section; the per-quant axis stays the GEMM family.

**Blocked by:** 02 (Re-home the GEMM engine into the engine root).

**Status:** resolved

- [x] Linear owns all five forward forms plus output-spec/exec-plan/tune and the run-mode ops delegating to the captured engine
- [x] Module base has the default no-op Run and bind; bind captures the engine reference; forward-before-bind asserts
- [x] All linear call sites use module forwards; no executor-form linear calls remain
- [x] Build green; dense and grouped (MoE) smoke runs produce meaningful output

## Answer

`LinearWeight` is now the executable `Linear` (`src/turbomind/models/linear.{h,cc}`, renamed from `linear_weight.*`): it owns the five Forward forms (dense, dense+act-scales, grouped, grouped+act-scales, plan-replay), `GetOutputSpec`/`GetExecPlan`/`Tune`, and the run-mode ops (`set_measure`/`Export`/`Import`/`GetTuningSeq`) delegating to the engine captured in `bind(Context&)` (`engine_ = &ctx.linear_engine()`). Any engine use before bind asserts (`TM_CHECK` in a private `engine()` accessor — it fired live during bring-up and caught the one real gap, below). The registry string stays frozen via a new `TM_MODULE_REGISTER_AS("LinearWeight", Linear, core::LinearConfig)`; `type()` and `LinearConfig`'s default name are unchanged, so Python's `create_module` is untouched.

`core::Module` grew the default no-op `Run(BatchOp, int, TensorMap&)` and the recursing `bind(::turbomind::Context&)` (opaque forward declarations for `turbomind::Context` and `BatchOp`; the global qualification avoids the `core::Context` name collision). The bind walk runs engine-side in `TurboMind::Impl::ProcessWeights` right after the prepare walk — no Python driver change in this ticket. `MoeWeight` overrides `bind` to descend into its internally-built `block_` (built in `prepare()` via `LinkLinearExperts`, invisible to the X-macro child walk) — without this the grouped forwards assert pre-bind.

All linear call sites flipped to weight-is-executor (`mlp.w2->Forward(...)`, `block.w1w3->Forward(...)`, `w_qkv->Forward(...)`): FFN, both MoE impls, attention + MLA projections, GatedDeltaNet, LanguageModel output GEMM, both ViTs. Executor members (`LlamaLinear& linear_`) deleted from all eight holder classes; `turbomind::Context::linear` deleted (GEMMEngine stays, `linear_engine()` accessor). `LlamaLinear` shrank to the standalone Python-path shim (default ctor owning a private engine; plan query/replay/tune/import/export only; dead engine-path forms removed) — it retires wholesale in ticket 04 when `linear.py` switches to `_tm.Context()` + bind.

Verified: `ninja` green; dense `Qwen/Qwen3-8B` 256 meaningful tokens, load 6.62–6.67 s (baseline 6.99 s); grouped MoE `Qwen/Qwen3-30B-A3B-Thinking-2507` 256 meaningful tokens, load 16.31 s (baseline 16.25 s); standalone `_tm.LlamaLinear()` construct/plan/close OK; `grep linear_\.Forward src/` empty. Two-axis review (standards + spec sub-agents): no contract breaches, all scope additions judged necessary; clang-format (v11.1.0, matching pre-commit) applied to all touched src/ files. Stage debts recorded: run-mode ops on any Linear mutate per-rank engine state (parity with old surface); the bind walk precedes the future `register_cache` pass (lifecycle ordering lands with stage 3).
