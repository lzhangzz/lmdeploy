# Executable modules: design spec

Merge TurboMind's weight storage and forward execution into one executable module class per module type. Status: assembled from the resolved tickets of the [wayfinder map](map.md), presented for user review. Deep sources: the [drafts](drafts/), the [Linear prototype](prototype/linear-module-interface.md), and the [prior-art research](research/prior-art.md).

## Problem

Today the weight side (data-only `core::Module` classes in `src/turbomind/models/`: `LinearWeight`, `AttentionWeight`, `FfnWeight`, `MoeWeight`, `DeltaNetWeight`) and the execution side (per-kind executors in `src/turbomind/models/llama/`: `LlamaLinear`, `UnifiedAttentionLayer`, `LlamaFfnLayer`, `MoeFfnLayer` × 2, `GatedDeltaNetLayer`) meet only at `Forward(input, const XxxWeight&, …)` call sites. Consequences: FFN GEMM orchestration copy-pasted 3×, twin metadata plumbing (`copy_metadata_to`, `AliasLinear`), geometry stored on weights but re-derived by executors, and no single place that owns a module end to end. Commit bc9a55488 already unified linear *execution* (`gemm::Family`, `WeightPlan`/`ExecPlan`); this spec finishes the job on the module axis.

## Concepts

- **Module** (canonical): the unit owning both a module's weights and its Forward; the single thing Python constructs by name. **Layer** is the legacy executor name and retires. (Glossary: `CONTEXT.md`.)
- **Context** (`turbomind::Context`, surviving): the per-rank infrastructure holder — GEMM engine, build-time stream/allocator, device props, comm, the warm-up flag. It holds no module state. `core::Context` (thread-local scope) is unchanged and keeps its name. *(Amended: the kind-state slots moved to the bind-pass map — ADR 0001.)*
- **Kind state**: the per-rank shared state of a kind (`AttentionState`, `DeltaNetState`) or of one Implementation (`MoeDefaultState`, `MoeDeepEPState`); created by the first module of the kind during the bind pass and shared through the BindEnv map. There is no `LinearState`; Linear captures the GEMM engine.
- **Implementation**: a concrete executable Module subclass of a kind, holding the kind's weights and whole Forward, selected at construction from the engine configuration (Moe: `MoeDefaultImpl`, `MoeDeepEPImpl`).
- **Binding surface**: the Python-visible construction surface (type strings, child/param names); frozen while internals change.

Hard constraints (user-set): existing checkpoints load as-is; `src/turbomind/engine/README.md` BatchOp/cache contract meanings hold; no perf regression (≥98% gate); the Python binding surface stays stable.

## Architecture

One executable base per module type — `Linear`, `Ffn`, `Moe`, `Attention`, `DeltaNet` — each keeping its frozen registry string (`"LinearWeight"`, …) so Python's `create_module` is untouched; where a kind has multiple implementations, they are Module subclasses under the single frozen string, selected at construction by the registry factory from the ambient per-rank `EngineParam` (Moe: `MoeDefaultImpl`/`MoeDeepEPImpl`). *(Amended: implementations are module subclasses, not internal branches — ADR 0001.)* `gemm::Family` remains the per-quant method axis (vLLM-style two-axis composition already in-tree); no method-object layer — an Implementation is a Module. Per-kind executor classes retire; their state survives as kind states on the bind-pass map or the Context-held GEMM engine. The decoder keeps the between-module glue (fused norms, allreduce, staging) and shrinks mechanically to module calls.

Lifecycle (normative pass order):

```text
Python builds tree -> prepare() walk -> register_cache(CacheRegistry&) walk
  -> bind(BindEnv&) walk -> category finalization -> first Run
```

(Amended post-landing: bind precedes category finalization — the kind states
created during bind register their bind-time ranges, e.g. the DeltaNet
recurrent grid, while the registry is still open. The normative invariant is
finalization after *all* registrations, which holds.)

## 1. Linear (`prototype/linear-module-interface.md` + amendment)

```cpp
class Linear: public core::Module {
public:
    const char* type() const override { return "LinearWeight"; }  // frozen
    Linear() = default;
    Linear(const core::LinearConfig& cfg);

    void prepare() override;               // WeightPlan::pack -> Family::Pack
    void set_plan(gemm::WeightPlan plan);  // stamped by Python after Gemm::GetWeightPlan
    void bind(turbomind::Context& ctx)     { engine_ = &ctx.linear_engine(); }

    // weight argument dropped: it is *this
    void Forward(const Tensor& input, Ref<Tensor> output);
    void Forward(const Tensor& input, const Tensor& input_scales, Ref<Tensor> output, Ref<Tensor> output_scales);
    void Forward(const Tensor& input, const Buffer_<int>& indices, const Buffer_<int>& offsets, Ref<Tensor> output);
    void Forward(const Tensor& input, const Tensor& input_scales, const Buffer_<int>& indices,
                 const Buffer_<int>& offsets, Ref<Tensor> output, Ref<Tensor> output_scales);
    void Forward(const gemm::ExecPlan& plan, const Tensor& input, const Tensor& input_scales,
                 const Buffer_<int>& indices, const Buffer_<int>& offsets,
                 Ref<Tensor> output, Ref<Tensor> output_scales);

    gemm::OutputSpec GetOutputSpec(const Tensor& input, const Buffer_<int>& indices = {}) const;
    std::optional<gemm::ExecPlan> GetExecPlan(const Tensor& input, const Buffer_<int>& indices = {},
                                              const Buffer_<int>& offsets = {});
    std::optional<gemm::ExecPlan> Tune(const Tensor& input, const Tensor& input_scales,
                                       const Buffer_<int>& indices, const Buffer_<int>& offsets,
                                       Ref<Tensor> output, Ref<Tensor> output_scales);

    // params (frozen X-macro): weight, bias, scales, global_scale, zeros
    // metadata: family, three DataFormats, epilogue, k_desc/q_desc, dims — as today

private:
    gemm::Arguments GetArguments(const Tensor&, const gemm::OutputSpec&, Ref<Tensor>&, Ref<Tensor>&);
    GEMMEngine* engine_{};   // captured from Context in bind(); never owned
};
```

`LlamaLinear` and its pimpl retire (the Impl contents become the Context-held engine). Call sites: `mlp.w2->Forward(gating, inter_scales, param.output, unused_scales)`; grouped MoE forwards against `moe->block()`; attention projections land mechanically in stage 3.

## 2. Ffn and Moe (`drafts/ffn-moe-design.md`)

- `Ffn::Forward(input, output)` = w1w3 → activation → w2 with the FP8-fused-SiLU and w1/w3-separate branches internal; grouped overload takes a `GroupedRoute` (indices/offsets/f2E/num_valid_tokens). The three verbatim orchestration copies collapse into this one body. `LlamaFfnLayer` retires.
- `Moe` is the executable base: gate → route (`invokeMoeGate_V2`/`NoAuxTC`; `SampleUniform` under warm-up, driven by the engine-owned `is_warm_up` flag) → gather → `block_->Forward(route)` → combine → shared fold (`shared->Forward`, `shared_gate->Forward`), with a pure-virtual `Forward`. `MoeDefaultImpl` and `MoeDeepEPImpl` are Module subclasses — each the executable module holding its layer's weights, bodies verbatim from the retired executor paths (`ForwardDefault`/`ForwardA2A`; the A2A in-place contract documented). Selected at construction by the `"MoeWeight"` factory from the ambient `EngineParam` (`moe_a2a_backend` string only; unknown → fatal; bf16 invariant on the DeepEP side). *(Amended: reverses the earlier "internal branch" decision — ADR 0001.)*
- Per-implementation kind state: `MoeDefaultState` (routing buffers) and `MoeDeepEPState` (`TokenDispatcher`, topk buffers, scales-copy stream/events), created once per rank by the first module during the bind pass and shared via the BindEnv map (keys `"moe.default"`/`"moe.deepep"`); a re-bind re-creates them (DeepEP re-JIT accepted). The merged `MoeState` struct and its Context slot are deleted.
- `experts` ModuleList stays the Python surface; `prepare()` keeps building the internal `block_` via `LinkLinearExperts`; `AliasRouted`/`meta_pack_` aliasing unchanged. Classes `Ffn`/`Moe` with frozen `"FfnWeight"`/`"MoeWeight"` strings.

## 3. Attention and DeltaNet (`drafts/attention-design.md`)

- Per-layer `Attention` modules. Per-layer state = weights + registered KV byte offset + rope params (`RopeKernelParam` computed in the ctor). The per-phase `AttentionData` is *batch* state — block ptrs are per-request bases shared across layers; layers apply their own `prefix_offset_` (addressing identical to today).
- `register_cache(CacheRegistry&)` is a dedicated pass (see lifecycle): each module registers its own per-layer bytes (prefix category); layout equals today's single L-sized registration, so cache layout and checkpoints are unchanged.
- `AttentionState` (kind state): per-phase `AttentionBatch` slots (batch-version stamp; `Attention::kSetup` fills only when stale — order-independent), fp32 workspace, pinned staging, CP context, aux stream + events.
- `DeltaNet` is its own module on the same pattern: per-layer conv registration in `register_cache` (checkpoint category), the recurrent block grid registered once per rank by `DeltaNetState` during `bind` — before finalization, because block-rows are shared across layers when `L_b > 1` (post-landing amendment) — `contracts.gated-deltanet` semantics preserved (per-layer conv offsets, recurrent composite parts, TP×CP folding). MLA is an internal branch of `Attention::Forward`. Classes `Attention`/`DeltaNet`, frozen `"AttentionWeight"`/`"DeltaNetWeight"`.

## 4. Engine services (`drafts/context-contract.md`)

- `turbomind::Context` survives as the per-rank infrastructure holder: **GEMM engine** (`gemm::Gemm` + dispatch cache + workspace + measure policy), build-time stream/allocator, device props, comm. `is_warm_up` stays the engine-owned flag on `turbomind::Context`, exposed by reference through BindEnv and read per run (post-landing amendment: the `env.warm_up()` accessor was never introduced).
- `bind(BindEnv&)` after initialization; BindEnv (defined beside `Context` in `models/llama/context.h`, forward-declared from core like `Context`) carries `const EngineParam& engine`, `turbomind::Context& context`, `CacheRegistry& registry`, `int& is_warm_up`, the shared-state map (`std::unordered_map<std::string, std::shared_ptr<void>>`, alive only during the walk), and `int& phases`. The first module of a kind creates its kind state on the map; the rest fetch. `Impl::Bind` builds the env and walks — no kind-module scan, no pre-walk state creation, no Context state slots. Modules never hold per-module engine copies or per-kind workspace duplicates.
- Tuning ops on the engine object; `Linear::Tune` the thin module method for the `linear.py` pin. Dispatch-cache mutation executor-thread-only; measure mode only during warm-up; engine-thread ops metadata-only (`contracts.executor-only`, `contracts.object-address`).
- Standalone: `linear.py` constructs a `turbomind::Context` and calls `mod.bind(ctx)` after `prepare`; the pybind shim keeps that Python surface and assembles a degenerate BindEnv (default `EngineParam`, empty registry, local flags, fresh map) internally.

## 5. BatchOp and decoder mapping (`drafts/batchop-decoder-mapping.md`)

- Glue stays in the decoder: fused `AllreduceResidualRMSnorm`, per-layer `ffn_group` mapping, hidden-state plumbing, next-layer norm staging. `DecoderLayerWeight` stays the Python-built data composite. Zero behavior change.
- `core::Module` grows default no-op `Run(BatchOp op, int phase, TensorMap& env)`; the generic walk hits every layer child per op; kinds implement subsets (post-landing amendment: Attention — kAdd rope init + kSetup fill + kPrepare borrows; DeltaNet — kSetup fill + kPrepare state prepare/clears; kUnprep reaches every layer child through the decoder's walk but no kind implements kUnprep-specific behavior — only the shared wave stamp advances; Linear/Ffn/Moe implement no Run ops). `kForward` keeps the decoder's orchestrated loop.
- `LanguageModel::Run` dispatch unchanged (single BatchOp entry per `concepts.module`); its embedding/output GEMMs become owned `Linear` modules.
- README amendments land same-change in stage 3: class renames (`contracts.cache-content`, `unified-attention`, `gated-deltanet`), registration timing (`ownership.module-cache`, `contracts.category-registration`: "during construction" → "during the register_cache pass"), new lifecycle paragraph. `concepts.module` needs no change.

## 6. Python binding surface (`drafts/python-binding-rules.md`)

- **Frozen**: registry strings, config classes/fields, `add_child_raw` child names, param names + `PackedTensor` semantics, module methods (`set_plan`/`param`/`prepare`/`set_meta_pack`/`LinkLinearExperts`), `_tm.create_module`/`WeightQuery`, `WeightPlan` fields, `linear.py` public names, driver names (`create_root`/`export`/`process_weight`/`gemm`).
- **The effort's entire Python delta**: `linear.py`'s construction block (`_tm.LlamaLinear()` → `_tm.Context()` + `mod.bind(ctx)`) and `turbomind.py` gaining `register_cache()`/`bind()` calls; a review follow-up also re-qualified `interns2_mobius.py` docstrings to the `Moe` class (mechanical, no code). `TurboMind.gemm` keeps its name, returning the engine object. Builders: zero changes, byte-identical across all stages (gate-checked). `weight_format.py` untouched.

## 7. Migration and verification (`drafts/migration-gates.md`)

Sequential stages on `modules`, bounded coexistence inside a stage, retired classes deleted by their successor's end:

| Stage | Carries | Extra gate |
|---|---|---|
| 1 Linear | Linear merge, `bind`/default `Run`, all call-site swaps, the Python delta | — |
| 2 FFN/MoE | `Ffn`/`Moe`, impl classes retire, `MoeState` | footprint + startup check |
| 3 Attention/GDN | `Attention`/`DeltaNet`, real `register_cache`, `AttentionState`, README contract-sync | prefix-cache + checkpoint variants |

Every gate: clean `ninja` build; frozen-list grep + builder import smoke; four core models via `scripts/test_turbomind_model.py` (`Qwen/Qwen3-8B`, `Qwen/Qwen3-8B-AWQ`, `Qwen/Qwen3-30B-A3B-Thinking-2507`, `Qwen/Qwen3.5-35B-A3B-FP8`; cache dir `/mnt_cfs/huggingface_hub/hub/`; ≥128 meaningful tokens, never batched); perf = `benchmark/profile_throughput.py --backend turbomind` (ticket 01's recorded substitution: `benchmark_decode.py` drives the pytorch backend) ≥98% of the recorded pre-stage-1 baseline — or, where a recorded number is proven environmentally stale by re-measuring the unmodified baseline commit, of same-session fresh references — plus load time within noise (post-landing amendment). Final gate adds `meta-Llama/Llama-3.1-70B-Instruct` at TP≥2. Baselines recorded in `perf/baseline.md` before stage 1. Checkpoints load unmodified at every gate.

## Risk register

1. **Fused-path perf drift** — the collapse of the 3× FFN orchestration must preserve the FP8 fused-SiLU scheduling exactly; mitigated by the per-model perf gate including the FP8 MoE.
2. **README contract-sync discipline** — stage 3 touches the normative doc; a missed same-change update violates `checklist.contract-sync`; mitigated by the compiled amendment list (section 5).
3. **Coexistence debt** — bounded coexistence can linger; mitigated by the rule that every retired class dies by its successor stage's end.
4. **`register_cache` ordering** — category finalization must not run before every module registers; the pass order is normative and engine-driven, not Python-driven.
5. **TP/comm coverage** — only the final gate exercises TP≥2 comm paths (allreduce/staging); a comm regression before that gate ships one stage late; accepted.
6. **Attention batch fill races** — the version-stamp guard is order-independent by design; stage 3 gate includes cache-behavior variants to prove it.
7. **Binding drift** — Python renames sneaking in; mitigated by the frozen-list grep at every gate.

## Decision log

Route and rationale live in the tickets, indexed by the [map](map.md): prior-art research → context contract (later reworked) → Linear interface → FFN/MoE → Attention → BatchOp/decoder → ExecContext rework (amending context contract and MoE transport) → Python rules → migration gates. Superseded decisions (ExecContext v1, per-module MoE transport) are preserved in-place in their assets for the record. A later redesign (ADR 0001) supersedes the internal-branch MoE merge and the Context kind-state slots: implementations are Module subclasses selected at construction, and kind states rendezvous on the bind-pass map.
