# PROTOTYPE: merged Linear module interface (validated)

**Throwaway interface draft** for [Merged Linear module interface](../issues/03-merged-linear-module-interface.md).
Answers: *what does the merged Linear look like, and do the real call sites read well against it?* — user-validated 2026-09-17; see Resolved decisions at the end.

Assumption stated up front (prototype-skill rule): the artifact lives here under `.scratch/` rather than as a stray header inside `src/turbomind/` to keep the in-tree build clean; paths below reference the real files it would replace. Not compiled, not in CMake.

Sources merged: `src/turbomind/models/linear_weight.h` (storage) + `src/turbomind/models/llama/LlamaLinear.{h,cu}` (execution). Execution state (GEMM engine, workspace, dispatch policy) follows `drafts/context-contract.md`: it lives in `ExecContext`, injected via `bind()`.

Design stance from the prior-art findings: the two-axis composition vLLM uses already exists here — `gemm::Family` is the per-quant method axis (supports/pack/convert_input/output_spec, unified by bc9a55488). The merge completes the *module* axis: one class owns params + forward. No new method-object layer is introduced; that would duplicate `Family`.

## The header

```cpp
// PROTOTYPE — merged Linear: owns the weights AND the forward.
// Replaces models/linear_weight.h + models/llama/LlamaLinear.{h,cu}.
#pragma once

#include <istream>
#include <optional>
#include <ostream>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/weight_plan.h"
#include "src/turbomind/models/exec_context.h"  // drafts/context-contract.md

namespace turbomind {

class Linear: public core::Module {
public:
    // Registry name frozen for the Python binding surface: builders create
    // this module by the string "LinearWeight" via core::LinearConfig.
    const char* type() const override
    {
        return "LinearWeight";
    }

    Linear() = default;
    Linear(const core::LinearConfig& cfg);

    // ----- lifecycle (unchanged from LinearWeight) -----

    void prepare() override;               // WeightPlan::pack -> Family::Pack (executor-free, load time)
    void set_plan(gemm::WeightPlan plan);  // stamped by Python after Gemm::GetWeightPlan
    bool is_graph_compatible() const;

    // ----- ExecContext injection (one bind pass, post-load, pre-first-Run) -----

    void bind(ExecContext& xc);            // default impl stores xc_; forward-before-bind asserts

    // ----- execution (absorbs LlamaLinear, weight argument dropped: it is *this) -----

    /// Dense: y = x*W^T (+b). Output allocated when null.
    void Forward(const Tensor& input, Ref<Tensor> output);

    /// Dense with dynamic act-scale companions.
    /// input_scales: when input is already FP8, used as GEMM U (skips QuantizeSymm).
    /// output_scales: when output_dtype() is FP8, filled with group-128 scales (W).
    void Forward(const Tensor& input, const Tensor& input_scales, Ref<Tensor> output, Ref<Tensor> output_scales);

    /// Grouped (MoE): indices/offsets select experts packed by LinkLinearExperts.
    void Forward(const Tensor& input, const Buffer_<int>& indices, const Buffer_<int>& offsets, Ref<Tensor> output);

    /// Grouped with dynamic act-scale companions.
    void Forward(const Tensor&       input,
                 const Tensor&       input_scales,
                 const Buffer_<int>& indices,
                 const Buffer_<int>& offsets,
                 Ref<Tensor>         output,
                 Ref<Tensor>         output_scales);

    /// Plan-replay: runs a plan returned by GetExecPlan (Python path).
    void Forward(const gemm::ExecPlan& plan,
                 const Tensor&         input,
                 const Tensor&         input_scales,
                 const Buffer_<int>&   indices,
                 const Buffer_<int>&   offsets,
                 Ref<Tensor>           output,
                 Ref<Tensor>           output_scales);

    gemm::OutputSpec GetOutputSpec(const Tensor& input, const Buffer_<int>& indices = {}) const;

    std::optional<gemm::ExecPlan> GetExecPlan(const Tensor&       input,
                                              const Buffer_<int>& indices = {},
                                              const Buffer_<int>& offsets = {});

    std::optional<gemm::ExecPlan> Tune(const Tensor&       input,
                                       const Tensor&       input_scales,
                                       const Buffer_<int>& indices,
                                       const Buffer_<int>& offsets,
                                       Ref<Tensor>         output,
                                       Ref<Tensor>         output_scales);

    // ----- per-rank run-mode ops: on the module, delegating to ExecContext -----

    void set_measure(bool measure);  // -> xc_->set_measure(...)
    [[maybe_unused]] int Export(std::ostream& os);
    [[maybe_unused]] int Import(std::istream& is);
    std::vector<int> GetTuningSeq() const;

    // ----- weight-side metadata (unchanged shape, now on the executable) -----

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(weight);
    }

    const gemm::Family* family{};        // stamped by WeightPlan::pack at prepare
    DataFormat          weight_format{};  // from cfg.format
    DataFormat          input_format{};   // derived in ctor
    DataFormat          output_format{};  // derived in ctor

    DataType input_dtype() const
    {
        return input_format.dtype;
    }
    DataType output_dtype() const
    {
        return output_format.dtype;
    }

    int      input_dim  = 0;
    int      output_dim = 0;
    DataType data_type{};  // model activation dtype, from cfg.data_type

    gemm::Epilogue     epilogue{};
    gemm::MatrixLayout k_desc{};
    gemm::MatrixLayout q_desc{};

#define LINEAR_WEIGHT_CHILDREN(X)

#define LINEAR_WEIGHT_PARAMS(X) \
    X(weight)                   \
    X(bias)                     \
    X(scales)                   \
    X(global_scale)             \
    X(zeros)

    TM_MODULE_DECLARE(Linear, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)

private:
    // absorbed from LlamaLinear::Impl
    gemm::Arguments GetArguments(const Tensor& input, const gemm::OutputSpec& spec, Ref<Tensor>& output, Ref<Tensor>& output_scales);

    ExecContext* xc_{};  // injected by bind(); never owned

    bool                            has_bias_ = false;
    bool                            prepared_ = false;
    std::optional<gemm::WeightPlan> plan_;
};

}  // namespace turbomind
```

What died: `LlamaLinear` (class, pimpl, `gemm_`, `workspace_`, `dispatch_policy_` — the engine and workspace live in `ExecContext`; the measure policy lives there too but is driven through `Linear::set_measure`), the `const LinearWeight&` argument on every overload, and `Context::linear`.

## Call sites, before → after

**Dense FFN** (`src/turbomind/models/llama/LlamaFfnLayer.cc:28-91`; `linear_` member deleted):

```cpp
// before
TM_SCOPE_CALL(linear_.Forward(param.input, *fused, mix));
...
TM_SCOPE_CALL(linear_.Forward(gating, inter_scales, *mlp.w2, param.output, unused_scales));

// after — the weight IS the executor
TM_SCOPE_CALL(fused->Forward(param.input, mix));
...
TM_SCOPE_CALL(mlp.w2->Forward(gating, inter_scales, param.output, unused_scales));
```

**Grouped MoE** (`MoeFfnDefaultImpl::Forward`, `moe_ffn_layer.cc:236-267`; same edit in `MoeFfnA2AImpl`):

```cpp
// before
linear_.Forward(intermediate_states, scales, *moe.block(), expert_ids, expert_offset,
                temp_buffer, temp_scales);

// after
moe->block().Forward(intermediate_states, scales, expert_ids, expert_offset,
                     temp_buffer, temp_scales);
```

**Attention projections** (`unified_attention_layer.cc:393,440`): `linear_.Forward(input, *w_qkv, qkv)` → `w_qkv->Forward(input, qkv)` — the `AttentionWeight` children are already `LinearWeight`, so this lands mechanically in ticket 05.

## Python surface mapping (`lmdeploy/turbomind/linear.py`)

Unchanged public names, now bound on the module instead of a wrapper: `get_weight_plan` (still `Gemm::GetWeightPlan` via the standalone `ExecContext`), `prepare_weight` (`set_plan` + `prepare`), `get_exec_plan` → `mod.GetExecPlan(...)`, `tune` → `mod.Tune(...)`, `forward_dense`/`forward_moe` → the dense/grouped `Forward` overloads. The standalone path constructs its own `ExecContext` (per the context contract) instead of receiving the engine's.

## Resolved decisions (user-validated)

1. **Shape accepted as canonical**: one `Linear`, `gemm::Family` as the per-quant axis, no new method-object layer, pimpl retired into `ExecContext`. FFN/MoE and Attention module designs reuse this shape.
2. **Warm-up/tuning ops all on the module**: `set_measure`, `Export`, `Import`, `GetTuningSeq` are `Linear` methods delegating to the injected `ExecContext` (no facade object).
3. **Naming**: C++ class `Linear`; `type()` keeps returning `"LinearWeight"` so Python's create-by-name is untouched.
4. **Call-site style**: weight-is-executor (`mlp.w2->Forward(...)`); executor members (`linear_`) disappear from every layer class.

## Amendment (Rework the ExecContext design)

The `ExecContext`/`bind(ExecContext&)`/`xc_` machinery above is superseded: `turbomind::Context` survives as the engine root, and `Linear::bind(Context&)` captures a reference to the GEMM engine living in `Context` (`engine_ = &ctx.linear_engine()`). There is no `LinearState`. The standalone `linear.py` path constructs a `Context` and binds it after `prepare`. Everything else in this prototype (params, Forward overloads, frozen `"LinearWeight"` string, call sites, Python surface mapping) stands.
