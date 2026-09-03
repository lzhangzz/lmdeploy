# Split weight planning from exec planning

Status: draft. This document does not authorize implementation.

## Goal

Separate the two decisions currently conflated by the word "plan":

- `WeightPlan` selects a family and describes how a source weight is packed.
- `ExecPlan` describes one concrete GEMM problem, publishes its output allocation metadata, and retains the selected kernel launch.

The current C++ `GemmPlan` is already a weight plan. Rename it without changing its responsibility. Replace the public allocation-only `OutputSpec` query with an execution-planning query that performs kernel dispatch once and returns an opaque kernel-bearing `ExecPlan`.

The intended public flow is:

```python
weight_plan = linear.get_weight_plan(weight_format=weight_format, dtype=torch.bfloat16)
weight = linear.prepare_weight(source, scales=scales, zeros=zeros, plan=weight_plan)

exec_plan = linear.get_exec_plan(x, weight)
out, out_scales = linear(x, weight, plan=exec_plan)
```

When the caller omits an output destination, `Linear` allocates it with `torch.empty_strided` from the plan's output specification. These returned tensors use Torch's allocator and the current Torch CUDA stream. Supplying preallocated Torch tensors remains supported. Native TurboMind allocation must not escape the API boundary.

## Object boundaries

| Object | Decision lifetime | State |
|---|---|---|
| `WeightPlan` | Source format through weight preparation | selected `Family`, `WeightBridge`, epilogue/fusion choice, packed output format, family shape constraints |
| `Weight` / `LinearWeight` | Prepared weight lifetime | packed tensors plus the family/input/output metadata published by `WeightPlan::pack` |
| `ExecPlan` | One concrete GEMM descriptor | `OutputSpec`, exact `GemmDesc`, and selected `LaunchSpec` |

The following state must not be added to `ExecPlan`:

- input, weight, routing, output, or scale pointers;
- ownership of any Torch or TurboMind allocation;
- graph-compatibility metadata;
- source-weight normalization or packing metadata.

`is_graph_compatible` remains a property of prepared `Weight`, because it is family-level information. `WeightPlan.shape_constraints` remains the source of padding constraints.

## Lifetime and reuse contracts

`WeightPlan` and `ExecPlan` are created and consumed by the same open `Linear` on the same CUDA device. This remains an unchecked precondition.

An `ExecPlan` is reusable when all fields represented by its `GemmDesc` remain unchanged:

- M, N, K, and expert count;
- family, input/weight/output dtypes, quantization descriptors, and epilogue;
- dense, blocked, or indexed striding mode;
- operand orders and pack descriptors.

Tensor addresses and grouped offset values may change between executions. An indexed plan may be reused with a different indices tensor of the same length because dispatch depends on indexed mode and work-row count, not on the index values.

The plan retains a pointer to a kernel owned by the originating `gemm::Gemm` registry. Destroying the originating `Linear`, using the plan with another `Linear`, or changing any descriptor field before reuse is undefined behavior. Cache import and tuning never mutate an existing plan; call `get_exec_plan` again or use the plan returned by `tune` to observe a newer cache decision.

## C++ file and type split

### Rename `GemmPlan` to `WeightPlan`

Rename:

- `src/turbomind/kernels/gemm/plan.h` to `src/turbomind/kernels/gemm/weight_plan.h`;
- `src/turbomind/kernels/gemm/plan.cc` to `src/turbomind/kernels/gemm/weight_plan.cc`;
- every `GemmPlan` reference to `WeightPlan`;
- the pybind class `_tm.GemmPlan` to `_tm.WeightPlan`.

The complete weight-planning declaration is:

```cpp
#pragma once

#include <array>

#include "src/turbomind/kernels/gemm/family.h"

namespace turbomind {
class LinearWeight;
}

namespace turbomind::gemm {

struct WeightQuery {
    DataFormat weight_format;
    DataType   data_type{};
    DataType   input_dtype{kNull};
    DataType   output_dtype{kNull};
    bool       grouped{};
};

class WeightPlan {
public:
    const Family& family() const noexcept
    {
        return *family_;
    }

    std::array<int, 4> shape_constraints() const noexcept
    {
        return {family_->min_k(), family_->min_n(), family_->align_k(), family_->align_n()};
    }

    int gate_up(ActivationType act_type, int projection_n);
    void pack(LinearWeight& linear, cudaStream_t stream) const;

private:
    friend class Gemm;

    const Family* family_{};
    WeightBridge  bridge_{};
    Epilogue      epilogue_{Epilogue::kNone};
    DataFormat    output_format_{};
};

}  // namespace turbomind::gemm
```

Rename the `Gemm` entry point without changing its selection logic:

```cpp
std::optional<WeightPlan> GetWeightPlan(const WeightQuery& query) const;
```

The definition becomes `Gemm::GetWeightPlan`; there is no remaining `PlanWeight` spelling.

`LinearWeight` changes only its type names:

```cpp
void set_plan(gemm::WeightPlan plan);

std::optional<gemm::WeightPlan> plan_;
```

The implementation of family selection, `gate_up`, and `pack` otherwise stays unchanged.

### Separate `OutputSpec` and add `ExecPlan`

Move `OutputSpec` from `family.h` into the neutral `src/turbomind/kernels/gemm/output_spec.h`. Both `family.h` and `exec_plan.h` include this value-type header; `family.h` must not include `exec_plan.h`.

The complete `output_spec.h` is:

```cpp
#pragma once

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/layout.h"

namespace turbomind::gemm {

struct OutputSpec {
    core::Layout layout;
    DataType     dtype{kNull};
    core::Layout scales_layout;
    DataType     scales_dtype{kNull};
};

}  // namespace turbomind::gemm
```

The complete new `exec_plan.h` is:

```cpp
#pragma once

#include <utility>

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/output_spec.h"

namespace turbomind {
class LlamaLinear;
}

namespace turbomind::gemm {

class ExecPlan {
public:
    const OutputSpec& output_spec() const noexcept
    {
        return output_spec_;
    }

private:
    friend class Gemm;
    friend class ::turbomind::LlamaLinear;

    ExecPlan(GemmDesc desc, LaunchSpec launch): desc_{std::move(desc)}, launch_{launch} {}

    OutputSpec       output_spec_;
    const GemmDesc   desc_;
    const LaunchSpec launch_;
};

}  // namespace turbomind::gemm
```

`LaunchSpec`, not merely `Kernel*`, is retained because `swizzle` and `splits` are part of the selected executable configuration. `GemmDesc` is retained privately as the exact cache identity of the immutable snapshot. Candidate estimates and measurements are not exposed. `Gemm` constructs the descriptor and launch, and `LlamaLinear` fills `output_spec_` before returning the object. No public mutator exists and no member changes after the completed plan is published.

`family.h` includes only `output_spec.h` for this type. Existing `plain_output_spec` and `fp8_output_spec` continue returning `OutputSpec` and otherwise remain unchanged.

Rename every `OutputSpec` field access in `plain_output_spec`, `fp8_output_spec`, `LlamaLinear`, and the Python binding:

```cpp
spec.layout = apply_output_epilogue(std::move(layout), epilogue);
spec.dtype = format.dtype;
spec.scales_layout = core::Layout{{cdiv(output_dim, kGroupSize), rows}, {round_up(rows, static_cast<core::ssize_t>(kRowAlignment)), 1}};
spec.scales_dtype = kFloat;
```

## Descriptor-only exec planning

Execution planning must not allocate, quantize, copy, or launch CUDA work. It builds the same `Operation` and `MatrixLayout` inputs used by execution and passes them through the existing `Context::Init(Operation, A, U, B, V, C, D)` path. `LlamaLinear` must never construct a `GemmDesc` directly.

For every current family, the GEMM A operand has:

- dtype `weight.input_format.dtype`;
- row-major order;
- flattened source rows, or `indices.size()` rows for indexed MoE;
- K equal to the input's last dimension;
- no pack descriptor.

Dynamic activation quantization changes A's dtype and produces U scales, but it does not change the GEMM dimensions, order, striding mode, or A/U pack descriptors. Kernel selection therefore does not require a temporary quantized tensor or its allocation layout.

Use one operation builder from both planning and execution:

```cpp
Operation GetOperation(const LinearWeight& weight) const
{
    Operation operation{};
    operation.dispatch = dispatch_policy_;
    operation.epilogue = weight.epilogue;
    operation.quant_a = MakeQuantDesc(weight.input_format);
    operation.quant_b = MakeQuantDesc(weight.weight_format);
    operation.batch_dim = 0;
    operation.family = weight.family->id;
    return operation;
}
```

Use one output-specification function for ordinary planning and the allocation phase of tuning:

```cpp
OutputSpec GetOutputSpec(const Tensor& input, const LinearWeight& weight, const Buffer_<int>& indices) const
{
    const bool grouped = weight.k_desc.ld == 0;
    const bool indexed = static_cast<bool>(indices) && indices.size() > 0;
    const int k = input.shape(-1);
    const int m = indexed ? indices.size() : input.size() / k;

    auto output_shape = input.shape();
    if (grouped) {
        output_shape = {m, weight.output_dim};
    }
    else {
        output_shape.back() = weight.output_dim;
    }
    return weight.family->output_spec(core::Layout{std::move(output_shape)}, weight.epilogue);
}
```

Add the following implementation to `LlamaLinear::GetExecPlan`. It constructs only the provisional `MatrixLayout` metadata. The canonical `GemmDesc` is produced inside `Context::Init`:

```cpp
std::optional<gemm::ExecPlan> GetExecPlan(const Tensor& input, const LinearWeight& weight, const Buffer_<int>& indices, const Buffer_<int>& offsets)
{
    using namespace gemm;

    const bool indexed = static_cast<bool>(indices) && indices.size() > 0;
    const int k = input.shape(-1);
    const int m = indexed ? indices.size() : input.size() / k;

    OutputSpec output_spec = impl_->GetOutputSpec(input, weight, indices);

    MatrixLayout desc_A{weight.input_format.dtype, kRowMajor, m, k, k};
    MatrixLayout desc_U{};
    MatrixLayout desc_B = weight.k_desc;
    MatrixLayout desc_V = weight.q_desc;
    MatrixLayout desc_D{output_spec.dtype, kRowMajor, m, weight.output_dim, static_cast<int>(output_spec.layout.stride(-2))};

    if (offsets) {
        desc_A.num = desc_U.num = desc_D.num = desc_B.num;
        desc_A.offsets = desc_U.offsets = desc_D.offsets = const_cast<int*>(offsets.data());
    }
    if (indexed) {
        desc_A.idxs = desc_U.idxs = const_cast<int*>(indices.data());
    }

    Gemm::Arguments args{};
    args.operation = impl_->GetOperation(weight);
    args.Adesc = desc_A;
    args.Udesc = desc_U;
    args.Bdesc = desc_B;
    args.Vdesc = desc_V;
    args.Cdesc = desc_D;
    args.Ddesc = desc_D;
    args.workspace = impl_->workspace_;
    auto plan = impl_->gemm_.GetExecPlan(args);
    if (plan) {
        plan->output_spec_ = std::move(output_spec);
    }
    return plan;
}
```

`gemm::Gemm::GetExecPlan` returns the selected immutable GEMM state directly as `std::optional<ExecPlan>`. `LlamaLinear::GetExecPlan` completes the private `output_spec_` member before the object is returned to any caller. There is no public mutator; after publication, the plan is immutable. `OutputSpec` is not passed into `Gemm`, stored in `Gemm::Arguments`, or used by dispatch. Descriptor-only planning leaves `Arguments::stream` null because `GetExecPlan` does not launch CUDA work or read the stream. Consequently, Python `get_exec_plan` does not enter `_activate()` merely to provide a TurboMind stream.

This metadata follows the existing runtime construction:

- dense A/B/D are flat;
- grouped unindexed A and D are blocked because the real offsets pointer is present;
- grouped indexed A is indexed and D remains blocked;
- grouped B is identified by the linked weight's `k_desc.ld == 0`;
- N remains the stored full weight width for fused SiLU even when `OutputSpec.layout` is compacted.

Planning accepts the real offsets and indices buffers only so `Context::Init` derives the same striding modes as execution. `ExecPlan` does not retain either pointer; `Context::Init` reduces their presence to `Striding` values in `GemmDesc`. Offset and index contents do not participate in dispatch.

Expose the following public C++ entry point:

```cpp
gemm::OutputSpec GetOutputSpec(const Tensor& input, const LinearWeight& weight, const Buffer_<int>& indices = {}) const;
std::optional<gemm::ExecPlan> GetExecPlan(const Tensor& input, const LinearWeight& weight, const Buffer_<int>& indices = {}, const Buffer_<int>& offsets = {});
```

`GetOutputSpec` delegates to the existing family-owned calculation. `GetExecPlan` uses the descriptor-only implementation above without allocating or flattening the input tensor.

## GEMM exec planning and dispatch

Add the flat, non-owning aggregate inside the public section of `gemm::Gemm`:

```cpp
struct Arguments {
    Operation    operation{};
    float        alpha{1.f};
    const void*  A{};
    MatrixLayout Adesc{};
    const void*  U{};
    MatrixLayout Udesc{};
    const void*  B{};
    MatrixLayout Bdesc{};
    const void*  V{};
    MatrixLayout Vdesc{};
    const void*  global_scale{};
    MatrixLayout global_scale_desc{};
    float        beta{};
    const void*  C{};
    MatrixLayout Cdesc{};
    void*        D{};
    MatrixLayout Ddesc{};
    void*        W{};
    MatrixLayout Wdesc{};
    Workspace    workspace{};
    cudaStream_t stream{};
};

[[nodiscard]] int Run(const ExecPlan& plan, const Arguments& args);

private:
friend class ::turbomind::LlamaLinear;

std::optional<ExecPlan> GetExecPlan(const Arguments& args);
std::optional<ExecPlan> Tune(const Arguments& args);
```

`Arguments` owns no allocation. It is assembled for one call and is never retained by `Gemm`, `ExecPlan`, the dispatch cache, or a kernel. The two construction methods are private because the low-level plan does not receive its allocation metadata from `Gemm`; only `LlamaLinear` may complete and publish it.

Read verbose logging configuration once when `Gemm::Impl` is constructed, and reuse one selection printer for heuristic/cache selection and tuning selection:

```cpp
void PrintSelection(const char* kind, const GemmDesc& desc, const LaunchSpec& spec) const
{
    if (verbose_) {
        std::cout << "[Gemm] " << kind
                  << " " << to_string(desc)
                  << " " << spec.kernel->name()
                  << " family=" << spec.kernel->desc().family
                  << " backend=" << spec.kernel->desc().backend
                  << " splits=" << spec.splits
                  << " swizzle=" << spec.swizzle << "\n";
    }
}

const bool verbose_{std::getenv("TM_GEMM_VERBOSE") != nullptr};
```

Remove all reads of `TM_GEMM_TUNE_VERBOSE`. The single `verbose_` value controls both selected-plan and tuning-candidate output.

`GetExecPlan` is then:

```cpp
std::optional<ExecPlan> Gemm::GetExecPlan(const Arguments& args)
{
    Context context{*impl_->props_};
    if (!context.Init(args.operation, args.Adesc, args.Udesc, args.Bdesc, args.Vdesc, args.Cdesc, args.Ddesc)) {
        return std::nullopt;
    }

    LaunchSpec launch = impl_->Dispatch(context, args.operation.dispatch, args.workspace.barriers_size, args.workspace.partials_size);
    if (!launch.kernel) {
        return std::nullopt;
    }

    impl_->PrintSelection("plan", context.desc(), launch);
    return ExecPlan{context.desc(), launch};
}
```

Planning consults imported or measured cache records and otherwise selects and caches the existing heuristic winner. That launch is final for the returned immutable object. Tuning does not consume this object.

## Execution and tuning

Replace the long `Gemm::Run` parameter list with `const ExecPlan&` and `const Gemm::Arguments&`. Remove context construction, dispatch, measurement, and cache mutation from `Run`; it always launches `plan.launch_`.

Keep the existing `Kernel::Launch` virtual interface unchanged. Expand `Gemm::Arguments` into that interface in exactly one internal function in `gemm.cu`:

```cpp
int Gemm::Impl::Launch(const LaunchSpec& spec, const Arguments& args, cudaStream_t stream)
{
    auto workspace = args.workspace;
    return spec.kernel->Launch(args.operation,
                               args.alpha,
                               args.A,
                               args.Adesc,
                               args.U,
                               args.Udesc,
                               args.B,
                               args.Bdesc,
                               args.V,
                               args.Vdesc,
                               args.global_scale,
                               args.global_scale_desc,
                               args.beta,
                               args.C,
                               args.Cdesc,
                               args.D,
                               args.Ddesc,
                               args.W,
                               args.Wdesc,
                               spec.swizzle,
                               spec.splits,
                               workspace,
                               stream);
}
```

`Run` is then:

```cpp
int Gemm::Run(const ExecPlan& plan, const Arguments& args)
{
    if (!plan.launch_.kernel) {
        TM_LOG_FATAL("No feasible kernel found for the problem: {}", to_string(plan.desc_));
        return -1;
    }
    return impl_->Launch(plan.launch_, args, args.stream);
}
```

`Tune` uses the same `Context::Init(Operation, A, U, B, V, C, D)` path and constructs a new immutable plan without performing a separate final-winner execution. `kReuse` on the supplied operation means an exact cached launch may be returned without measurement; otherwise `Tune` measures, stores, and returns the winner:

```cpp
std::optional<ExecPlan> Gemm::Tune(const Arguments& args)
{
    Context context{*impl_->props_};
    if (!context.Init(args.operation, args.Adesc, args.Udesc, args.Bdesc, args.Vdesc, args.Cdesc, args.Ddesc)) {
        return std::nullopt;
    }

    if (args.operation.dispatch & DispatchPolicy::kReuse) {
        if (auto selected = impl_->cache_.Find(context.desc())) {
            impl_->PrintSelection("plan", context.desc(), *selected);
            return ExecPlan{context.desc(), *selected};
        }
    }

    const auto launch = [&](LaunchSpec spec, cudaStream_t stream) { return impl_->Launch(spec, args, stream); };

    std::optional<LaunchSpec> selected = impl_->Measure(context, args.workspace.barriers_size, args.workspace.partials_size, launch, args.stream);
    if (!selected) {
        return std::nullopt;
    }
    impl_->PrintSelection("tune", context.desc(), *selected);
    return ExecPlan{context.desc(), *selected};
}
```

Refactor `Gemm::Impl::Measure` to return the measured winner instead of an integer and remove its current exact-cache early return. Cache-aware versus unconditional tuning is decided by `Operation::dispatch` in `Tune`, before calling `Measure`. Change the exact-match branch of `DispatchCache::Insert` to replace the stored launch so the newly measured winner becomes the record returned by later `GetExecPlan` calls:

```cpp
if (p != idxs.end() && p->first == batch_size) {
    specs[p->second] = spec;
    return false;
}
```

Route cache import through that same insertion path instead of appending duplicate indices and then stable-deduplicating them:

```cpp
int Import(std::istream& is)
{
    std::vector<std::pair<GemmDesc, LaunchSpec>> entries;
    ImportDispatchCache(is, entries, kernels_);
    Summary(entries);

    for (const auto& [desc, spec] : entries) {
        Insert(desc, spec);
    }

    return entries.size();
}
```

Remove the old manual append, sort, deduplicate, and unreferenced-spec compaction code from `Import`. An existing `ExecPlan` continues retaining its copied `LaunchSpec`; import changes only the cache entry observed by a subsequent `GetExecPlan`.

Remove the unused `top_k` argument from `Measure`; candidate preselection continues using `tuning_.top_k`. The complete control flow is:

```cpp
template<class LaunchFunc>
std::optional<LaunchSpec> Measure(Context& ctx, size_t barriers_size, size_t partials_size, LaunchFunc launch_func, cudaStream_t stream)
{
    const auto candidates = Find(ctx, barriers_size, partials_size, tuning_.top_k);

    std::vector<LaunchSpec> specs;
    for (const auto& candidate : candidates) {
        auto swizzled = ctx.Swizzle(candidate, tuning_.swizzle);
        specs.insert(specs.end(), swizzled.begin(), swizzled.end());
    }

    specs = Sampler{*measurer_, tuning_.clusters}.Run(std::move(specs), launch_func, stream);

    if (verbose_) {
        for (const auto& spec : specs) {
            std::cout << "[tune] " << to_string(ctx.desc()) << " " << spec.kernel->name() << " swizzle=" << spec.swizzle << " splits=" << spec.splits << " measured=" << spec.measured << "\n";
        }
    }

    if (specs.empty()) {
        std::cerr << "No valid kernel found for the problem\n";
        return std::nullopt;
    }

    cache_.Insert(ctx.desc(), specs.front());
    return specs.front();
}
```

Both heuristic selection and measurement enter through the existing `Context::Init(Operation, A, U, B, V, C, D)`. The descriptor stored in `ExecPlan::desc_` is only the canonical result copied from `context.desc()`; it is never independently assembled or used to initialize another `Context`.

Remove the current verbose block from `Run`. Execution of an existing plan performs no selection and emits no `[Gemm] plan` or `[Gemm] tune` line.

## `LlamaLinear` integration

`LlamaLinear::Impl::Forward` receives `const gemm::ExecPlan& plan`. It still performs input conversion, constructs the actual pointer-bearing `MatrixLayout` objects, and launches on the current TurboMind stream. It no longer calls `GetOutputSpec` or asks `Gemm` to dispatch.

Output allocation uses the passed plan:

```cpp
const auto& output_spec = plan.output_spec();

Tensor& D = output;
if (!D) {
    D = Tensor{output_spec.layout, output_spec.dtype, kDEVICE};
}
Tensor D_gemm = D.view({-1, D.shape(-1)});

Tensor& W = output_scales;
if (output_spec.scales_dtype != kNull && !W) {
    W = Tensor{output_spec.scales_layout, output_spec.scales_dtype, kDEVICE};
}
```

Use `D_gemm` for the runtime D pointer, leading dimension, and descriptor rows. The GEMM descriptor's N remains `weight.output_dim`; fused SiLU continues writing a compact logical output through the existing epilogue contract.

Extract input conversion, output preparation, descriptor construction, and argument assembly into one reused `LlamaLinear::Impl` method. `A` and `U` are caller-owned local tensors so any conversion allocations outlive the returned non-owning arguments:

```cpp
Gemm::Arguments GetArguments(Tensor& A, Tensor& U, const OutputSpec& output_spec, const Tensor& input, const Tensor& input_scales, const LinearWeight& weight, const Buffer_<int>& indices, const Buffer_<int>& offsets, Tensor& output, Tensor& output_scales);
```

`GetArguments` preserves the rank-two-or-higher dense-input contract in one place before calling the rank-two `GetOperandA` implementation:

```cpp
Tensor in = input.view({-1, input.shape(-1)});
std::tie(A, desc_A, U, desc_U) = GetOperandA(weight, in, input_scales, indices, offsets);
```

Explicit forwarding, production forwarding, and tuning all pass their original input to `GetArguments`; none repeats this view operation. Input rank of at least two remains an unchecked public API precondition, so descriptor-only planning may use `output_spec.layout.stride(-2)` directly. Rank-one input is neither supported nor tested.

The following assembly occurs only inside `GetArguments`, after the normalized `GetOperandA` call, `GetOperandB`, and output descriptor construction:

```cpp
Gemm::Arguments args{};
args.operation = GetOperation(weight);
args.A = A.raw_data();
args.Adesc = desc_A;
args.U = U.data_or((void*)nullptr);
args.Udesc = desc_U;
args.B = B.raw_data();
args.Bdesc = desc_B;
args.V = V.data_or((void*)nullptr);
args.Vdesc = desc_V;
args.global_scale = global_scale.data_or((void*)nullptr);
args.global_scale_desc = global_scale_desc;
args.C = D_gemm.raw_data();
args.Cdesc = desc_D;
args.D = D_gemm.raw_data();
args.Ddesc = desc_D;
args.W = W_ptr;
args.Wdesc = desc_W;
args.workspace = workspace_;
args.stream = core::Context::stream().handle();

return args;
```

Explicit-plan forwarding uses that method and executes only the supplied plan:

```cpp
Tensor A;
Tensor U;
Gemm::Arguments args = GetArguments(A, U, plan.output_spec(), input, input_scales, weight, indices, offsets, output, output_scales);
const int ec = gemm_.Run(plan, args);
if (ec) {
    TM_LOG_ERROR("{}: {}", __PRETTY_FUNCTION__, ec);
}
```

The existing production `Forward` overloads also build the real pointer-bearing arguments first. During warm-up they ask `Tune` for a plan; otherwise they ask `GetExecPlan`. Both branches execute through the same plan-only `Run`:

```cpp
OutputSpec output_spec = impl_->GetOutputSpec(input, weight, indices);
Tensor A;
Tensor U;
Gemm::Arguments args = impl_->GetArguments(A, U, output_spec, input, input_scales, weight, indices, offsets, output.get(), output_scales.get());

std::optional<ExecPlan> exec_plan;
if (impl_->dispatch_policy_ & DispatchPolicy::kMeasure) {
    exec_plan = impl_->gemm_.Tune(args);
}
else {
    exec_plan = impl_->gemm_.GetExecPlan(args);
}

TM_CHECK(exec_plan);
exec_plan->output_spec_ = std::move(output_spec);
const int ec = impl_->gemm_.Run(*exec_plan, args);
if (ec) {
    TM_LOG_ERROR("{}: {}", __PRETTY_FUNCTION__, ec);
}
```

The explicit tuning entry point forces measurement by using `kMeasure`, attaches the allocation metadata, executes the returned winner exactly once, and returns that same immutable plan:

```cpp
OutputSpec output_spec = impl_->GetOutputSpec(input, weight, indices);
Tensor A;
Tensor U;
Gemm::Arguments args = impl_->GetArguments(A, U, output_spec, input, input_scales, weight, indices, offsets, output.get(), output_scales.get());
args.operation.dispatch = DispatchPolicy::kMeasure;
auto exec_plan = impl_->gemm_.Tune(args);
if (!exec_plan) {
    return std::nullopt;
}
exec_plan->output_spec_ = std::move(output_spec);
if (impl_->gemm_.Run(*exec_plan, args)) {
    return std::nullopt;
}
return exec_plan;
```

Do not repeat either `Gemm::Arguments` field assembly or the expanded `Kernel::Launch` argument list outside these two single implementation sites.

Add explicit `Forward` overloads taking `const ExecPlan&` for the Python binding. Retain the existing production overload signatures, but make them construct a local plan from the real arguments as shown above. There is no compatibility `Run` overload and no execution path that omits `ExecPlan`.

Add a separate tuning entry point. It receives already allocated output tensors, performs input conversion once, builds the same runtime descriptors as `Forward`, calls `gemm::Gemm::Tune`, executes the returned plan through `Run`, and returns it. The supplied outputs therefore contain the requested result when this method returns:

```cpp
std::optional<gemm::ExecPlan> Tune(const Tensor& input, const Tensor& input_scales, const LinearWeight& weight, const Buffer_<int>& indices, const Buffer_<int>& offsets, Ref<Tensor> output, Ref<Tensor> output_scales);
```

`LlamaLinear::Tune` recomputes the same `OutputSpec` used by the private allocation query, uses it while assembling the actual output descriptors, and attaches it to the returned plan before publication. It does not pass the specification into `gemm::Gemm::Tune`.

Keep the C++-only warm-up control and its existing call sites in `TurboMind::WarmUp`, but use the existing combined policy value to distinguish cache-aware warm-up tuning from unconditional explicit tuning:

```cpp
void LlamaLinear::set_measure(bool measure)
{
    impl_->dispatch_policy_ = measure ? gemm::DispatchPolicy::kAppend : gemm::DispatchPolicy::kReuse;
}
```

`kAppend` is already `kMeasure | kReuse`: production warm-up enters `Tune`, where the reuse bit permits an exact cached plan to bypass measurement. Python `Linear.tune` sets only `kMeasure`, so it always measures. Remove `set_measure` only from pybind; do not remove the C++ method or `DispatchPolicy::kMeasure` usage.

No CUDA kernel, family registration, weight packing, or engine scheduler behavior changes in this plan.

## One `Gemm` owner in the Python API

The current Python `Linear` creates both `_tm.Gemm()` and `_tm.LlamaLinear()`. An `ExecPlan` from the first object would contain a pointer into a different registry from the one used by the second object.

Remove the standalone `_gemm`. Bind weight and exec planning through `LlamaLinear`, whose internal `gemm_` also performs execution:

```cpp
.def("get_weight_plan", [](LlamaLinear& self, const gemm::WeightQuery& query) { return self.gemm().GetWeightPlan(query); }, py::arg("query"))
.def("get_exec_plan", [](LlamaLinear& self, const LinearWeight& weight, std::shared_ptr<core::Tensor> input, std::shared_ptr<core::Tensor> indices_tensor, std::shared_ptr<core::Tensor> offsets_tensor) {
    core::Tensor input_tensor = TensorFromShared(input, "input");
    auto as_int_buffer = [](const std::shared_ptr<core::Tensor>& tensor) {
        return tensor && *tensor ? Buffer_<int>{static_cast<int*>(tensor->raw_data()), tensor->size(), tensor->device()} : Buffer_<int>{};
    };
    Buffer_<int> indices = as_int_buffer(indices_tensor);
    Buffer_<int> offsets = as_int_buffer(offsets_tensor);
    return self.GetExecPlan(input_tensor, weight, indices, offsets);
}, py::arg("weight"), py::arg("input"), py::arg("indices") = py::none(), py::arg("offsets") = py::none())
```

Removing `Linear._gemm` does not remove the `_tm.Gemm` type. Production model building and dtype resolution still use independently owned GEMM registries. Retain that binding with the renamed weight getter:

```cpp
py::class_<gemm::Gemm>(m, "Gemm")
    .def(py::init<>())
    .def("get_weight_plan", &gemm::Gemm::GetWeightPlan, py::arg("query"))
    .def("data_types", &gemm::Gemm::DataTypes, py::arg("weight_format"));
```

Migrate the production builder call in `lmdeploy/turbomind/builders/_base.py` without retaining a `plan_weight` alias:

```python
plan = self._ctx.gemm.get_weight_plan(query)
```

Bind `OutputSpec` only as the private allocation intermediate used by `Linear.tune`:

```cpp
py::class_<gemm::OutputSpec>(m, "_GemmOutputSpec")
    .def_property_readonly("output_shape", [](const gemm::OutputSpec& spec) { return spec.layout.shape(); })
    .def_property_readonly("output_stride", [](const gemm::OutputSpec& spec) { return spec.layout.stride(); })
    .def_readonly("output_dtype", &gemm::OutputSpec::dtype)
    .def_property_readonly("output_scales_shape", [](const gemm::OutputSpec& spec) { return spec.scales_layout.shape(); })
    .def_property_readonly("output_scales_stride", [](const gemm::OutputSpec& spec) { return spec.scales_layout.stride(); })
    .def_readonly("output_scales_dtype", &gemm::OutputSpec::scales_dtype);
```

Add a private `_get_output_spec` binding that takes the prepared weight, input, and optional indices and delegates to `LlamaLinear::GetOutputSpec`. It exists only so Python can allocate Torch output buffers before measurement. Add `tune` binding arguments matching the existing dense/grouped forward bindings; it delegates to `LlamaLinear::Tune` and returns the new immutable native plan. The private `OutputSpec` is not passed back into the tuning binding.

Bind the C++ exec plan read-only. Do not expose its kernel pointer, `GemmDesc`, swizzle, or split count through the public Python API:

```cpp
py::class_<gemm::ExecPlan>(m, "GemmExecPlan")
    .def_property_readonly("output_shape", [](const gemm::ExecPlan& plan) { return plan.output_spec().layout.shape(); })
    .def_property_readonly("output_stride", [](const gemm::ExecPlan& plan) { return plan.output_spec().layout.stride(); })
    .def_property_readonly("output_dtype", [](const gemm::ExecPlan& plan) { return plan.output_spec().dtype; })
    .def_property_readonly("output_scales_shape", [](const gemm::ExecPlan& plan) { return plan.output_spec().scales_layout.shape(); })
    .def_property_readonly("output_scales_stride", [](const gemm::ExecPlan& plan) { return plan.output_spec().scales_layout.stride(); })
    .def_property_readonly("output_scales_dtype", [](const gemm::ExecPlan& plan) { return plan.output_spec().scales_dtype; });
```

Change the dense and grouped pybind forward methods to receive `const gemm::ExecPlan& plan` and call the explicit C++ `Forward` overload. The binding receives non-null output tensors because the Python `Linear` method allocates missing destinations before entering native code. The binding itself does not allocate storage. Remove the `set_measure` binding.

## Public Python API

Export `ExecPlan` beside `Linear`, `Weight`, and `WeightPlan`:

```python
__all__ = ['ExecPlan', 'Linear', 'Weight', 'WeightPlan', 'is_available']
```

The wrapper contains only its native implementation:

```python
class ExecPlan:
    __slots__ = ('_impl',)

    @property
    def output(self) -> torch.Tensor:
        spec = self._impl
        return torch.empty_strided(spec.output_shape, spec.output_stride, dtype=_to_torch_dtype(spec.output_dtype), device='meta')

    @property
    def output_scales(self) -> torch.Tensor | None:
        spec = self._impl
        if spec.output_scales_dtype == _tm().DataType.TYPE_INVALID:
            return None
        return torch.empty_strided(spec.output_scales_shape, spec.output_scales_stride, dtype=_to_torch_dtype(spec.output_scales_dtype), device='meta')
```

`Linear` retains only the device and the single native executor:

```python
class Linear:
    __slots__ = ('device', '_impl')
```

`Linear.__init__`, `close`, `get_weight_plan`, record import/export, and tuning construction all use `self._impl`. `get_weight_plan` calls `self._impl.get_weight_plan(query)`.

Rename the existing public method `Linear.plan_weight` to `Linear.get_weight_plan` without changing its arguments, source-format normalization, fusion handling, or returned Python `WeightPlan`. Rename the corresponding native binding to `get_weight_plan`; do not retain `plan_weight` as an alias.

Replace `output_spec` and `output_spec_moe` with one getter. The prepared weight determines dense versus grouped mode. Grouped planning receives the same `offsets` and optional `indices` tensors that will be passed to `forward_moe`, allowing `Context::Init` to derive blocked or indexed striding from the real descriptors:

```python
def get_exec_plan(self, x: torch.Tensor, weight: Weight, *, offsets: torch.Tensor | None = None, indices: torch.Tensor | None = None) -> ExecPlan:
    tm = _tm()
    impl = self._impl.get_exec_plan(weight._impl, tm.from_dlpack_with_strides(x), None if indices is None else tm.from_dlpack_with_strides(indices), None if offsets is None else tm.from_dlpack_with_strides(offsets))
    if impl is None:
        raise NotImplementedError('no GEMM kernel accepts the execution problem')
    plan = ExecPlan()
    plan._impl = impl
    return plan
```

No output tensor is accepted by the planning call. The returned meta tensors describe allocation only and carry no storage.

`Linear` allocates only a missing destination. A caller-provided `out` or `out_scales` is passed through unchanged. Normal execution uses the allocation metadata already attached to the native plan. Tuning obtains the private native `OutputSpec` only for the same allocation function; it does not pass that object into the native tuning call:

```python
def _allocate_output(self, spec, out: torch.Tensor | None, out_scales: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if out is None:
        out = torch.empty_strided(spec.output_shape, spec.output_stride, dtype=_to_torch_dtype(spec.output_dtype), device=self.device)
    if out_scales is None and spec.output_scales_dtype != _tm().DataType.TYPE_INVALID:
        out_scales = torch.empty_strided(spec.output_scales_shape, spec.output_scales_stride, dtype=_to_torch_dtype(spec.output_scales_dtype), device=self.device)
    return out, out_scales

def __call__(self, x: torch.Tensor, weight: Weight, *, plan: ExecPlan, out: torch.Tensor | None = None, input_scales: torch.Tensor | None = None, out_scales: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    with self._activate():
        out, out_scales = self._allocate_output(plan._impl, out, out_scales)
        self._impl.forward_dense(plan._impl, _tm().from_dlpack_with_strides(x), weight._impl, _tm().from_dlpack_with_strides(out), None if input_scales is None else _tm().from_dlpack_with_strides(input_scales), None if out_scales is None else _tm().from_dlpack_with_strides(out_scales))
    return out, out_scales

def forward_moe(self, x: torch.Tensor, weight: Weight, *, plan: ExecPlan, offsets: torch.Tensor, out: torch.Tensor | None = None, indices: torch.Tensor | None = None, input_scales: torch.Tensor | None = None, out_scales: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    with self._activate():
        out, out_scales = self._allocate_output(plan._impl, out, out_scales)
        self._impl.forward_moe(plan._impl, _tm().from_dlpack_with_strides(x), weight._impl, None if indices is None else _tm().from_dlpack_with_strides(indices), _tm().from_dlpack_with_strides(offsets), _tm().from_dlpack_with_strides(out), None if input_scales is None else _tm().from_dlpack_with_strides(input_scales), None if out_scales is None else _tm().from_dlpack_with_strides(out_scales))
    return out, out_scales
```

Tuning constructs and returns a new plan; it never receives an old plan:

```python
def tune(self, x: torch.Tensor, weight: Weight, *, offsets: torch.Tensor | None = None, indices: torch.Tensor | None = None, out: torch.Tensor | None = None, input_scales: torch.Tensor | None = None, out_scales: torch.Tensor | None = None) -> tuple[ExecPlan, torch.Tensor, torch.Tensor | None]:
    tm = _tm()
    input_impl = tm.from_dlpack_with_strides(x)
    indices_impl = None if indices is None else tm.from_dlpack_with_strides(indices)
    offsets_impl = None if offsets is None else tm.from_dlpack_with_strides(offsets)
    input_scales_impl = None if input_scales is None else tm.from_dlpack_with_strides(input_scales)
    with self._activate():
        output_spec = self._impl._get_output_spec(weight._impl, input_impl, indices_impl)
        out, out_scales = self._allocate_output(output_spec, out, out_scales)
        impl = self._impl.tune(input_impl, weight._impl, indices_impl, offsets_impl, tm.from_dlpack_with_strides(out), input_scales_impl, None if out_scales is None else tm.from_dlpack_with_strides(out_scales))
    if impl is None:
        raise NotImplementedError('no GEMM kernel accepts the execution problem')
    plan = ExecPlan()
    plan._impl = impl
    return plan, out, out_scales
```

Because allocation occurs inside `_activate`, it uses `self.device`, Torch's device allocator, and the Torch CUDA stream current for that device when the method was called. The returned tensors are ordinary caller-visible Torch tensors; the native binding neither owns nor retains them after the call.

As in the approved standalone API contract, mismatching a plan with different tensors, dimensions, modes, dtypes, weights, or executor is undefined behavior. Do not add defensive comparisons of the runtime arguments against `ExecPlan::desc_`.

## Tuning behavior in Python

Tuning constructs a new immutable plan and returns the Torch buffers used to measure and execute it:

```python
exec_plan, out, out_scales = linear.tune(x, weight)
linear(x, weight, plan=exec_plan, out=out, out_scales=out_scales)
```

The Torch allocations happen before the native tuning call and are not included in the GEMM measurer's event timings. No pre-existing plan is passed or ignored. `Gemm::Tune` measures and constructs the plan from the winner; `LlamaLinear::Tune` then executes that same winner once through `Run`. The following ordinary forward uses the immutable selected launch without dispatch.

Remove the `Linear.tuning()` context manager. Tuning is plan construction, not mutable executor state.

## Fixture and benchmark migration

Do not add a parallel test harness or new test files. Adapt the existing fixture around the new API.

Remove `prepare_stream` and `execution_stream` from `LinearFixture`. The refactored fixture does not bind itself to fixed streams; construction, weight preparation, batch preparation, reference execution, GEMM execution, and tuning all use the Torch stream current when the caller invokes them. With no external stream context, this is the device's default stream.

Construct weights directly:

```python
self.linear = Linear(self.device)
self._build_weights()
```

The public fixture operations no longer switch streams:

```python
def prepare_batch(self, batch_size: int) -> None:
    self._prepare_batch(batch_size)

def run_reference(self) -> None:
    self._run_reference()

def run_linear_forward(self):
    return self._run_linear_forward()
```

Remove the corresponding custom-stream synchronization from `LinearFixture.close()`.

`LinearFixture._prepare_batch` creates and retains one plan after preparing the input and routing metadata. It leaves both output destinations empty so the first `Linear` call exercises the new allocation path:

```python
if case.expert_num > 0:
    indices = self.f2n if case.moe_indexed else None
    self.exec_plan = self.linear.get_exec_plan(self.x_original, self.w_quant, offsets=self.offsets, indices=indices)
else:
    self.exec_plan = self.linear.get_exec_plan(self.x_original, self.w_quant)

self.output = None
self.output_scales = None
```

The existing reference path determines whether FP8 output scales are required from `self.exec_plan.output_scales`, not from the initially empty `self.output_scales`.

The forward path stores the tensors returned by `Linear`; the first call allocates and later calls reuse them:

```python
def _run_linear_forward(self):
    case = self.case
    if case.expert_num > 0:
        self.output, self.output_scales = self.linear.forward_moe(self.x_original, self.w_quant, plan=self.exec_plan, offsets=self.offsets, indices=self.f2n if case.moe_indexed else None, input_scales=self.input_scales, out=self.output, out_scales=self.output_scales)
    else:
        self.output, self.output_scales = self.linear(self.x_original, self.w_quant, plan=self.exec_plan, input_scales=self.input_scales, out=self.output, out_scales=self.output_scales)
    return self.output, self.output_scales
```

Batch preparation replaces the plan whenever M or routing mode changes. Remove the fixture-only `_allocate_from_meta` helper after its final use disappears.

Add one fixture method that replaces the heuristic plan with the immutable measured plan and retains the buffers returned by the tuning call:

```python
def tune(self):
    case = self.case
    indices = self.f2n if case.moe_indexed else None
    self.exec_plan, self.output, self.output_scales = self.linear.tune(self.x_original, self.w_quant, offsets=self.offsets, indices=indices, out=self.output, input_scales=self.input_scales, out_scales=self.output_scales)
```

In `benchmark.py`, replace the context-manager block with the explicit constructor:

```python
if tune:
    fx.tune()
```

Timed iterations reuse both the newly returned plan and its Torch output allocations. Record events and synchronize the caller's current stream rather than a fixture-owned stream:

```python
stream = torch.cuda.current_stream(fx.device)
```

The fixture must no longer call `output_spec` or `output_spec_moe`. Remove those public methods rather than retaining aliases that bypass execution planning.

## Production forwarding

The Python standalone API receives explicit exec plans. Existing C++ engine callers continue using the current `LlamaLinear::Forward` overload signatures. Those overloads construct a local plan from their real arguments and execute it through the same `Run(const ExecPlan&, const Arguments&)` path.

Production warm-up retains `set_measure(true/false)` and its measured-cache export behavior. The refactor therefore does not require scheduler, `BatchOp`, or cache-management changes and does not change any contract in `src/turbomind/engine/README.md`.

No exec plan is stored in `LinearWeight`: execution state is shape- and routing-specific and does not belong to prepared weight metadata.

## Files in scope

- `src/turbomind/kernels/gemm/plan.h` -> `weight_plan.h`
- `src/turbomind/kernels/gemm/plan.cc` -> `weight_plan.cc`
- `src/turbomind/kernels/gemm/output_spec.h` (new)
- `src/turbomind/kernels/gemm/exec_plan.h` (new)
- `src/turbomind/kernels/gemm/family.h`
- `src/turbomind/kernels/gemm/kernel/e4m3.h`
- `src/turbomind/kernels/gemm/dispatch_cache.h`
- `src/turbomind/kernels/gemm/dispatch_cache.cu`
- `src/turbomind/kernels/gemm/gemm.h`
- `src/turbomind/kernels/gemm/gemm.cu`
- `src/turbomind/kernels/gemm/CMakeLists.txt`
- `src/turbomind/models/linear_weight.h`
- `src/turbomind/models/linear_weight.cc`
- `src/turbomind/models/llama/LlamaLinear.h`
- `src/turbomind/models/llama/LlamaLinear.cu`
- `src/turbomind/python/linear_bind.cpp`
- `lmdeploy/turbomind/builders/_base.py`
- `tests/turbomind/linear/linear.py`
- `tests/turbomind/linear/fixture.py`
- `tests/turbomind/linear/benchmark.py`
- `tests/turbomind/linear/run_sched_cmp.py`
- existing includes and call sites found by the required `rg` checks

Do not change CUDA kernels, kernel registries, family definitions, converters, packers, weight-format resolution, or engine scheduling.

## Implementation order

1. Add neutral `output_spec.h`, move `OutputSpec` into it without behavioral changes, and include it independently from `family.h` and the new `exec_plan.h`.
2. Rename C++ `GemmPlan` and its files to `WeightPlan`; rename `PlanWeight` to `GetWeightPlan`; update CMake, includes, `LinearWeight`, both `LlamaLinear` and `_tm.Gemm` pybind entry points, and the production builder call in `lmdeploy/turbomind/builders/_base.py`.
3. Add the flat, non-owning `Gemm::Arguments` aggregate and the single internal adapter to `Kernel::Launch`.
4. Add `Gemm::GetExecPlan(const Arguments&)` on top of the existing `Context::Init(Operation, A, U, B, V, C, D)` path; return `std::optional<ExecPlan>` directly.
5. Make `Gemm::Run(const ExecPlan&, const Arguments&)` launch only the plan's retained `LaunchSpec`.
6. Make measurement return a winner, allow it to replace an exact cache record, and add `Gemm::Tune(const Arguments&)` as a separate immutable-plan constructor.
7. Add the reused `LlamaLinear::Impl::GetArguments`, then add `LlamaLinear::GetOutputSpec`, `GetExecPlan`, `Tune`, explicit-plan forwarding, and production local-plan construction. Keep the C++ `set_measure` warm-up control and map its enabled state to `kAppend`.
8. Move Python planning onto the `LlamaLinear`-owned `Gemm`; expose `get_weight_plan`, `get_exec_plan`, and `tune`; remove `Linear._gemm`, `Linear.tuning`, and the pybind exposure of `set_measure`.
9. Add the public Python `ExecPlan`, replace allocation-only queries, and require it during ordinary execution.
10. Migrate the existing fixture and benchmark flow, and change `run_sched_cmp.py` documentation and environment assignment from `TM_GEMM_TUNE_VERBOSE` to `TM_GEMM_VERBOSE`.
11. Build and run the verification below.

## Verification

Do not add new test files or benchmark cases. Reuse the existing tests and explicit benchmark-case selection.

### Static checks

```bash
rg -n "GemmPlan|PlanWeight|gemm/plan\\.h|gemm/plan\\.cc|output_spec\\(|output_spec_moe\\(" src/turbomind lmdeploy/turbomind tests/turbomind/linear
rg -n "\\.plan_weight\\(|\\.exec_plan\\(|\\.exec_plan_moe\\(|get_exec_plan_moe|def tuning" src/turbomind lmdeploy/turbomind tests/turbomind/linear
rg -n '"set_measure"' src/turbomind/python tests/turbomind/linear
rg -n "ConstMatrixRef|MatrixRef" src/turbomind/kernels/gemm src/turbomind/models/llama
rg -n "_gemm" tests/turbomind/linear/linear.py
rg -n "GetWeightPlan|GetExecPlan|Tune|get_weight_plan|get_exec_plan|\.tune\(|WeightPlan|ExecPlan" src/turbomind tests/turbomind/linear
rg -n "TM_GEMM_TUNE_VERBOSE" src/turbomind lmdeploy/turbomind tests/turbomind
git diff --check
```

The first three commands must report no stale public `GemmPlan`, `PlanWeight`, old plan-file include, old Python output query, old Python getter spelling, or standalone Python `_gemm`. Legitimate internal `Family::output_spec` and `OutputSpec` references remain. The `TM_GEMM_TUNE_VERBOSE` command must also report no matches; historical documents under `plans/` are intentionally outside that active-code check.

### Build

Do not set `PYTHONPATH` for compilation:

```bash
cd /data/lmdeploy-gemm/build
ninja
```

### Python syntax and existing unit coverage

Set the in-tree path before Python commands:

```bash
PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python -m py_compile lmdeploy/turbomind/builders/_base.py tests/turbomind/linear/linear.py tests/turbomind/linear/fixture.py tests/turbomind/linear/benchmark.py tests/turbomind/linear/run_sched_cmp.py tests/turbomind/linear/test_linear.py
```

Before every CUDA-backed command, use `get_gpu_usage`, choose an empty SM90 GPU, and run outside the sandbox. Then run:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm pytest tests/turbomind/linear/test_linear.py -q
```

In the existing `tests/turbomind/linear/test_linear.py`, reuse the existing dense BF16 case for focused rank-three coverage; do not add a new case or test file:

```python
@cuda_required
@tm_required
def test_rank_three_dense_input():
    case = case_by_name()['llama2_7b_o__bf16_bf16_bf16']
    fx = LinearFixture(case)
    try:
        fx.prepare_batch(6)
        x = fx.x_original.view(2, 3, case.input_dim)
        fx.x_original = x
        fx.x_source = x
        fx.exec_plan = fx.linear.get_exec_plan(x, fx.w_quant)
        fx.output = None
        fx.output_scales = None
        fx.run_reference()
        fx.run_linear()
        assert fx.output.shape == (2, 3, case.output_dim)
        fx.check_tolerances(fx.compare())
    finally:
        fx.close()
```

Import `case_by_name` beside the existing `expand_suite` import. This test exercises descriptor-only planning, output allocation, centralized input flattening, execution, and restoration of the rank-three output shape. Do not add rank-one coverage because rank at least two is an unchecked precondition.

### Explicit exec-plan correctness

Run existing dense, grouped, fused, U4-zero-synthesis, and native-FP8-output cases with explicit selection. A requested case must fail rather than skip if unsupported:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python -m tests.turbomind.linear.bench_linear --suite custom --case qwen35_35b_a3b_shared_gate_up__bf16_bf16_bf16__fuse_silu,llama2_7b_o__bf16_u4k128_bf16,qwen35_35b_a3b_shared_gate_up__e4m3k128_e4m3b128_bf16__fuse_silu,mixtral_8x7b_gate_up__bf16_bf16_bf16__fuse_silu --batch 1 --iters 0
```

Confirm that every case passes its existing numerical tolerances, that the first forward allocates its missing Torch output, and that native FP8 output allocates and returns its output-scale tensor. Confirm that later forwards reuse the returned destinations when they are passed back explicitly.

### Dispatch and tuning

Use a fresh process and leave validation enabled so an ordinary heuristic plan executes before explicit tuning:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" TM_GEMM_TUNE='top_k=0,min_iter=1,max_iter=1,swizzle=[0,1,2,3]' TM_GEMM_VERBOSE=1 PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python -m tests.turbomind.linear.bench_linear --suite custom --case llama2_7b_o__bf16_u4k128_bf16 --batch 1 --tune --iters 100
```

Verify from logs that:

- candidate `[tune]` lines appear even if validation previously populated an exact heuristic record, proving explicit tuning measures rather than accepting that record;
- one `[Gemm] plan` line identifies the ordinary selected launch and one `[Gemm] tune` line identifies the measured winner;
- `tune` constructs an immutable plan and performs the final winner launch;
- all 100 timed forwards reuse the retained exec plan without emitting another selection line;
- there is no second heuristic dispatch or "No feasible kernel" failure.

### Cache import ordering

Use an explicit base path and the benchmark's emitted per-case suffix:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" TM_GEMM_TUNE='top_k=0,min_iter=1,max_iter=1,swizzle=[0,1,2,3]' PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python -m tests.turbomind.linear.bench_linear --suite custom --case llama2_7b_o__bf16_u4k128_bf16 --batch 1 --tune --no-validate --iters 0 --export /tmp/gemm_exec_plan_records
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" TM_GEMM_VERBOSE=1 PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python -m tests.turbomind.linear.bench_linear --suite custom --case llama2_7b_o__bf16_u4k128_bf16 --batch 1 --iters 0 --import /tmp/gemm_exec_plan_records.llama2_7b_o__bf16_u4k128_bf16__tp1__ep1
```

Run each command outside the sandbox after checking that the selected GPU remains empty. Confirm the second process imports the record before `_prepare_batch` creates the exec plan, selects the imported kernel, and passes correctness. Importing after constructing a plan leaves that immutable plan unchanged; a later `get_exec_plan` observes the imported record.

Also verify replacement in one process. The source registry deliberately exports swizzle 3, while the target registry first constructs a swizzle-0 plan for the same descriptor:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python - <<'PY'
import os

import torch

from tests.turbomind.linear.cases import case_by_name
from tests.turbomind.linear.fixture import LinearFixture

os.environ['TM_GEMM_VERBOSE'] = '1'
case = case_by_name()['llama2_7b_o__bf16_u4k128_bf16']
device = torch.device('cuda')
record = '/tmp/gemm_exec_plan_import_after_plan'

os.environ['TM_GEMM_TUNE'] = 'top_k=0,min_iter=1,max_iter=1,swizzle=[3]'
source = LinearFixture(case, device=device)
try:
    source.prepare_batch(4096)
    source.tune()
    source.linear.export_records(record)
finally:
    source.close()

os.environ['TM_GEMM_TUNE'] = 'top_k=0,min_iter=1,max_iter=1,swizzle=[0]'
target = LinearFixture(case, device=device)
try:
    target.prepare_batch(4096)
    original = target.exec_plan
    target.linear.import_records(record)
    target.exec_plan = target.linear.get_exec_plan(target.x_original, target.w_quant)
    target.run_reference()
    target.run_linear()
    target.check_tolerances(target.compare())
    target.exec_plan = original
    target.run_linear()
    target.check_tolerances(target.compare())
finally:
    target.close()
PY
```

Run this command outside the sandbox after checking the selected GPU again. Require the target's first `[Gemm] plan` line to report `swizzle=0` and its post-import `[Gemm] plan` line to report `swizzle=3`. Executing the retained `original` plan after import must emit no new selection line and must still pass correctness.

### Model smoke

Use `scripts/test_turbomind_model.py` unchanged with the locally cached BF16 `Qwen/Qwen3-8B` model. Select an empty SM90 GPU with `get_gpu_usage`, then run every command below outside the sandbox:

```bash
set -euo pipefail
WARMUP_RECORDS=$(mktemp /tmp/turbomind_warmup_gemm_records.XXXXXX)
WARMUP_LOG=$(mktemp /tmp/turbomind_warmup_gemm_log.XXXXXX)
env -u TM_GEMM_IMPORT -u TM_GEMM_TUNE TM_GEMM_VERBOSE=1 TM_GEMM_EXPORT="$WARMUP_RECORDS" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python scripts/test_turbomind_model.py --model-id Qwen/Qwen3-8B --cache-dir /mnt_cfs/huggingface_hub/hub/ --gpus "$EMPTY_SM90_GPU" --max-new-tokens 128 --prompt "Explain why immutable execution plans are useful in a CUDA inference engine." 2>&1 | tee "$WARMUP_LOG"
test -s "$WARMUP_RECORDS"
rg -m1 '^\[Gemm\] tune ' "$WARMUP_LOG"
```

The model process must succeed, at least one `[Gemm] tune` line must prove that `set_measure(true)` reached the production `kAppend -> Tune` path, and the exported record file must be nonempty. Inspect the generated response in `WARMUP_LOG`; the requested generation length is 128 tokens, and the response must contain meaningful human language relevant to the prompt. A successful exit with gibberish is a failure.

## Completion criteria

- C++ `GemmPlan` has become `WeightPlan`, with no change to family or packing selection.
- The plan getters are consistently named `GetWeightPlan` and `GetExecPlan` in C++, and `get_weight_plan` and `get_exec_plan` in Python.
- `OutputSpec` fields are `layout`, `dtype`, `scales_layout`, and `scales_dtype`, with no redundant `output_` prefix.
- `OutputSpec` lives in `output_spec.h`; `family.h` and `exec_plan.h` include that neutral header independently, and `family.h` does not include `exec_plan.h`.
- Runtime operands are carried by the flat, non-owning `Gemm::Arguments`; no `ConstMatrixRef`, `MatrixRef`, or equivalent per-operand wrapper is introduced.
- `Gemm::GetExecPlan(const Arguments&)` and `Gemm::Tune(const Arguments&)` return `std::optional<ExecPlan>` directly; neither method receives `OutputSpec`, and `OutputSpec` is not a member of `Gemm::Arguments`.
- The expanded `Kernel::Launch` argument list exists in exactly one adapter in `gemm.cu`.
- A public Python `ExecPlan` contains only one opaque native implementation.
- The native exec plan contains `OutputSpec`, `GemmDesc`, and the selected `LaunchSpec` but no tensor pointers or allocations.
- `context.cu` remains the only construction path for `GemmDesc`; both initial planning and tuning use the existing `Context::Init(Operation, A, U, B, V, C, D)` interface.
- Python planning and execution use the same `Gemm` registry owned by `LlamaLinear`.
- Output allocation metadata and kernel selection are produced by one planning call.
- Omitting `out` allocates and returns a Torch tensor with the plan's exact shape, stride, dtype, device, and current-stream ordering.
- Omitting `out_scales` allocates it only when the plan specifies an output-scale tensor.
- Caller-provided output destinations remain supported and are returned unchanged.
- `ExecPlan` is immutable after construction.
- `Gemm::Run` remains `[[nodiscard]]`; explicit and production forwarding preserve nonzero launch-error logging.
- Ordinary execution launches the retained kernel without redispatch.
- Tuning never accepts or ignores an existing plan; it returns a new immutable `ExecPlan` containing the measured winner.
- `Linear.tuning()` and the pybind exposure of `LlamaLinear::set_measure` no longer exist; the C++ method remains for production warm-up.
- Existing production `LlamaLinear::Forward` callers construct a local plan and execute through the sole `Run(const ExecPlan&, const Arguments&)` path.
- Existing linear tests, explicit SM90 cases, tuning, cache import, the full build, and the model smoke pass.
