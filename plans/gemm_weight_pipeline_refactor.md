# GEMM-owned weight processing plan

## Status

**Implemented and verified on SM90.**

## Design

The GEMM module owns every weight-representation decision. The loader asks,
per TP-local weight, how the weight must be represented; the GEMM module
answers from its device registries; the loader arranges tensors and calls
the returned pack operation. Three functions cover the whole pipeline:

```text
(registry, weight_format, input_dtype)   -> kernel family + WeightBridge
family.gate_up(activation_type, projection_n) -> gate/up arrangement
family.Pack(LinearWeight, WeightBridge) -> packed weights + descriptors
```

Every family is a concrete `Family` object. The object stores the
family's metadata and its family-owned capability and packing callbacks.
The object definition only names callbacks; it never contains their bodies.
Related objects share function templates, while unique callbacks are named
functions beside the family. There are no concrete family subclasses.

Principles:

- Every value has exactly one producer and at least one non-diagnostic
  consumer. No field is carried "for later".
- The GEMM module owns format, kernel-family, and packing policy. Python and
  model code contain no architecture checks, format tables, group-size
  constants, or kernel alignment constants.
- Selection is a hard filter with priority ordering. Any unsatisfiable
  condition is an error at load time; there are no silent fallbacks to other
  formats.
- The query is shape-agnostic — it carries no K/N at all; shapes enter
  only where they are consumed: the loader's padding, the gate/up
  derivation's width argument, and `Pack`'s alignment checks. The query
  plans one flat GEMM, carries no epilogue
  and no gate/up distinction. Its `data_type` is an operand contract —
  the dtype of every floating-point operand — not a model-format decision.
- Shape gaps are the family's responsibility, exposed as data: each kernel
  family reports the K/N divisibility and minimum extents over which its
  kernel family provides coverage (`align_k`/`align_n` and
  `min_k`/`min_n`). The loader meets them by zero-padding before TP slicing
  with the existing group-padding helpers (`pad_output_groups` /
  `pad_input_groups`, the mechanism behind `_pad_ffn_for_tp`) — but
  only an axis whose producer and consumer both run at the padded width
  may grow: the FFN intermediate, padded on w1/w3's output and w2's input
  alike, with `config.inter_size` carrying the padded width as today.
  Every other axis must natively meet the family's alignment and minimum
  extent — a miss is
  a load error, and there is no runtime activation padding or output
  slicing anywhere. Dimensions are the padded dimensions everywhere:
  `LinearWeight.input_dim`/`output_dim`, the `MatrixLayout` descriptors,
  and every runtime GEMM extent agree by construction. Padded elements
  contribute exactly zero (weight 0, zero-point 0, scales 0, bias 0).
  Python contains no kernel alignment constants.
- One model component — a single LLM or a single ViT — is required to run on
  one CUDA arch across all of its local and remote devices (TP/EP/CP/DP or
  none); a VLM's LLM and ViT may run on different archs. This is an unchecked
  deployment precondition: startup performs no local architecture check and
  no cross-node architecture exchange. Under that precondition every active
  device of a component holds an identical registry. The registry is a pure function of the
  compiled kernel set, the arch (`is_arch_compatible`), and
  `sharedMemPerBlockOptin` (the registry's smem filter) — and
  `sharedMemPerBlockOptin` is an architecture property, identical for all
  devices of the same CUDA arch — so equal arch implies identical
  registries and first-device answers are authoritative.
- Runtime dispatch consumes the actual `MatrixLayout` values produced by
  `LlamaLinear` and the family restriction stored in `GemmDesc.family`. It
  never consults the `GemmPlan`: normal plan-produced outputs carry the fixed
  `Family.id`, which restricts filtering, measurement, and cache lookup
  to kernels whose `family().id` has the same value. The MoE router's FP32
  output is part of its weight query, so it selects a source-preserving cuBLAS
  family and carries that family's nonzero id like every other planned linear.

```text
ModelLoader.data_type ------------------------+
                                              |
Linear.weight_format --make_data_format()--+   |
                                           |  |
TP-local weight shape (K, N) --------------+--+
                                              |
                                              v
                                        WeightQuery
                                              |
                           kernel registry (families pack their own weights)
                                              |
                                              v
                                        GemmPlan
              +-------------------------------+--------------+
              |                               |              |
       plan.gate_up()                  plan.pack()      LinearWeight
       gate/up arrangement             family Pack()     runtime fields
              |                               |              |
              +-------------------------------+--------------+
                                              |
                                              v
                                 runtime GEMM dispatch from
                                 actual MatrixLayout values
```

## Terms

A **logical weight format** is the checkpoint/load-storage `DataFormat`
produced by `WeightFormat.make_data_format(data_type)`. It describes
the weight, scale, and zero tensors before persistent kernel packing.

A **kernel family** is one concrete `Family` object implementing the
GEMM weight pipeline: the family's persistent weight layout, packing, and
fusion behavior are baked into the registration unit that owns the object.
It has one fixed, nonzero integer `id` and one fixed `priority`; `0` is
reserved for unrestricted runtime dispatch. Capability and packing are
object callbacks. Input conversion and the MLP fusion query are common
`Family` operations driven by the object's formats and fusion data.
Tile size,
stages, raster order, swizzle, split-K, epilogue, and runtime addressing are
properties of the family, not of a runtime key. The family reports the
epilogues it can execute (`supported_epilogues`) and the actual output
format per epilogue (`output_format(epilogue)`); neither is a stored field.

The **data type** is the query field naming the resolved component
dtype — the dtype of floating-point weights, token embeddings, and hidden
states. It is always set. It is the default unfused output dtype.

The **output type** is the query field naming a required unfused output dtype.
`kNull` means the data type. The MoE gate sets it to `kFloat`; every other
weight query leaves it unset. It is a filter, not a preference: a family whose
unfused output differs from the resolved requirement is ineligible.

The **input type** is the query field naming the preferred GEMM operand-A
dtype — a preference, never a filter. `kNull` — unspecified — leaves the
ordering untouched: selection walks all eligible families in
`priority` order, and the FP8 preference on capable hardware
falls out of priority rather than a special case. An explicit input type
reorders: eligible families whose input format has that dtype walk first,
the rest keep their relative priority below — nothing is rejected. For an
FP8 weight, explicit BF16 therefore prefers weight-only W8A16 execution
even where W8A8 exists, and explicit FP8 prefers W8A8 where registered
and runs W8A16 where not. The preference never relaxes
the data-type contract: a weight-only family whose floating-point input differs
from the data type is ineligible regardless of the input type, and its output
must match the resolved output requirement — so an FP16 model never selects a
BF16-input family.

A **floating-point activation format** contains model-dtype values, block sizes
`{1, 1}`, and no scale or zero tensor. A **dynamic FP8 activation format**
contains E4M3 values, K-axis block sizes `{128, 1}`, FP32 scales, and no zero
tensor.

An **epilogue** is a bitmask (`Epilogue::kNone == 0`); multiple bits may be
set. The enum class gains bitwise `|`/`&` operators so masks compose and
membership is testable. The query never sees epilogue values. They appear
only inside kernel family implementations — a family's fusion query answers
with the epilogue it implements — and as the selected mask stored privately
in `GemmPlan`.
The loader passes the activation type, never an epilogue; the kernel family
owns the activation-to-epilogue pairing.

`TM_GEMM_WEIGHT_PACK` is a debug knob (unset = auto, 0 = force plain
layouts, 1 = force packed layouts). It is honored inside
`Family::supports`: a family whose persistent layout requires
non-trivial packing reports itself unsupported under 0, a source-preserving
family under 1. Selection and the data-type enumeration therefore see
the same
reduced registry; Python and model code never read the knob.

## Data model

Add `src/turbomind/kernels/gemm/family.h`; `family.cc` contains the
non-inline `Family` method definitions:

```cpp
#pragma once

#include <cstdint>
#include <optional>

#include <cuda_runtime.h>

#include "src/turbomind/core/data_format.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind {
class LinearWeight;
}

namespace turbomind::gemm {

// The concrete re-expression closing the gap between an accepted stored
// weight format and the family's kernel-facing layout. Produced by
// Family::supports after it verifies the constraints; consumed by
// Family::Pack, which performs exactly what the bridge says. A
// trivial bridge (operator bool() == false) means the stored format needs
// no re-expression.
struct WeightBridge {
    // Per-dimension scale/zero replication, {K, N}: stored {bk, bn} scaling
    // replicated by {bk / g, bn} lands on the kernel's per-group-{g, 1}
    // scales — {g, 1} alone covers plain group duplication, an N factor
    // covers block-scaled storage. The converse — merging finer stored
    // groups into coarser ones — is requantization and is never expressed
    // by a bridge.
    int2 replicate_scales{1, 1};
    // Scale/zero target dtypes. kNull is an explicit "no conversion is
    // needed" result: Pack leaves that tensor's dtype unchanged and checks
    // that its actual normalized dtype is already the family's required
    // dtype. A non-kNull value names the target of a packing-time cast;
    // Pack reads the actual tensor dtype, which may differ from the declared
    // format's (e.g. AWQ normalizes qparams to FP16 while a BF16 query
    // declares BF16), and casts it to that target.
    DataType convert_scales{kNull};
    DataType convert_zeros{kNull};

    // False when the bridge is trivial: no replication, no conversion.
    explicit operator bool() const noexcept
    {
        return replicate_scales.x != 1 || replicate_scales.y != 1
               || convert_scales != kNull || convert_zeros != kNull;
    }
};

// One kernel family's concrete object. The registration unit supplies the
// family-owned capability and packing callbacks.
class Family {
public:
    const std::uint32_t id;
    const int priority;

    const DataFormat& input_format() const noexcept { return input_format_; }

    // The base, unfused shape contract of a TP-local GEMM, per axis.
    // align_k/align_n are divisibility requirements; min_k/min_n are
    // minimum extents. The loader zero-pads to them — but only an axis padded at
    // both producer and consumer may grow (the FFN intermediate); every
    // other axis must natively meet them or the load fails (see the
    // principles). Fusion-specific gate_up_block requirements are
    // deliberately excluded: optional fusion must not increase the model's
    // intermediate width.
    int align_k() const noexcept { return align_k_; }
    int align_n() const noexcept { return align_n_; }
    int min_k() const noexcept { return min_k_; }
    int min_n() const noexcept { return min_n_; }

    // Execution-mode capabilities of the family's kernels, as family
    // constants: blocked grouped execution (B from the expert pointer
    // tables, per-group A extents) and indexed (routed-row) A input —
    // e.g. the SM90 FP8 families gather A/U in-kernel; most don't.
    bool grouped() const noexcept { return grouped_; }
    bool indexed_input() const noexcept { return indexed_input_; }

    // The component data type this family requires. A floating-point input
    // supplies the answer; for a quantized-input family, the floating-point
    // unfused output supplies it. A family must expose at least one of those
    // two floating-point formats.
    DataType data_type() const;

    Epilogue supported_epilogues() const;

    // Actual output format for the given epilogue. The base has only the
    // unfused output; requesting an epilogue outside supported_epilogues()
    // is an error.
    DataFormat output_format(Epilogue epilogue) const;

    // Joint capability query: does this family pack `weight_format`, honor
    // the data-type and requested-output contracts, and — when `grouped` is
    // set — execute blocked grouped. On
    // success the answer is the `WeightBridge` closing the gap between the
    // stored format and the family's kernel-facing layout; on failure,
    // std::nullopt. The base implementation checks the shared policy — the
    // knob, the execution mode, and the data-type contract — and answers
    // with a trivial bridge.
    //
    // requires_packing_ is the family constant: whether its persistent
    // layout differs from load-storage. Because the knob is honored here,
    // PlanWeight selection and the data-type enumeration see the same
    // reduced registry. The execution mode is not part of enumeration:
    // dense and expert weights share the component's dtype resolution, so
    // `DataTypes` passes grouped=false and a mode-incapable family fails
    // only the per-weight MoE query.
    //
    // The common method checks shared policy, then calls the family-owned
    // callback to verify the weight-side constraints and fill the bridge:
    //
    // - weight dtype and encoding must be exactly what `Pack` implements;
    // - the stored quant group must be a multiple of the kernel group —
    //   the bridge records the K replication factor;
    // - block-scaled storage is accepted when each block scale can be
    //   replicated to the kernel's group scaling — the bridge records both
    //   replication factors;
    // - scale/zero dtypes are accepted when convertible to the kernel's
    //   dtype (among float/BF16/FP16) — the bridge records a target when the
    //   normalized tensor needs a conversion and leaves it kNull when no
    //   conversion is needed;
    // - format-internal alignment (block sizes, scale tiling) must be
    //   compatible with the kernel's layout.
    //
    // Anything else — a different weight dtype, storage groups finer than
    // the kernel's, an encoding `Pack` does not implement — is std::nullopt.
    // The bridge is the complete recipe: `Pack` performs exactly what it
    // says, no more.
    //
    // Answering is an obligation: a family that accepts a format must serve
    // it at every shape, because selection is shape-agnostic. The family
    // states what its kernels need through `align_k`/`align_n` and
    // `min_k`/`min_n`; the loader pads the FFN intermediate axis to them and
    // hard-checks every other axis. A shape that cannot meet the contract is
    // a load error, never a fallback to another data type.
    std::optional<WeightBridge>
    supports(const DataFormat& weight_format, DataType data_type,
             DataType output_dtype, bool grouped) const;

    // Pack `linear`'s weight tensors into the family's persistent layout:
    // applies `bridge` (scale/zero replication, scale/zero dtype
    // conversion), replaces weight/scales/zeros/global-scale, writes
    // k_desc/q_desc, and updates linear.weight_format from the queried
    // load-storage format to the packed kernel format. `bridge` must come
    // from this family's
    // `supports` answer for `linear.weight_format`. Any failure is fatal
    // (TM_CHECK): there is no partial-failure contract to honor.
    void Pack(LinearWeight& linear, const WeightBridge& bridge,
              cudaStream_t stream) const;

    // Runtime input conversion: produce the kernel input (A, U) from the
    // incoming activation. The base implementation covers floating-point-input
    // families: the incoming dtype must equal input_format().dtype and the
    // tensor and scales pass through. For a quantized input, it quantizes
    // data-type input with QuantizeSymm and passes an already-quantized
    // input with its scales through. No padding: the
    // activation's K already equals the weight's packed K — the loader
    // pads the FFN intermediate at both ends and rejects every other
    // misalignment, so dims and descriptor extents agree by construction.
    void ConvertInput(Tensor& A, Tensor& U, const LinearWeight& weight,
                      const Tensor& input, const Tensor& input_scales,
                      cudaStream_t stream) const;

    // MLP fusion query: the w1/w3 arrangement for a gate/up pair of the
    // given per-projection width under the given activation. The base
    // implementation uses the object's gate_up_block and fused output.
    int gate_up(ActivationType, int, Epilogue& epilogue) const;

    Family(std::uint32_t id, int priority,
           DataFormat input_format, DataFormat output_format,
           int align_k, int align_n, int min_k, int min_n,
           bool requires_packing, bool grouped, bool indexed_input,
           std::optional<WeightBridge> (*supports)(const DataFormat&, bool),
           void (*pack)(LinearWeight&, const WeightBridge&, cudaStream_t),
           int gate_up_block = 0,
           DataFormat fused_output = {});

private:
    DataFormat input_format_;
    DataFormat output_format_;
    int        align_k_{}, align_n_{};
    int        min_k_{}, min_n_{};
    bool       requires_packing_{};
    bool       grouped_{};
    bool       indexed_input_{};
    std::optional<WeightBridge> (*supports_)(const DataFormat&, bool){};
    void (*pack_)(LinearWeight&, const WeightBridge&, cudaStream_t){};
    int        gate_up_block_{};
    DataFormat fused_output_{};
};

}  // namespace turbomind::gemm
```

Add `src/turbomind/kernels/gemm/plan.h`; `plan.cc` contains only the
non-inline `GemmPlan` method definitions:

```cpp
#pragma once

#include "src/turbomind/kernels/gemm/family.h"

namespace turbomind {
class LinearWeight;
}

namespace turbomind::gemm {

struct WeightQuery {
    DataFormat weight_format;  // logical (load-storage) format
    DataType data_type{};      // resolved component dtype; always set
    DataType input_dtype{kNull};  // preferred operand-A dtype; kNull = unset
    DataType output_dtype{kNull}; // required output; kNull = data_type
    bool grouped{};  // MoE expert weight: requires blocked grouped execution
};

class GemmPlan {
public:
    const Family& family() const noexcept { return *family_; }

    // Derive the w1/w3 arrangement for a gate/up pair of the given TP-local
    // padded per-projection width. Takes the family's offer, validates that
    // the fused arrangement tiles the width, and degrades to the unfused
    // answer on failure.
    int gate_up(ActivationType act_type, int projection_n);

    void pack(LinearWeight& linear, cudaStream_t stream) const;

private:
    friend class Gemm;

    // The family pointer names a persistent object that outlives the plan.
    const Family* family_{};
    // The selected family's supports() answer for the queried format;
    // applied by pack().
    WeightBridge bridge_{};
    Epilogue epilogue_{Epilogue::kNone};
    DataFormat output_format_{};
};

}  // namespace turbomind::gemm
```

Extend the existing `Gemm` interface with the query entry point:

```cpp
class Gemm {
public:
    Gemm();
    ~Gemm();

    // The query entry point. Failure is a value: std::nullopt when no
    // registered family can serve the query. Nothing in the GEMM query path
    // throws.
    std::optional<GemmPlan> PlanWeight(const WeightQuery& query) const;

    // The data types supported for a non-grouped query of this weight
    // format on this device: the declared data type of every registered
    // family that accepts the format with grouped=false. This is not a
    // component-wide executability guarantee. The input-type preference
    // does not participate — it reorders, never rejects, so the returned
    // set is independent of it. Empty = unsupported.
    std::vector<DataType> DataTypes(const DataFormat& weight_format) const;

    // ... Run(...) unchanged ...
};
```

Each weight query produces one `GemmPlan` through the component's
authoritative first-device `Gemm`; that plan is attached to every device
instance of that linear. Different linears issue their own queries and may
receive different plans. A plan is never serialized, so it carries no
version or registry fingerprint — just the selected family (process-static,
outlives the plan), the `WeightBridge` the family answered with, and the
selected epilogue and output format. The plan is not device-bound.

## Kernel family registration

`KernelDesc` remains the runtime launch descriptor. Each registration unit
defines its concrete `Family` object beside the kernels that belong to
it. Every tile variant receives a reference to that object through the base
constructor:

```cpp
class Kernel {
public:
    const Family& family() const noexcept { return family_; }

protected:
    explicit Kernel(const Family& family)
        : family_{family}
    {
    }

    const Family& family_;  // the family object outlives all kernels
};
```

Related family objects share function templates. Unique callbacks are named
functions beside the family; callback bodies are never embedded in the object
definition. For example, the two SM80 floating-point families share their support
and pack functions:

```cpp
template<DataType Dtype>
std::optional<WeightBridge> supports_fp(const DataFormat& format, bool)
{
    return format == DataFormat{Dtype} ? std::optional{WeightBridge{}} :
                                         std::nullopt;
}

template<class Arch, Order WeightOrder, uint32_t WeightPack, DataType Dtype>
void pack_fp(LinearWeight& linear, const WeightBridge& bridge,
                         cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    PackWeight(linear,
               GetImpl<Arch, WeightOrder, WeightPack,
                       uint16_t, uint16_t>(),
               stream);
    linear.q_desc        = {};
    linear.weight_format = Dtype;
}

constexpr auto f16_packer =
    pack_fp<Arch<80>, kRowMajor,
                        HMMA_16816 | OPERAND_B | 1, kHalf>;
constexpr auto bf16_packer =
    pack_fp<Arch<80>, kRowMajor,
                        HMMA_16816 | OPERAND_B | 1, kBfloat16>;

const Family f16{13, 190, kHalf, kHalf,
                 32, 8, 1, 1, true, true, true,
                 supports_fp<kHalf>,
                 f16_packer};
const Family bf16{14, 200, kBfloat16, kBfloat16,
                  32, 8, 1, 1, true, true, true,
                  supports_fp<kBfloat16>,
                  bf16_packer};
```

Each `Registrar` binds one family. `Registry` constructs that callback's
`Collector` with the family, so every existing `Collector::add<Config>()`
inside the callback uses the same family without repeating it and without
adding a carrier to the tile configuration. The complete `Collector` change
is:

```cpp
#include <type_traits>

class Collector {
public:
    explicit Collector(const Family& family)
        : family_{family}
    {
    }

    template<class T>
    void add()
    {
        if constexpr (std::is_base_of_v<Kernel, T>) {
            kernels_.emplace_back(std::make_unique<T>(family_));
        }
        else {
            kernels_.emplace_back(
                std::make_unique<KernelImpl<typename T::Kernel>>(
                    family_));
        }
    }

    std::vector<std::unique_ptr<Kernel>> release()
    {
        return std::move(kernels_);
    }

private:
    const Family& family_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
};
```

`Registrar` stores the family with the existing callback, and registry
construction passes it to `Collector`:

```cpp
inline std::vector<std::pair<const Family*, RegisterFn>>&
gKernelFactories()
{
    static std::vector<std::pair<const Family*, RegisterFn>> v;
    return v;
}

struct Registrar {
    Registrar(const Family& family, RegisterFn fn)
    {
        gKernelFactories().emplace_back(&family, std::move(fn));
    }
};
```

```cpp
for (auto& [family, register_fn] : gKernelFactories()) {
    Collector collector{*family};
    register_fn(collector, arch_);
    for (auto& kernel : collector.release()) {
        Add(std::move(kernel));
    }
}
```

The SM90 GMMA families (BF16, FP8 W8A8, mixed u4/e4m3/mxfp4/nvfp4, MXFP4×FP8
folded/unfolded) construct `KernelImplSm90*` in local `add` functions today.
Those functions keep their current template arguments and pass the concrete
kernel type to the same family-bound `Collector::add<T>()`; the four
`KernelImplSm90*` constructors gain the family argument. For example:

```cpp
template<class Gemm>
void add(Collector& c)
{
    c.add<KernelImplSm90MxFp4Fp8<Gemm>>();
}
```

The two cuBLAS implementations keep their direct `Registrar` path. Their
constructors gain the same family argument: the existing `CublasKernel`
initializer list becomes
`Kernel{family}, cublas_{}`, and the existing `CublasGroupedKernel` initializer
list becomes the same. Their constructor
bodies are otherwise unchanged. Six registrar entries bind the six local
family objects; the existing identifier `reg` becomes the array containing
those entries:

```cpp
Registrar reg[]{
    {dense_f16,
     [](Collector& c, int) {
         c.add<CublasKernel>();
     }},
    {dense_bf16,
     [](Collector& c, int arch) {
         if (arch >= Sm80::value) {
             c.add<CublasKernel>();
         }
     }},
    {f16_f32,
     [](Collector& c, int) {
         c.add<CublasKernel>();
     }},
    {bf16_f32,
     [](Collector& c, int arch) {
         if (arch >= Sm80::value) {
             c.add<CublasKernel>();
         }
     }},
#if defined(ENABLE_CUBLAS_GROUPED)
    {grouped_f16,
     [](Collector& c, int arch) {
         if (Sm100::is_compatible(arch)) {
             c.add<CublasGroupedKernel>();
         }
     }},
    {grouped_bf16,
     [](Collector& c, int arch) {
         if (Sm100::is_compatible(arch)) {
             c.add<CublasGroupedKernel>();
         }
     }},
#endif
};
```

Families 104 and 105 are the dense source-preserving FP16-input/FP32-output
and BF16-input/FP32-output cuBLAS families. The BF16 specializations are absent
before SM80, matching the existing compute-capability-8+ BF16 contract. FP32
is not added to `Gemm::DataTypes`: `data_type()` comes from the floating-point
input for these families, and an ordinary `output_dtype == kNull` query resolves
to the component data type and therefore does not select their FP32 output.

`Registry` stores the distinct family pointers explicitly. The public
accessor and private storage added to the existing `Registry` are:

```cpp
[[nodiscard]] const std::vector<const Family*>& families() const
{
    return families_;
}

std::vector<const Family*> families_;
```

Inside `Registry::Add`, after the existing architecture and shared-memory
filters accept the kernel and before the original/transposed kernels are
stored, collect and validate its family directly:

```cpp
const Family* family = &kernel->family();
if (std::find(families_.begin(), families_.end(), family) == families_.end()) {
    TM_CHECK(family->id != 0);
    for (const Family* other : families_) {
        TM_CHECK(other->id != family->id);
    }
    families_.push_back(family);
}
```

Registry construction then collects the distinct `&family_` instances from
the kernels it already holds — one per kernel family; a family with no
kernels on this device is simply absent from the device's family list. The
registry stores each kernel twice (original plus a `TransposedKernel`
wrapper), so `TransposedKernel` forwards `family_` from the kernel it wraps
and the collection dedupes.
There is no separate family registration call and no global family registry.
Every family object supplies a fixed nonzero literal `id`. Registry
construction rejects `id == 0` and rejects the same `id` on different family
objects. An assigned id is permanent because it is part of the persistent
dispatch-cache key.
Family metadata consistency — one priority, one output format, one fusion
behavior, and one acceptance rule per family — holds by construction because
the family is a single object. Two families answering the same query is not an
error; priority decides.

A `gate_up` override's fused output format must be a well-formed kernel
input format — a floating-point data-type format or dynamic FP8 `{128,1}`
with FP32 scales. The kernel combination ensures that the fused output format
is consumable by the down projection; `GemmPlan::gate_up` does not query the
down plan.

`priority` is explicit family-selection policy, replacing scattered Python
and `FfnWeight::prepare()` architecture checks. Higher values win among the
families that pass selection. The data-type preference is part of the value:
within each tier the BF16 family takes the tier value and the FP16 sibling
ten less. Equal priorities across the registry are not an error — whole
tiers share a value — but a tie for the selected position after applying
the input-type preference is reported as a query failure. Registration
order never breaks a tie.
The initial priorities preserve the currently selected native
representation:

- W8A8 and W4A8 folded: 300
- native SM90 representations: 250
- kernel-specific transformed formats: 200 BF16 / 190 FP16
- group-duplicating variants of the above: 150 BF16 / 140 FP16
- source-preserving fallbacks: 100 BF16 / 90 FP16

When legacy U4 dense and grouped layouts otherwise overlap at the same tier,
the grouped-layout sibling is one point lower. Dense queries therefore select
the dense-specialized layout; grouped queries exclude that dense-only object
and select the grouped layout.

### Complete family catalog

The following IDs are the complete initial assignment. They are permanent
dispatch-cache keys and must not be renumbered when another family is added.
Registration paths are relative to `src/turbomind/kernels/gemm/kernel/`.
`grouped/indexed` records the two published booleans. Every SM70--SM90
grouped kernel family remains eligible for dense problems and supports
indexed input. The two grouped cuBLAS families are grouped-only and require
the explicit gather path.
Alignment/minimum tuples are
`(align_k, align_n; min_k, min_n)`. Fusion entries give
`activation / gate_up_block / fused output format`.

#### SM70

| ID  | Participating registration                                         | Source; data/input/output                                                                                                                                                                                                                                   | Group; priority; alignment/minimum; grouped/indexed | Packing operation; fusion                                                                                                                                                          |
| --- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1` | `sm70_884_16.cu`, `Config_F16`                                     | `DataFormat{kHalf}`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`                                                                                                                                                                                        | none; 190; `(16,8;1,1)`; `true/true`                | required: existing s884 `B                                                                                                                                                         | 1` FP16 layout conversion; none |
| `2` | `sm70_884_8.cu`, `Config_E4M3`                                     | E4M3 values, block sizes `{128,1}` or `{128,128}`, trivial-float scales, and no zeros; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`                                                                                                                      | 128; 190; `(128,8;1,1)`; `true/true`                | required: replicate blockwise scales along N when needed, convert them to FP16, then apply the existing s884 row-major `B                                                          | 1`weight and column-major`V     | 1` qparam conversions; none |
| `3` | the G32 `Config_U4_d` and `Config_U4_g` blocks in `sm70_884_4.cu`  | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 32 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`  | 32; 140; `(32,8;1,1)`; `true/true`                  | required: replicate scales/zeros by `weight_format.block_sizes[0] / 32`, keep normalized FP16 qparams (`convert_scales = convert_zeros = kNull`), then existing s884 row-major `B  | 1`weight and column-major`V     | 1` qparam conversions; none |
| `4` | the G128 `Config_U4_d` and `Config_U4_g` blocks in `sm70_884_4.cu` | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 128 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}` | 128; 190; `(128,8;1,1)`; `true/true`                | required: replicate scales/zeros by `weight_format.block_sizes[0] / 128`, keep normalized FP16 qparams (`convert_scales = convert_zeros = kNull`), then the same s884 row-major `B | 1`weight and column-major`V     | 1` qparam conversions; none |
| `5` | the `Config_MXF4` block in `sm70_884_4.cu`                         | E2M1 values, exactly `{32,1}`, UE8M0 scales, and no zeros; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`                                                                                                                                                  | 32; 190; `(32,8;1,1)`; `true/true`                  | required: existing s884 row-major `B                                                                                                                                               | 1`weight and column-major`V     | 1` UE8M0 conversions; none  |

#### SM75

| ID   | Participating registration                        | Source; data/input/output                                                                                                                                                                                                                                   | Group; priority; alignment/minimum; grouped/indexed | Packing operation; fusion                                                                                                            |
| ---- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `6`  | `sm75_16816_16.cu`, `Config_F16`                  | `DataFormat{kHalf}`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`                                                                                                                                                                                        | none; 190; `(32,8;1,1)`; `true/true`                | required: existing s16816 row-major `B                                                                                               | 1` FP16 conversion; none    |
| `7`  | `sm75_16816_8.cu`, `Config_E4M3`                  | E4M3 values, block sizes `{128,1}` or `{128,128}`, trivial-float scales, and no zeros; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`                                                                                                                      | 128; 190; `(128,8;1,1)`; `true/true`                | required: replicate blockwise scales along N when needed, convert them to FP16, then apply the existing s16816 column-major `A       | 1`weight and`U              | 1` qparam conversions; none |
| `8`  | the G32 `Config_U4_d` block in `sm75_16816_4.cu`  | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 32 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`  | 32; 140; `(32,8;1,1)`; `false/false`                | required: replicate by `weight_format.block_sizes[0] / 32`, keep normalized FP16 qparams, then existing dense column-major s16816 `B | 2`weight and column-major`V | 1` qparam conversions; none |
| `9`  | the G32 `Config_U4_g` block in `sm75_16816_4.cu`  | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 32 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`  | 32; 139; `(32,8;1,1)`; `true/true`                  | required: replicate by `weight_format.block_sizes[0] / 32`, keep normalized FP16 qparams, then grouped row-major s16816 `B           | 2`weight and column-major`V | 1` qparam conversions; none |
| `10` | the G128 `Config_U4_d` block in `sm75_16816_4.cu` | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 128 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}` | 128; 190; `(128,8;1,1)`; `false/false`              | required: replicate by `weight_format.block_sizes[0] / 128`, keep normalized FP16 qparams, then dense column-major s16816 `B         | 2`weight and column-major`V | 1` qparam conversions; none |
| `11` | the G128 `Config_U4_g` block in `sm75_16816_4.cu` | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 128 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}` | 128; 189; `(128,8;1,1)`; `true/true`                | required: replicate by `weight_format.block_sizes[0] / 128`, keep normalized FP16 qparams, then grouped row-major s16816 `B          | 2`weight and column-major`V | 1` qparam conversions; none |
| `12` | the `Config_MXF4` block in `sm75_16816_4.cu`      | E2M1 values, exactly `{32,1}`, UE8M0 scales, and no zeros; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`                                                                                                                                                  | 32; 190; `(32,8;1,1)`; `true/true`                  | required: existing s16816 column-major `A                                                                                            | 1`weight and`U              | 1` UE8M0 conversions; none  |

#### SM80

| ID   | Participating registration                                   | Source; data/input/output                                                                                                                                                                                                                                   | Group; priority; alignment/minimum; grouped/indexed | Packing operation; fusion                                                                                                      |
| ---- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `13` | the `half` `Config_F16_g` block in `sm80_16816_16.cu`        | `DataFormat{kHalf}`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`                                                                                                                                                                                        | none; 190; `(32,8;1,1)`; `true/true`                | required: existing s16816 row-major `B                                                                                         | 1` FP16 conversion; none    |
| `14` | the `nv_bfloat16` `Config_F16_g` block in `sm80_16816_16.cu` | `DataFormat{kBfloat16}`; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}`                                                                                                                                                                        | none; 200; `(32,8;1,1)`; `true/true`                | required: existing s16816 row-major `B                                                                                         | 1` BF16 conversion; none    |
| `15` | `sm80_16816_8.cu`, both `Config_E4M3` N variants             | E4M3 values, block sizes `{128,1}` or `{128,128}`, trivial-float scales, and no zeros; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}`                                                                                                          | 128; 200; `(128,8;1,1)`; `true/true`                | required: replicate blockwise scales along N when needed, convert them to BF16, then apply the existing s16816 column-major `A | 1`weight and`U              | 1` qparam conversions; none |
| `16` | the G32 `Config_U4_d` block in `sm80_16816_4.cu`             | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 32 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`  | 32; 140; `(32,8;1,1)`; `false/false`                | required: replicate by `weight_format.block_sizes[0] / 32`, keep normalized FP16 qparams, then dense column-major s16816 `B    | 2`weight and column-major`V | 1` qparam conversions; none |
| `17` | the G32 `Config_U4_g` block in `sm80_16816_4.cu`             | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 32 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`  | 32; 139; `(32,8;1,1)`; `true/true`                  | required: replicate by `weight_format.block_sizes[0] / 32`, keep normalized FP16 qparams, then grouped row-major s16816 `B     | 2`weight and column-major`V | 1` qparam conversions; none |
| `18` | the G128 `Config_U4_d` block in `sm80_16816_4.cu`            | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 128 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}` | 128; 190; `(128,8;1,1)`; `false/false`              | required: replicate by `weight_format.block_sizes[0] / 128`, keep normalized FP16 qparams, then dense column-major s16816 `B   | 2`weight and column-major`V | 1` qparam conversions; none |
| `19` | the G128 `Config_U4_g` block in `sm80_16816_4.cu`            | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 128 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}` | 128; 189; `(128,8;1,1)`; `true/true`                | required: replicate by `weight_format.block_sizes[0] / 128`, keep normalized FP16 qparams, then grouped row-major s16816 `B    | 2`weight and column-major`V | 1` qparam conversions; none |
| `20` | the active `Config_MXF4` N16/N8 blocks in `sm80_16816_4.cu`  | E2M1 values, exactly `{32,1}`, UE8M0 scales, and no zeros; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}`                                                                                                                                      | 32; 200; `(32,8;1,1)`; `true/true`                  | required: existing s16816 column-major `A                                                                                      | 1`weight and`U              | 1` UE8M0 conversions; none  |

#### SM90

| ID   | Participating registration                                                                                                                    | Source; data/input/output                                                                                                                                                                                                                                               | Group; priority; alignment/minimum; grouped/indexed             | Packing operation; fusion                                                                                                                                                                                                        |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `21` | `sm90_16816_8.cu`, both `Config_E4M3` N variants                                                                                              | E4M3 values, block sizes `{128,1}` or `{128,128}`, trivial-float scales, and no zeros; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}`                                                                                                                      | 128; 200; `(128,8;1,1)`; `true/true`                            | required: replicate blockwise scales along N when needed, convert them to BF16, then apply the legacy s16816 column-major `A                                                                                                     | 1`weight and`U              | 1` qparam conversions; none |
| `22` | the G32 `Config_U4_d` block in `sm90_16816_4.cu`                                                                                              | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 32 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`              | 32; 140; `(32,8;1,1)`; `false/false`                            | required: replicate by `weight_format.block_sizes[0] / 32`, keep normalized FP16 qparams, then legacy dense column-major s16816 `B                                                                                               | 2`weight and column-major`V | 1` qparam conversions; none |
| `23` | the G32 `Config_U4_g` block in `sm90_16816_4.cu`                                                                                              | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 32 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`              | 32; 139; `(32,8;1,1)`; `true/true`                              | required: replicate by `weight_format.block_sizes[0] / 32`, keep normalized FP16 qparams, then legacy grouped row-major s16816 `B                                                                                                | 2`weight and column-major`V | 1` qparam conversions; none |
| `24` | the G128 `Config_U4_d` block in `sm90_16816_4.cu`                                                                                             | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 128 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`             | 128; 190; `(128,8;1,1)`; `false/false`                          | required: replicate by `weight_format.block_sizes[0] / 128`, keep normalized FP16 qparams, then legacy dense column-major s16816 `B                                                                                              | 2`weight and column-major`V | 1` qparam conversions; none |
| `25` | the G128 `Config_U4_g` block in `sm90_16816_4.cu`                                                                                             | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 128 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`             | 128; 189; `(128,8;1,1)`; `true/true`                            | required: replicate by `weight_format.block_sizes[0] / 128`, keep normalized FP16 qparams, then legacy grouped row-major s16816 `B                                                                                               | 2`weight and column-major`V | 1` qparam conversions; none |
| `26` | the active `Config_MXF4` N16/N8 blocks in `sm90_16816_4.cu`                                                                                   | E2M1 values, exactly `{32,1}`, UE8M0 scales, and no zeros; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}`                                                                                                                                                  | 32; 200; `(32,8;1,1)`; `true/true`                              | required: legacy s16816 column-major `A                                                                                                                                                                                          | 1`weight and`U              | 1` UE8M0 conversions; none  |
| `27` | `sm90_64n16_16.cu`, all flat/blocked/indexed registrations                                                                                    | `DataFormat{kBfloat16}`; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}`                                                                                                                                                                                    | none; 250; `(1,1;1,1)`; `true/true`                             | required: transpose source `(K,N)` into the existing physical `(N,K)` representation and publish its column-major `(K,N)` descriptor; SiLU / 64 / `DataFormat{kBfloat16}`                                                        |
| `28` | `sm90_64n32_8.cu`, both v3 and weight-as-A flat/blocked/indexed registrations                                                                 | E4M3 values, block sizes `{128,128}`, trivial-float scales, and no zeros; `kBfloat16 / DataFormat{kFloat8_e4m3,{128,1},kFloat} / DataFormat{kBfloat16}`                                                                                                                 | weight 128, activation 128; 300; `(1,1;1,1)`; `true/true`       | required: convert scales to FP32, transpose the weight and converted scales, and publish column-major descriptors; `ConvertInput` performs or passes through dynamic FP8; SiLU / 128 / `DataFormat{kFloat8_e4m3,{128,1},kFloat}` |
| `29` | all active registrations in `sm90_64n16_4.cu`, `_flat_col.cu`, `_indexed_col.cu`, `_indexed_row.cu`, `_blocked_col.cu`, and `_blocked_row.cu` | `weight_format.dtype == kUint4`, `weight_format.block_sizes.size() == 2`, `weight_format.block_sizes[0] % 128 == 0`, `weight_format.block_sizes[1] == 1`, and both qparam dtypes pass `IsTrivialFloatType`; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}` | 128; 250; `(128,128;128,1)`; `true/true`                        | required: replicate scales/zeros by `weight_format.block_sizes[0] / 128`, set both bridge conversion targets to `kBfloat16`, then `PackSm90U4Weight` and `PackSm90U4QParams`; SiLU / 64 / `DataFormat{kBfloat16}`                |
| `30` | all active flat/blocked/indexed registrations in `sm90_64n16_mxfp4.cu`                                                                        | E2M1 values, exactly `{32,1}`, UE8M0 scales, and no zeros; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}`                                                                                                                                                  | 32; 250; `(64,128;128,1)`; `true/true`                          | required: `PackSm90Fp4PrmtWeight` and `PackSm90MxFp4QParams`; SiLU / 64 / `DataFormat{kBfloat16}`                                                                                                                                |
| `31` | all active flat/blocked/indexed registrations in `sm90_64n16_nvfp4.cu`                                                                        | `DataFormat{kFloat4_e2m1,{16,1},kFloat8_e4m3}` plus FP32 global scale; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}`                                                                                                                                      | 16; 250; `(64,128;128,1)`; `true/true`                          | required: `PackSm90Fp4PrmtWeight`, `PackSm90Fp4QParams`, and preserved global scale; SiLU / 64 / `DataFormat{kBfloat16}`                                                                                                         |
| `32` | all active flat/blocked/indexed registrations in `sm90_64n16_e4m3.cu`                                                                         | E4M3 values, block sizes `{128,128}`, trivial-float scales, and no zeros; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}`                                                                                                                                   | 128; 250; `(128,128;128,1)`; `true/true`                        | required: convert scales to FP32, then `PackSm90Fp8E4M3Weight` and `PackSm90Fp8E4M3Scales`; SiLU / 64 / `DataFormat{kBfloat16}`                                                                                                  |
| `33` | all active flat/blocked/indexed registrations in `sm90_64n32_mxfp4_fp8_folded.cu`                                                             | E2M1 values, exactly `{32,1}`, UE8M0 scales, and no zeros; `kBfloat16 / DataFormat{kFloat8_e4m3,{128,1},kFloat} / DataFormat{kBfloat16}`                                                                                                                                | weight 32, activation 128; 300; `(128,64;256,1)`; `true/true`   | required: `PackSm90MxFp4Fp8FoldedWeight` and `PackSm90MxFp4Fp8FoldedQParams`; `ConvertInput` performs or passes through dynamic FP8; SiLU / 128 / `DataFormat{kFloat8_e4m3,{128,1},kFloat}`                                      |
| `34` | `sm90_64n32_mxfp4_fp8_unfolded.cu`                                                                                                            | E2M1 values, exactly `{32,1}`, UE8M0 scales, and no zeros; `kBfloat16 / DataFormat{kFloat8_e4m3,{128,1},kFloat} / DataFormat{kBfloat16}`                                                                                                                                | weight 32, activation 128; 250; `(128,64;256,1)`; `false/false` | required: `PackSm90MxFp4Fp8UnfoldedWeight` and `PackSm90MxFp4Fp8UnfoldedQParams`; `ConvertInput` performs or passes through dynamic FP8; none                                                                                    |

#### cuBLAS

| ID    | Participating registration                                                  | Source; data/input/output                                                            | Group; priority; alignment/minimum; grouped/indexed | Packing operation; fusion                      |
| ----- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------- | ---------------------------------------------- |
| `100` | dense `CublasKernel`                                                        | `DataFormat{kHalf}`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`                 | none; 90; `(1,1;1,1)`; `false/false`                | source-preserving descriptor publication; none |
| `101` | dense `CublasKernel`, registered only for SM80+                             | `DataFormat{kBfloat16}`; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}` | none; 100; `(1,1;1,1)`; `false/false`               | source-preserving descriptor publication; none |
| `102` | `CublasGroupedKernel` on compatible SM100 builds                            | `DataFormat{kHalf}`; `kHalf / DataFormat{kHalf} / DataFormat{kHalf}`                 | none; 90; `(1,1;1,1)`; `true/false`, grouped only   | source-preserving descriptor publication; none |
| `103` | `CublasGroupedKernel` on compatible SM100 builds                            | `DataFormat{kBfloat16}`; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kBfloat16}` | none; 100; `(1,1;1,1)`; `true/false`, grouped only  | source-preserving descriptor publication; none |
| `104` | dense `CublasKernel` for an explicit FP32 output                            | `DataFormat{kHalf}`; `kHalf / DataFormat{kHalf} / DataFormat{kFloat}`                | none; 90; `(1,1;1,1)`; `false/false`                | source-preserving descriptor publication; none |
| `105` | dense `CublasKernel` for an explicit FP32 output, registered only for SM80+ | `DataFormat{kBfloat16}`; `kBfloat16 / DataFormat{kBfloat16} / DataFormat{kFloat}`    | none; 100; `(1,1;1,1)`; `false/false`               | source-preserving descriptor publication; none |

Within a Config family split into per-kernel-group families, the exact-group
family takes the higher tier and the duplicating family the lower one, so an
exactly matching stored group never pays duplication.

The one-argument `DataFormat` constructor accepts any `DataType`, sets block
sizes to `{1, 1}`, and leaves scales and zeros absent. Dynamic FP8 is expressed
directly with the complete constructor; there are no activation-format helper
functions:

```cpp
inline bool operator==(const DataFormat& a, const DataFormat& b) noexcept
{
    return a.dtype == b.dtype && a.block_sizes == b.block_sizes
           && a.scales.dtype == b.scales.dtype
           && a.zeros.dtype == b.zeros.dtype;
}

inline bool operator!=(const DataFormat& a, const DataFormat& b) noexcept
{
    return !(a == b);
}
```

The cuBLAS families are source-preserving, floating-point family objects. The
dense objects reject grouped queries through `grouped_ = false`; the grouped
objects additionally reject non-grouped queries in their shared support
template. Their complete definitions are:

```cpp
template<DataType Dtype, bool Grouped>
std::optional<WeightBridge>
supports(const DataFormat& format, bool grouped)
{
    if (Grouped && !grouped) {
        return std::nullopt;
    }
    return format == DataFormat{Dtype} ?
               std::optional{WeightBridge{}} : std::nullopt;
}

template<DataType Dtype, bool Grouped>
void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t)
{
    TM_CHECK(!bridge);
    TM_CHECK_EQ(linear.weight.dtype(), Dtype);
    linear.k_desc = MatrixLayout{Dtype,
                                 kRowMajor,
                                 linear.input_dim,
                                 linear.output_dim,
                                 linear.output_dim,
                                 0,
                                 0,
                                 nullptr,
                                 nullptr};
    linear.q_desc = {};
}

const Family dense_f16{100, 90, kHalf, kHalf,
                       1, 1, 1, 1, false, false, false,
                       supports<kHalf, false>, pack<kHalf, false>};
const Family dense_bf16{101, 100, kBfloat16, kBfloat16,
                        1, 1, 1, 1, false, false, false,
                        supports<kBfloat16, false>, pack<kBfloat16, false>};
const Family grouped_f16{102, 90, kHalf, kHalf,
                         1, 1, 1, 1, false, true, false,
                         supports<kHalf, true>, pack<kHalf, true>};
const Family grouped_bf16{103, 100, kBfloat16, kBfloat16,
                          1, 1, 1, 1, false, true, false,
                          supports<kBfloat16, true>, pack<kBfloat16, true>};
const Family f16_f32{104, 90, kHalf, kFloat,
                     1, 1, 1, 1, false, false, false,
                     supports<kHalf, false>, pack<kHalf, false>};
const Family bf16_f32{105, 100, kBfloat16, kFloat,
                      1, 1, 1, 1, false, false, false,
                      supports<kBfloat16, false>, pack<kBfloat16, false>};
```

The BF16 dense object is registered only for SM80 and later. The grouped
objects are registered only when the compatible SM100 grouped-cuBLAS path is
built.
The reusable U4 metadata check and scale-replication bridge live once in
`kernel/u4.h`:

```cpp
template<int GroupSize>
std::optional<WeightBridge>
supports_u4(const DataFormat& format, bool)
{
    if (format.dtype != kUint4) {
        return std::nullopt;
    }
    if (format.block_sizes.size() != 2) {
        return std::nullopt;
    }
    if (format.block_sizes[1] != 1) {
        return std::nullopt;
    }
    if (format.block_sizes[0] % GroupSize != 0) {
        return std::nullopt;
    }
    if (!IsTrivialFloatType(format.scales.dtype)) {
        return std::nullopt;
    }
    if (!IsTrivialFloatType(format.zeros.dtype)) {
        return std::nullopt;
    }
    WeightBridge bridge;
    bridge.replicate_scales.x = format.block_sizes[0] / GroupSize;
    return bridge;
}
```

Each family translation unit keeps its own packer because the packed layout
and pack implementations belong to that family. For example, the SM80
U4/FP16 family uses this local packing template for dense/grouped layouts and
group sizes 32/128:

```cpp
template<int GroupSize, bool Grouped>
void pack_u4(LinearWeight& linear, const WeightBridge& bridge,
             cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    if constexpr (Grouped) {
        PackWeight(linear,
                   GetImpl<Arch<80>, kRowMajor,
                           HMMA_16816 | OPERAND_B | 2,
                           uint16_t, uint4_t>(),
                   stream);
    }
    else {
        PackWeight(linear,
                   GetImpl<Arch<80>, kColMajor,
                           HMMA_16816 | OPERAND_B | 2,
                           uint16_t, uint4_t>(),
                   stream);
    }
    PackQParams(linear,
                GetImpl<Arch<80>, kColMajor,
                        HMMA_16816 | OPERAND_V | 1,
                        uint32_t, uint32_t>(),
                QuantDesc{QuantType::kK, GroupSize},
                stream);
    linear.weight_format =
        DataFormat{kUint4, {GroupSize, 1}, kHalf, kHalf};
}

const Family u4_d_32{16, 140, kHalf, kHalf, 32, 8, 1, 1,
                     true, false, false, supports_u4<32>, pack_u4<32, false>};
const Family u4_g_32{17, 139, kHalf, kHalf, 32, 8, 1, 1,
                     true, true, true, supports_u4<32>, pack_u4<32, true>};
const Family u4_d_128{18, 190, kHalf, kHalf, 128, 8, 1, 1,
                      true, false, false, supports_u4<128>, pack_u4<128, false>};
const Family u4_g_128{19, 189, kHalf, kHalf, 128, 8, 1, 1,
                      true, true, true, supports_u4<128>, pack_u4<128, true>};
```

The dense G128 object instantiates `GroupSize = 128` and publishes
`align_k = 128`, accepting stored groups divisible by 128 and replicating
scales/zeros down to 128. The grouped G32/G128 pair uses the grouped
row-major packed-B layout, sets `grouped_ = true`, and remains eligible for
dense problems. The dense pair uses the dense column-major packed-B layout
and rejects grouped queries through the base `grouped_ = false` contract.
For dense selection, each dense family is one priority point above its
grouped-layout sibling, so the specialized dense layout wins without an
ambiguous tie. For a stored-128 dense weight the exact-group dense family has
`priority = 190` (transformed, FP16), while the dense G32 family — which pays
scale/zero duplication for such a weight — at 140. A K that misses the
G128 granularity is padded by the loader where the axis grows at both
ends (the FFN intermediate); anywhere else K must meet the granularity
natively or the load fails — there is no shape-driven walk-down to G32.
For a
stored-32 or stored-64 weight only the G32 pair answers. Grouped selection
filters out the dense objects and follows the same exact-group priority rule
within the grouped pair. The MXFP4 and
NVFP4 families have a fixed group size; their support callbacks accept
exactly that group size, with no duplication case.

The SM90 W4A8 folded family adds the fused side: a fused
block width of 128 and a fused output value of
`DataFormat{kFloat8_e4m3, {128, 1}, kFloat}` for
`kGatedSilu`. The common `gate_up` operation returns `128` for SiLU and zero
for other activations. `GemmPlan::gate_up` performs the
`projection_n % 128` validation. The family has `priority = 300`.

## Packing

The gap between the stored weight and the kernel-facing layout is an
explicit value: `supports` verifies the constraints and answers with a
`WeightBridge` naming every re-expression — scale/zero replication (group
duplication or block-to-group), scale/zero dtype conversion — and `Pack`
performs exactly what the bridge says. There is no central format table:
verification, the bridge, and the conversion live next to each other in
the family, and neither dispatches on architecture or epilogue. Shape
feasibility is not the packer's concern: `is_feasible` owns it (see the
selection algorithm).

The scale/zero dtype cast is a packing operation, not a copy-path one: the
loader's copy preserves the normalized tensor dtypes (AWQ scales stay FP16
regardless of the declared format), and `Pack` casts an actual tensor only
when the corresponding bridge target is non-`kNull`. `kNull` positively
means that no conversion is needed; `Pack` leaves the tensor unchanged and
checks that its actual normalized dtype is already the family's required
dtype. The bridge is derived from the format's normalization contract, not
merely by comparing the target with the declared candidate dtype. Thus the
FP16 U4 family above leaves both fields `kNull`, whereas a native BF16 U4
family sets both targets to `kBfloat16` unconditionally because the actual
normalized AWQ/GPTQ qparams are FP16 even when the query declares BF16.
The same rule applies to FP8 checkpoints: `FP8Format.normalize` preserves the
checkpoint's actual FP16/BF16 scale dtype, while the native SM90 FP8 families
require FP32 scales. Their family-local `supports` callbacks therefore set
`convert_scales = kFloat` before their pack callbacks transpose or repack the
weight and scale tensors.

Packing does not consume bias. Python copies bias to `LinearWeight.bias`,
and `LinearWeight::prepare` continues to normalize its floating dtype.

The current `LayoutConverter` implementations can remain internal building
blocks of the families' `Pack` implementations during migration.
`GetConverters` is removed once every current path is covered by some
family's `supports`/`Pack` pair; no `Pack` implementation may call
`GetConverters` or perform equivalent secondary selection.

## Logical weight format construction

Remove `ResolveLinearWeightFormat`. The checkpoint-side `WeightFormat`
already owns the weight dtype, block sizes, zero-point presence, and tensor
normalization, so it constructs the queried load-storage `DataFormat`
directly. `Family::supports` remains the only authority that decides
whether a family accepts that descriptor.

Remove the unused `QuantParamDesc::transposed` field. No producer sets it and
no packing or dispatch path consumes it; physical qparam orientation remains
in `LinearWeight::q_desc`:

```cpp
struct QuantParamDesc {
    DataType dtype{};

    bool present() const noexcept
    {
        return dtype != kNull;
    }
};

struct DataFormat {
    DataType         dtype{};
    std::vector<int> block_sizes;
    QuantParamDesc   scales{};
    QuantParamDesc   zeros{};

    DataFormat() = default;

    explicit DataFormat(DataType dtype)
        : dtype{dtype}, block_sizes{1, 1}
    {
    }

    DataFormat(DataType         dtype,
               std::vector<int> block_sizes,
               DataType         scales_dtype = kNull,
               DataType         zeros_dtype  = kNull)
        : dtype{dtype},
          block_sizes{std::move(block_sizes)},
          scales{scales_dtype},
          zeros{zeros_dtype}
    {
    }

    bool is_quantized() const noexcept;

    int rank() const noexcept
    {
        return static_cast<int>(block_sizes.size());
    }
};
```

`data_format.h` includes `<utility>` for the move into `block_sizes`. The
zero-argument constructor remains available to C++ value members, but Python
binds only the constructor that completely describes the format. The resolver
binding is deleted:

```cpp
py::class_<turbomind::QuantParamDesc>(m, "QuantParamDesc")
    .def_readwrite("dtype", &turbomind::QuantParamDesc::dtype)
    .def("present", &turbomind::QuantParamDesc::present);

py::class_<turbomind::DataFormat>(m, "DataFormat")
    .def(py::init<turbomind::DataType>(),
         py::arg("dtype"))
    .def(py::init<turbomind::DataType,
                  std::vector<int>,
                  turbomind::DataType,
                  turbomind::DataType>(),
         py::arg("dtype"),
         py::arg("block_sizes"),
         py::arg("scales_dtype") = turbomind::kNull,
         py::arg("zeros_dtype") = turbomind::kNull)
    .def_readwrite("dtype", &turbomind::DataFormat::dtype)
    .def_readwrite("block_sizes", &turbomind::DataFormat::block_sizes)
    .def_readwrite("scales", &turbomind::DataFormat::scales)
    .def_readwrite("zeros", &turbomind::DataFormat::zeros)
    .def("is_quantized", &turbomind::DataFormat::is_quantized)
    .def("rank", &turbomind::DataFormat::rank);
```

The existing `WeightFormat.make_data_format` handles trivial and integer
formats from fields those formats already publish:

```python
def make_data_format(self, data_type) -> _tm.DataFormat:
    if self.has_zero_point:
        return _tm.DataFormat(
            data_type if self.weight_dtype is None else self.weight_dtype,
            [self.block_in or 1, self.block_out or 1],
            data_type,
            data_type)
    return _tm.DataFormat(
        data_type if self.weight_dtype is None else self.weight_dtype,
        [self.block_in or 1, self.block_out or 1])
```

`FP8Format` distinguishes blockwise and groupwise checkpoints from the
weight/scale shapes. Both use the same class; `block_out` is `128` for
blockwise scales and `1` for groupwise scales:

```python
class FP8Format(WeightFormat):
    def __init__(self, *, block_out: int):
        if block_out not in (1, 128):
            raise ValueError(f'unsupported_fp8_block_out_{block_out}')
        super().__init__(block_in=128, block_out=block_out)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        scales = available.get('.weight_scale_inv')
        if scales is None or scales.dtype not in (
                torch.float16, torch.bfloat16, torch.float32):
            return False
        weight = available.get('.weight')
        if weight is None:
            return False
        if weight.dtype not in (torch.float8_e4m3fn, torch.uint8):
            return False
        if weight.dim() < 2 or scales.dim() < 2:
            return False
        expected = (
            (weight.shape[-2] + self.block_out - 1) // self.block_out,
            (weight.shape[-1] + self.block_in - 1) // self.block_in,
        )
        return scales.shape[-2:] == expected

    def make_data_format(self, data_type) -> _tm.DataFormat:
        return _tm.DataFormat(_tm.DataType.TYPE_FP8_E4M3,
                              [self.block_in, self.block_out],
                              _tm.DataType.TYPE_FP32)
```

`MXFP4Format` also overrides the method because its scale storage is part of
its checkpoint-format contract:

```python
class MXFP4Format(WeightFormat):
    name           = 'mxfp4'
    suffix_map     = {
        '.blocks': 'weight',
        '.scales': 'scales',
        '.bias': 'bias',
    }
    weight_dtype   = _tm.DataType.TYPE_FP4_E2M1
    has_zero_point = False

    def __init__(self):
        super().__init__(block_in=32, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        scales = available.get('.scales')
        if scales is None or scales.dtype != torch.uint8:
            return False
        weight = available.get('.blocks')
        if weight is None or weight.dtype != torch.uint8:
            return False
        return weight.numel() == scales.numel() * 16

    def make_data_format(self, data_type) -> _tm.DataFormat:
        return _tm.DataFormat(_tm.DataType.TYPE_FP4_E2M1,
                              [32, 1],
                              _tm.DataType.TYPE_UINT8)
```

No family name, architecture, packing layout, or execution mode enters this
construction. AWQ/GPTQ/compressed-tensor qparams continue to declare the
candidate data type here even though their normalization produces actual
FP16 tensors; the selected family's `WeightBridge` remains responsible for
the actual-dtype conversion contract.

## Selection algorithm

`Gemm::PlanWeight` operates on its device's current kernel `Registry`:

1. The query carries no shape. Each family's `supports` call performs the
   exact logical-format validation; `PlanWeight` adds no separate format
   table or validation helper.
2. Keep the registered family instances whose
   `family->supports(query.weight_format, query.data_type,
query.output_dtype, query.grouped)` answers with a
   bridge: the family enforces the
   data-type contract on its floating-point input, resolves `kNull` output to
   `query.data_type`, enforces the resulting output requirement, and verifies
   the weight-side constraints in the same family callback. A family answering
   `std::nullopt` drops out; if none remains, the query returns
   `std::nullopt`.
3. When `query.input_dtype` is set, move families whose input format has that
   dtype ahead of those without it. Within those two groups, order by
   `priority`. A tie for the selected position in the preferred
   group is a query failure. If no family is preferred, the same rule applies
   to the non-preferred group. `kNull` creates one group and leaves priority
   as the only ordering:
   FP8-input families win on hardware that registers them simply
   by outranking the alternatives.
4. Take the first remaining family. Selection never walks `is_feasible`, and
   registration performs no shape-coverage proof. Alignment, minimum
   extents, grouped execution, indexed input, and epilogue support are
   trusted family declarations. Runtime `is_feasible` remains their
   enforcement point; a false declaration is an implementation bug and a
   fatal dispatch miss.

The complete implementation iterates the registry's distinct family pointers
directly. It performs selection in one pass and introduces no second
selector, helper, pair, tuple, or per-query kernel scan:

```cpp
std::optional<GemmPlan> Gemm::PlanWeight(const WeightQuery& query) const
{
    const Family* selected{};
    WeightBridge selected_bridge{};
    bool selected_preferred{};
    bool ambiguous{};

    for (const Family* family : impl_->registry_.families()) {
        auto bridge = family->supports(query.weight_format,
                                       query.data_type,
                                       query.output_dtype,
                                       query.grouped);
        if (!bridge) {
            continue;
        }

        const bool preferred =
            query.input_dtype != kNull
            && family->input_format().dtype == query.input_dtype;

        if (!selected
            || preferred > selected_preferred
            || (preferred == selected_preferred
                && family->priority > selected->priority)) {
            selected = family;
            selected_bridge = *bridge;
            selected_preferred = preferred;
            ambiguous = false;
        }
        else if (preferred == selected_preferred
                 && family->priority == selected->priority) {
            ambiguous = true;
        }
    }

    if (!selected || ambiguous) {
        return std::nullopt;
    }

    GemmPlan plan;
    plan.family_      = selected;
    plan.bridge_      = selected_bridge;
    plan.output_format_ = selected->output_format(Epilogue::kNone);
    return plan;
}
```

The loader's half of the shape contract: before TP slicing, an axis that
may grow — the FFN intermediate, padded at both producer and consumer —
is zero-padded so that each rank's shard meets the selected family's
`align_k`/`align_n` and `min_k`/`min_n`, with the existing group-padding
helpers. Every other axis must already meet them; a miss is a load error. There is no
runtime activation padding or output slicing: the padded width is the
module's dimension everywhere. Storage extents equal the module dims and meet
the family alignment and minimum extents by construction.

The kernel group is a family constant the family keeps to itself; the plan
carries the bridge, not group state. `Pack` applies the bridge —
replicating scales/zeros down to the family's kernel group when the stored
group is coarser. `Pack` updates `linear.weight_format` to the packed
kernel group, so the existing `MakeQuantDesc(weight.weight_format)` runtime
path supplies the packed group to `Operation::quant_b` and dispatch's
group-size equality admits only matching kernels.

`is_feasible` remains the single owner of all divisibility and
minimum-dimension constraints, but planning no longer consults it. Its
consumer is runtime dispatch. Today's base check covers tile alignment
(`desc.m/n/k % desc_.align`) and quant group-size _equality_ between the
problem and the kernel descriptor; this refactor adds K divisibility by the
quant group to the base check — `desc.k % desc.quant_b.group_size` for
quantized kernels — whose values are already carried by `GemmDesc` (today
the SM90 mixed kernels obtain the same constraint only indirectly, through
`align.z = lcm(CTA_K, group_size)`). Genuinely kernel-specific constraints
(a pipeline's minimum K depth, fused-epilogue capability) live in that
kernel's own override, next to its existing epilogue check. Each family's
`align_k`/`align_n` summarize the divisibility from `align` (and the quant
group for K), while `min_k`/`min_n` summarize minimum extents from the
overrides. Fusion-specific constraints do not enter either base contract.
This absorbs
the shape conditions that today sit in the `GetConverters` gates — they
were always kernel constraints — and replaces the triplicated SM90 gates in
Python, `GetConverters`, and converter runtime checks.

The selected family converts the incoming activation into its own input
format at runtime through its `ConvertInput` operation: floating-point-input
families pass the tensor through; quantized-input families quantize
data-type activations. A quantized activation is never fed to a kernel
requiring floating-point input — there is no dequantization path anywhere.

An explicit `input_dtype` can never fail a query by itself: it only
reorders eligible families. A dtype no registered family consumes (for
example FP8 with an FP16 model) simply matches nothing and leaves the
priority order in effect. Gate/up and down queries are independent: no field
crosses between them, and the down query never sees the gate/up output
format. The kernel combination ensures their compatibility.

## Data-type resolution

The data type — the floating-point type storing floating-point weights and
intermediate data (token embeddings, hidden states) — is resolved before any
weight query, per model component, from exactly two inputs: the component's
weight formats (from the quantization config) and the registered kernel
families. No device-capability heuristics, no
candidate guessing: each family declares the data type it requires, and
the registry enumerates the set supported per weight format with
`grouped=false`. This is a candidate set for dtype resolution, not proof
that every later per-weight execution mode is supported. The HF
config's `dtype`/`torch_dtype` does not participate in the enumeration — it
is publisher-verified numerics metadata, not executability evidence — but it
breaks the BF16/FP16 tie when the request is `'auto'`. For VLM configs the
lookup descends into `text_config`, then `llm_config`, exactly as today:

```cpp
std::vector<DataType> Gemm::DataTypes(const DataFormat& weight_format) const
{
    std::vector<DataType> dtypes;
    for (const Family* family : impl_->registry_.families()) {  // this device
        const DataType dt = family->data_type();
        if (family->supports(weight_format, dt, /*grouped=*/false)
            && std::find(dtypes.begin(), dtypes.end(), dt) == dtypes.end()) {
            dtypes.push_back(dt);
        }
    }
    return dtypes;
}
```

The family answers through its own `supports` callback — the same call
`PlanWeight` filters with — so enumeration and non-grouped selection agree
on formats.
Shapes are covered by the family obligation: answering `supports` commits
the family to bridging any shape gap at pack time or through its kernel
variants (see the `supports` contract), so the shape-free enumeration
cannot select a data type whose non-grouped query fails for shape alone.
Grouped capability is outside this enumeration.
Because the bridge can record scale/zero dtype conversion, the storage
dtype of a checkpoint's scales does not affect the answer: families accept
any float-family storage and convert at pack time, so the set is over
data types only.

A candidate data type remains in the resolution set when the registry
serves every one of its weight formats materialized with that candidate
under `grouped=false`; the config chooses from that set. This is necessary
but not sufficient for every later per-weight query. An explicit `dtype`
request must be in the set — with the legacy FP16 escape hatch preserved: a
requested dtype that is absent falls back to FP16 with a warning when FP16
is present, and fails the load otherwise. `'auto'` takes the model config's
publisher dtype when it is in the set, BF16 before FP16 otherwise. The lookup
descends into `text_config` or `llm_config` and prefers `dtype` over
`torch_dtype`; only `_resolve_dtype`'s `is_bf16_supported` check,
`use_native_sm90_u4`, `_U4_MODEL_FORMATS`, and `_build_resolver`'s FP16 forcing
are replaced by this enumeration after `quantization_config` parsing.
`get_tm_config` runs before any engine context exists. Reuse the existing
`_resolve_dtype` and `_build_resolver` functions: `_resolve_dtype` keeps its
publisher-config traversal and replaces only its device-capability rule with
the executable set:

```python
def _resolve_dtype(requested: str, hf_model_cfg,
                   executable: set[str]) -> str:
    dtype = requested
    if dtype == 'auto':
        if getattr(hf_model_cfg, 'text_config', None):
            hf_model_cfg = hf_model_cfg.text_config
        elif getattr(hf_model_cfg, 'llm_config', None):
            hf_model_cfg = hf_model_cfg.llm_config

        config_dtype = getattr(hf_model_cfg, 'dtype', None)
        if config_dtype is None:
            config_dtype = getattr(hf_model_cfg, 'torch_dtype', None)

        TORCH_DTYPE_MAP = {
            torch.bfloat16: 'bfloat16',
            torch.float16: 'float16',
        }
        dtype = TORCH_DTYPE_MAP.get(config_dtype)
        if dtype not in executable:
            dtype = ('bfloat16'
                     if 'bfloat16' in executable
                     else 'float16')

    if dtype not in executable:
        if 'float16' not in executable:
            raise RuntimeError(
                'no executable data type for this device')
        logger.warning('data type downgraded to float16')
        dtype = 'float16'
    return dtype
```

`_build_resolver` retains its existing `model_format` branches verbatim. The
only removed code is `use_bf16_u4` and the blanket U4 FP16 forcing. After the
existing branches append `TrivialFormat()`, the function enumerates and
selects the component dtype:

```python
def _build_resolver(model_format: str | None,
                    group_size: int | None,
                    requested: str,
                    hf_model_cfg,
                    device) -> tuple[WeightFormatResolver, torch.dtype]:
    formats: list[WeightFormat] = []
    if model_format in (None, 'hf'):
        pass
    elif model_format == 'awq':
        formats.append(AWQFormat(block_in=group_size))
    elif model_format == 'gptq':
        formats.append(GPTQFormat(block_in=group_size))
    elif model_format == 'compressed-tensors':
        formats.append(CompressedTensorFormat(block_in=group_size))
    elif model_format == 'fp8':
        formats.extend((FP8Format(block_out=128),
                        FP8Format(block_out=1)))
    elif model_format == 'mxfp4':
        formats.append(MXFP4Format())
    else:
        raise ValueError(f'unknown model_format: {model_format!r}')
    formats.append(TrivialFormat())

    with torch.cuda.device(device):
        probe = _tm.Gemm()
        executable: set[str] = set()
        for dtype in ('bfloat16', 'float16'):
            data_type = _torch_dtype_to_cpp(getattr(torch, dtype))
            for fmt in formats:
                weight_format = fmt.make_data_format(data_type)
                if data_type not in probe.data_types(weight_format):
                    break
            else:
                executable.add(dtype)

    dtype = _resolve_dtype(requested, hf_model_cfg, executable)
    torch_dtype = getattr(torch, dtype)
    return WeightFormatResolver(
        data_type=_torch_dtype_to_cpp(torch_dtype),
        formats=formats), torch_dtype
```

Each call creates one temporary `Gemm`, which is destroyed when
`_build_resolver` returns. There is no `device_gemm`, process-wide cache, or
dtype-probe handoff.

The text component uses the checkpoint model format:

```python
requested_dtype = engine_config.dtype
resolver, dtype = _build_resolver(
    engine_config.model_format,
    group_size,
    requested_dtype,
    hf_model_cfg,
    engine_config.devices[0])
engine_config.dtype = str(dtype).split('.')[1]
```

After `model_cls` is selected, the existing vision branch performs a separate
trivial-format pass with the original request:

```python
init_kwargs = {}
if getattr(model_cls, '_vision', False):
    init_kwargs['language_model_only'] = (
        engine_config.language_model_only)
    if not engine_config.language_model_only:
        vision_resolver, _ = _build_resolver(
            None,
            None,
            requested_dtype,
            hf_model_cfg,
            engine_config.devices[0])
        init_kwargs['vision_resolver'] = vision_resolver
model = model_cls(cfg, resolver=resolver, **init_kwargs)
```

The aggregate model gives vision its own `Context` carrying that resolver's
dtype. The same devices, engine-owned `Gemm` reference, and input-dtype
preference are reused:

```python
def bind_runtime(self, *, ctx, root_handles,
                 attn_tp, mlp_tp, ep, model_tp):
    self.text_model.bind_runtime(
        ctx=ctx,
        root_handles=root_handles,
        attn_tp=attn_tp,
        mlp_tp=mlp_tp,
        ep=ep,
        model_tp=model_tp)

    if self.vision_model is not None:
        ctx = Context(
            ctx.devices,
            ctx.gemm,
            data_type=self.vision_model._resolver.data_type,
            gemm_input_dtype=ctx.gemm_input_dtype)
        self.vision_model.bind_runtime(
            ctx=ctx,
            root_handles=root_handles,
            attn_tp=attn_tp,
            mlp_tp=mlp_tp,
            ep=ep,
            model_tp=model_tp)
```

`Qwen3_5VisionModel._restore_dtype` and every call to it are deleted. Ordinary
`VisionModelBuilder`, `Builder`, `AttentionBuilder`, and `LayerNormBuilder`
construction now receives the vision-typed context and needs no dtype repair.

The result is component-local: within text or vision, weights, embeddings,
and hidden states all take that component's resolved dtype, so no
BF16↔FP16 conversion appears between its operations. A BF16 AWQ text
component keeps BF16 on SM90 — the native U4 family and the 16816 FP16 family
both answer, so the set is `{bf16, fp16}` — and runs FP16 on SM80, where the
only U4 families are FP16, without any Python architecture check. An AWQ group
size the native family does not pack (gs≠128 on SM90) simply never enters the
set for that format. A group size no family accepts empties the set — a load
error, not a fallback.

This reports non-grouped format capability, not component-wide
executability: the registry enumerates,
the model level decides — the data type also binds embedding, attention,
and normalization, which the registry knows nothing about. Per-weight
queries afterwards still hard-fail. In particular, a grouped query may fail
for the resolved dtype even when another candidate dtype would have served
that grouped query. The loader deliberately does not retry dtype resolution
or downgrade after such a failure.

Enumeration needs a concrete device: kernel constructors read real device
properties, so a `Registry` can only be built against one. Under the unchecked
same-architecture deployment precondition, the resolver enters `devices[0]`
explicitly — never the ambient current device — and its result is
authoritative for the component. Resolution runs
once per component: a VLM performs a text pass and a vision pass, each with
that component's weight formats and its own resolved config dtype. The LLM
and the
ViT currently share the same device set, so a VLM queries the same
first device for both; each component uses its own temporary dtype probe and
its own weight formats.

### Planning `Gemm` instances

`Gemm` construction binds the current device and builds its registry. The
temporary first-device dtype probe is destroyed before engine construction.
Loading then uses the first device's engine-owned `LlamaLinear::Impl::gemm_`.
Under the unchecked same-architecture deployment precondition, that engine
instance is authoritative for every `PlanWeight`.
The other engine contexts retain their own runtime `Gemm` instances as they do
today; loading does not construct another persistent planning instance or a
device-indexed plan set.

## Gate/up derivation

`GemmPlan::gate_up(act_type, projection_n)` takes the family's fusion offer,
validates it, writes the selected epilogue and output format directly into
the plan, and returns the applicable block width. The plan carries no shape —
the caller passes the TP-local padded per-projection width:

```cpp
int GemmPlan::gate_up(ActivationType act_type, int projection_n)
{
    epilogue_ = Epilogue::kNone;
    output_format_ = family_->output_format(Epilogue::kNone);

    Epilogue epilogue = Epilogue::kNone;
    const int gate_up_block =
        family_->gate_up(act_type, projection_n, epilogue);
    if (!gate_up_block) {
        return 0;  // the family fuses nothing
    }

    const DataFormat output_format = family_->output_format(epilogue);

    // (a) The fused arrangement must tile the padded projection width.
    // gate_up_block is deliberately excluded from the family's base
    // alignment and from FFN padding: optional fusion must not increase
    // the intermediate width. A miss therefore degrades to unfused.
    if (projection_n % gate_up_block) {
        return 0;
    }

    epilogue_ = epilogue;
    output_format_ = output_format;
    return gate_up_block;
}
```

The offer lives in the family. The base `Family::gate_up` sets its
epilogue output argument to `Epilogue::kNone` and returns `0` for every
activation and width. A family that fuses SiLU overrides it: for its published
activation it sets the output argument to `Epilogue::kGatedSilu` and returns
its fused block width. `GemmPlan::gate_up` obtains the corresponding output
format through the family's existing `output_format(epilogue)`. For anything
else the family delegates to the base answer. An unmatched activation or
untilable width degrades to unfused, never to a different epilogue. The loader
passes the activation type — a fact it owns — never an epilogue. Adding fused
support for another activation on another kernel family changes that family's
override and nothing else; there is no central activation-to-epilogue dispatch
table.

The tiling validation degrades to unfused rather than failing: an unfused
combined GEMM always remains executable because the family was selected for
the same format with the unfused epilogue. Gate/up and down planning stay
independent; the kernel combination ensures the fused output is consumable by
the down path.

There is no separate w1/w3 fallback: queries are shape-free, so the
combined-width query fails only when no family serves the format at all —
the projection-width query would fail identically. The FFN always builds
the combined w1w3 linear.

## Packing and `LinearWeight`

`GemmPlan::pack` is the returned packing function, invoked by
`LinearWeight::prepare` from the plan attached by the loader after module
creation — never from Python. For gate/up, `gate_up()` has already written the
selected epilogue and output format into that same plan. The packing is
identical in fused and unfused cases — the w1/w3 tensors arrive already
interleaved by `_block_pack_w1w3`, and the family packs the combined weight as
a plain (K, 2N) weight. Each module calls the same plan under its own current
CUDA device and stream. Any failure in `Pack` is fatal
(`TM_CHECK`): packing happens once per weight at load, and a failed pack
means the model cannot run, so there is no partial-mutation contract —
the process dies immediately. The plan must be attached to the same linear
from which its query was constructed, and the loader must satisfy the shape
contract before attachment. These are trusted preconditions:
`GemmPlan::pack` does not recheck the linear's data type, source format,
storage extents, alignment, or minimums.

The complete plan-level operation calls the selected family directly, then
publishes runtime metadata:

```cpp
void GemmPlan::pack(LinearWeight& linear, cudaStream_t stream) const
{
    family_->Pack(linear, bridge_, stream);
    linear.family = family_;
    linear.input_format = family_->input_format();
    linear.epilogue = epilogue_;
    linear.output_format = output_format_;
}
```

`family_->Pack` applies the bridge (scale/zero replication and scale/zero
dtype conversion), replaces the weight/scales/zeros/global-scale tensors,
writes the weight and qparam `MatrixLayout` descriptors into `linear` carrying
the padded extents, and updates `linear.weight_format` to the packed kernel
format.

The runtime metadata becomes:

```cpp
class LinearWeight: public core::Module {
public:
    void set_plan(gemm::GemmPlan plan);

    const gemm::Family* family{};  // process-static family instance

    DataFormat weight_format{};
    DataFormat input_format{};
    DataFormat output_format{};

    gemm::Epilogue epilogue{gemm::Epilogue::kNone};
    gemm::MatrixLayout k_desc{};
    gemm::MatrixLayout q_desc{};

    int input_dim{};
    int output_dim{};
    DataType data_type{};

private:
    std::optional<gemm::GemmPlan> plan_;
};
```

The loader attaches one plan by value after construction. The attachment
method and `prepare()` path are:

```cpp
void LinearWeight::set_plan(gemm::GemmPlan plan)
{
    plan_ = std::move(plan);
}

void LinearWeight::prepare()
{
    if (!weight) {
        return;
    }

    TM_CHECK(plan_);
    plan_->pack(*this, core::Context::stream().handle());
    EnsureFloatDtype(bias, data_type);
}
```

`LinearWeight::prepare` no longer selects formats or converters. For a
weight created with a plan, it executes the plan's `pack` — the single call
site replacing today's converter invocation — then normalizes bias dtype.
`plan_` is optional because `MoeWeight::prepare` creates default-constructed
linked `LinearWeight` views after child preparation and fills them through
`copy_metadata_to`; those linked views never execute their own `prepare()`.
Every nonempty loader-created weight that does execute `prepare()` must have a
plan, enforced by the `TM_CHECK(plan_)` immediately before packing.
`FfnWeight::prepare` no longer changes child input/output formats
or epilogues; it only recurses into children. `MoeWeight::LinkLinearExperts`
builds the expert pointer tables as today; the experts of one MoE layer are
loaded through one path with one weight format on one arch, so their packed
metadata is identical by construction and needs no cross-check. The linked
view copies the selected family pointer through the existing metadata-copy
operation:

```cpp
dst.family = family;
```

## Runtime consumption

`LlamaLinear` uses the fields written by `pack`:

```cpp
Operation operation{};
operation.dispatch = dispatch_policy_;
operation.epilogue = weight.epilogue;
operation.quant_a = MakeQuantDesc(weight.input_format);
operation.quant_b = MakeQuantDesc(weight.weight_format);
operation.batch_dim = 0;

Tensor& D = output;
if (!D) {
    const int dim = (weight.epilogue & Epilogue::kGatedSilu) != Epilogue::kNone
                        ? weight.output_dim / 2
                        : weight.output_dim;
    D = Tensor{{desc_A.rows, dim}, weight.output_dtype(), kDEVICE};
}
operation.family = weight.family->id;
```

`Operation` carries the integer restriction. Every `LinearWeight` consumed by
`LlamaLinear` was created with a `GemmPlan`, so every call writes the selected
family's nonzero id. The MoE router's FP32 result selected its FP32-output
cuBLAS family during loading; runtime does not infer dispatch policy from
`D.dtype()`:

```cpp
struct Operation {
    DispatchPolicy dispatch;
    Epilogue epilogue;
    QuantDesc quant_a;
    QuantDesc quant_b;
    int batch_dim;
    std::uint32_t family{};
};
```

`get_gemm_desc` copies `operation.family` into a new first field on
`GemmDesc`; transpose leaves it unchanged:

```cpp
struct GemmDesc {
    std::uint32_t family{};
    int arch;
    DataType type_a;
    DataType type_b;
    DataType type_c;
    Order order_a;
    Order order_b;
    Order order_c;
    Striding striding_a;
    Striding striding_b;
    Striding striding_c;
    Pack pack_a;
    Pack pack_b;
    Pack pack_u;
    Pack pack_v;
    QuantDesc quant_a;
    QuantDesc quant_b;
    Epilogue epilogue;
    int batch_dim;
    int group_axis;
    int m;
    int n;
    int k;
    int num;
};
```

`Kernel::is_feasible` rejects another family before its existing descriptor
checks for implementations that call the base:

```cpp
if (desc.family && desc.family != family().id) {
    return false;
}
```

`CublasKernel::is_feasible` and `CublasGroupedKernel::is_feasible` are
direct overrides that do not call the base. Each begins with the same check
before its existing cuBLAS-specific conditions:

```cpp
if (desc.family && desc.family != family().id) {
    return false;
}
```

The check deliberately stays in these two overrides; `Context::Filter`
continues to call `k->is_feasible(g)` without a separate family-id condition.

The fused arrangement is a family constant, not runtime state: a kernel
family publishes exactly one gate/up arrangement through its `gate_up`
override, its fused kernels are compiled for that block width, and the
group count follows inside the kernel from `N / (2 * block)`. The fixed id on
`GemmDesc` guarantees that runtime dispatch considers only the family whose
arrangement the loader produced. The per-kernel
`KernelDesc::supports_fused_silu` bool is widened to an
`Epilogue supported_epilogues` mask, initialized in the kernel
constructors from the compile-time capability constant that the
`is_feasible` overrides test today (`Gemm::kSupportsFusedSilu`); the
overrides then test the requested epilogue's membership in the descriptor
mask. No arrangement field is needed: the required family identity plus the
epilogue determines the arrangement any dispatched kernel will consume.
`TransposedKernel` clears `kGatedSilu` from its copied descriptor mask because
the M/N swap does not preserve gate/up pairing along N. Its feasibility check
validates the wrapper's mask before delegating the remaining kernel-specific
checks to the wrapped kernel.

The existing dispatch-cache API remains unchanged because `family` is
part of `GemmDesc`. It is added to the `GemmDesc` comparison tuple, so cache
lookup, lower-bound reuse, insertion, export, and import are family-specific.
The fixed integer is serialized as part of the existing `Record::gemm`.
Import additionally requires a matching kernel to belong to that id; `0`
preserves the existing unrestricted behavior:

```cpp
for (const auto& p : kernels) {
    if (p->desc() == record.kernel
        && (!record.gemm.family
            || p->family().id == record.gemm.family)) {
        spec.kernel = p;
        break;
    }
}
if (spec.kernel) {
    entries.emplace_back(record.gemm, spec);
}
```

Input conversion is a common family operation, not a central format or
architecture dispatch. `pack`
stores the family pointer in `LinearWeight`; forward makes one call:

```cpp
weight.family->ConvertInput(A, U, weight, input, input_scales, stream);
```

The base implementation checks the incoming dtype matches the family's input
dtype and passes the tensor and scales through. The W8A8 family overrides
it: data-type input is quantized with `QuantizeSymm` (per-128-block FP8
plus FP32 scales); dynamic-FP8 input with its scales passes through. A
quantized tensor is never fed to a floating-point-input kernel, so no
dequantization path exists — a fused epilogue's output format is always a
valid kernel input format for the consumer's family.

Extents need no runtime reconciliation: the loader pads the FFN
intermediate axis at both ends (`config.inter_size` carries the padded
width as today) and rejects every other misalignment at load, so the
activation's K equals the weight's packed K and C's width equals the
weight's packed N. `LlamaLinear` builds the GEMM problem from the live
tensors and `weight.k_desc`/`weight.q_desc` exactly as today, with
`weight.input_dim`/`weight.output_dim` agreeing with the descriptor
extents by construction.

After conversion, runtime addressing continues to come from actual
`MatrixLayout` fields: `idxs != nullptr` means routed rows; `ld == 0` or
`offsets != nullptr` means grouped storage; otherwise the matrix is flat.
The routed-row fallback is static and family-driven — no probing, no arch
gate. The family reports whether its kernels accept an indexed A
(`indexed_input()`); when it does not, `GetOperandA` takes the existing
gather path — `invokeMoeDispatch` for A, `invokeMoeDispatchScales` for its
scales (LlamaLinear.cu:101/104 today, gated to SM100+BF16; the gate is
deleted) — clearing the indices so the GEMM runs blocked over the
`offsets` descriptor, i.e. a blocked grouped GEMM over the device-side
expert pointer/offset tables built by `LinkLinearExperts`:

```cpp
// GetOperandA, MoE path — the family decides, no probe, no SM100 gate:
if (has_indices && !weight.family->indexed_input()) {
    Tensor A_e{{m, k}, A.dtype(), kDEVICE};
    invokeMoeDispatch(A_e, A, indices.data(), m, num_valid_tokens, st);
    if (U) {
        Tensor U_e;
        invokeMoeDispatchScales(U_e, U, indices.data(), m, num_valid_tokens, st);
        U = U_e;
    }
    A = A_e;
    indices = {};  // applied by the gather; descriptor runs blocked
}
```

The blocked kernel exists by the family's trusted `grouped()` contract: the
MoE builder marks its queries `grouped = true`, so a family that does not
advertise grouped kernels can never be selected for an expert weight. As
stated in the selection algorithm, the deliberately narrow registration
check does not separately prove grouped coverage. A false advertisement is
an implementation bug, and `Gemm::Run`'s fatal miss stays fatal rather than
becoming a fallback trigger. No per-group flat loop, and no host-side copy
of the expert base pointers is retained — the device tables already carry
everything the blocked form needs. This fallback is a runtime-only
adaptation (implementation-order item 7); it neither invalidates nor
modifies the persistent GEMM plan.

## Python binding

Bind the query fields and the read-only result values Python consumes. The
existing `Epilogue` binding gains arithmetic support for mask membership:

```cpp
py::enum_<gemm::Epilogue>(m, "Epilogue", py::arithmetic())
    .value("kNone", gemm::Epilogue::kNone)
    .value("kChannelCombination", gemm::Epilogue::kChannelCombination)
    .value("kGatedSilu", gemm::Epilogue::kGatedSilu);

py::enum_<ActivationType>(m, "ActivationType")
    .value("kSilu", ActivationType::kSilu)
    .value("kSiluGptOss", ActivationType::kSiluGptOss)
    .value("kGeluPytorchTanh", ActivationType::kGeluPytorchTanh)
    .value("kGelu", ActivationType::kGelu);

py::class_<gemm::Family>(m, "Family")
    .def_property_readonly("input_format", &gemm::Family::input_format,
                           py::return_value_policy::reference_internal)
    .def_property_readonly("align_k", &gemm::Family::align_k)
    .def_property_readonly("align_n", &gemm::Family::align_n)
    .def_property_readonly("min_k", &gemm::Family::min_k)
    .def_property_readonly("min_n", &gemm::Family::min_n)
    .def_property_readonly("grouped", &gemm::Family::grouped)
    .def_property_readonly("indexed_input", &gemm::Family::indexed_input)
    .def("data_type", &gemm::Family::data_type)
    .def("supported_epilogues", &gemm::Family::supported_epilogues)
    .def("output_format", &gemm::Family::output_format,
         py::arg("epilogue"));

py::class_<gemm::WeightQuery>(m, "WeightQuery")
    .def(py::init<>())
    .def_readwrite("weight_format", &gemm::WeightQuery::weight_format)
    .def_readwrite("data_type", &gemm::WeightQuery::data_type)
    .def_readwrite("input_dtype", &gemm::WeightQuery::input_dtype)
    .def_readwrite("grouped", &gemm::WeightQuery::grouped);

py::class_<gemm::GemmPlan>(m, "GemmPlan")
    .def_property_readonly("family", &gemm::GemmPlan::family,
                           py::return_value_policy::reference_internal)
    .def("gate_up", &gemm::GemmPlan::gate_up, py::arg("act_type"),
         py::arg("projection_n"))
    .def("pack",
         [](const gemm::GemmPlan& plan, LinearWeight& linear) {
             plan.pack(linear, core::Context::stream().handle());
         },
         py::arg("linear"),
         py::call_guard<py::gil_scoped_release>());

py::class_<gemm::Gemm>(m, "Gemm")
    .def(py::init<>())
    .def("plan_weight", &gemm::Gemm::PlanWeight, py::arg("query"))
    .def("data_types", &gemm::Gemm::DataTypes, py::arg("weight_format"));
```

These bindings make the selected family inspectable. Python does not write it
or feed it into another query.

Keep the existing one-argument `create_module(config)` binding unchanged.
Restore the existing owning `LinearWeight(config)` constructor for the
standalone linear harness and bind `set_plan` separately:

```cpp
py::class_<LinearWeight, core::Module>(m, "LinearWeight")
    .def(py::init<const core::LinearConfig&>(), py::arg("config"))
    .def("set_plan", &LinearWeight::set_plan, py::arg("plan"));
```

The direct constructor is Python-owned. The generic `create_module` result
remains non-owning because production transfers it to its parent through
`add_child_raw`. Both paths call `set_plan(plan)` before `prepare()`. No
constructor or attachment API accepts separate gate/up metadata.

## `Gemm` ownership

The persistent planning instances are the `Gemm` objects already owned by the
engine's per-device `LlamaLinear`. Add this public declaration to
`LlamaLinear.h`:

```cpp
gemm::Gemm& gemm() noexcept;
```

Its definition returns the existing member and does not allocate another
instance:

```cpp
gemm::Gemm& LlamaLinear::gemm() noexcept
{
    return impl_->gemm_;
}
```

Add the forward declaration and public method declaration to `turbomind.h`:

```cpp
namespace gemm {
class Gemm;
}

gemm::Gemm& gemm(int index);
```

Add the same declaration to `TurboMind::Impl`, then define both methods:

```cpp
gemm::Gemm& gemm(int index);

gemm::Gemm& TurboMind::Impl::gemm(int index)
{
    return contexts_[index]->linear->gemm();
}

gemm::Gemm& TurboMind::gemm(int index)
{
    return impl_->gemm(index);
}
```

The Python binding returns an engine-owned reference:

```cpp
.def("gemm",
     &TurboMind::gemm,
     py::return_value_policy::reference_internal,
     "index"_a)
```

`ModelLoader._bind_runtime` places the first-device reference in the existing builder
`Context`:

```python
class Context:
    def __init__(self, devices, gemm, data_type, gemm_input_dtype):
        self.devices = devices
        self.gemm = gemm
        self.data_type = data_type
        self.gemm_input_dtype = gemm_input_dtype
        self._active_mask_stack = [(True,) * len(devices)]


mc = self.model_comm
gemm_input_dtype = {
    None: _tm.DataType.TYPE_INVALID,
    'float16': _tm.DataType.TYPE_FP16,
    'bfloat16': _tm.DataType.TYPE_BF16,
    'float8_e4m3': _tm.DataType.TYPE_FP8_E4M3,
}[self.engine_config.gemm_input_dtype]
ctx = Context(
    [mc.context(g) for g in range(self.gpu_count)],
    mc.gemm(0),
    data_type=self.data_type,
    gemm_input_dtype=gemm_input_dtype)
```

`TurboMind.create` and `_create_weight` construct every engine `Context` — and
therefore every `LlamaLinear::Impl::gemm_` — before `ModelLoader` calls
`_bind_runtime`. The returned reference remains owned by `model_comm` for the
engine lifetime. `self._ctx.gemm` is therefore the first device's
engine-owned planning instance, not the temporary dtype probe.

## Python loading flow

Add one query constructor in `Builder`; it has no architecture, kernel, or
shape knowledge — the query is format-level:

```python
def _make_gemm_query(self, linear: Linear, *, grouped: bool = False
                     ) -> _tm.WeightQuery:
    query = _tm.WeightQuery()
    # the builder config carries the component's resolved dtype — the
    # builder received that component's Context, so this is correct for
    # ViT and LLM alike
    query.data_type = self.config.data_type
    query.weight_format = linear.weight_format.make_data_format(
        self.config.data_type)
    # already mapped by ModelLoader; kNull when the user left it unset
    query.input_dtype = self._ctx.gemm_input_dtype
    # output_dtype remains its kNull default: require data_type output
    # MoE expert weights only: the family must have blocked grouped kernels
    query.grouped = grouped
    return query
```

The builder does not copy the preference into its module config or another
field. `_make_gemm_query` reads the already mapped value directly from
`self._ctx.gemm_input_dtype`.

The builder context retains the first device's engine-owned `Gemm` reference.
`self._ctx.gemm` is authoritative for every weight query
under the unchecked same-architecture deployment precondition. It is
intentionally not the temporary instance used earlier for dtype enumeration.

`_make_gemm_query` sets `data_type` from `self.config.data_type` — the
component's resolved dtype, copied by `Builder.__init__` from its context — and
`input_dtype` from `self._ctx.gemm_input_dtype`. `ModelLoader` maps the
user's `TurbomindEngineConfig.gemm_input_dtype` string once (`None` → `kNull`)
before constructing `Context`; the builder and its module config never copy
this engine setting. Python carries no family-selection rule for the input
preference: for floating-point weights the data-type contract in base `supports` pins the
input to the data type, and the knob is a preference that only reorders
eligible families — `bfloat16`/`float16` prefers weight-only W8A16 execution
even where W8A8 exists, `float8_e4m3` prefers W8A8 where a W8A8 family is
registered and runs W8A16 where not.
The FFN flow passes `grouped=self.config.is_expert`; every non-expert caller
uses the default `False`.

The MoE builder supplies the only explicit output requirement. Router weights
are trivial floating-point weights, so this selects family 104 or 105 and keeps
their source-preserving layout:

```python
def add_gate(self, name, linear):
    query = self._make_gemm_query(linear)
    query.output_dtype = _tm.DataType.TYPE_FP32
    self._add_linear(name, linear, split_side=None,
                     plan=self._query_gemm(query))
```

For each call, query the authoritative first-device instance and return that
linear's plan:

```python
def _query_gemm(self, query: _tm.WeightQuery):
    with self._ctx.devices[0]:
        # plan_weight never throws; None here is a hard load error
        plan = self._ctx.gemm.plan_weight(query)
    if plan is None:
        raise RuntimeError('no GEMM kernel family for this component')
    return plan
```

Change `_add_linear` to accept an optional `plan`. When the caller does not
supply one, `_add_linear` produces the generic query itself. After planning,
the loader hard-checks
the global weight against the selected family's TP-local alignment and
minimum extents — padding happens only in the FFN flow (below); every other
weight must already meet them:

```python
def _add_linear(self, name: str, linear: Linear,
                split_side: SplitSide | None = None,
                plan=None):
    if plan is None:
        plan = self._query_gemm(
            self._make_gemm_query(linear))
    # Hard shape check — padding is the FFN flow's job (below); a miss here
    # is a load error, never a silent fallback.
    family = plan.family
    fmt = linear.weight_format
    tp = self.tp.size if split_side else 1
    k = int(linear.tensors['weight'].shape[0])
    n = int(linear.tensors['weight'].shape[-1])
    min_k = family.min_k
    min_n = family.min_n
    align_k = math.lcm(family.align_k, fmt.block_in or 1)
    align_n = math.lcm(family.align_n, fmt.block_out or 1)
    if split_side == SplitSide.INPUT:
        min_k *= tp
        align_k *= tp
    elif split_side == SplitSide.OUTPUT:
        min_n *= tp
        align_n *= tp
    if (k < min_k or k % align_k or n < min_n or n % align_n):
        raise RuntimeError(
            f'{name}: {family} requires K >= {min_k}, K % {align_k} == 0, '
            f'N >= {min_n}, and N % {align_n} == 0; got K={k} N={n}')
```

There is no padding outside the FFN flow: an axis that misses its alignment
or minimum extent is a load error — e.g. an input-split weight whose local K
misses `align_k` or `min_k` — never a silent fallback.

The rest of `_add_linear` keeps its existing Python `WeightFormat.pack`, TP
shard, allocation, and copy order. `LinearConfig` carries the local dims
from the tensor shapes exactly as today — where the FFN flow padded, those
are the padded dims, and the descriptor extents agree with them by
construction. After generic module creation, Python attaches the one selected
plan. Packing and runtime-metadata publication remain deferred to
`LinearWeight::prepare`:

```python
with context:
    module = _tm.create_module(linear_config)
    module.set_plan(plan)
    for kind, tensor in tensors.items():
        shard = _shard(tensor, kind_split_dims[kind], tp, rank)
        _copy_shard_to_param(module, kind, shard,
                             alloc_shape=packed[kind].alloc_shape,
                             alloc_dtype=packed[kind].alloc_dtype)
    handles.append(module)
```

`LinearWeight::prepare` executes the attached plan's `pack` inside the
module's own device context. This is the same call site that invokes
converters today; the pack steps themselves are unchanged.

Before module creation, `_add_linear` builds `LinearConfig` from the
linear's tensor shapes as today — the plan carries no dims. The same plan is
attached to every active device replica of that linear under the unchecked
same-architecture deployment precondition; another linear may have another
plan.

## Linear harness flow

`fixture.py` imports the existing lazy `_tm` accessor with the other symbols
it already imports from `linear.py`:

```python
from .linear import (
    Linear,
    Weight,
    _tm,
    activation_needs_quantize,
    dequantize_symm,
    device_context,
    link_experts,
    quantize_symm,
)
```

Importing the accessor does not load the extension; the first `_tm()` call
retains the existing lazy-load behavior.

The test fixture creates one `Gemm` in its existing device context and keeps
it until every weight has been destroyed:

```python
with self.on_tm_stream():
    self.gemm = _tm().Gemm()
    self.linear = Linear()
```

In `close()`, the existing weight fields and expert lists are cleared first;
then the fixture releases `self.gemm`, followed by the stream boundary and
device context:

```python
self.w_original = None
self.w_quant = None
self.w_dequant = None
self.e_original = []
self.e_quant = []
self.e_dequant = []
self.gemm = None
self._stream_boundary = None
ctx = getattr(self, '_ctx', None)
if ctx is not None:
    self._ctx = None
    ctx.__exit__(None, None, None)
```

`Weight` requires the already selected `plan`; there is no planless
wrapper path. Its `LinearConfig.format` comes from that plan. The underlying
`LinearWeight(config)` constructor is Python-owned, and the wrapper attaches
the plan immediately afterward:

```python
class Weight:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 data_type: str,
                 weight_type: str,
                 group_size: int = 0,
                 has_bias: bool = False,
                 *,
                 weight_format,
                 plan) -> None:
        tm = _tm()
        dt = to_tm_dtype(data_type)
        wt = to_tm_dtype(weight_type)

        cfg = tm.LinearConfig()
        cfg.input_dim = input_dim
        cfg.output_dim = output_dim
        cfg.data_type = dt
        cfg.format = weight_format
        cfg.has_bias = has_bias

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._data_type = data_type
        self._weight_type = weight_type
        self._group_size = group_size
        self._has_bias = has_bias
        self.plan = plan
        self.gate_up_block = 0
        self._impl = tm.LinearWeight(cfg)
        self._impl.set_plan(plan)
```

The existing tensor allocations follow that block unchanged.

The existing `_allocate_weight_triple` constructs the two storage descriptors,
queries the fixture's `self.gemm`, mutates the quantized weight's plan for
gate/up when requested, and passes each plan into `Weight` before any tensor
allocation or copy:

```python
def _allocate_weight_triple(
        self, grouped: bool = False) -> tuple[Weight, Weight, Weight]:
    tm = _tm()
    c = self.case
    dt = to_tm_dtype(c.data_type)
    wt = to_tm_dtype(c.weight_type)

    with self.on_tm_stream():
        query = tm.WeightQuery()
        query.weight_format = tm.DataFormat(dt)
        query.data_type = dt
        query.input_dtype = dt
        query.grouped = False
        plan = self.gemm.plan_weight(query)

        w_original = Weight(
            c.input_dim,
            c.output_dim,
            c.data_type,
            c.data_type,
            0,
            weight_format=query.weight_format,
            plan=plan)
        w_dequant = Weight(
            c.input_dim,
            c.output_dim,
            c.data_type,
            c.data_type,
            0,
            weight_format=query.weight_format,
            plan=plan)

        query = tm.WeightQuery()
        if c.weight_type in ('bf16', 'fp16'):
            query.weight_format = tm.DataFormat(wt)
        elif c.weight_type == 'fp8_e4m3':
            query.weight_format = tm.DataFormat(
                wt, [128, 128], tm.DataType.TYPE_FP32)
        elif c.weight_type == 'uint4':
            query.weight_format = tm.DataFormat(
                wt, [c.group_size, 1], dt, dt)
        elif c.weight_type == 'fp4_e2m1':
            query.weight_format = tm.DataFormat(
                wt, [c.group_size, 1], tm.DataType.TYPE_UINT8)
        query.data_type = dt
        query.input_dtype = to_tm_dtype(c.input_type)
        query.grouped = grouped
        plan = self.gemm.plan_weight(query)

        gate_up_block = 0
        if c.fuse_silu:
            gate_up_block = plan.gate_up(
                tm.ActivationType.kSilu, c.output_dim // 2)
            if not gate_up_block:
                raise NotImplementedError(
                    'selected family does not fuse SiLU')

        w_quant = Weight(
            c.input_dim,
            c.output_dim,
            c.data_type,
            c.weight_type,
            c.group_size,
            weight_format=query.weight_format,
            plan=plan)
        w_quant.gate_up_block = gate_up_block
        return w_original, w_quant, w_dequant
```

The dense path calls `_allocate_weight_triple()` and the MoE path calls
`_allocate_weight_triple(grouped=True)`. The harness deletes the later calls
to `set_grouped`, `set_input_type`, and `set_epilogue`; it also deletes those
three `Weight` methods and `_apply_fuse_silu_epilogue`. An explicitly fused
case raises `NotImplementedError` immediately when the selected family returns
block `0`, before weight generation or either block operation. This derives
from `Exception`, so the existing `LinearFixture.__init__` handler calls
`close()` before re-raising it.

The pytest caller converts that cleaned-up unsupported case into a skip:

```python
def test_smoke_linear_correctness(run):
    try:
        fx = LinearFixture(run.case)
    except NotImplementedError as e:
        pytest.skip(str(e))
    try:
        fx.prepare_batch(run.batch_size)
        fx.run_reference()
        fx.run_linear()
        fx.check_tolerances(fx.compare())
    finally:
        fx.close()
```

The standalone benchmark logs the unsupported case and continues to the next
case group instead of terminating:

```python
for case_runs in by_case.values():
    try:
        fx = LinearFixture(case_runs[0].case, device=device)
    except NotImplementedError as e:
        case = case_runs[0].case
        logger.warning(
            f'Unsupported benchmark case {case.name} '
            f'(tp={case.tp}, ep={case.ep}): {e}')
        continue
```

This replaces only the constructor line immediately before the benchmark's
existing `try`/`finally`; its per-case body and `finally: fx.close()` remain
unchanged.

Gate/up tensor arrangement and the reference epilogue use the block returned
by `GemmPlan::gate_up`, not the deleted format/hardware table:

```python
w = self._make_random_weight(w_quant.gate_up_block)

def _make_random_weight(self, gate_up_block: int) -> torch.Tensor:
    c = self.case
    dtype = self._torch_dtype()
    scale = _weight_fill_scale(c.input_dim)
    if c.fuse_silu:
        inter = c.output_dim // 2
        w1 = torch.randn(
            c.input_dim, inter, device=self.device, dtype=dtype) * scale
        w3 = torch.randn(
            c.input_dim, inter, device=self.device, dtype=dtype) * scale
        return block_pack_w1w3(w1, w3, gate_up_block)
    return torch.randn(
        c.input_dim, c.output_dim, device=self.device, dtype=dtype) * scale
```

```python
if c.fuse_silu:
    block = self.w_quant.gate_up_block
    self.d_original = apply_block_fused_silu(self.d_original, block)
    self.d_dequant = apply_block_fused_silu(self.d_dequant, block)
    if (self.w_quant._impl.output_format.dtype
            == _tm().DataType.TYPE_FP8_E4M3):
        _, _, self.d_original = quantize_symm_row_fp8(self.d_original)
        _, _, self.d_dequant = quantize_symm_row_fp8(self.d_dequant)
```

`fused_silu_block` and its four fixed constants are removed from
`tests/turbomind/linear/reference.py`.

The linked expert view reuses the first expert's selected objects:

```python
e0 = experts[0]
fused = Weight(
    e0._input_dim,
    e0._output_dim,
    e0._data_type,
    e0._weight_type,
    e0._group_size,
    has_bias=e0._has_bias,
    weight_format=e0._impl.weight_format,
    plan=e0.plan)
fused.gate_up_block = e0.gate_up_block
e0._impl.copy_metadata_to(fused._impl)
```

The owning `py::init<const core::LinearConfig&>()` binding remains available
to the standalone harness, which immediately calls the separately bound
`set_plan`. The obsolete `set_grouped` and
`set_fp8_fused_silu_output` bindings are removed with their implementations.

## FFN flow

`FfnBuilder.add_ffn` keeps `_pad_ffn_for_tp` as its one padding step —
adjusted to take the selected family's base alignment and minimum extents in
place of the hardcoded `_GEMM_K_ALIGN` lcm. The intermediate width is w2's
TP-local K and half of combined w1w3's TP-local N, so both sides of the
base shape contract are applied explicitly:

```python
def _pad_ffn_for_tp(w1: Linear, w2: Linear, w3: Linear, tp: int,
                    family) -> tuple[Linear, Linear, Linear]:
    """Pad w1/w3 output dim and w2 input dim so every rank's shard meets
    the family's base shape contract, TP divisibility, and format blocks."""
    fmt = w1.weight_format
    unit = math.lcm(fmt.block_in or 1, fmt.block_out or 1)
    raw = int(w1.tensors['weight'].size(-1))

    # For local projection P: w2 has K=P and combined w1w3 has N=2P.
    projection_align_n = family.align_n // math.gcd(family.align_n, 2)
    projection_min_n = (family.min_n + 1) // 2
    gran = math.lcm(family.align_k, projection_align_n, unit)
    minimum = max(family.min_k, projection_min_n)

    div = gran * tp
    target = max(raw, minimum * tp)
    target = ((target + div - 1) // div) * div
    groups = raw // unit
    target_groups = target // unit
    w1 = pad_output_groups(w1, src_groups=groups,
                           dst_groups=target_groups)
    w3 = pad_output_groups(w3, src_groups=groups,
                           dst_groups=target_groups)
    w2 = pad_input_groups(w2, src_groups=groups,
                          dst_groups=target_groups)
    return w1, w2, w3
```

The old `tp <= 1` early exit goes away — the base shape contract applies at
any TP. `gate_up_block` is not an input to `_pad_ffn_for_tp` and is not
folded into `align_n`: after mandatory padding, `gate_up()` checks the
resulting TP-local projection width and degrades to unfused when it does not
tile the optional fused arrangement. Fusion therefore adds no intermediate
padding; growth can come only from the base GEMM, format-block, TP, and
minimum-extent requirements.
w1, w2, and w3
are required to share the component's logical weight format. The resolver
selects a format per prefix independently, so this is deliberately only a
caller precondition, not a property the flow checks. Divergent w1/w2/w3
formats are explicitly undefined behavior by design: the loader does not
detect them, does not promise a load error, and does not provide recovery
or fallback. Under the precondition, w1/w3 share shapes and one
shape-free query selects the plan for their combined w1w3 linear — the query
itself knows nothing about gating.

The w1w3 query is shape-free, so it comes first — before any padding. There is
no combined-width trial and no separate w1/w3 fallback, and a combined GEMM
is always plannable when the format is servable. After `gate_up()` mutates
that plan, w2 performs the same shape-free query independently. Under the
shared-format precondition the two queries have identical format, data type,
input-type preference, and grouped mode, so deterministic selection returns
the same family; the separate w2 plan retains the unfused epilogue and output
format. The intermediate runs at the padded width as today:

```python
act_type = getattr(self.config, 'act_type', 0)
if isinstance(act_type, int):
    act_type = {0: 'silu', 1: 'gpt-oss'}.get(act_type, 'silu')
act_type = (_tm.ActivationType.kSiluGptOss if act_type == 'gpt-oss'
            else _tm.ActivationType.kSilu)

# shape-free w1w3 query before padding
plan = self._query_gemm(
    self._make_gemm_query(w1, grouped=self.config.is_expert))

w1, w2, w3 = _pad_ffn_for_tp(w1, w2, w3, self.tp.size,
                             plan.family)
proj = w1.tensors['weight'].size(-1)  # padded global projection width

gate_up_block = plan.gate_up(act_type, proj // self.tp.size)

self.config.inter_size = proj
self.config.fuse_silu = gate_up_block != 0
# groups tiles the global padded width; unfused: one
# [all gate | all up] pair per rank
groups = (proj // gate_up_block
          if gate_up_block else self.tp.size)
w1w3 = _block_pack_w1w3(w1, w3, groups=groups)
self._add_linear('w1w3', w1w3, SplitSide.OUTPUT, plan)

# w2 has its own authoritative plan
plan = self._query_gemm(
    self._make_gemm_query(w2, grouped=self.config.is_expert))
self._add_linear('w2', w2, SplitSide.INPUT, plan)
```

`_block_pack_w1w3` already applies the same logical group count to weight,
scale, zero, and bias tensors while respecting their different physical
widths through `@transform_output_dim`. The loader derives the count from
the plan's block width and the padded width. For an unfused plan, it
creates the TP-local `[all gate | all up]` arrangement. For a fused plan, it
creates the exact registered block width.

## File-level migration

Add:

- `src/turbomind/kernels/gemm/family.h` and `family.cc`: `WeightBridge`,
  the concrete `Family` object type, and the common family operations.
  These files contain no concrete family objects.
- `src/turbomind/kernels/gemm/plan.h` and `plan.cc`: `WeightQuery`,
  `GemmPlan`, and only the plan operations. `family.cc` and `plan.cc` are
  added to the `gemm2_core` target in
  `src/turbomind/kernels/gemm/CMakeLists.txt`.
- `lmdeploy/messages.py`: the `gemm_input_dtype` field on
  `TurbomindEngineConfig` (`None` default; allow only `None`, `float16`,
  `bfloat16`, and `float8_e4m3` in `__post_init__`).

Change:

- `src/turbomind/core/data_format.h/.cc`: add the two `DataFormat`
  constructors shown above and `operator==` and `operator!=` for exact comparison,
  including dtype, block sizes, and both quant-parameter dtypes; remove
  `QuantParamDesc::transposed` and the `ResolveLinearWeightFormat`
  declaration and definition while retaining `DataFormat::is_quantized`.
- GEMM kernel families: define one concrete `Family` object per row in
  the complete family catalog beside its registrations, with exactly its
  permanent ID, priority, formats, shape contract, execution flags, packing
  operation, and fusion behavior. Related objects share local function
  templates; unique callbacks are written directly in the object definition.
  Tile variants reference that object. Each family declares its
  `align_k`/`align_n`
  divisibility (from its kernels' `align` and quant group) separately from
  its `min_k`/`min_n` minimum extents (from minimum-depth overrides).
  Fusion-specific block widths are not included. The per-family format structs in
  `sm90_mixed_pack.h` stay as the families' compile-time traits, unchanged.
  Existing SM70/75/80/90-16816 `Collector::add<Config>()` calls remain
  unchanged. The SM90 GMMA local `add` functions pass their concrete
  `KernelImplSm90*` type to the same `Collector::add<T>()`. All kernel constructors —
  `KernelImpl`, the four `KernelImplSm90*`, both cuBLAS implementations, and
  the `TransposedKernel` wrapper — take or forward only the family reference.
- `src/turbomind/kernels/gemm/registrar.h`: bind one `Family` reference
  in each `Registrar` entry, construct `Collector` with that family, and have
  its existing `add<Config>()` pass the bound family to `KernelImpl`.
- `src/turbomind/kernels/gemm/cublas.cu`: add four local family objects for
  dense/grouped `kHalf` and `kBfloat16`, with fixed ids 100–103 and the
  source-preserving priorities
  90/100. Register the dense FP16 specialization on every arch and the dense
  BF16 specialization only when `arch >= Sm80::value`; on compatible SM100
  builds, register a grouped cuBLAS kernel for both grouped specializations.
  Both direct `is_feasible` overrides begin with the selected-family id check
  before their existing conditions.
- `src/turbomind/kernels/gemm/registry.h/.cu`: store the distinct surviving
  family pointers in `families_`, expose them through `families()`, construct
  each collector from its registrar entry's family, and reject zero or
  duplicate family ids while collecting them.
- `src/turbomind/kernels/gemm/types.h`: `Epilogue` gains bitwise `|`/`&`
  operators; `KernelDesc::supports_fused_silu` is widened to an
  `Epilogue supported_epilogues` mask; `Operation.family` carries the
  selected family's fixed id.
- `src/turbomind/kernels/gemm/desc.h`: add the fixed integer
  `GemmDesc.family`; transpose preserves it.
- `src/turbomind/kernels/gemm/kernel.cu` and the kernel implementations:
  base `is_feasible` first rejects a nonzero `GemmDesc.family` that differs
  from `family().id`, then applies the layout-geometry
  constraints from the deleted `GetConverters` gates (quant-group K
  divisibility added to the base check, kernel-specific minimums in the
  overrides; epilogue membership tested against the descriptor mask). The
  two cuBLAS overrides perform the same id rejection directly because they
  do not call the base.
- `src/turbomind/kernels/gemm/convert_v3.cu`: expose its concrete conversions
  as the owning family's `Pack` implementations; remove selection policy.
- `src/turbomind/kernels/gemm/gemm.h/.cu`: implement `PlanWeight` exactly as
  shown over `impl_->registry_.families()` and implement `DataTypes` over the
  same family collection.
- `src/turbomind/kernels/gemm/dispatch_cache.cu`: include
  `GemmDesc.family` in the comparison tuple and require imported
  `KernelDesc` matches to have that id when it is nonzero. Bump
  `kDispatchCacheVersion` — `GemmDesc` and the widened `KernelDesc` change
  the serialized record layout, and old cache files must be rejected by
  version rather than silently dropped on size mismatch.
- `src/turbomind/python/linear_bind.cpp`: bind the query, `GemmPlan`, and the
  pack operations; retain the owning `LinearWeight(config)` constructor, bind
  `LinearWeight::set_plan`, and remove the obsolete grouped/fused-output
  policy bindings.
- `src/turbomind/python/bind.cpp`: bind both `DataFormat` constructors shown
  above, remove the `ResolveLinearWeightFormat` binding, keep the existing
  generic `create_module(config)` binding unchanged, and bind
  `TurboMind::gemm(index)` as an engine-owned reference.
- `src/turbomind/models/llama/LlamaLinear.h/.cu`: expose the existing
  `Impl::gemm_` through `gemm()` without allocating another instance.
- `src/turbomind/turbomind.h/.cc`: expose
  `contexts_[index]->linear->gemm()` through `TurboMind::gemm(index)`.
- `lmdeploy/turbomind/weight_format.py`: construct load-storage
  `DataFormat` values directly in the existing `make_data_format` method as
  shown above; `FP8Format` and `MXFP4Format` supply their format-specific
  overrides.
- `lmdeploy/turbomind/builders/_base.py`: retain the first device's
  engine-owned `Gemm` reference and the mapped `gemm_input_dtype` in
  `Context`, read the latter
  directly while building queries through the first-device instance, return
  one plan per linear query, create every production linear through the
  existing generic factory, and call `set_plan` on every active device replica
  before parent attachment.
- `lmdeploy/turbomind/model_loader.py`: populate `Context.gemm` from
  `model_comm.gemm(0)` after the engine contexts have been created, map the
  engine config's `gemm_input_dtype` string to `_tm.DataType`, and pass it to
  `Context`.
- `lmdeploy/turbomind/converter.py`: resolve the data type by trialing each
  candidate dtype against every weight format through a temporary
  first-device `Gemm` under `torch.cuda.device(devices[0])`; retain the
  existing nested publisher-config lookup and `dtype`-before-`torch_dtype`
  preference in `_resolve_dtype`, replace that function's architecture check
  with the executable set, and reuse `_build_resolver` for separate text and
  vision passes. Remove the U4 architecture heuristic and destroy each
  temporary `Gemm` when `_build_resolver` returns.
- `lmdeploy/turbomind/models/qwen3_5.py`: bind the text and vision source
  models with separate `Context` values carrying their respective resolver
  dtypes. Delete `Qwen3_5VisionModel._restore_dtype` and every call to it;
  construct the existing vision builders normally with the vision context.
- `tests/turbomind/linear/linear.py`: replace the removed binding call with
  the complete `DataFormat` constructor matching the test's requested
  weight type and group size; require `plan` in `Weight`, retain the returned
  gate/up block width, construct an owning `LinearWeight(config)`, immediately
  call `set_plan`, reuse the first expert's values in `link_experts`, and
  remove the three superseded policy methods.
- `tests/turbomind/linear/fixture.py`: own one `self.gemm`, query plans in
  `_allocate_weight_triple` before module construction, pass `grouped=True`
  from the MoE path, mutate the quantized plan with `gate_up`, and use its
  returned block width for arrangement and reference output handling as shown
  above. Import the existing lazy `_tm` accessor from `.linear` for those
  calls. Raise `NotImplementedError` when an explicitly fused case receives
  block `0`; the existing constructor handler closes the fixture before
  re-raising.
- `tests/turbomind/linear/test_linear.py`: catch `NotImplementedError` around
  `LinearFixture` construction and call `pytest.skip` after cleanup.
- `tests/turbomind/linear/benchmark.py`: catch `NotImplementedError` around
  `LinearFixture` construction, log the unsupported case and reason, and
  continue to the next case group.
- `tests/turbomind/linear/reference.py`: remove `fused_silu_block` and its
  fixed format/hardware block table.
- `src/turbomind/core/test_data_format.cc`: construct the existing test
  descriptors with the `DataFormat` constructor instead of calling the
  removed resolver; retain the existing `is_quantized`, rank, block-size,
  scale, and zero assertions.
- `lmdeploy/turbomind/builders/ffn.py`: consume only the returned gate/up
  arrangement; remove hardware/format fusion policy and the separate w1/w3
  fallback (shape-free queries make it vacuous). `_pad_ffn_for_tp` takes
  the selected family's `align_k`/`align_n` and `min_k`/`min_n` instead of
  its `_GEMM_K_ALIGN` kernel constant and loses the `tp <= 1` early exit.
  It does not consume `gate_up_block`; `gate_up()` runs after mandatory
  padding and falls back to unfused when that width does not tile.
- `src/turbomind/models/linear_weight.h/.cc`: add the owned optional `plan_`
  and one-argument `set_plan` exactly as shown; require and execute the plan in
  `prepare`, stop selecting converters, and copy `family` in
  `copy_metadata_to`.
- `src/turbomind/models/ffn_weight.cc`: stop changing formats and epilogues.
- `src/turbomind/models/llama/LlamaLinear.cu`: consume planned formats,
  convert activations through the family's `ConvertInput`, and retain
  runtime-only addressing adaptation. The SM100 cuBLAS-grouped gate at
  :96 is deleted — the routed-row gather decision is static, driven by
  `weight.family->indexed_input()`; `quant_b` continues to come from
  `MakeQuantDesc(weight.weight_format)`. `operation.family` is always set to
  `weight.family->id`; output dtype never disables the family restriction.
  The epilogue equality tests in output sizing become membership tests
  once `Epilogue` is a mask (`weight.epilogue == Epilogue::kGatedSilu` →
  `& Epilogue::kGatedSilu`).
- the FFN builder: pass `grouped=self.config.is_expert`, mutate the w1w3 plan
  through `gate_up`, and query an independent plan for w2.
- `src/turbomind/models/moe_weight.cc`: link experts without format or
  descriptor cross-checks; expert metadata is identical by construction,
  and the existing `copy_metadata_to` call carries `family` into each
  linked view.

Remove during migration:

- `ResolveLinearWeightFormat` and its Python binding after every
  `WeightFormat` and existing test caller constructs `DataFormat` directly.
- `ConverterRequest`, `ConverterSet`, and `GetConverters`.
- `fuse_w1w3`, `_should_fuse_silu`, `_fused_silu_block`, the Python SM
  checks, and the fixed 64/128 fusion constants in `ffn.py`.
- The Python architecture and metadata checks driving data-type choice:
  `_resolve_dtype`'s `is_bf16_supported` policy, `use_native_sm90_u4`
  (device capability, `has_sm90_mixed_kernel`, group size),
  `_U4_MODEL_FORMATS`, and the blanket FP16 forcing in `_build_resolver`.
- `Qwen3_5VisionModel._restore_dtype` and all of its call sites; the vision
  source model receives a context with its resolved dtype instead.
- `DeriveActivationFormats` and `set_fp8_fused_silu_output`
  (`linear_weight.cc`), and the SM90 fused-eligibility filter in
  `FfnWeight::prepare` that rewrites child input formats and epilogues.

## Implementation order

1. Add and bind both `DataFormat` constructors shown above, switch
   every `WeightFormat.make_data_format` implementation and existing test
   caller to that constructor, then remove
   `ResolveLinearWeightFormat` and its binding.
2. Add the concrete `Family` objects in catalog ID order, including
   the four cuBLAS objects. Put each object beside its registrations. Share
   repeated support and pack behavior through local function templates, and
   write unique callbacks directly in the object definition. Move each
   current source-to-target conversion into the pack callback and produce the
   matching `WeightBridge` in the support callback (scale/zero replication
   and scale/zero dtype conversion).
3. Split the existing mixed registration callbacks along the catalog
   boundaries, bind one catalog family to each `Registrar`, and update
   `Collector` and the kernel constructors so every registered kernel receives
   that bound family.
4. Implement `Gemm::PlanWeight`, `Gemm::DataTypes`, and the returned
   `GemmPlan::pack` operations.
5. Expose each engine context's existing `LlamaLinear::Impl::gemm_` through
   `model_comm.gemm(index)`, store only `model_comm.gemm(0)` in
   `Context.gemm`, bind the query/result and `LinearWeight::set_plan`, and
   switch generic `_add_linear` to query through that single planning
   instance, create, and attach each linear's plan to every active device
   replica before parent attachment. Switch the linear harness to its
   fixture-owned `Gemm`, owning `LinearWeight(config)` constructor, and
   immediate `set_plan` call before removing the superseded policy setters.
6. Switch FFN gate/up arrangement and fused-SiLU selection to the block width
   returned while mutating the w1w3 plan, keeping the w2 query independent.
7. Make `LinearWeight::prepare` execute the attached plan's `pack` and
   `FfnWeight::prepare` recursion-only, then remove the superseded selectors.
8. Restrict runtime dispatch with `GemmDesc.family`, complete runtime
   activation conversion (`ConvertInput` implementations), and complete
   runtime-only grouped fallbacks before enabling the design for MoE.

## Verification after approval and implementation

No verification is run while this plan is unapproved. After implementation:

1. Build the affected TurboMind targets in the existing `build` directory.
2. Verify one family instance per family at registry construction and that
   `Registry::families()` contains each surviving family pointer exactly once;
   verify every registrar binds exactly one family and every kernel collected
   from it references that family. Verify all 38 catalog rows use their exact
   permanent IDs and that every active kernel registration appears in exactly
   one row;
   both `PlanWeight` and `DataTypes` iterate that collection without scanning
   `Registry::kernels()`. Verify
   each family's `supports` override calls the base, verifies its
   constraints, and answers with a `WeightBridge` naming exactly the
   re-expressions its `Pack` performs — scale/zero replication and
   scale/zero dtype conversion — and `std::nullopt` for anything `Pack`
   cannot re-express losslessly.
3. Verify query results with the knob unset, force-plain (`=0` skips every
   family requiring non-trivial packing), and force-packed (`=1`),
   `kNull` default ordering by priority (FP8 weight
   selecting W8A8 where a W8A8 family is registered, weight-only otherwise),
   preference behavior of an explicit input type (FP8 weight preferring
   W8A16 under an explicit BF16 request even where W8A8 exists; an explicit
   FP8 request preferring W8A8 where registered and still serving W8A16
   where not — the query never fails on the preference alone),
   the `gemm_input_dtype` knob reaching the query,
   the data-type contract (an FP16 model never selects a BF16-input
   family, even under `kNull`; a W8A8 family whose BF16 unfused output
   mismatches an FP16 data type is rejected),
   shape-free queries (the combined w1w3 linear is always built; w1w3 and w2
   use two separate plans from identical queries, deterministically select the
   same family, and only the w1w3 plan is mutated by `gate_up()`),
   `gate_up()` validation (a fused offer degrades to unfused when the padded
   projection width does not tile the fused block),
   loader padding confined to the FFN intermediate axis (w1/w3 output and
   w2 input padded alike, to `tp`, format-block, base-family-alignment, and
   minimum-extent requirements, no kernel constants and no fusion-block
   requirement) with every other axis hard-checked — a weight that misses
   either base requirement is a load error naming the family and the
   requirement,
   `ConvertInput` pass-through/quantization with no padding or slicing,
   and ambiguous priority rejection.
4. Verify data-type resolution by enumeration: BF16 AWQ yields
   `{bf16, fp16}` where both U4 families are registered (SM90) and `{fp16}`
   where U4 families are FP16-only (SM80); `'auto'` descends through
   `text_config` or `llm_config`, prefers `dtype` over `torch_dtype`, and
   takes that publisher dtype when it is in the set (a BF16-published
   checkpoint takes BF16, an FP16-published one FP16, on SM90), choosing BF16
   otherwise; an
   explicit request outside the executable set downgrades to FP16
   with a warning or fails the load when FP16 is also absent. Verify the
   text pass uses the checkpoint format set, the vision pass uses only
   `TrivialFormat`, and both pass the original user request through the same
   `_resolve_dtype` publisher-config traversal. Verify the aggregate model
   binds text and vision with contexts carrying their respective resolver
   dtypes, and that `Qwen3_5VisionModel._restore_dtype` and all of its calls
   are absent.
   Verify no local architecture check or cross-node architecture exchange is
   added; equal architecture across all component devices remains an unchecked
   deployment precondition.
   Verify the resolver enters `devices[0]` explicitly (never the
   ambient current device), creates one temporary `Gemm` per resolver pass,
   and releases each before `TurboMind.create`. Verify model loading creates no additional
   planning instance: `Context.gemm` is a reference to `model_comm.gemm(0)`,
   which is the `Gemm` already owned by the first engine context's
   `LlamaLinear`. Verify each linear query returns one `GemmPlan`,
   that plan is attached to every active device replica of that linear,
   different linears may hold different plans, and `pack` runs under each
   module's current device and stream without a captured-device check.
   Verify generic `create_module(config)` remains unchanged, every
   loader-created linear immediately receives `set_plan(plan)`, and
   `prepare()` requires `plan_` before packing. Verify default-constructed MoE
   linked views are created only after child preparation, receive packed
   metadata through `copy_metadata_to`, and do not execute their own
   `prepare()`.
   Verify the linear fixture's one `Gemm` serves every plan query, and verify
   dense, grouped, fused-SiLU, and linked-expert harness weights are all
   created with their selected plan.
5. Verify every plan's `Pack` emits the packed descriptors its family format
   requires. Verify every concrete family has a fixed nonzero `id`, registry
   construction rejects duplicate ids, every planned linear call copies that
   id to `GemmDesc.family`, runtime filtering admits only kernels with the same
   nonzero id, and dispatch-cache export/import preserves the restriction.
   Verify both direct cuBLAS
   `is_feasible` overrides reject a different nonzero family id, the FP16 and
   BF16 planned paths admit only their matching specialization, BF16 is not
   enumerated through cuBLAS on SM70/SM75, ordinary queries do not select the
   FP32-output families, and an MoE gate query selects family 104 or 105 and
   executes with that exact id.
6. Verify gate/up tensor arrangement for the unfused arrangement and
   64-wide and 128-wide blocks, including scales, zeros, and bias. Verify an
   explicitly fused harness case raises `NotImplementedError` before weight
   generation when `gate_up` returns block `0`, constructor cleanup runs, the
   pytest caller reports a skip, and the standalone benchmark logs a warning
   before continuing to the next case group.
7. Verify `ConvertInput`: pass-through for floating-point-input families,
   BF16-to-dynamic-FP8 quantization for the W8A8 family, dynamic-FP8
   pass-through with scales, and a fused FP8 gate/up output feeding the
   W8A8 down family directly.
8. Verify dense and MoE runtime paths: every non-cuBLAS grouped family uses
   the in-kernel indexed path, while grouped cuBLAS uses the gather path
   (`invokeMoeDispatch` + `invokeMoeDispatchScales`, blocked grouped GEMM
   over the `offsets` descriptor). Verify both produce the same outputs and
   a `grouped` query never selects a family without grouped kernels.
9. Run `scripts/test_turbomind_model.py` unchanged with responses of at
   least 128 tokens and confirm the response is meaningful and relevant.
   Check GPU availability before every GPU command.

## Approval boundary

Approval of this document would authorize implementation of this design and
its listed verification only. Until explicit approval, the plan remains the
only modified file.
