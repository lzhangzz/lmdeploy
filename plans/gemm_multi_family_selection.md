# SM90 BF16 dense cuBLAS tuning candidate

## Status

**Draft. Do not implement without explicit approval.**

The scope is deliberately narrow: make the existing dense BF16 cuBLAS kernel
available beside the native kernels in SM90 BF16 family 27. cuBLAS continues
to reject grouped GEMM and every non-empty epilogue.

This plan does not add secondary-family metadata or general multi-family
selection.

## Resulting family

Family 27 remains the only owner of SM90 BF16 planning, packing, padding, and
gate/up arrangement. Its kernel list becomes:

```text
family 27:
    native dense SM90 BF16 kernels
    native grouped SM90 BF16 kernels
    native fused-SiLU SM90 BF16 kernels
    existing dense cuBLAS kernel
```

The dense cuBLAS kernel is feasible only for flat, non-quantized GEMM with
`Epilogue::kNone`. Therefore:

```text
dense BF16, no epilogue:      native SM90 or cuBLAS
dense BF16, gated SiLU:       native SM90 only
grouped BF16, any epilogue:   native SM90 only
```

Family 101 remains the source-preserving dense BF16 cuBLAS family used when
family 27 is unavailable or weight packing is disabled.

No fields are added to `Family`, `Operation`, `GemmDesc`, `GemmPlan`,
`LinearWeight`, or `LlamaLinear`.

## Build the existing SM90 BF16 family

Enable the existing translation unit in
`src/turbomind/kernels/gemm/CMakeLists.txt`:

```diff
     kernel/sm90_64n16_e4m3.cu
-#    kernel/sm90_64n16_16.cu
+    kernel/sm90_64n16_16.cu
```

The source already excludes its contents when compiled with CUDA older than
12.3. The `gemm2_sm90` target is already created only when SM90 code is being
built.

## Kernel-owned availability

Registrar callbacks declare kernels; they do not inspect the runtime device.
Remove the architecture parameter from `RegisterFn`:

```cpp
using RegisterFn = std::function<void(Collector&)>;
```

`Registry` invokes each callback without an architecture:

```cpp
for (auto& [family, register_fn] : gKernelFactories()) {
    Collector collector{*family};
    register_fn(collector);
    for (auto& kernel : collector.release()) {
        Add(std::move(kernel));
    }
}
```

Add a kernel availability contract. The base implementation retains the
existing compiled-architecture behavior:

```cpp
// kernel.h
virtual bool is_available(int arch) const noexcept;
```

```cpp
// kernel.cu
bool Kernel::is_available(int arch) const noexcept
{
    return is_arch_compatible(desc_.arch, arch);
}
```

`Registry::Add` uses that contract instead of calling
`is_arch_compatible` itself:

```cpp
if (!kernel->is_available(arch_)) {
    is_valid = false;
}
```

Every registrar lambda under `src/turbomind/kernels/gemm` drops its unused
`int arch` parameter. Its registration body otherwise remains unchanged:

```diff
-Registrar reg(family, [](Collector& c, int /*arch*/) {
+Registrar reg(family, [](Collector& c) {
```

## Reuse the dense cuBLAS kernel

Keep `CublasKernel` implemented in `cublas.cu`. Expose only a registration
function and its non-null default availability predicate:

```cpp
// cublas.h
#pragma once

namespace turbomind::gemm {

class Collector;

inline bool always_available(int)
{
    return true;
}

void add_cublas(Collector& collector,
                bool (*available)(int) = always_available);

}  // namespace turbomind::gemm
```

Allow `Collector::add` to forward constructor arguments to concrete `Kernel`
classes. Configuration-based registrations continue to accept no arguments:

```cpp
template<class T, class... Args>
void add(Args&&... args)
{
    if constexpr (std::is_base_of_v<Kernel, T>) {
        kernels_.emplace_back(
            std::make_unique<T>(family_, std::forward<Args>(args)...));
    }
    else {
        static_assert(sizeof...(Args) == 0);
        kernels_.emplace_back(
            std::make_unique<KernelImpl<typename T::Kernel>>(family_));
    }
}
```

`CublasKernel` stores and applies the predicate without changing
`KernelDesc::arch`:

```cpp
CublasKernel(const Family& family, bool (*available)(int)):
    Kernel{family}, cublas_{}, available_{available}
{
    // Existing constructor body remains unchanged.
}

bool is_available(int arch) const noexcept override
{
    return Kernel::is_available(arch) && available_(arch);
}

bool (*available_)(int);
```

The registration function encapsulates the private kernel class:

```cpp
void add_cublas(Collector& collector, bool (*available)(int))
{
    collector.add<CublasKernel>(available);
}
```

Define the BF16 availability predicate once and reuse it for families 101 and
105:

```cpp
bool bf16_available(int arch)
{
    return arch >= Sm80::value;
}
```

The existing dense cuBLAS registrations become declarative:

```cpp
Registrar reg[]{
    {dense_f16, [](Collector& c) {
         add_cublas(c);
     }},
    {dense_bf16, [](Collector& c) {
         add_cublas(c, bf16_available);
     }},
    {f16_f32, [](Collector& c) {
         add_cublas(c);
     }},
    {bf16_f32, [](Collector& c) {
         add_cublas(c, bf16_available);
     }},
#if defined(ENABLE_CUBLAS_GROUPED)
    {grouped_f16, [](Collector& c) {
         c.add<CublasGroupedKernel>();
     }},
    {grouped_bf16, [](Collector& c) {
         c.add<CublasGroupedKernel>();
     }},
#endif
};
```

Include `cublas.h` from `kernel/sm90_64n16_16.cu`. Its registrar adds an SM90
available instance to family 27:

```cpp
Registrar reg(bf16, [](Collector& c) {
    add_cublas(c, Sm90::is_compatible);

    // Existing native family-27 registrations remain unchanged below.
});
```

`Collector` constructs every kernel with the collector's family. The SM90
call therefore creates a family-27 `CublasKernel`; the calls in `cublas.cu`
continue to create family-100, family-101, family-104, and family-105
instances. Planning sees a family only when at least one of its kernels passes
kernel-owned availability. No alternate eligibility rule is needed:

```cpp
if (desc.family && desc.family != family().id) {
    return false;
}
```

## Layout and epilogue behavior

Do not add another cuBLAS launch path. The existing dense implementation
already derives `CUBLAS_OP_N` or `CUBLAS_OP_T` from the actual operand
descriptors and requires column-major output. The registry already creates a
`TransposedKernel` for row-major output.

For the normal `LlamaLinear` call, family 27 publishes:

```cpp
MatrixLayout Bdesc{kBfloat16, kColMajor, K, N, K};
MatrixLayout Ddesc{kBfloat16, kRowMajor, M, N, N};
```

The transposed wrapper presents the existing cuBLAS kernel with:

```text
A := transpose(Bdesc), logical N x K, row-major
B := transpose(Adesc), logical K x M, column-major
D := transpose(Ddesc), logical N x M, column-major
```

This computes `D^T = B^T A^T`, which is the transpose of the requested
`D = A B`. The family-27 weight contains ordinary BF16 values and has no
operand pack tag, so the existing cuBLAS feasibility checks accept it.

Keep these existing hard rejections unchanged:

```cpp
if (std::tie(desc.striding_a, desc.striding_b, desc.striding_c)
    != std::tuple{Striding::kFlat, Striding::kFlat, Striding::kFlat}) {
    return false;
}
if (desc.epilogue != Epilogue::kNone) {
    return false;
}
if (desc.num > 1 || desc.group_axis >= 0) {
    return false;
}
```

No temporary output, activation kernel, grouped cuBLAS change, or native SM90
mainloop change is part of this plan.

## Tuning behavior

Do not change estimates, `Find`, `Context::Populate`, `Sampler`, clustering,
or `top_k`.

The default tuner setting `top_k=0` already returns every feasible cluster.
`Sampler` first measures one leader from every cluster. Its `clusters` setting
only controls which clusters have their remaining members measured; it does
not remove the measured leaders. The dense cuBLAS kernel is therefore in the
tuning set and remains eligible to win once it is registered in family 27.

On an untuned cache miss, the existing backend launch spec has estimated costs
`{0, 0}`. The existing `Find(..., top_k=1)` path therefore selects cuBLAS
before native family-27 kernels. This is accepted behavior for the tuning
candidate design; the plan does not impose native-first untuned dispatch.

An explicitly positive `top_k` continues to perform its existing global
preselection. This plan does not override that user-selected pruning policy.

## Kernel cache identity

The same `CublasKernel` implementation is now instantiated for multiple
families. Record the bound family in `KernelDesc` so cache import restores the
correct instance:

```cpp
struct KernelDesc {
    int       arch;
    uint32_t  family;
    OpClass   op_class;
    uint32_t  algo;
    Order     raster;
    DataType  type_a;
    DataType  type_b;
    DataType  type_c;
    Order     order_a;
    Order     order_b;
    Order     order_c;
    Striding  striding_a;
    Striding  striding_b;
    Striding  striding_c;
    Pack      pack_a;
    Pack      pack_b;
    Pack      pack_u;
    Pack      pack_v;
    QuantDesc quant_a;
    QuantDesc quant_b;
    int       policy_a;
    int       policy_b;
    int3      cta_tile;
    int3      mma_tile;
    int3      atom_layout;
    int2      cluster_shape;
    int3      align;
    int2      c_tile;
    int       stages;
    bool      split_k;
    Epilogue  supported_epilogues;
    int       group_axis;
    int       backend;
    bool      transpose;
};
```

Make the base constructor out of line because `Family` is forward-declared in
`kernel.h`:

```cpp
// kernel.h
explicit Kernel(const Family& family);
```

```cpp
// kernel.cu, where Family is complete
Kernel::Kernel(const Family& family): family_{family}, desc_{}, info_{}
{
    desc_.family = family.id;
}
```

`transpose(KernelDesc)` copies the family unchanged. Kernel descriptor
equality includes it immediately after `arch`:

```cpp
return std::tie(d.arch,
                d.family,
                d.op_class,
                d.algo,
                d.raster,
                d.type_a,
                d.type_b,
                d.type_c,
                d.order_a,
                d.order_b,
                d.order_c,
                d.striding_a,
                d.striding_b,
                d.striding_c,
                d.pack_a,
                d.pack_b,
                d.pack_u,
                d.pack_v,
                d.quant_a,
                d.quant_b,
                d.policy_a,
                d.policy_b,
                d.cta_tile,
                d.mma_tile,
                d.atom_layout,
                d.cluster_shape,
                d.align,
                d.c_tile,
                d.stages,
                d.split_k,
                d.supported_epilogues,
                d.backend,
                d.transpose,
                d.group_axis);
```

Cache export and import otherwise remain unchanged. Increase
`kDispatchCacheVersion` from 5 to 6 because `KernelDesc` changes.

## Runtime evidence

Use the existing `TM_GEMM_TUNE_VERBOSE` switch to enable the currently
commented dispatch line. Include the bound family and backend so cache-miss and
cache-import behavior can be verified directly:

```cpp
if (std::getenv("TM_GEMM_TUNE_VERBOSE")) {
    std::cout << "[Gemm] dispatch: " << spec.kernel->name()
              << " family=" << spec.kernel->desc().family
              << " backend=" << spec.kernel->desc().backend
              << " split_k=" << spec.splits
              << " swizzle=" << spec.swizzle << std::endl;
}
```

This does not change the existing `[tune]` line format or its analysis
scripts.

## Verification after approval

1. Build with `ninja` from `build/` without setting `PYTHONPATH`.
2. Verify the registrar migration statically:

   ```bash
   rg -n 'Collector&[^)]*,[[:space:]]*int' \
       src/turbomind/kernels/gemm
   ```

   It must find no registrar callback retaining the removed architecture
   parameter. Inspect every changed `Registrar` declaration and confirm its
   registration body is otherwise unchanged. Do not reconfigure the build for
   additional CUDA architectures.

   Residual risk: the SM70, SM75, and SM80 registrar signature edits are
   verified by source inspection only. They are not compiled by the configured
   `90a-real` build, per the explicit decision not to reconfigure additional
   architectures. The configured build type-checks the shared `RegisterFn`
   interface, registry invocation, cuBLAS registrars, and SM90 registrars.
3. Before every GPU command, run `get_gpu_usage` and select an empty SM90 GPU.
   Run every benchmark and model-test command outside the sandbox. Before the
   runtime commands, set:

   ```bash
   export EMPTY_SM90_GPU=<physical GPU ID reported by get_gpu_usage>
   export CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU"
   export PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm
   ```

4. In a fresh process outside the sandbox, export an untuned family-27 cuBLAS
   entry while validating its output:

   ```bash
   env -u TM_GEMM_WEIGHT_PACK \
       -u TM_GEMM_TUNE \
       -u TM_GEMM_IMPORT \
       -u TM_GEMM_EXPORT \
       TM_GEMM_TUNE_VERBOSE=1 \
   python -m tests.turbomind.linear.bench_linear \
       --suite full \
       --case llama2_7b_o__bf16_bf16_bf16 \
       --batch 1 \
       --tp 1 \
       --ep 1 \
       --exact-parallel \
       --iters 0 \
       --export /tmp/gemm27.records
   ```

   The run must pass output validation and report `family=27 backend=1`. The
   benchmark emits exactly:

   ```text
   /tmp/gemm27.records.llama2_7b_o__bf16_bf16_bf16__tp1__ep1
   ```
5. In a fresh process outside the sandbox, tune the dense case without running
   validation first:

   ```bash
   env -u TM_GEMM_WEIGHT_PACK \
       -u TM_GEMM_IMPORT \
       -u TM_GEMM_EXPORT \
       TM_GEMM_TUNE='top_k=0' \
       TM_GEMM_TUNE_VERBOSE=1 \
   python -m tests.turbomind.linear.bench_linear \
       --suite full \
       --case llama2_7b_o__bf16_bf16_bf16 \
       --batch 1 \
       --tp 1 \
       --ep 1 \
       --exact-parallel \
       --tune \
       --no-validate \
       --iters 0
   ```

   In `[tune]` lines, a cuBLAS kernel name begins `sm0_` and a native SM90
   kernel name begins `sm90_`. Confirm that both prefixes were measured.
6. In another fresh process outside the sandbox, tune the gated-SiLU case:

   ```bash
   env -u TM_GEMM_WEIGHT_PACK \
       -u TM_GEMM_IMPORT \
       -u TM_GEMM_EXPORT \
       TM_GEMM_TUNE='top_k=0' \
       TM_GEMM_TUNE_VERBOSE=1 \
   python -m tests.turbomind.linear.bench_linear \
       --suite full \
       --case llama2_7b_gate_up__bf16_bf16_bf16__fuse_silu \
       --batch 1 \
       --tp 1 \
       --ep 1 \
       --exact-parallel \
       --tune \
       --no-validate \
       --iters 0
   ```

   Confirm that `[tune]` lines contain native `sm90_` kernels and no `sm0_`
   kernel.
7. In another fresh process outside the sandbox, tune the grouped case:

   ```bash
   env -u TM_GEMM_WEIGHT_PACK \
       -u TM_GEMM_IMPORT \
       -u TM_GEMM_EXPORT \
       TM_GEMM_TUNE='top_k=0' \
       TM_GEMM_TUNE_VERBOSE=1 \
   python -m tests.turbomind.linear.bench_linear \
       --suite full \
       --case mixtral_8x7b_gate_up__bf16_bf16_bf16 \
       --batch 16 \
       --tp 1 \
       --ep 1 \
       --exact-parallel \
       --tune \
       --no-validate \
       --iters 0
   ```

   Confirm that `[tune]` lines contain native `sm90_` kernels and no `sm0_`
   kernel.
8. In a fresh process outside the sandbox, validate family 101 with its own
   row-major weight. Scope `TM_GEMM_WEIGHT_PACK=0` to this command only:

   ```bash
   env -u TM_GEMM_TUNE \
       -u TM_GEMM_IMPORT \
       -u TM_GEMM_EXPORT \
       TM_GEMM_WEIGHT_PACK=0 \
       TM_GEMM_TUNE_VERBOSE=1 \
   python -m tests.turbomind.linear.bench_linear \
       --suite full \
       --case llama2_7b_o__bf16_bf16_bf16 \
       --batch 1 \
       --tp 1 \
       --ep 1 \
       --exact-parallel \
       --iters 0
   ```

   The run must pass output validation and report `family=101 backend=1`.
9. In another fresh process outside the sandbox, import the exact file emitted
   by step 4:

   ```bash
   env -u TM_GEMM_WEIGHT_PACK \
       -u TM_GEMM_TUNE \
       -u TM_GEMM_IMPORT \
       -u TM_GEMM_EXPORT \
       TM_GEMM_TUNE_VERBOSE=1 \
   python -m tests.turbomind.linear.bench_linear \
       --suite full \
       --case llama2_7b_o__bf16_bf16_bf16 \
       --batch 1 \
       --tp 1 \
       --ep 1 \
       --exact-parallel \
       --iters 0 \
       --import \
       /tmp/gemm27.records.llama2_7b_o__bf16_bf16_bf16__tp1__ep1
   ```

   The run must pass output validation and report `family=27 backend=1`,
   proving that cache import restored the family-27 instance rather than the
   family-101 instance.
10. After checking `get_gpu_usage` again, run this model smoke outside the
    sandbox. The script overwrites `CUDA_VISIBLE_DEVICES`, so pass the exact
    physical GPU selected earlier:

    ```bash
    env -u TM_GEMM_WEIGHT_PACK \
        -u TM_GEMM_TUNE \
        -u TM_GEMM_IMPORT \
        -u TM_GEMM_EXPORT \
    python scripts/test_turbomind_model.py \
        --model-id Qwen/Qwen3-8B \
        --cache-dir /mnt_cfs/huggingface_hub/hub/ \
        --gpus "$EMPTY_SM90_GPU" \
        --max-new-tokens 128 \
        --prompt "Explain why unit tests are useful in software engineering."
    ```

    Explicitly verify that the response contains meaningful human words
    relevant to the prompt.
11. Run `git diff --check`.
