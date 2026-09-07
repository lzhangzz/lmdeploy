# Unified GEMM kernel configuration and registration

## Status

Approved for implementation on 2026-09-07. Implementation has not started.

## Goal and registration contract

Use the same structural configuration and registration expression in every GEMM catalog:

```cpp
add<K<_128x256_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
```

The first three parameters of every `K` and its underlying factory's `Type` are **configuration, pipeline stages, raster order**, in that order. The remaining parameters describe variations supported by the selected implementation. They must not freeze cache policies, split-K, epilogue geometry, multicast, MMA configuration, or other existing tuning choices inside an opaque alias.

Each catalog is a free function taking the bound kernel alias as a template-template parameter `K`. Native SM90 uses `template<template<class, auto...> class K>`; SM70–SM80 use `template<template<class, int, Order, class, class, auto...> class K>` to account for their cache-policy type arguments. Bind a concrete factory's `Type` when selecting the catalog function for a `Registrar`. Individual entries use `add<K<...>>(c)` without `typename`, `template`, or `::Type`.

Factories bind architecture, data types, packed format, quantization group sizes, and the selected implementation once. They only construct types; the complete catalog stays in its free function and can be instantiated for multiple families. The W8A8 catalog uses separate `ConfigV3` and `ConfigWA` factories for activation-as-A and weight-as-A FP8. Its single existing registrar invokes the V3 catalog, then the WA catalog, passing each factory's `Type` as `K` and preserving the current order.

The named `_128x256_1x2` configuration contains tile geometry, the distribution of math workers, and exactly one active producer/math register budget. Registers are written in its definition. There is no lookup of registers by shape or striding, no second unused register pair, and no register argument on each registration line.

Other requirements:

- Use `_128x256_1x2`, with no `Tile_`, architecture, data-type, or stage prefix/suffix. Use `_128x256x32_1x8x1` where the K dimensions are variable, as in SM70–SM80.
- Stages remain independent of configuration aliases; reuse an alias across stage counts.
- Reuse one configuration alias across raster, striding, fusion, and multicast variants when its geometry and active register budget are the same.
- When budgets differ, use separate local scopes with the same short alias spelling. Each scope states its own budget explicitly.
- Every catalog registration is one physical line. Do not wrap code to fit a width limit.
- Do not add a universal option parser, a `Policy` wrapper, a second register-policy lookup, registration macros, or tag classes for individual scalar parameters.
- Do not change the registered kernel set, registration order, family IDs, priorities, packing, feasibility, scheduler behavior, or the configured architecture/source lists.
- Code in a kernel that consumes configuration constants may be adjusted to the new interface. MMA operations, copy operations, synchronization, and mainloop ordering retain their current semantics.

## Scope

The migration covers the active SM70, SM75, SM80, and SM90 catalogs and the retained SM90 sources that are currently disabled in CMake. SM100/SM120 currently use the SM80 catalogs and receive their configuration migration through those files.

| Files | Change |
| --- | --- |
| `src/turbomind/kernels/gemm/kernel/config.h` | Shared structural types defined below. |
| `src/turbomind/kernels/gemm/registrar.h` | One registration function and one host-kernel construction contract. |
| `src/turbomind/kernels/gemm/arch/config_sm70_s884.h`, `config_sm75_s16816.h`, `config_sm80_s16816.h` | Adapt existing family configuration factories to the shared configuration and parameter ordering. |
| `src/turbomind/kernels/gemm/kernel/sm70_*.cu`, `sm75_*.cu`, `sm80_*.cu`, `sm90_16816_*.cu` | Migrate registrations, including currently disabled SM90 sources. |
| `src/turbomind/kernels/gemm/kernel/sm90_64n32_8.cu` | Separate V3 and WA factories, each passed as `K` to its catalog with the common `Config, Stages, Raster` prefix. |
| `src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h` and the U4/MXFP4/NVFP4/E4M3 catalogs | Shared format-bound mixed-kernel factory. |
| `src/turbomind/kernels/gemm/kernel/sm90_64n16_16.cu` | BF16 configurations and registrations. |
| `src/turbomind/kernels/gemm/kernel/sm90_64n32_mxfp4_fp8_folded.cu`, `sm90_64n32_mxfp4_fp8_unfolded.cu` | Folded and unfolded configuration consumers. |
| The corresponding `gemm_universal_sm90_*.h` files | Read geometry, stages, registers, and optional tunables through the new compile-time parameters. |
| Existing `sm90_*traits.h` and `kernel/sm90_*_config.h` files | Remove superseded tile records; retain mathematical traits that kernels still use. |
| `src/turbomind/kernels/gemm/cublas.cu` | Route host-kernel construction through the shared registration function. |

`arch/config_simt.h` has no live catalog in this tree. Do not create a SIMT catalog or new factory for it.

## 1. Shared configuration types

Add this complete header at `src/turbomind/kernels/gemm/kernel/config.h`:

```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm::config {

template<int... Dimensions>
struct Shape;

template<int M_, int N_>
struct Shape<M_, N_> {
    static constexpr int M = M_;
    static constexpr int N = N_;
};

template<int M_, int N_, int K_>
struct Shape<M_, N_, K_> {
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;
};

template<int Producer_, int Math_>
struct Registers {
    static constexpr int Producer = Producer_;
    static constexpr int Math = Math_;
};

template<class... Parts>
struct Config;

template<class Tile_, class Groups_>
struct Config<Tile_, Groups_> {
    using Tile = Tile_;
    using Groups = Groups_;
};

template<class Tile_, class Groups_, class Registers_>
struct Config<Tile_, Groups_, Registers_>: Config<Tile_, Groups_> {
    using RegisterConfig = Registers_;
};

}  // namespace turbomind::gemm::config
```

The `config` namespace describes these shared structural types and avoids two existing names: the conversion configuration `turbomind::gemm::Config` in `convert.cuh`, and `sm80_s16816::detail`, which legacy catalogs import through their namespace directives. Do not rename or reuse the conversion configuration.

The common types have no stages, raster order, striding, architecture, data type, or derived register policy. SM70–SM80 instantiate the two-part configuration and have no dummy register data. SM90 instantiates the three-part configuration.

`Shape<M, N, K>` in the legacy path follows the existing CTA and thread-group axes of that implementation. `Shape<M, N>` in native SM90 follows the public tile axes: M is batch, N is output. Preserve the existing transpose inside weight-as-A traits; do not exchange the configuration's axes to match WGMMA operands.

The aliases live inside catalog functions or other local scopes. Import the shared types used by a catalog once at the start of its registration function, then use the short names in every configuration alias. Keep these explicit using-declarations inside the function so `Config` resolves to the shared structural type even when the enclosing GEMM namespace contains the conversion `Config`:

```cpp
template<template<class, auto...> class K>
void register_kernels(Collector& c)
{
    using config::Config;
    using config::Registers;
    using config::Shape;

    {
        using _128x256_1x2 = Config<Shape<128, 256>, Shape<1, 2>, Registers<72, 216>>;
        add<K<_128x256_1x2, 3, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_128x256_1x2, 4, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_128x256_1x2, 3, kColMajor, Striding::kBlocked, true>>(c);
    }
    {
        using _128x256_1x2 = Config<Shape<128, 256>, Shape<1, 2>, Registers<120, 192>>;
        add<K<_128x256_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_128x256_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);
    }
}
```

This illustrates independent tunables; it is not permission to add these combinations to the production catalog. Migrate only existing entries.

## 2. One registration operation

`K<...>` resolves through the bound factory's `Type` to the concrete host kernel implementing `gemm::Kernel`. The legacy result is `KernelImpl<GemmUniversal<...>>`; native results use their existing `KernelImplSm90*` classes. Host launch wrappers remain responsible for their existing launch ABIs.

In `registrar.h`, replace `Collector` with this definition and add the shared free function. Keep `RegisterFn`, `gKernelFactories`, and `Registrar` as they are. The header needs `<memory>`, `<utility>`, and `<vector>` for this code; it no longer needs the legacy `kernel_impl.h` include or the `is_base_of` branch.

```cpp
class Collector {
public:
    explicit Collector(const Family& family): family_{family} {}

    template<class T, class... Args>
    void add(Args&&... args)
    {
        kernels_.emplace_back(std::make_unique<T>(family_, std::forward<Args>(args)...));
    }

    std::vector<std::unique_ptr<Kernel>> release()
    {
        return std::move(kernels_);
    }

private:
    const Family& family_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
};

template<class T, class... Args>
void add(Collector& c, Args&&... args)
{
    c.add<T>(std::forward<Args>(args)...);
}
```

All catalog entries use this free `add`. Delete `add_v3`, `add_wa`, `add_kernel`, and the mixed/folded/unfolded registration forwarding functions after their callers migrate. Include each required `kernel_impl*.h` at the factory that actually instantiates it.

cuBLAS has no exposed tile/stage configuration. It uses the same registration operation with its existing concrete host type and constructor arguments:

```cpp
void add_cublas(Collector& collector, bool (*available)(int))
{
    add<CublasKernel>(collector, available);
}
```

Change both grouped cuBLAS registrations to `add<CublasGroupedKernel>(c)`. Keep their compile guards and availability behavior, including the family-27 `add_cublas(c, Sm90::is_compatible)` entry.

## 3. SM70–SM80 family factories

Adapt the existing `Config_*` factory names and replace their old catalog API.

The existing builders `Sm70_s884`, `Sm75_s16816`, and `Sm80_s16816` continue to construct their MMA, mainloop, epilogue, and scheduler. Each builder maps `Config::Tile` and `Config::Groups` once. All its family factories reuse that mapping.

Remove `raster_order` from each builder's outer template parameter list and move it to the third parameter of its nested `Type`. Its `Kernel` alias now denotes the existing host wrapper. Include `kernel_impl.h` in these configuration headers rather than obtaining it indirectly from `registrar.h`.

For SM80, the complete replacement nested `Type` is below. `SMEM_M`, `SMEM_N`, `SMEM_K`, `MODE_A`, `MODE_B`, `MODE_C`, `Arch`, operand/transform types, `mma_iter_order`, `order_C`, `Tc`, and `group_axis` are the existing outer builder members/parameters. The existing outer definition continues to provide them.

```cpp
template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int GroupSizeU = 1, int GroupSizeV = 1, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true>
struct Type {
    using Tile = typename Config_::Tile;
    using Groups = typename Config_::Groups;

    static constexpr int CTA_M = Tile::M;
    static constexpr int CTA_N = Tile::N;
    static constexpr int CTA_K = Tile::K;

    using Partition = Blocked<Groups::M, Groups::N, kColMajor>;
    using MMA_Map = gemm::MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, Groups::K>;
    using MMA = Tiled_MMA_v2<SM80_MMA_16x8x16_F32_F16_F16_F32_TN<Dtype>, MMA_Map, mma_iter_order>;
    using Mainloop = MainloopSm80_v2<MMA, A, IteratorSm80<MODE_A, PolicyA>, TransformA, U, GroupSizeU, B, IteratorSm80<MODE_B, PolicyB>, TransformB, V, GroupSizeV, Stages, FusePrefetch>;

    static constexpr int CHUNK_K = std::lcm(std::lcm(GroupSizeU, GroupSizeV), CTA_K);
    using Scheduler = SchedulerSm70<Raster, CTA_M, CTA_N, CTA_K, CHUNK_K, SplitK, group_axis>;

    static constexpr int TILE_C_M = EpiM == -1 ? CTA_M : EpiM;
    static constexpr int TILE_C_N = EpiN == -1 ? CTA_N : EpiN;
    using Epilogue = gemm::Epilogue_<Tc, CTA_M, CTA_N, TILE_C_M, TILE_C_N, MMA::kThreadCount, Rearrange<MMA>, Operand_C<float, order_C>, MODE_C, SplitK>;
    using Kernel = KernelImpl<GemmUniversal<Arch, Mainloop, Epilogue, Scheduler>>;
};
```

SM70/SM75 use this same parameter prefix and geometry mapping, retaining their existing `MainloopSm70` and MMA definitions. Their nested `Type` has no `FusePrefetch` argument; keep the current fixed `true` passed to `MainloopSm70`. Their final aliases are respectively `KernelImpl<GemmUniversal<Sm70, Mainloop, Epilogue, Scheduler>>` and `KernelImpl<GemmUniversal<Sm75, Mainloop, Epilogue, Scheduler>>`.

For SM80 U4, these are the complete replacements for the current two factory aliases in `arch/config_sm80_s16816.h`. They also serve the retained SM90 s16816 U4 translation unit through the existing `Arch` parameter:

```cpp
template<class Arch, class T, int GroupSize>
struct Config_U4_d {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true>
    using Type = typename Sm80_s16816<Arch, T, kColMajor, Operand_A<half, kRowMajor>, Transform_Default, VoidOperand, Operand_B_Pack<uint4_t, kColMajor, 2>, Transform_HMMA_16816<1, 0>, Operand_UV_Pack<uint32_t, true>, kRowMajor, half, -1>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, GroupSize, EpiM, EpiN, FusePrefetch>::Kernel;
};

template<class Arch, class T, int GroupSize>
struct Config_U4_g {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true>
    using Type = typename Sm80_s16816<Arch, T, kColMajor, Operand_A<T, kRowMajor>, Transform_Default, VoidOperand, Operand_B_Pack<uint4_t, kRowMajor, 2>, Transform_HMMA_16816<1, 0>, Operand_UV_Pack<uint32_t, true>, kRowMajor, T, 0>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, GroupSize, EpiM, EpiN, FusePrefetch>::Kernel;
};
```

Bind group size in concrete factory aliases such as `Config_U4_d<Sm80, half, 32>` and `Config_U4_d<Sm80, half, 128>`, then pass each alias's `Type` to the shared `register_u4_d<K>` catalog. Replace the function's old group-size parameter with `K`; the complete ordered entry list remains shared. The grouped factory binds the grouped family's different packed B layout. Neither factory fixes raster, cache policies, split-K, stages, epilogue dimensions, or prefetch. The final section shows the legacy function declaration and bindings.

SM80 floating-point, E4M3, and MXFP4 factories follow the same interface. Keep the SMEM operand-N choice free for E4M3/MXFP4: the current `C8` and `Cg` paths share a family and must remain expressible through one factory. The complete replacements are:

```cpp
template<class Arch, class T>
struct Config_F16_g {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true>
    using Type = typename Sm80_s16816<Arch, T, kColMajor, Operand_A<T, kRowMajor>, Transform_Default, VoidOperand, Operand_B_Pack<T, kRowMajor, 1>, Transform_Default, VoidOperand, kRowMajor, T, 0>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, 1, EpiM, EpiN, FusePrefetch>::Kernel;
};

template<class Arch, class T>
struct Config_E4M3 {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true, int GroupAxis = 1, int OperandN = 16>
    using Type = typename Sm80_s16816<Arch, T, kRowMajor, Operand_A_Pack<fp8_e4m3_t, kColMajor, 1>, Transform_HMMA_16816<0, 1>, Operand_UV_Pack<uint16_t, false>, Operand_B<T, kRowMajor, OperandN>, Transform_Default, VoidOperand, kColMajor, T, GroupAxis>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 128, 1, EpiM, EpiN, FusePrefetch>::Kernel;
};

template<class Arch, class T>
struct Config_MXF4 {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true, int GroupAxis = 1, int OperandN = 16>
    using Type = typename Sm80_s16816<Arch, T, kRowMajor, Operand_A_Pack<fp4_e2m1_t, kColMajor, 1>, Transform_HMMA_16816<0, 1>, Operand_UV_Pack<uint8_t, false>, Operand_B<T, kRowMajor, OperandN>, Transform_Default, VoidOperand, kColMajor, T, GroupAxis>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 32, 1, EpiM, EpiN, FusePrefetch>::Kernel;
};
```

The constants `(128, 1)` for E4M3 and `(32, 1)` for MXFP4 above are existing packed qparameter group sizes. They are not epilogue dimensions or fixed tuning policy. The epilogue remains `EpiM, EpiN`.

Apply the same factory-to-builder argument mapping to the existing SM70/SM75 factories. Their builder `Type` ends at `EpiM, EpiN`, so do not expose a dummy prefetch parameter. Preserve these family-specific differences:

| Factory | Fixed qparameter groups U/V | Free parameters following `Config, Stages, Raster, PolicyA, PolicyB, SplitK, EpiM, EpiN` |
| --- | --- | --- |
| SM70 U4 | `1, GroupSize`, bound once for family 3 or 4 | `GroupAxis`, selecting the existing dense or grouped path. |
| SM70 floating point | `1, 1` | `GroupAxis`. |
| SM70 E4M3 | `1, 128` | `GroupAxis`. |
| SM70 MXFP4 | `1, 32` | `GroupAxis`. |
| SM75 U4 dense/grouped | `1, GroupSize`, with the current family-specific B layout | No additional parameter. |
| SM75 floating point | `1, 1` | `GroupAxis`. |
| SM75 E4M3 | `128, 1` | `GroupAxis`. |
| SM75 MXFP4 | `32, 1` | `GroupAxis`. |

For SM70 U4, `Config_U4_d` and `Config_U4_g` currently differ only in the group axis: the `GetOperand` types in the dense alias resolve to the same operands used by the grouped alias. Keep one existing factory spelling and expose `GroupAxis` on its `Type`; remove the other alias once it has no callers. This preserves both implementations in each SM70 U4 family without introducing two new configuration classes.

The complete argument order passed to their builder `Type` is:

```cpp
Config_, Stages, Raster, PolicyA, PolicyB, SplitK, GroupSizeU, GroupSizeV, EpiM, EpiN
```

`GroupSizeU` and `GroupSizeV` denote the exact constants/bound family argument in the table, not additional catalog parameters. Preserve the current iterator choice, MMA iteration order, operand packing, and transpose of each factory.

## 4. Native SM90 parameter consumption

Native device kernels consume the shared configuration directly. Do not create another tile adapter containing duplicated geometry/stages/register fields.

The following are complete replacement template declarations for the existing device classes. They specify their new parameter lists; they do not replace the existing kernel bodies with empty structs.

```cpp
template<class Config_, int Stages_, Order Raster, Striding Mode, bool Silu, int MulticastA, int MulticastB, int MaxOpN, int EpiStages_>
struct GemmUniversalSm90_v3;

template<class Config_, int Stages_, Order Raster, Striding Mode, bool Silu, int MulticastA, int MulticastB>
struct GemmUniversalSm90_Fp8Wa;

template<class Format_, class Config_, int Stages_, Order Raster, Striding Mode, bool Silu, int MulticastA, int MulticastB, int MmaN, bool SeparateMmaAtoms, int EpiM, int EpiStages_>
struct GemmUniversalSm90Mixed;

template<class Config_, int Stages_, Order Raster, Striding Mode, bool Silu, int MulticastA, int MulticastB, int L2HintW, int MmaN, bool SeparateMmaAtoms, int EpiM, int EpiStages_>
struct GemmUniversalSm90_Bf16;

template<class Config_, int Stages_, Order Raster, Striding Mode, bool Silu, int MulticastA, int MulticastB, int MmaN, int EpilogueStages>
struct GemmUniversalSm90MxFp4Fp8Folded;

template<class Config_, int Stages_, Order Raster, int MmaN>
struct GemmUniversalSm90MxFp4Fp8Unfolded;
```

Replace the corresponding class template prefixes, including forward declarations and all instantiations, in one migration. Keep tuning defaults on the factory `Type` aliases in section 5; device templates receive every argument explicitly. Remove defaults from the old device-template parameter lists.

For native 2D configurations, the directly consumed members are:

```cpp
using Tile = typename Config_::Tile;
using Groups = typename Config_::Groups;
using RegisterConfig = typename Config_::RegisterConfig;

static constexpr int TILE_M = Tile::M;
static constexpr int TILE_N = Tile::N;
static constexpr int Stages = Stages_;
static constexpr int kProducerRegs = RegisterConfig::Producer;
static constexpr int kMathRegs = RegisterConfig::Math;
```

Native BF16 retains its configurable K extent and therefore uses a 3D `Shape` and `TILE_K = Tile::K`. V3/WA/folded/unfolded keep their existing K128 constant. Mixed keeps `kSm90MixedTileK` (K64). Do not add a K tuning argument to a kernel whose copy/packing contract fixes K.

V3 and WA consume `Groups::M` and `Groups::N` as their existing `WG_M` and `WG_N`. BF16, mixed, folded, and unfolded consume the existing public-axis CuTe layout with the following exact construction:

```cpp
using WGLayout = cute::Layout<cute::Shape<cute::Int<Groups::M>, cute::Int<Groups::N>>>;
```

This CuTe construction belongs inside the kernel. Registration lines use the short configuration aliases.

For kernels with a `Mode` parameter, derive the existing grouped flag exactly once:

```cpp
static constexpr bool is_grouped_gemm = Mode != Striding::kFlat;
static constexpr Striding kStridingA = Mode;
static constexpr Striding kStridingB = is_grouped_gemm ? Striding::kBlocked : Striding::kFlat;
static constexpr Striding kStridingC = is_grouped_gemm ? Striding::kBlocked : Striding::kFlat;
static constexpr bool kIndexedGather = Mode == Striding::kIndexed;
```

Use these existing member names wherever the old template boolean was consumed. Keep the unfolded kernel dense-only.

Map the remaining common scalar parameters to the existing kernel members directly:

```cpp
static constexpr Order kRasterOrder = Raster;
static constexpr bool kSupportsFusedSilu = Silu;
static constexpr int kMulticastA = MulticastA;
static constexpr int kMulticastB = MulticastB;
```

Update uses of the removed template-parameter spellings throughout each existing body, including `Grouped`, `StridingA`, `raster_order`, and the old stage/epilogue arguments. Retain derived values such as cluster size, chunk size, and grouped multicast restrictions.

### Register budget data path

Remove `kProducerRegsTma`, `kMathRegsTma`, `kProducerRegsIndexed`, and `kMathRegsIndexed` from migrated configurations and kernels. Both producer branches use the one selected `kProducerRegs`; both math branches use `kMathRegs`:

```cpp
cutlass::arch::warpgroup_reg_dealloc<kProducerRegs>();
```

```cpp
cutlass::arch::warpgroup_reg_alloc<kMathRegs>();
```

Keep each call at its current location in the producer/math code. Do not relocate it across synchronization or merge surrounding branches as part of this refactor.

Preserve the existing register-budget assertions for the active specialization. For example, V3/WA currently require `producer + 2 * math == 504` for two math WGs, while mixed/BF16 use `<= 504`. Do not change these to one guessed universal inequality. Preserve the existing single-WG and three-WG constraints where those implementations support them.

### Other existing configuration fields

| Existing source of a constant | New source |
| --- | --- |
| V3 `Tile::kMaxOpN` | Free `MaxOpN` parameter. |
| V3 `Tile::kEpiStorageStages` | Free `EpiStages_` parameter. |
| WA `Tile::kMaxOpN` | Remove the redundant cap field: it is only an assertion bound, while the actual MMA N already equals `WG_TILE_M`. Preserve `WG_TILE_M == OP_N` and all geometry constraints. |
| Mixed/BF16 `kMmaN` | Free `MmaN` parameter; zero retains the existing per-WG default. |
| Mixed/BF16 `kSeparateMmaAtoms` | Free `SeparateMmaAtoms` parameter. |
| Mixed/BF16 `kEpiM` | Free `EpiM` parameter; zero retains the current default calculation. |
| Mixed/BF16 `kEpiPipeStages` | Free `EpiStages_` parameter; zero retains the current default calculation. |
| BF16 weight L2 hint | Free `L2HintW` parameter. |
| Folded/unfolded `kMmaN` | Free `MmaN` parameter. |
| Folded epilogue stages | Free `EpilogueStages` parameter. |

The folded kernel currently accepts `EpilogueTileM`/`EpilogueTileN` but does not use them: its actual epilogue geometry is `std::gcd(32, TILE_M)` by 128. Remove the unused arguments rather than presenting them as tunable controls. Preserve the actual geometry. This plan does not implement new epilogue tile variations.

Delete the now-unused `MixedMmaN`, `MixedSeparateMmaAtoms`, `MixedEpiM`, and `MixedEpiPipeStages` detection helpers. Parameters provide those values directly. Keep unrelated dequantization, MMA, shared-memory, and copy traits.

## 5. Complete native family factories

### SM90 W8A8

Place these two factory classes in the existing anonymous namespace of `sm90_64n32_8.cu`:

```cpp
struct ConfigV3 {
    template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1, int MaxOpN = 128, int EpiStages = 1>
    using Type = KernelImplSm90<GemmUniversalSm90_v3<Config_, Stages, Raster, Mode, Silu, MulticastA, MulticastB, MaxOpN, EpiStages>>;
};

struct ConfigWA {
    template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1>
    using Type = KernelImplSm90<GemmUniversalSm90_Fp8Wa<Config_, Stages, Raster, Mode, Silu, MulticastA, MulticastB>>;
};
```

Move the current V3 entries into `register_v3<K>` and the current WA entries into `register_wa<K>`, both in the existing anonymous namespace. Keep one W8A8 registrar that invokes them in that order. These representative entries show the complete function and binding syntax; migration retains every existing entry in each function, including the final WA blocks for two math warp groups:

```cpp
template<template<class, auto...> class K>
void register_v3(Collector& c)
{
    using config::Config;
    using config::Registers;
    using config::Shape;

    using _128x256_2x1 = Config<Shape<128, 256>, Shape<2, 1>, Registers<88, 208>>;
    add<K<_128x256_2x1, 4, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_128x256_2x1, 4, kColMajor, Striding::kIndexed, true, 1, 2>>(c);
}

template<template<class, auto...> class K>
void register_wa(Collector& c)
{
    using config::Config;
    using config::Registers;
    using config::Shape;

    using _8x128_1x1 = Config<Shape<8, 128>, Shape<1, 1>, Registers<40, 168>>;
    add<K<_8x128_1x1, 4, kRowMajor, Striding::kFlat>>(c);
    add<K<_8x128_1x1, 4, kRowMajor, Striding::kFlat, false, 1, 2>>(c);
}

Registrar reg(w8a8, [](Collector& c) {
    register_v3<ConfigV3::Type>(c);
    register_wa<ConfigWA::Type>(c);
});
```

Each `Type` directly names its concrete device implementation and host wrapper. `ConfigV3::Type` exposes the named `MaxOpN` and `EpiStages` parameters with defaults `128` and `1`; `ConfigWA::Type` ends at `MulticastB`. Registration lines use the selected `K` and pass striding as the fourth argument. All tuning arguments are named template parameters on the factory; the catalog's `auto...` accepts their differing signatures and preserves their defaults.

The existing V3 128x256 case uses `MaxOpN=128, EpiStages=1`. Existing 128x192 entries explicitly pass `192, 2`; existing 64x256 entries explicitly pass `128, 2`. Copy these values from the current catalog and traits during migration, not from the example's defaults.

### SM90 mixed U4/MXFP4/NVFP4/E4M3

Replace the forwarding function in `sm90_64n16_mixed_reg.h` with this reusable factory in its existing `detail` namespace:

```cpp
template<class Format>
struct C {
    template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1, int MmaN = 0, bool SeparateMmaAtoms = false, int EpiM = 0, int EpiStages = 0>
    using Type = KernelImplSm90Mixed<GemmUniversalSm90Mixed<Format, Config_, Stages, Raster, Mode, Silu, MulticastA, MulticastB, MmaN, SeparateMmaAtoms, EpiM, EpiStages>>;
};
```

Pass the bound factory alias into the shared free catalog. For U4, these two representative entries show the explicit MMA-N override and the defaulted indexed case, with both dtype bindings:

```cpp
template<template<class, auto...> class K>
void register_kernels(Collector& c)
{
    using config::Config;
    using config::Registers;
    using config::Shape;

    using _384x128_1x2 = Config<Shape<384, 128>, Shape<1, 2>, Registers<40, 232>>;
    add<K<_384x128_1x2, 3, kRowMajor, Striding::kFlat, false, 1, 1, 192>>(c);

    using _128x256_1x2 = Config<Shape<128, 256>, Shape<1, 2>, Registers<120, 192>>;
    add<K<_128x256_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
}

using BF16 = detail::C<Sm90U4Format<32, kBfloat16>>;
using FP16 = detail::C<Sm90U4Format<32, kHalf>>;

Registrar reg[]{
    {bf16, register_kernels<BF16::Type>},
    {f16, register_kernels<FP16::Type>},
};
```

Keep the complete existing ordered U4 catalog in this one function; replace its dtype template parameter with `K`. The factory only constructs types. No group size or compute dtype appears on each `add` line, and neither dtype requires a duplicate catalog. The other mixed catalogs bind their current `Format` to the same factory and pass its `Type` to their free catalog in the same way.

`192` is the existing MMA-N setting. It is not inferred from the alias spelling or hidden in a policy table. The indexed entry omits that argument and retains the factory default.

### SM90 BF16

Place this factory in the existing anonymous namespace of `sm90_64n16_16.cu`:

```cpp
struct C {
    template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1, int L2HintW = 0, int MmaN = 0, bool SeparateMmaAtoms = false, int EpiM = 0, int EpiStages = 0>
    using Type = KernelImplSm90Bf16<GemmUniversalSm90_Bf16<Config_, Stages, Raster, Mode, Silu, MulticastA, MulticastB, L2HintW, MmaN, SeparateMmaAtoms, EpiM, EpiStages>>;
};
```

BF16 retains its configurable K dimension through `Shape<M, N, K>`. Keep the current catalog's multicast values at `(1, 1)`; exposing the existing device-template arguments does not add catalog entries. Preserve its L2-hint entry explicitly.

Move the existing registration body into `register_kernels<K>` with the same native template-template declaration shown above, and bind it with `Registrar reg(bf16, register_kernels<C::Type>);`. Keep `add_cublas(c, Sm90::is_compatible)` as the function's first registration, followed by the existing ordered tunable entries using `add<K<...>>(c)`.

### SM90 folded MXFP4 x FP8

Place this factory in the existing folded translation unit:

```cpp
struct C {
    template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1, int MmaN = Config_::Tile::M / Config_::Groups::M, int EpiStages = 2>
    using Type = KernelImplSm90MxFp4Fp8<GemmUniversalSm90MxFp4Fp8Folded<Config_, Stages, Raster, Mode, Silu, MulticastA, MulticastB, MmaN, EpiStages>>;
};
```

Use `Registers<40, 232>` for existing flat/blocked entries and `Registers<72, 216>` for existing indexed entries. The existing 192x128 entry passes MMA N96 explicitly. Small plain-output entries with one epilogue stage pass that value explicitly. Preserve all fusion and multicast restrictions enforced by the kernel.

Move the existing catalog into `register_kernels<K>` and bind it with `Registrar reg(folded, register_kernels<C::Type>);`. The three existing 256x128 entries must explicitly pass `MmaN=128` after `MulticastA, MulticastB`; the factory default would evaluate to `256 / 1 = 256`. Preserve these entries in their current order within that function, using its local `config` imports:

```cpp
using _256x128_1x2 = Config<Shape<256, 128>, Shape<1, 2>, Registers<40, 232>>;
add<K<_256x128_1x2, 4, kRowMajor, Striding::kFlat, false, 1, 1, 128>>(c);
add<K<_256x128_1x2, 4, kRowMajor, Striding::kFlat, false, 2, 1, 128>>(c);
add<K<_256x128_1x2, 4, kRowMajor, Striding::kFlat, false, 1, 2, 128>>(c);
```

### SM90 unfolded MXFP4 x FP8

Place this factory in the retained unfolded translation unit:

```cpp
struct C {
    template<class Config_, int Stages, Order Raster, int MmaN = Config_::Tile::M / Config_::Groups::M>
    using Type = KernelImplSm90MxFp4Fp8<GemmUniversalSm90MxFp4Fp8Unfolded<Config_, Stages, Raster, MmaN>>;
};
```

It retains its existing dense-only behavior. Its sole current configuration uses `Shape<64, 128>`, `Shape<1, 2>`, `Registers<40, 232>`, stage 3, and row-major raster. Use the same native `register_kernels<K>` function declaration and bind it with `Registrar reg(unfolded, register_kernels<C::Type>);`. Do not add unsupported grouped/fusion arguments or enable its CMake source as part of migration.

## 6. Catalog migration and preservation

Preserve the complete ordered list of active kernel descriptors per family. Scope configuration aliases around existing ordered blocks; do not regroup registrations by shape if doing so changes their order.

For every existing entry, record and preserve:

1. Family and underlying device implementation, including V3 versus WA.
2. Public tile geometry, internal thread/warp-group geometry, and K geometry where variable.
3. Pipeline stages, raster, group axis or striding, split-K, fused-SiLU capability, and multicast.
4. Cache policies, prefetch, operand-N variant, MMA-N and separate-atom settings, epilogue geometry/stages, and L2 hint where used.
5. The active producer/math register counts selected by that entry's actual input path.

The two SM90 register-pair fields must not simply be copied into the new type under different names. A flat/blocked entry copies the former TMA pair; an indexed entry copies the former indexed pair. Unfolded copies only its TMA pair. The register instructions in the resulting specialization receive exactly the same integers as before.

Keep currently commented and CMake-disabled entries disabled. Convert their syntax to the new interface where they are retained as candidate documentation, but do not create configuration objects solely for unused candidates. Existing referenced mathematical traits remain; obsolete named tile aliases and tile-base records are removed after all uses migrate.

No new cache version is needed if descriptor values and ordering are preserved. C++ mangled symbol names will change; they are not dispatch-cache identity. Do not change `KernelDesc`, human-readable kernel names, or cache serialization to accommodate the new template spelling.

## 7. Implementation sequence

1. Record the current source state, configured architecture list, active kernel descriptor order, complete ordered source inventories for the SM70/SM75/SM80 catalogs, and representative compiler resource reports before modifying implementation.
2. Add the shared structural configuration header and adapt the existing family factories.
3. Update native SM90 template declarations and their constant reads together with their factory aliases. Preserve active register values and call locations.
4. Migrate the SM70, SM75, SM80, native SM90, retained SM90 s16816, and retained unfolded catalogs to free functions accepting the bound alias `K`, with `add<K<...>>(c)` on each entry. Pass each concrete factory's `Type` at its existing registrar position.
5. Route cuBLAS construction through the shared `add` and remove the old `Collector` configuration/host-type branch after the last old catalog use is gone.
6. Remove superseded tile records, registration helpers, and configuration-field detection helpers. Do not remove reusable mathematical traits.
7. Perform the verification below and report unsupported hardware or uncompiled sources explicitly.

## 8. Verification

All implementation verification in this plan is transient or uses existing tests. Do not add a permanent test file or benchmark case. Drafting has not built the project or run GPU workloads. Validate the shared structural definitions and the separate V3/WA factory interfaces with the compile checks below.

### Static and compile verification

Use the current C++17 standard. Check the complete shared definitions, representative instantiations of both `ConfigV3::Type` and `ConfigWA::Type`, V3's default and explicit `MaxOpN`/`EpiStages` values, and representative factories for the other families with the repository's actual host/CUDA compiler. Instantiate the native and legacy catalog template-template parameters with the actual factory aliases, including omitted trailing arguments, explicit overrides, and both U4 dtype/group-size bindings. A transient compile check may contain `static_assert`s for the exact geometry, stage, active registers, and mapped host-wrapper type; do not introduce APIs for those checks.

Before implementation, capture baseline descriptors and resource reports with the current build. After implementation, compare the ordered descriptors and counts per family, not just which kernel happened to win one dispatch. Use a transient native program through the existing registry accessors if the Python API does not expose the required data. Retain no production logging or test-only binding changes.

The current `90a-real` build excludes SM70/SM75/SM80 catalogs, and `Registry` filters kernels by device architecture. Independently of the runtime descriptor dump, capture a complete ordered source inventory for each of those catalogs before editing. Resolve family bindings, configuration aliases, and template defaults into the parameters listed in section 6, including architecture, data types, packing, and quantization groups. Record each entry's position within its source and family, preserve duplicate entries, and record retained disabled candidates separately with their disabled status.

After migration, derive the same inventories from the new factories and registrations and compare the complete ordered records and counts per source and family. Require identical values, ordering, multiplicity, and enabled/disabled status. This source comparison requires no legacy GPU hardware; compiling the migrated translation units is a separate check and does not establish catalog preservation.

Run static searches over the GEMM source tree to confirm:

- Every tunable catalog uses `add<K<...>>(c)`, with the concrete factory's `Type` bound at the catalog invocation. Every such `Type` starts with configuration, stages, and raster; no individual entry needs a dependent-name qualifier.
- No catalog still uses `add_v3`, `add_wa`, `add_kernel`, the old mixed forwarding function, or `c.add` directly.
- No migrated configuration contains stages or both TMA/indexed register pairs.
- No references remain to removed tile aliases, detection helpers, or old template argument orders.
- No active catalog entry, family field, CMake architecture selection, or enabled-source list changed.

Build from the existing `build` directory using `ninja`, without setting `PYTHONPATH` for the build. The current tree uses CUDA 12.8 and `90a-real`; that does not compile the edited SM70/SM75/SM80 files. Compile those affected translation units as well with the supported CUDA 12.8 SM70/SM75/SM80 targets, using the existing CMake target configuration or transient compiler invocations derived from its compile commands. Syntax-check retained disabled SM90 translation units with the SM90 target's definitions/includes without enabling their registrars in the production library. Preserve and restore the user's configured architecture list if it is temporarily changed for verification.

For representative native FP8, BF16, U4, MXFP4, NVFP4, folded, and retained unfolded specializations, compare baseline and new ptxas register counts, stack/spills, and shared-memory sizes at identical parameters. Inspect changed SASS if resource reports or correctness differ. A configuration-only refactor must not silently introduce a spill or change active register budgets.

### Existing SM90 correctness coverage

Before **each GPU command**, query `get_gpu_usage`, select an empty SM90 GPU, and run outside the sandbox. Set `PYTHONPATH=/data/lmdeploy-gemm` for Python. The physical GPU ID obtained from the usage query is supplied as `EMPTY_SM90_GPU` in these commands.

```bash
env -u TM_GEMM_TUNE -u TM_GEMM_WEIGHT_PACK CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm python -m pytest tests/turbomind/linear/test_linear.py -q -rs
```

The default smoke subset is not sufficient: it currently omits U4 and several other types. Exercise all existing type/shape combinations without timing:

```bash
env -u TM_GEMM_TUNE -u TM_GEMM_WEIGHT_PACK CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm python -m tests.turbomind.linear.bench_linear --suite full --tp 1 --ep 1 --warmup 0 --iters 0
```

Compare skips and unsupported-case warnings with the baseline. Do not treat an all-skipped family as correctness coverage. Require previously supported FP16/BF16 U4, native FP8, BF16, mixed, grouped, indexed, and fused cases to remain supported. Existing unsupported API combinations remain unsupported; this refactor must not enable registrations or add format support to make them pass.

NVFP4 has an accepted numerical-validation limitation for this refactor: `LinearFixture._weight_format()` rejects NVFP4 because the standalone Linear API does not support it, so the full suite skips every NVFP4 case. The BF16 tuner cases and Qwen3-8B model smoke below also provide no NVFP4 numerical coverage. Retain NVFP4 compilation, ordered descriptor comparison, and compiler register/spill/shared-memory comparisons. Report NVFP4 numerical correctness as unverified when completing the refactor; passing the shared mixed-kernel tests does not establish format-specific NVFP4 numerical correctness.

Run a fresh-process tuner check over existing dense and grouped cases to confirm candidate construction and validate the selected execution plan. The benchmark CLI validates before tuning, so use the existing `LinearFixture` in a transient inline check to run and validate the plan returned by `tune()`. Clear `TM_GEMM_IMPORT` explicitly so an inherited cache-import setting cannot disable tuning. After querying GPU usage again, run outside the sandbox:

```bash
env -u TM_GEMM_IMPORT -u TM_GEMM_WEIGHT_PACK CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm TM_GEMM_TUNE='top_k=0,clusters=0,swizzle=[0,1,2,3],min_iter=1,max_iter=1' TM_GEMM_VERBOSE=1 python - <<'PY'
from tests.turbomind.linear.cases import case_by_name
from tests.turbomind.linear.fixture import LinearFixture

cases = case_by_name()
for case_name in (
    'llama2_7b_o__bf16_bf16_bf16',
    'qwen3_30b_a3b_down__bf16_bf16_bf16',
    'qwen3_30b_a3b_gate_up__bf16_bf16_bf16__fuse_silu',
):
    fx = LinearFixture(cases[case_name])
    try:
        fx.prepare_batch(128)
        fx.tune()
        fx.run_reference()
        fx.run_linear()
        fx.check_tolerances(fx.compare())
        print(f'{case_name}: tuned output validation passed')
    finally:
        fx.close()
PY
```

These existing cases cover dense, blocked grouped, and indexed grouped fused execution. Run the baseline and migrated checks on the same GPU with identical problem parameters and tuner settings. `top_k=0` removes the initial candidate limit, and `clusters=0` disables the sampler's timing-dependent selection of which clusters to expand, so every supplied candidate is measured individually.

For each problem, compare the before/after multisets of `(KernelDesc, swizzle, splits)`, preserving duplicate counts while ignoring measured times and timing order. Resolve verbose kernel names through the captured registry descriptor inventory; if a name maps to distinct descriptors, use a transient native program to obtain the actual candidate descriptor records. Continue checking registration order separately through the ordered registry and source inventories. Do not change tile/stage/register choices based on the measured times in this refactor.

### Native model smoke

Query GPU usage again, then run the existing script as-is outside the sandbox:

```bash
env -u TM_GEMM_TUNE -u TM_GEMM_WEIGHT_PACK PYTHONPATH=/data/lmdeploy-gemm python scripts/test_turbomind_model.py --model-id Qwen/Qwen3-8B --cache-dir /mnt_cfs/huggingface_hub/hub/ --gpus "$EMPTY_SM90_GPU" --tp 1 --max-new-tokens 128 --prompt "Explain how matrix multiplication is used in transformer language models, with concrete examples and enough detail for a full response."
```

Confirm this model/cache entry still exists in `/data/models.json` when implementing. Inspect the generated text and require meaningful human-language output relevant to the prompt. A successful process exit alone does not establish model correctness.

## Completion criteria

- All tunable GEMM catalogs use the shared `Config`, named local geometry aliases, the common `Config, Stages, Raster` parameter prefix, and `add<K<...>>(c)` in free catalog functions.
- Each family retains every existing implementation and tunable variation through its factories and their parameters. Factories only construct types; their bound `Type` aliases select catalog instantiations. The single W8A8 registrar invokes the V3 catalog followed by the WA catalog.
- Register budgets are explicit in named configurations and contain exactly the active producer/math pair. They do not appear on individual registration lines or behind a lookup trait.
- One geometry/register configuration can be reused with different stages and other free tuning arguments.
- The family registry, packing, descriptors, active kernel set/order, and numeric computation retain their existing behavior; complete before/after source inventories match for every SM70/SM75/SM80 catalog.
- Obsolete tile records, parameter-detection helpers, and family-specific registration forwarding functions have no remaining users and are removed.
- All affected supported architecture sources compile, SM90 validation passes for cases supported by the existing harness with no new unsupported cases, compiler resource comparisons show no unexplained regression, and the native model smoke produces a relevant response.
- Completion explicitly reports the accepted NVFP4 numerical-validation limitation and the compilation, descriptor, and compiler resource checks performed for that family.

## Representative legacy catalog binding

Legacy catalogs use the same `add<K<...>>(c)` spelling, with cache policies still passed as types. The first five template-template parameters describe configuration, stages, raster, and the two cache policies; `auto...` accepts the remaining scalar parameters. Defaults remain on the selected factory's `Type`. This example shows two existing SM80 U4 dense entries; migrate the complete ordered list into the same shared function:

```cpp
template<template<class, int, Order, class, class, auto...> class K>
void register_u4_d(Collector& c)
{
    using config::Config;
    using config::Shape;

    using _128x256x32_1x8x1 = Config<Shape<128, 256, 32>, Shape<1, 8, 1>>;
    add<K<_128x256x32_1x8x1, 3, kColMajor, D, D, true, 128, 128>>(c);
    add<K<_128x256x32_1x8x1, 4, kColMajor, D, D, true, 128, 128>>(c);
}

using D32 = sm80_s16816::Config_U4_d<Sm80, half, 32>;
using D128 = sm80_s16816::Config_U4_d<Sm80, half, 128>;
```

Replace the callbacks at their existing positions in the registrar array:

| Existing family | Catalog function |
| --- | --- |
| `u4_d_32` | `register_u4_d<D32::Type>` |
| `u4_d_128` | `register_u4_d<D128::Type>` |

Apply the same pattern to the grouped catalog with its own factory and complete entry list. Preserve the array order `u4_d_32`, `u4_g_32`, `u4_d_128`, `u4_g_128`, `mxfp4`; do not move the dense families together. Existing catalogs whose entry lists differ, such as SM70 U4 group sizes 32 and 128, remain separate free functions accepting `K`.
