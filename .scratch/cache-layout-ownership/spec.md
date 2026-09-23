# Spec: Module-owned cache layouts, planner-owned page composition

Status: decided (wayfinder session, 2026-09-22). Every design decision below was settled with the owner; this spec is the implementation contract. An implementing agent should not need to make further design decisions.

ADR: `docs/adr/0001-module-owned-cache-layouts.md`. Glossary terms: `CONTEXT.md` (Cache Layout, Object Cache Plan, Page Tuning).

## Destination

Split the two concerns that `object_cache_plan.cc` currently fuses:

- **Cache layout** (module-owned): the shape a module's cache state takes. `UnifiedAttentionLayer` owns `AttentionCacheLayout`; `GatedDeltaNetLayer` owns `GDNCacheLayout`. Each module computes its layout from its own weights plus `EngineParam`, via static functions on the module class. Offsets live in the layout descriptors, never in the weights.
- **Object cache plan** (planner-owned): composing module layouts under one page size, and page tuning (selecting the page size and which module-offered GDN layout to adopt under LMCache divisibility). The planner selects among module-offered candidates; the only layout fields it writes are the adopted candidate's page-fit paddings.

`object_cache_plan.{h,cc}` survives as the planning unit only: `ObjectCachePlan`, `CacheLayouts`, `ComposeObjectCachePlan`, `TuneObjectCacheLayout`, `CommonCacheRegionSize`. `CreateDefaultAttentionCachePlan`, `CreateDefaultGdnCachePlan`, `CreateDefaultObjectCachePlan`, and the `TuneObjectCacheLayout(const ModelWeight&, const EngineParam&)` wrapper are deleted.

## Ownership rules

1. The module is the only writer of its layout's geometry and offsets. The planner calls module/static layout functions with its choices; the only layout fields it ever assigns are the page-fit paddings (`conv_part_bytes`, `recurrent_part_bytes`) of the adopted candidate — page fit is planner policy because the page size is.
2. Layout computation consumes weights + `EngineParam` before any layer instance exists, so it is exposed as static member functions on the module classes, fanned out by `LanguageModel::ComputeCacheLayouts` so no caller below the model unit knows individual modules.
3. Page tuning ranks module-offered candidates by the existing waste/fit key. The candidate enumeration (including the merge-only guard `num_blocks <= layer_count`) is module policy and moves into the GDN module.
4. `TM_GDN_BLOCK_CONFIG`, when set, is honored: the module offers exactly that one layout and tuning adapts page and padding around it. (Behavior change relative to today, where tuning overrides the pin; see Verification.)
5. `block::Layout` (`kernels/attention/block.h`) stays in kernels. The attention layout computation calls it; the planner never sees it.
6. Scope: the two llama modules only. No `engine/` moves, no new abstraction for hypothetical third modules.

## Behavior bar

Byte-identical outputs on every path except one:

- Default path (no LMCache): identical `AttentionCacheLayout`, `GDNCacheLayout`, and `page_size`.
- Tuned path (LMCache), `TM_GDN_BLOCK_CONFIG` unset: identical chosen geometry, padded part bytes, and `page_size`.
- Tuned path, `TM_GDN_BLOCK_CONFIG` set: intentional diff. Today the tuner overrides the pinned `(layers, heads)` (`object_cache_plan.cc:216-230, 290-291` ignore the pin); after this change the pin is the only candidate and the planner tunes page size and part padding around it.

Nothing in the engine, scheduler, registry, allocator, or LMCache registration changes. `ObjectAllocator` still receives `plan.page_size`; `CacheCategory` registrations keep their sizes; `lmcache.cc` pool derivation and the `page_size % part_bytes == 0` check are untouched.

## 1. New header: `src/turbomind/models/llama/attention_cache_layout.h`

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstddef>
#include <vector>

namespace turbomind {

struct AttentionCacheLayout {
    size_t              object_bytes{};       // S: one KV block across all full-attention layers
    std::vector<size_t> layer_offsets_bytes;  // byte offsets in full-attention weight order
};

}  // namespace turbomind
```

## 2. New header: `src/turbomind/models/llama/gdn_cache_layout.h`

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstddef>
#include <vector>

namespace turbomind {

struct GDNCacheLayout {
    int layers_per_block{};
    int heads_per_block{};
    int num_layer_groups{};
    int num_head_groups{};
    int num_blocks{};

    size_t           conv_bytes{};
    size_t           conv_part_bytes{};   // == conv_bytes until page tuning pads it
    std::vector<int> conv_state_offsets;  // Element offsets, in the same order as the GDN weights.

    size_t           recurrent_total_bytes{};   // layers * local_v_heads * cell bytes
    size_t           recurrent_block_bytes{};   // layers_per_block * heads_per_block * cell bytes (exact)
    size_t           recurrent_part_bytes{};    // block bytes until page tuning pads it
    std::vector<int> recurrent_state_offsets;   // Element offsets, in the same order as the GDN weights.
};

}  // namespace turbomind
```

Header-only: the layout is plain data. The page-fit arithmetic stays in the planner (section 7), which owns the page size it pads against; cell facts (`cell elements`, `cell bytes`, `local_v_heads`) are intermediates of `ComputeLayouts` (section 4.2) and are not stored on the layout.

## 3. `UnifiedAttentionLayer` changes

### 3.1 Header (`unified_attention_layer.h`)

- Add include: `#include "src/turbomind/models/llama/attention_cache_layout.h"`.
- Drop the forward declaration `struct AttentionCachePlan;` (line 41).
- Replace the constructor parameter type `const AttentionCachePlan& cache_plan` with `const AttentionCacheLayout& cache_layout`.
- `ForwardParam::layer_id` narrows from the global layer counter (today it feeds only debug labels, `unified_attention_layer.cc:334-392`) to the module-local index into the layout offset vectors; the decoder supplies the mapped value (section 9). Annotate the field: `int layer_id;  // module-local index into the layout offset vectors`.
- Add the static layout function and the stored layout:

```cpp
    // Computes this module's cache layout from its weights. Callable before any
    // layer exists; the only writer of AttentionCacheLayout.
    static AttentionCacheLayout ComputeCacheLayout(const std::vector<AttentionWeight*>& weights,
                                                   const EngineParam&                   engine);
```

```cpp
    AttentionCacheLayout layout_;  // module-owned; replaces per-weight cache_block_offset
```

### 3.2 `unified_attention_layer.cc`

Constructor body: replace lines 124-128 with:

```cpp
    TM_CHECK_GE(weights.size(), 1);
    TM_CHECK_EQ(weights.size(), cache_layout.layer_offsets_bytes.size());
    layout_               = cache_layout;
    prefix_cache_offset_  = registry.prefix().Register(cache_layout.object_bytes, /*alignment=*/1);
```

(The `weights[i]->cache_block_offset = ...` loop is deleted; no other line in the constructor changes.)

`core_attention` (the function at line 401; `p` is the `ForwardParam`): replace lines 463-469 with:

```cpp
        const size_t offset_bytes = layout_.layer_offsets_bytes.at(p.layer_id);
        TM_CHECK_LE(offset_bytes, INT_MAX);
        const int cache_block_offset = static_cast<int>(offset_bytes);

        // decode only
        params.block_iter_params = BlockIteratorParams{(char**)d.block_ptrs.data(),  //
                                                       d.block_ptrs_offsets.data() + offset,
                                                       cache_block_offset,
                                                       engine_param_.cache_block_seq_len};
```

(Check the `size_t` layout value before narrowing, as today: `unified_attention_layer.cc:463` checks the `size_t` field and only then casts to `int` at the use site.)

Define the moved static (verbatim body of today's `CreateDefaultAttentionCachePlan`, `object_cache_plan.cc:59-86`, with the return type renamed). Add includes `src/turbomind/kernels/attention/block.h` and `src/turbomind/core/data_type.h` if not already present; `BlockConfig` moves here from `object_cache_plan.cc:26-41`:

```cpp
// clang-format off
namespace {
struct BlockConfig {
    int  head_dim_;
    int  head_num_;
    int  block_len_;
    int  t_bits_;
    int  q_bits_;
    bool share_kv_;

    int  t_bits() const { return t_bits_; }
    int  q_bits() const { return q_bits_; }
    int  head_dim() const { return head_dim_; }
    int  head_num() const { return head_num_; }
    int  block_len() const { return block_len_; }
    bool is_share_kv() const { return share_kv_; }
};
}  // namespace
// clang-format on

AttentionCacheLayout UnifiedAttentionLayer::ComputeCacheLayout(const std::vector<AttentionWeight*>& weights,
                                                               const EngineParam&                   engine)
{
    TM_CHECK(!weights.empty());

    const int dtype_bits = byte_size(engine.data_type, 8);
    const int quant_bits = engine.quant_policy ? engine.quant_policy : dtype_bits;

    auto get_block_config = [&](const AttentionWeight& w) {
        BlockConfig b{w.head_dim,
                      w.kv_head_num / w.tp_size,
                      engine.cache_block_seq_len,
                      dtype_bits == quant_bits ? 0 : dtype_bits,
                      quant_bits,
                      w.head_dim == 576};
        return b;
    };

    AttentionCacheLayout result;
    result.layer_offsets_bytes.reserve(weights.size());
    result.object_bytes = 0;  // byte size (quantization aware)
    for (int i = 0; i < weights.size(); ++i) {
        block::Layout layout{get_block_config(*weights[i])};
        result.layer_offsets_bytes.push_back(result.object_bytes);
        result.object_bytes += layout.layer_size();
    }
    return result;
}
```

## 4. `GatedDeltaNetLayer` changes

### 4.1 Header (`GatedDeltaNetLayer.h`)

- Add include: `#include "src/turbomind/models/llama/gdn_cache_layout.h"`.
- Drop the forward declaration `struct GdnCachePlan;` (line 19).
- `ForwardParam` gains a layer index (mirrors `UnifiedAttentionLayer::ForwardParam::layer_id`), and the commented-out `// int layer_id;` placeholder is replaced for real:

```cpp
    struct ForwardParam {
        int                   phase;
        Tensor                input;
        Tensor                output;
        const DeltaNetWeight* weights;
        int                   layer_id;  // module-local index into the layout offset vectors
    };
```

- Constructor parameter `const GdnCachePlan& cache_plan` becomes `const GDNCacheLayout& cache_layout`.
- Add statics and the stored layout; delete the `layer_index_` member:

```cpp
    // Computes this module's offered layouts: front() is the default. With
    // TM_GDN_BLOCK_CONFIG set, exactly one layout (the pin is honored, never overridden).
    static std::vector<GDNCacheLayout> ComputeLayouts(const std::vector<DeltaNetWeight*>& weights,
                                                      const EngineParam&                  engine);
```

```cpp
    GDNCacheLayout layout_;  // module-owned; replaces per-weight conv/linear state offsets
```

### 4.2 `GatedDeltaNetLayer.cc`

Constructor: replace lines 73-82 and 114-121 with:

```cpp
    TM_CHECK_EQ(weights.size(), cache_layout.conv_state_offsets.size());
    TM_CHECK_EQ(weights.size(), cache_layout.recurrent_state_offsets.size());

    layout_            = cache_layout;
    layers_per_block_  = cache_layout.layers_per_block;
    heads_per_block_   = cache_layout.heads_per_block;
    num_head_groups_   = cache_layout.num_head_groups;
    num_layer_groups_  = cache_layout.num_layer_groups;
    num_blocks_        = cache_layout.num_blocks;
    block_bytes_       = cache_layout.recurrent_part_bytes;
    conv_total_bytes_  = cache_layout.conv_part_bytes;

    rec_base_ = registry.checkpoint().Register({{block_bytes_, 1, static_cast<size_t>(num_blocks_)}});
    registry.checkpoint().Register(conv_total_bytes_, 1);
```

The `weights[layer]->conv_state_offset = ...` / `linear_state_offset = ...` loop and the `layer_index_` fill are deleted. Every other constructor line (the `require_mode` planning checks, the info log, buffer allocation) is unchanged.

Consumers, in `Forward(ForwardParam param)` (the parameter is named `param` in this file, unlike attention's `p`):

- line 361, `invokeFusedConv1dSiLU(..., weights.conv_state_offset, ...)`: the argument becomes `layout_.conv_state_offsets.at(param.layer_id)`.
- line 382, `const int layer = layer_index_.at(param.weights);`: becomes `const int layer = param.layer_id;`.
- line 384, `const int64_t state_layer_offset = weights.linear_state_offset;`: becomes
  `const int64_t state_layer_offset = layout_.recurrent_state_offsets.at(param.layer_id);`.

Define the moved static (expands today's `CreateDefaultGdnCachePlan`, `object_cache_plan.cc:88-150`, plus the candidate enumeration from `TuneObjectCacheLayout`'s geometry loop, `object_cache_plan.cc:216-230`). Cell facts and `local_v_heads` are function locals, not layout fields. Add includes `<cstdio>` (for `std::sscanf`) and `src/turbomind/kernels/core/math.h` (for `ceil_div`) — neither is available in this file today; `<cstdlib>` (`std::getenv`) and `src/turbomind/core/data_type.h` (`byte_size`) already are:

```cpp
std::vector<GDNCacheLayout> GatedDeltaNetLayer::ComputeLayouts(const std::vector<DeltaNetWeight*>& weights,
                                                               const EngineParam&                  engine)
{
    TM_CHECK(!weights.empty());
    const auto& first   = *TM_CHECK_NOTNULL(weights.front());
    const int   tp_size = engine.attn_tp_size * engine.attn_cp_size;

    TM_CHECK_EQ(first.num_k_heads % tp_size, 0);
    TM_CHECK_EQ(first.num_v_heads % tp_size, 0);
    for (const auto* weight_ptr : weights) {
        const auto& weight = *TM_CHECK_NOTNULL(weight_ptr);
        TM_CHECK_EQ(weight.num_k_heads, first.num_k_heads);
        TM_CHECK_EQ(weight.num_v_heads, first.num_v_heads);
        TM_CHECK_EQ(weight.key_head_dim, first.key_head_dim);
        TM_CHECK_EQ(weight.value_head_dim, first.value_head_dim);
        TM_CHECK_EQ(weight.d_conv, first.d_conv);
        TM_CHECK_EQ(weight.data_type, first.data_type);
    }

    const int layer_num     = static_cast<int>(weights.size());
    const int local_k_heads = first.num_k_heads / tp_size;
    const int local_v_heads = first.num_v_heads / tp_size;

    // Fixed byte facts shared by every candidate geometry.
    const size_t recurrent_cell_elements = first.key_head_dim * first.value_head_dim;
    const size_t recurrent_cell_bytes    = byte_size(engine.state_dtype, recurrent_cell_elements);

    GDNCacheLayout base;
    base.recurrent_total_bytes = weights.size() * local_v_heads * recurrent_cell_bytes;

    const int    key_dim                 = local_k_heads * first.key_head_dim;
    const int    value_dim               = local_v_heads * first.value_head_dim;
    const size_t conv_dim                = 2 * key_dim + value_dim;
    const size_t conv_elements_per_layer = conv_dim * first.d_conv;
    const size_t conv_elements           = weights.size() * conv_elements_per_layer;
    base.conv_bytes           = byte_size(first.data_type, conv_elements);
    base.conv_part_bytes      = base.conv_bytes;
    base.conv_state_offsets.reserve(weights.size());
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        base.conv_state_offsets.push_back(layer * conv_elements_per_layer);
    }

    auto make_layout = [&](int layers_per_block, int heads_per_block) {
        GDNCacheLayout result   = base;
        result.layers_per_block = layers_per_block;
        result.heads_per_block  = heads_per_block;
        result.num_layer_groups = ceil_div(layer_num, layers_per_block);
        result.num_head_groups  = ceil_div(local_v_heads, heads_per_block);
        result.num_blocks       = result.num_layer_groups * result.num_head_groups;
        result.recurrent_block_bytes =
            static_cast<size_t>(layers_per_block) * heads_per_block * recurrent_cell_bytes;
        result.recurrent_part_bytes = result.recurrent_block_bytes;
        result.recurrent_state_offsets.resize(weights.size());
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            result.recurrent_state_offsets[layer] =
                static_cast<int>(layer % layers_per_block * heads_per_block * recurrent_cell_elements);
        }
        return result;
    };

    if (const char* value = std::getenv("TM_GDN_BLOCK_CONFIG")) {
        int layers_per_block = 1;
        int heads_per_block  = local_v_heads;
        TM_CHECK_EQ(std::sscanf(value, "%d,%d", &layers_per_block, &heads_per_block), 2)
            << "expected TM_GDN_BLOCK_CONFIG=l,h (e.g. 4,16)";
        TM_CHECK_GT(layers_per_block, 0);
        TM_CHECK_GT(heads_per_block, 0);
        // The pin is honored, never overridden: it is the single offered layout.
        return {make_layout(layers_per_block, heads_per_block)};
    }

    std::vector<GDNCacheLayout> candidates;
    candidates.push_back(make_layout(1, local_v_heads));  // the default comes first
    for (int layers_per_block = 1; layers_per_block <= layer_num; ++layers_per_block) {
        for (int heads_per_block = 1; heads_per_block <= local_v_heads; ++heads_per_block) {
            const int num_blocks =
                ceil_div(layer_num, layers_per_block) * ceil_div(local_v_heads, heads_per_block);
            if (num_blocks > layer_num) {
                continue;  // merge-only guard: tuned state never fragments past the per-layer baseline
            }
            if (layers_per_block == 1 && heads_per_block == local_v_heads) {
                continue;  // already offered first
            }
            candidates.push_back(make_layout(layers_per_block, heads_per_block));
        }
    }
    TM_CHECK(!candidates.empty());
    return candidates;
}
```

Candidate order and byte-identity: today's search enumerates geometries `(l asc, h asc)` and keeps the first strictly-smallest key. The final two key elements are `(layers_per_block, heads_per_block)`, so two distinct geometries can never tie, and moving the default `(1, H_v)` to the front cannot change the winner. The padding arithmetic is unchanged: the tuner performs exactly the `Divisors(page_units)` + `lower_bound` steps of `object_cache_plan.cc:250-263` with the same per-page hoisting, and `recurrent_block_bytes` here equals the old `geometry.recurrent_bytes` (`l * h * cell_bytes`).

## 5. `LanguageModel` fan-out

`language_model.h`: include `src/turbomind/models/llama/object_cache_plan.h` (replacing the `struct ObjectCachePlan;` forward declaration; needed because `CacheLayouts` is returned by value), and declare:

```cpp
    // Fan-out to the modules' layout statics. The planner-side entry below
    // turbomind.cc never names an individual module.
    static CacheLayouts ComputeCacheLayouts(const ModelWeight& weights, const EngineParam& engine);
```

`language_model.cc` (add includes for `unified_attention_layer.h` and `GatedDeltaNetLayer.h` if not already visible):

```cpp
CacheLayouts LanguageModel::ComputeCacheLayouts(const ModelWeight& weights, const EngineParam& engine)
{
    std::vector<AttentionWeight*> attention_weights;
    std::vector<DeltaNetWeight*>  gdn_weights;
    for (int layer = 0; layer < weights.num_layer; ++layer) {
        const auto* layer_weights = TM_CHECK_NOTNULL(weights.layer(layer));
        if (layer_weights->attention) {
            attention_weights.push_back(layer_weights->attention.get());
        }
        if (layer_weights->linear_attn) {
            gdn_weights.push_back(layer_weights->linear_attn.get());
        }
    }

    CacheLayouts layouts;
    if (!attention_weights.empty()) {
        layouts.attention = UnifiedAttentionLayer::ComputeCacheLayout(attention_weights, engine);
    }
    if (!gdn_weights.empty()) {
        layouts.gdn_candidates = GatedDeltaNetLayer::ComputeLayouts(gdn_weights, engine);
    }
    return layouts;
}
```

The `LanguageModel` constructor chain (`LanguageModel`, `LanguageModel::Impl`, `UnifiedDecoder`) keeps taking `const ObjectCachePlan&`; only the layer constructors' parameter types change (steps 3.1, 4.1) and `unified_decoder.cc` passes `*cache_plan.attention` / `*cache_plan.gdn` exactly as today (lines 88-101).

## 6. `object_cache_plan.h` after

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/models/llama/attention_cache_layout.h"
#include "src/turbomind/models/llama/gdn_cache_layout.h"

namespace turbomind {

// What a model offers to planning: module-computed layouts, not raw weights.
struct CacheLayouts {
    std::optional<AttentionCacheLayout> attention;       // absent when the model has no full-attention layers
    std::vector<GDNCacheLayout>         gdn_candidates;  // front() is the default; size 1 when the env pin applies
};

struct ObjectCachePlan {
    size_t                              page_size{32 << 20UL};  // default 32MB
    std::optional<AttentionCacheLayout> attention;
    std::optional<GDNCacheLayout>       gdn;
};

// Equal byte budgets do not imply equal page counts: CUDA allocations can
// consume different alignment padding on each rank. Trim only the tail so the
// original allocation base remains available for CUDA IPC registration.
inline size_t CommonCacheRegionSize(comm::HostCommImpl* group, const void* base, size_t bytes, size_t page_size)
{
    if (!page_size)
        throw std::invalid_argument("cache page size must be positive");
    const auto remainder = reinterpret_cast<std::uintptr_t>(base) % page_size;
    const auto padding   = remainder ? page_size - remainder : 0;
    const auto local     = bytes > padding ? (bytes - padding) / page_size : 0;
    const auto pages     = comm::AllReduce(group, local, comm::RedOp::kMin);
    if (!pages)
        throw std::runtime_error("cache budget cannot fit one aligned page on every TP rank");
    return padding + pages * page_size;
}

// Composes module layouts into a plan with the default page size.
ObjectCachePlan ComposeObjectCachePlan(const CacheLayouts& layouts);

// Page tuning: selects the page size and the adopted GDN layout from the
// module-offered candidates. Geometry and offsets stay as the module wrote
// them; the planner only stamps the page-fit padding (conv/recurrent part
// bytes) of the adopted candidate.
std::optional<ObjectCachePlan> TuneObjectCacheLayout(ObjectCachePlan plan,
                                                     const std::vector<GDNCacheLayout>& gdn_candidates);

}  // namespace turbomind
```

`CommonCacheRegionSize` is unchanged from today (repeated here only to show the surviving header surface).

## 7. `object_cache_plan.cc` after

The whole file becomes:

```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/object_cache_plan.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

namespace {

std::vector<size_t> Divisors(size_t value)
{
    std::vector<size_t> lower;
    std::vector<size_t> upper;
    for (size_t divisor = 1; divisor <= value / divisor; ++divisor) {
        if (value % divisor == 0) {
            lower.push_back(divisor);
            if (divisor != value / divisor) {
                upper.push_back(value / divisor);
            }
        }
    }
    lower.insert(lower.end(), upper.rbegin(), upper.rend());
    return lower;
}

}  // namespace

ObjectCachePlan ComposeObjectCachePlan(const CacheLayouts& layouts)
{
    TM_CHECK(layouts.attention || !layouts.gdn_candidates.empty())
        << "model must offer at least one cache layout";
    ObjectCachePlan plan;
    plan.attention = layouts.attention;
    if (!layouts.gdn_candidates.empty()) {
        plan.gdn = layouts.gdn_candidates.front();
    }
    return plan;
}

std::optional<ObjectCachePlan> TuneObjectCacheLayout(ObjectCachePlan                    plan,
                                                     const std::vector<GDNCacheLayout>& gdn_candidates)
{
    constexpr size_t alignment     = 256;
    constexpr size_t max_page_size = 32 << 20UL;
    TM_CHECK(plan.attention || plan.gdn) << "plan must have at least one of attention or gdn";
    TM_CHECK_EQ(plan.gdn.has_value(), !gdn_candidates.empty())
        << "gdn candidates must accompany a gdn layout";

    size_t attention_bytes{};
    size_t attention_units{1};
    if (plan.attention) {
        attention_bytes = plan.attention->object_bytes;
        if (attention_bytes == 0 || attention_bytes > max_page_size || attention_bytes % alignment != 0) {
            return std::nullopt;
        }
        attention_units = attention_bytes / alignment;
    }

    if (!plan.gdn) {
        plan.page_size = max_page_size / attention_bytes * attention_bytes;
        return plan;
    }

    const size_t conv_units     = ceil_div(plan.gdn->conv_bytes, alignment);
    const size_t max_page_units = max_page_size / alignment;

    const size_t page_step_units = plan.attention ? attention_units : 1;
    const size_t min_page_units  = plan.attention ? attention_units : std::max<size_t>(conv_units, 1);

    struct Candidate {
        using Key = std::tuple<size_t, int, size_t, size_t, size_t, int, int>;

        Key            key;
        size_t         page_units{};
        size_t         conv_part_bytes{};
        size_t         recurrent_part_bytes{};
        GDNCacheLayout layout;
    };

    std::optional<Candidate> best;
    for (size_t page_units = min_page_units; page_units <= max_page_units; page_units += page_step_units) {
        // Hoisted per page, as today: the divisor set and the conv fit are
        // candidate-independent (conv_bytes is shared by every candidate).
        const auto divisors = Divisors(page_units);
        const auto conv_it  = std::lower_bound(divisors.begin(), divisors.end(), conv_units);
        if (conv_it == divisors.end()) {
            continue;
        }
        const size_t conv_part_bytes  = *conv_it * alignment;
        const size_t conv_waste_bytes = conv_part_bytes - plan.gdn->conv_bytes;

        for (const auto& candidate : gdn_candidates) {
            const size_t recurrent_units = ceil_div(candidate.recurrent_block_bytes, alignment);
            const auto   recurrent_it    = std::lower_bound(divisors.begin(), divisors.end(), recurrent_units);
            if (recurrent_it == divisors.end()) {
                continue;
            }
            const size_t recurrent_part_bytes = *recurrent_it * alignment;
            const size_t actual_page_units    = std::lcm(
                attention_units, std::lcm(conv_part_bytes / alignment, recurrent_part_bytes / alignment));
            const size_t recurrent_waste_bytes =
                static_cast<size_t>(candidate.num_blocks) * recurrent_part_bytes - candidate.recurrent_total_bytes;
            const Candidate::Key key{conv_waste_bytes + recurrent_waste_bytes,
                                     candidate.num_blocks,
                                     actual_page_units,
                                     conv_waste_bytes,
                                     recurrent_waste_bytes,
                                     candidate.layers_per_block,
                                     candidate.heads_per_block};
            if (!best || key < best->key) {
                best = Candidate{key, actual_page_units, conv_part_bytes, recurrent_part_bytes, candidate};
            }
        }
    }

    if (!best) {
        return std::nullopt;
    }

    // Page-fit padding is planner policy: stamp the two part-byte fields of the
    // adopted layout. Geometry and offsets stay as the module wrote them.
    best->layout.conv_part_bytes      = best->conv_part_bytes;
    best->layout.recurrent_part_bytes = best->recurrent_part_bytes;
    plan.gdn                          = std::move(best->layout);
    plan.page_size                    = best->page_units * alignment;
    return plan;
}

}  // namespace turbomind
```

Mapping to today's search, for the byte-identity argument: the loop keeps today's shape — `Divisors(page_units)` and the conv `lower_bound` are evaluated once per page and shared by all candidates (`object_cache_plan.cc:250-255`); only the recurrent `lower_bound` runs per candidate (`:256-263`). The divisor set, the "smallest divisor that fits" rule, the tie-break key, and the page stepping (`attention_units` steps so every page holds whole attention objects, `:235-236, :249`) are identical. The chosen layout is the adopted candidate with its two part-byte fields stamped, rather than an in-place mutation of `*plan.gdn` (`:287-299`); `recurrent_block_bytes` and both offset vectors already carry the candidate's values, which the old code recomputed to the same numbers.

## 8. Weight field deletions

- `src/turbomind/models/attention_weight.h:128-129`: delete the `// Set by runtime layer` comment and `size_t cache_block_offset{};`.
- `src/turbomind/models/delta_net_weight.h:73-75`: delete the `// Set at runtime` comment, `int conv_state_offset{};`, `int linear_state_offset{};`.

## 9. `unified_decoder.cc` module-local indices

`ForwardParam::layer_id` is the module-local layer index — position among the layers that have that module — because the layouts' offset vectors are in module-local gathered order while the forward loop's `layer` is global. For dense models the two coincide; for GDN hybrids they do not. Build the mapping in the existing ctor gather loop (lines 60-76):

```cpp
    attn_local_index_.assign(model_weight.num_layer, -1);
    gdn_local_index_.assign(model_weight.num_layer, -1);
    for (int i = 0; i < model_weight.num_layer; ++i) {
        auto layer = model_weight.layer(i);
        // ... moe/ffn gathers unchanged ...
        if (layer->linear_attn) {
            gdn_local_index_[i] = static_cast<int>(gdn_weights.size());
            gdn_weights.push_back(layer->linear_attn.get());
        }
        if (layer->attention) {
            attn_local_index_[i] = static_cast<int>(attn_weights.size());
            attn_weights.push_back(layer->attention.get());
        }
    }
```

New members in `unified_decoder.h`:

```cpp
    // Global layer -> module-local layer index (-1 when the layer lacks the module).
    // ForwardParam::layer_id carries the module-local index; the modules' layout
    // offset vectors are indexed by it.
    std::vector<int> attn_local_index_;
    std::vector<int> gdn_local_index_;
```

Both forward call sites (lines 266-272) pass the mapped index — note the attention call changes too (today it passes the global `layer`, used only for debug labels):

```cpp
        if (weights.at(layer)->linear_attn) {
            linear_attn_layer_->Forward({phase,
                                         local_hidden_states,
                                         local_hidden_states,
                                         weights.at(layer)->linear_attn.get(),
                                         gdn_local_index_[layer]});
        }
        else {
            auto* attn = weights.at(layer)->attention.get();
            attn_layer_->Forward(
                {phase, local_hidden_states, local_hidden_states, attn, attn_local_index_[layer]});
        }
```

The `layer_index_` member and its two uses in `GatedDeltaNetLayer.cc` disappear with step 4. The warm-up `continue` (`layer >= tune_layer_num_`) and the `global_token_num == 0` break skip suffixes only, so they never disturb the maps; skipped layers simply never forward.

## 10. `turbomind.cc` wiring

Replace lines 326-330 with:

```cpp
    const bool use_lmcache = !param.lmcache_addr.empty();
    auto layouts    = LanguageModel::ComputeCacheLayouts(*TM_CHECK_NOTNULL(weights_[index]->text_model_ptr()), param);
    auto cache_plan = ComposeObjectCachePlan(layouts);
    if (use_lmcache) {
        auto tuned = TuneObjectCacheLayout(std::move(cache_plan), layouts.gdn_candidates);
        TM_CHECK(tuned.has_value());
        cache_plan = std::move(*tuned);
    }
```

Everything after (allocator construction with `cache_plan.page_size`, `LanguageModel` construction, `lmcache.Register`) is unchanged.

## 11. Build

No build changes: both new headers are header-only and `object_cache_plan.cc` stays in the `models` target. Build with `ninja` from the `build` folder per repo convention.

## 12. Engine README updates

`src/turbomind/engine/README.md` is normative (`AGENTS.md` hard constraint); these sentences describe where offsets live and must be rewritten in the same change. Behavior clauses stay untouched.

Line 380, `UnifiedAttentionLayer` paragraph: replace the first sentence

> `UnifiedAttentionLayer` registers its KV byte requirement with the prefix category during construction and stores the returned byte offset.

with

> `UnifiedAttentionLayer` registers its KV byte requirement (its module-owned `AttentionCacheLayout` object size) with the prefix category during construction and stores the returned registry byte offset; per-layer KV offsets are read from the module-owned layout (`ForwardParam::layer_id` is the module-local layer index into it), not written into the weights.

Line 386, `GatedDeltaNetLayer` paragraph: replace `stores the relevant offsets (per-layer conv element offsets within part 0, computed by the module; the base part id rec_base for recurrent parts)` with `stores the relevant offsets (per-layer conv element offsets within part 0, carried in its module-owned GDNCacheLayout; the base part id rec_base for recurrent parts)`, and in the same paragraph replace `plus a per-layer in-block element offset linear_state_offset == (L%L_b)*H_b*cell_elems` with `plus a per-layer in-block element offset from the module-owned layout, linear_state_offset == (L%L_b)*H_b*cell_elems`.

## 13. Verification

**Byte-identity harness (throwaway, not committed).** Page tuning is offline math and needs no LMCache server: `TuneObjectCacheLayout` can be driven directly.

1. Before starting the refactor in this tree, add a small debug entry that, for a given model, prints every field of the composed plan and the tuned plan (attention `object_bytes` + each `layer_offsets_bytes` entry; gdn every scalar and both offset vectors; `page_size`), one line per field, tagged with the field name. Drive it for (a) the default path and (b) the tuned path, on the model configs below.
2. Run it, capture output (`plan-before-<model>.txt`).
3. Apply the refactor, port the harness to `ComputeCacheLayouts` + `ComposeObjectCachePlan` / `TuneObjectCacheLayout(plan, candidates)`, capture `plan-after-<model>.txt`.
4. Diff. Required result: zero diff for the default path and the unpinned tuned path. For `TM_GDN_BLOCK_CONFIG=<l,h>` set plus the tuned path, the expected and only diff is: `layers_per_block/heads_per_block/num_*_groups/num_blocks` (and offsets) equal the pin instead of the tuner's old override; `page_size`, `conv_part_bytes`, `recurrent_part_bytes` are tuned around the pin.
5. Delete the harness.

Model configs (from `/data/models.json`, HF offline per repo convention): one dense model, one with `head_dim == 576` (exercises the share-kv path), one GDN hybrid. For the GDN hybrid also run the pinned case with a few `(l, h)` values, including `1,<num_v_heads>` (the default geometry).

**End-to-end.** `scripts/test_turbomind_model.py` as-is, no batching loop, on the dense and the GDN hybrid configs; verify the responses are meaningful and at least 128 tokens per repo policy. The hybrid run is also what exercises the module-local `layer_id` indexing (section 9): a wrong index fails loudly via `.at()` or produces gibberish, both caught by the meaningful-response bar.

**Compile.** `ninja` from `build` with no new warnings in the touched files.

## 14. Out of scope

- Any change to values (default geometry, page search, padding rules) beyond the honored-pin case.
- Moving `block::Layout` out of `kernels/attention/block.h`.
- Relocating planning into `engine/`, or generalizing the layout pattern for modules that do not exist yet.
- The engine README's behavior clauses, the scheduler, `CacheRegistry`, `ObjectAllocator`, and LMCache integration: untouched consumers.
