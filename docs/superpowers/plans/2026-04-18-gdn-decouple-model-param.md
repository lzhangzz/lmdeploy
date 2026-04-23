# Decouple GatedDeltaNetLayer from ModelParam/AttentionParam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove GatedDeltaNetLayer's dependency on ModelParam and AttentionParam. Read dimensional fields from DeltaNetWeight at runtime; pass non-dimensional fields directly.

**Architecture:** DeltaNetWeight fields become public so Forward can read them. GatedDeltaNetLayer constructor shrinks to take only `norm_eps`, `state_dtype`, `layer_types` directly (plus `EngineParam`, `Context`, `phases`). Derived dimensions computed as locals in Forward.

**Tech Stack:** C++17, CUDA, ninja build

---

### Task 1: Make DeltaNetWeight fields public

**Files:**
- Modify: `src/turbomind/models/delta_net_weight.h:37-71`

- [ ] **Step 1: Move private fields to public section in DeltaNetWeight**

Replace the current class body in `src/turbomind/models/delta_net_weight.h` (lines 37–71):

```cpp
class DeltaNetWeight: public core::Module {
public:
    const char* type() const override { return "DeltaNetWeight"; }

    DeltaNetWeight() = default;

    DeltaNetWeight(const core::DeltaNetConfig& cfg);

    void prepare() override;

    // --- X-macro field lists ---
#define DELTA_NET_WEIGHT_CHILDREN(X) \
    X(LinearWeight, in_proj_all) \
    X(LinearWeight, out_proj)    \
    X(NormWeight,   norm)

#define DELTA_NET_WEIGHT_PARAMS(X) \
    X(conv1d) \
    X(A_log)  \
    X(dt_bias)

    TM_MODULE_DECLARE(DeltaNetWeight, DELTA_NET_WEIGHT_CHILDREN, DELTA_NET_WEIGHT_PARAMS)

    // --- Config fields (public for runtime access) ---
    int      hidden_dim_{};
    int      num_k_heads_{};
    int      num_v_heads_{};
    int      key_head_dim_{};
    int      value_head_dim_{};
    int      d_conv_{};
    bool     bias_{};
    int      tp_size_{};
    int      tp_rank_{};
    DataType data_type_{};
};
```

- [ ] **Step 2: Build to verify no breakage**

Run: `cd build && ninja`
Expected: clean build (no code reads these fields externally yet)

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/delta_net_weight.h
git commit -m "refactor(deltanet): make DeltaNetWeight fields public"
```

---

### Task 2: Rewrite GatedDeltaNetLayer header

**Files:**
- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.h`

- [ ] **Step 1: Update constructor declaration and remove dimensional members**

Replace the constructor declaration (lines 22–27) and the "Model dimensions" block (lines 38–53) in `src/turbomind/models/llama/GatedDeltaNetLayer.h`.

New constructor declaration:
```cpp
    GatedDeltaNetLayer(float                   norm_eps,
                       DataType                state_dtype,
                       const std::vector<int>& layer_types,
                       const EngineParam&      engine,
                       const Context&          ctx,
                       int                     phases);
```

Replace lines 38–53 (the `hidden_units_` through `state_dtype_` block) with:
```cpp
    // Config passed at construction
    int              tp_size_;
    int              num_linear_layers_;
    std::vector<int> layer_types_;
    float            norm_eps_;
    DataType         state_dtype_;
```

The full private section should read:
```cpp
private:
    void Setup(int phase, TensorMap& env);

    // Config passed at construction
    int              tp_size_;
    int              num_linear_layers_;
    std::vector<int> layer_types_;
    float            norm_eps_;
    DataType         state_dtype_;

    LlamaLinear& linear_;

    // Per-phase batch data (mirrors UnifiedAttentionLayer pattern)
    struct Data {
        std::vector<RequestCache*> rc;
        std::vector<int>           input_lens;
        int                        batch_size = 0;
        Buffer_<int>               q_offsets;
        Buffer_<int>               k_offsets;
        std::vector<Tensor>        conv_states;
        std::vector<Tensor>        recurrent_states;
        Buffer_<void*>             conv_state_ptrs;
        Buffer_<void*>             recurrent_state_ptrs;
    };
    std::vector<Data> data_;

    // staging buffers
    Buffer_<void*> conv_state_ptrs_buf_;
    Buffer_<void*> recurrent_state_ptrs_buf_;

    int          sm_count_{1};
    Buffer_<int> work_counter_;

    cudaStream_t aux_stream_{};
    cudaEvent_t  ev_before_{};
    cudaEvent_t  ev_after_{};
```

- [ ] **Step 2: Commit**

```bash
git add src/turbomind/models/llama/GatedDeltaNetLayer.h
git commit -m "refactor(gdn): rewrite constructor signature and remove dimensional members"
```

---

### Task 3: Rewrite GatedDeltaNetLayer constructor and Forward

**Files:**
- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.cc`

- [ ] **Step 1: Rewrite constructor body**

Replace the constructor (lines 12–70) in `src/turbomind/models/llama/GatedDeltaNetLayer.cc`:

```cpp
GatedDeltaNetLayer::GatedDeltaNetLayer(float                   norm_eps,
                                       DataType                state_dtype,
                                       const std::vector<int>& layer_types,
                                       const EngineParam&      engine,
                                       const Context&          ctx,
                                       int                     phases):
    tp_size_(engine.attn_tp_size),
    num_linear_layers_(0),
    norm_eps_(norm_eps),
    state_dtype_(state_dtype),
    linear_(*ctx.linear)
{
    layer_types_       = layer_types;
    for (auto t : layer_types_) {
        if (t == 1)
            ++num_linear_layers_;
    }

    if (num_linear_layers_ > 0) {
        conv_state_ptrs_buf_      = {engine.max_batch_size, kCPUpinned};
        recurrent_state_ptrs_buf_ = {engine.max_batch_size, kCPUpinned};
    }

    for (int i = 0; i < phases; ++i) {
        data_.emplace_back();
        if (num_linear_layers_ > 0) {
            data_.at(i).conv_state_ptrs      = empty_like(conv_state_ptrs_buf_, kDEVICE);
            data_.at(i).recurrent_state_ptrs = empty_like(recurrent_state_ptrs_buf_, kDEVICE);
        }
    }

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device);
    work_counter_ = {1, kDEVICE};

    check_cuda_error(cudaStreamCreateWithPriority(&aux_stream_, cudaStreamNonBlocking, -1));
    check_cuda_error(cudaEventCreateWithFlags(&ev_before_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&ev_after_, cudaEventDisableTiming));
}
```

- [ ] **Step 2: Remove dead dtype_ usage in Run()**

In the `Run()` method (line 83), remove the line:
```cpp
        const auto             dtype = dtype_;
```

The kAdd handler becomes:
```cpp
    if (op == BatchOp::kAdd) {
        Buffer_<RequestCache*> rc = env.at("requests").buffer();
        for (int i = 0; i < rc.size(); ++i) {}
    }
```

- [ ] **Step 3: Rewrite Forward to read from DeltaNetWeight**

Replace the `Forward` method (lines 146–344). Insert dimensional reads at the top of the `dispatch` lambda, and replace all member variable references with the local variables.

At the top of the `dispatch` lambda (after `using T = decltype(t);`), add:

```cpp
        const auto& w          = *p.weights;
        const int   num_k_heads = w.num_k_heads_ / tp_size_;
        const int   num_v_heads = w.num_v_heads_ / tp_size_;
        const int   key_head_dim = w.key_head_dim_;
        const int   value_head_dim = w.value_head_dim_;
        const int   d_conv      = w.d_conv_;
        const int   key_dim     = num_k_heads * key_head_dim;
        const int   value_dim   = num_v_heads * value_head_dim;
        const int   conv_dim    = key_dim * 2 + value_dim;
```

Then replace all occurrences of these former member variables inside `dispatch`:

| Old (member) | New (local) |
|---|---|
| `num_v_heads_` | `num_v_heads` |
| `conv_dim_` | `conv_dim` |
| `value_dim_` | `value_dim` |
| `key_head_dim_` | `key_head_dim` |
| `value_head_dim_` | `value_head_dim` |
| `num_k_heads_` | `num_k_heads` |
| `d_conv_` | `d_conv` |
| `state_dtype_` | `state_dtype_` (unchanged — still a member) |
| `norm_eps_` | `norm_eps_` (unchanged — still a member) |
| `layer_types_` | `layer_types_` (unchanged — still a member) |
| `hidden_units_` | (unused — remove) |
| `dtype_` | (unused — remove) |
| `key_dim_` | `key_dim` |
| `num_linear_layers_` | `num_linear_layers_` (unchanged — still a member) |

Also update the `TM_LOG_INFO` in the constructor to use `w.` prefix for the logged values. But since the constructor no longer has access to the weight, remove the `TM_LOG_INFO` line entirely (the values are no longer available at construction time).

- [ ] **Step 4: Build to verify**

Run: `cd build && ninja`
Expected: clean build (call site not yet updated — will fail at link. That's OK, we fix in Task 4.)

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/llama/GatedDeltaNetLayer.cc
git commit -m "refactor(gdn): read dims from DeltaNetWeight at runtime, drop ModelParam/AttentionParam"
```

---

### Task 4: Update call site in unified_decoder.cc

**Files:**
- Modify: `src/turbomind/models/llama/unified_decoder.cc:58-59`

- [ ] **Step 1: Update GatedDeltaNetLayer construction call**

Replace line 59 in `src/turbomind/models/llama/unified_decoder.cc`:

```cpp
        linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(model, attn, engine, attn_tp_size_, ctx, phases);
```

With:

```cpp
        linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(
            model.norm_eps, model.linear_state_dtype, model.layer_types,
            engine, ctx, phases);
```

- [ ] **Step 2: Build to verify clean compile**

Run: `cd build && ninja`
Expected: clean build with no warnings

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/llama/unified_decoder.cc
git commit -m "refactor(decoder): update GatedDeltaNetLayer call site"
```

---

### Task 5: Smoke test with a DeltaNet model

**Files:**
- None (verification only)

- [ ] **Step 1: Check GPU availability**

Run the `get_gpu_usage` MCP tool to confirm a free GPU.

- [ ] **Step 2: Test with a model that has linear attention layers**

Run:
```bash
cd /data/lmdeploy-modeling
python scripts/test_turbomind_model.py <deltanet-model-id> --prompt "Hello, how are you?" --request-output-len 128
```

Use a model from the model-server registry that uses DeltaNet layers (e.g., a Qwen3.5 variant with linear attention). Check `list_models` MCP tool for available models.

Expected: Model produces meaningful human-language response, no CUDA errors, no assertion failures.

- [ ] **Step 3: Commit (if any fixup was needed)**

Only commit if fixes were required during testing.
