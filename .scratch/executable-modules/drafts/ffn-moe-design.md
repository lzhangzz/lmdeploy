# Merged FFN and MoE module design

Draft spec section from [Merged FFN and MoE module design](../issues/04-merged-ffn-and-moe-module-design.md). Builds on the validated [Linear interface](../prototype/linear-module-interface.md) and `drafts/context-contract.md` (with the MoE amendment noted below).

## Decisions

1. **Ffn owns the whole FFN**: `Ffn::Forward` covers w1w3 → activation → w2, with the FP8-fused-SiLU and w1/w3-separate branches internal. Dense and grouped overloads share one body; `LlamaFfnLayer` retires.
2. **Moe owns routing and transport code**: `Moe::Forward` = gate → route → transport-gather → grouped FFN → combine → shared fold, including warm-up random routing. `MoeFfnLayer`, `MoeFfnDefaultImpl`, `MoeFfnA2AImpl` retire; default-vs-DeepEP is an internal branch.
3. **Transport/routing state is owned per module instance** (explicit user decision, deviating from the context contract's per-kind pools — see Amendment). Each Moe instance owns its `TokenDispatcher`, routing buffers, and factory-made scales-copy stream.
4. **Grouped block mechanism unchanged**: `experts` ModuleList stays the Python surface; `prepare()` builds the internal grouped `block_` via `LinkLinearExperts`; `AliasRouted`/`meta_pack_` aliasing unchanged.
5. **Naming**: C++ classes `Ffn` and `Moe`; `type()` keeps returning `"FfnWeight"`/`"MoeWeight"`.

## Ffn interface

```cpp
// PROTOTYPE — merged Ffn: the whole feed-forward, dense or grouped.
// Replaces models/ffn_weight.h + models/llama/LlamaFfnLayer.{h,cc} and the
// copy-pasted GEMM orchestration in both MoeFfn impls.
class Ffn: public core::Module {
public:
    const char* type() const override
    {
        return "FfnWeight";
    }

    Ffn() = default;
    Ffn(const core::FfnConfig& cfg);

    void prepare() override;
    void bind(ExecContext& xc);

    /// Dense: y = W2·act(W1x, W3x). Activation internal (fused-silu/FP8 and
    /// w1/w3-separate branches). Output allocated when null.
    void Forward(const Tensor& input, Ref<Tensor> output);

    /// Grouped (the MoE expert block): routing tables drive expert-grouped
    /// GEMMs and the grouped activation. Same internal branches.
    void Forward(const Tensor& input, const GroupedRoute& route, Ref<Tensor> output);

#define FFN_WEIGHT_CHILDREN(X) \
    X(Linear, w1)              \
    X(Linear, w3)              \
    X(Linear, w2)              \
    X(Linear, w1w3)

    // act_type / is_fused_silu / hidden_dim / inter_size / tp_* as today
};

/// Routing tables shared by the grouped overloads.
struct GroupedRoute {
    Buffer_<int>  indices;          // grouped row -> source token slot
    Buffer_<int>  offsets;          // per-local-expert row offsets [E+1]
    Buffer_<int>  f2E;              // grouped row -> expert id (activation grouping)
    const int*    num_valid_tokens; // device pointer: routed token count
};
```

The three verbatim copies of the w1w3-vs-w1/w3 + FP8-fused-SiLU block (`LlamaFfnLayer.cc:51-90`, `moe_ffn_layer.cc:236-267`, `moe_ffn_layer.cc:510-539`) collapse into this one body. Grouped activation keeps its variant (`Activation(gating, bias, f2E, act, num_valid_tokens, st)` vs dense `Activation(gating, inter, act, st)`) as an internal branch on `route` presence.

## Moe interface

```cpp
// PROTOTYPE — merged Moe: routing, transport, grouped experts, shared fold.
// Replaces models/moe_weight.h (execution side) + models/llama/moe_ffn_layer.{h,cc}.
class Moe: public core::Module {
public:
    const char* type() const override
    {
        return "MoeWeight";
    }

    Moe() = default;
    Moe(const core::MoeConfig& cfg);

    void prepare() override;  // experts -> LinkLinearExperts -> block_
    void bind(ExecContext& xc);

    /// Full MoE forward: gate, route, transport, grouped experts, combine,
    /// shared-expert fold. token_mask: [tokens]; invalid tokens route nowhere.
    void Forward(const Tensor& input, Ref<Tensor> output, const bool* token_mask);

#define MOE_WEIGHT_CHILDREN(X) \
    X(Linear, gate)            \
    X(Linear, shared_gate)     \
    X(Ffn,   shared)           \
    X(core::ModuleList, experts)

private:
    Tensor Gate(const Tensor& input);   // gate->Forward / shared_gate->Forward
    GroupedRoute Route(const Tensor& logits, const bool* token_mask, StreamRef st);
    // invokeMoeGate_V2 / NoAuxTC; under env.warm_up(): SampleUniform tables.

    // --- transport, owned per module instance (Amendment below) ---
    // default: table compaction buffers (f2n_, offsets_, masks_, en2f_, accum_)
    // deepep : std::unique_ptr<comm::TokenDispatcher> dispatcher_;
    //          StreamHandle scales_copy_stream_;  // factory-made handle

    Ffn* block_{};  // grouped expert block built by prepare(); never Python-visible

    // routing buffers, per module instance (Amendment below)
};
```

`Moe::Forward` flow (both transports share it; only gather/scatter differ):

```cpp
Tensor Moe::Forward(const Tensor& input, Ref<Tensor> output, const bool* token_mask)
{
    auto  logits = gate ? gate->Forward(input) : Tensor{};        // merged Linear
    auto  route  = Route(logits, token_mask, stream);             // warm-up: SampleUniform
    auto  xs     = Gather(input, route);                          // default: view; deepep: dispatcher_->Dispatch
    Tensor temp;
    block_->Forward(xs, route, temp);                             // THE grouped-FFN call
    auto  routed = Scatter(temp, route);                          // combine / A2A return
    if (shared) {
        Tensor shared_out{...};
        shared->Forward(Shard(input), shared_out);                // merged Ffn, dense
        routed += WeightedShared(shared_out, shared_gate ? Gate(input) : none);
    }
    output = routed;
}
```

The embedded `LlamaFfnLayer ffn_` member disappears: the shared expert is `shared->Forward(...)` directly.

## Amendment to the context contract

The context contract's never-hold rule ("per-kind persistent workspaces: request from `pools()`; shared across layer instances") is **waived for Moe transport/routing state by explicit decision**: each per-layer Moe instance owns its `TokenDispatcher`, routing buffers, and scales-copy stream (factory-made). Consequences, recorded so the spec review sees them:

- Dispatcher buffers scale with max tokens × hidden × topk and now multiply by layer count; the DeepEP JIT compile loop also runs per layer.
- The merged-model memory profile is intentionally worse than today's single per-rank dispatcher.
- Migration gates must therefore include a MoE-stage memory-footprint and startup-cost check (folded into Migration order and verification gates).

Streams remain factory-made (`make_stream`) — per-module ownership of a handle is consistent with the factory clause; only the pooling clause is waived.
