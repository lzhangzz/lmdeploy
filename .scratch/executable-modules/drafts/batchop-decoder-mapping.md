# BatchOp and decoder mapping

Draft spec section from [BatchOp and decoder mapping](../issues/06-batchop-and-decoder-mapping.md). Decides how executable modules plug into the engine loop with contract meanings held.

## Decisions

1. **Glue stays in the decoder.** `UnifiedDecoder`'s loop keeps the fused `AllreduceResidualRMSnorm`, the per-layer `ffn_group` comm mapping, global/local hidden-state plumbing, and the next-layer norm staging (layer L staging layer L+1's `attention_norm`; final layer staging `output_norm`). Only the executor calls swap to module forwards. `DecoderLayerWeight` remains the Python-built data composite — it does not become an executable composite. Zero behavior change; faithful to the locked "modules + minimal adaptation" scope.
2. **Default no-op `Run` on `core::Module`.** `virtual void Run(BatchOp op, int phase, TensorMap& env) {}` — the decoder walk calls it on every layer child for every op; each kind implements its subset and ignores the rest (contract-native, `contracts.batchop`).
3. **Versioned fill guard** for the pooled per-phase `AttentionBatch`: the pool slot carries a batch-version stamp; `Attention::kSetup` fills the batch-level parts only when stale (first module in the walk fills, the rest skip). Order-independent, no coordinator object.
4. **Two decoder paths, as today**: non-forward ops go through the generic `Run` walk; `kForward` keeps the decoder's orchestrated loop (ordering and fused glue are load-bearing).

## The shrunk decoder loop (kForward)

```cpp
// unified_decoder.cc — executor members (attn_layer_, ffn_layer_, moe_ffn_layer_,
// linear_attn_layer_) are gone; the loop drives modules through the data composite.
for (int layer = 0; layer < layer_num_; ++layer) {
    if (env.warm_up() && layer >= tune_layer_num) {
        continue;
    }
    auto* w = layers_[layer];  // DecoderLayerWeight, Python-built

    if (w->linear_attn) {
        w->linear_attn->Forward(delta_batch, local_hidden_states);      // was linear_attn_layer_->Forward
    }
    else {
        w->attention->Forward(attn_batch(phase), local_hidden_states);  // was attn_layer_->Forward({..., layer})
    }

    AllreduceResidualRMSnorm(...);  // unchanged: allreduce + residual + ffn_norm + ffn_group mapping

    if (w->moe_ffn) {
        w->moe_ffn->Forward(global_hidden_states, global_hidden_states, token_mask);  // was moe_ffn_layer_
    }
    else if (w->feed_forward) {
        TM_CHECK(!w->moe_ffn);  // staging invariant, unchanged
        w->feed_forward->Forward(global_hidden_states, global_hidden_states);        // was ffn_layer_
    }
    // next-layer norm staging: unchanged, decoder-owned
}
```

## Op walk (non-forward ops)

```cpp
void UnifiedDecoder::Run(BatchOp op, int phase, TensorMap& env)
{
    for (auto* layer : layers_) {
        layer->for_each_child([&](const char*, core::Module* child) {
            child->Run(op, phase, env);  // default no-op; kinds implement their subset
        });
    }
}
```

## Op coverage by module kind

| Module kind | kAdd | kSetup | kPrepare | kForward | kUnprep | kFetch/kUpdate/kDel |
|---|---|---|---|---|---|---|
| Linear, Ffn, Moe | — | — | — | `Forward` | — | — |
| Attention | — | fill `AttentionBatch` (versioned guard), resolve pointers | — | `Forward` | as today's `Run` mapping | — |
| DeltaNet | — | resolve frontier parts | clear state (position-0 requests) | `Forward` | frontier export | — |
| NormWeight and other data-only children | no-op everywhere (default) | | | | | |

`LanguageModel::Run`'s dispatch shape is unchanged (input processor / decoder / generation / output processor); it stays the single BatchOp entry per `concepts.module`. `LanguageModel`'s own `linear_` uses (embedding/output GEMMs) migrate to owned `Linear` modules — landing details belong to the migration order.

## Engine pass order (normative)

```text
Python builds tree -> prepare() walk -> register_cache(CacheRegistry&) walk
  -> category finalization -> bind(ExecContext&) walk -> first Run
```

## README amendments required by contract-sync

Per `checklist.contract-sync`, these land in the same change as the code that renames the classes (implementation stages, listed in the spec):

- `contracts.cache-content`: `UnifiedAttentionLayer`/`GatedDeltaNetLayer` → `Attention`/`DeltaNet` (per-layer modules).
- `contracts.unified-attention` and `contracts.gated-deltanet`: class renames plus the per-layer-module, pooled-batch, `register_cache`-pass shape.
- `ownership.module-cache` and `contracts.category-registration`: "during construction" → "during the `register_cache` pass" (registration timing moved out of constructors).
- New lifecycle paragraph: the pass order above, and the default no-op `Run` dispatch rule.
- `concepts.module` needs no change — the merged modules are exactly what it already describes.

## Amendment (Rework the ExecContext design)

The pass order's final injection step is `bind(Context&)` on the surviving `turbomind::Context` (not `ExecContext`): modules capture typed references — Linear the GEMM engine, Attention/Moe/DeltaNet their kind states. No other part of this mapping changes.
