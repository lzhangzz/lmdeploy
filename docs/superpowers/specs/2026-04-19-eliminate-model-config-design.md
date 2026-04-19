# Eliminate model_config Design

## Goal

Delete the `ModelParam`, `MLAParam`, and `MoeParam` structs. Stop parsing the `model_config` YAML section in C++ entirely. ModelWeight derives all model-level fields from its children after weight loading. Consumers read public fields from ModelWeight directly.

## Current State

`turbomind.cc` parses 39 fields from `model_config`, 1 field from `attention_config`, and 20 fields from `engine_config`. The `model_config` fields populate two structs:

| Struct | Fields | Consumers |
|---|---|---|
| `ModelParam` | ~30 fields | ModelWeight (constructor init only), LanguageModel, UnifiedDecoder, InputProcessor, OutputProcessor, SequenceManager |
| `MoeParam` | 12 fields + `Method` enum | ModelWeight (dead storage), UnifiedDecoder (expert_num check only), MoeWeight (stores its own copy from MoeConfig) |
| `MLAParam` | 4 fields (q_lora_rank, kv_lora_rank, qk_rope_dim, v_head_dim) | AttentionWeight (`mla_` member) |

**Key facts:**
- `ModelWeight::model_param_`, `moe_param_`, and `engine_param_` are all dead storage after construction -- never read via any accessor.
- `MoeFfnLayer` reads MoE params exclusively from `MoeWeight::moe_param()`, not from any standalone `MoeParam`.
- `LanguageModel::model_param()` is called only from `engine.cc` (2 call sites: SequenceManager and Validate).
- `LanguageModel::Impl::param_` has exactly one runtime read: `param_.hidden_units` at line 201.
- `quant_policy` is already parsed from `engine_config["quant_policy"]` (not model_config) but stored in ModelParam.
- `session_len` is parsed from `model_config["session_len"]` but stored in EngineParam.
- `data_type` is parsed from `model_config["data_type"]`. Python also generates `dtype` in engine_config YAML (via `asdict(TurbomindEngineConfig)`) but C++ ignores it.
- AttentionWeight stores both `MLAParam mla_{}` and individual fields (`kv_lora_rank_`, `q_lora_rank_`, etc.) from AttentionConfig -- the `MLAParam` is redundant.

## Changes

### 1. Delete ModelParam, MLAParam, MoeParam, HasLinearAttention

Remove from `llama_params.h`:
- `ModelParam` struct (lines 22-58)
- `MLAParam` struct (lines 15-20)
- `HasLinearAttention(ModelParam)` helper (lines 60-68)
- `MoeParam` struct including `Method` enum (lines 71-93)

Move `MoeParam::Method` enum to `MoeWeight` header (used by `get_moe_method()` and `MoeWeight::method()`). Call it `MoeMethod` or similar.

`EngineParam` gains two new fields:
- `quant_policy` (int, already parsed from `engine["quant_policy"]` on line 244, just store in EngineParam instead of ModelParam)
- `tune_layer_num` (int, parse from `engine["tune_layer_num"]` with default 1)

### 2. Eliminate MLAParam from AttentionWeight

AttentionWeight already has individual fields from AttentionConfig: `kv_lora_rank_`, `q_lora_rank_`, `qk_rope_dim_`, `v_head_dim_`. Delete `MLAParam mla_{};` member.

Update `is_mla()` to check `kv_lora_rank_ > 0` instead of `mla_.kv_lora_rank > 0`.

Update UnifiedAttentionLayer's forward path that reads `weights.mla_.kv_lora_rank` etc. to use the individual fields (`weights.kv_lora_rank_`, etc.).

### 3. ModelWeight derives fields from children

ModelWeight constructor changes from:
```cpp
ModelWeight(DataType, const ModelParam&, const EngineParam&, const MoeParam&)
```
to:
```cpp
ModelWeight(const EngineParam&)
```

Remove `data_type_` from constructor param (derived in prepare). Remove dead storage: `model_param_`, `moe_param_`, `engine_param_`. Keep only what's needed from EngineParam: `tp_size_` and `tp_rank_` (copied at construction).

Remove existing accessors (`hidden_units()`, `vocab_size_padded()`, `tp_size()`, `num_layers()`). Make derived fields public. Populated in `prepare()` after children are loaded:

| Field | Type | Derived from |
|---|---|---|
| `data_type_` | `DataType` | `layers[0]->attention->data_type_` |
| `hidden_units_` | `int` | `layers[0]->attention->hidden_dim_` |
| `vocab_size_` | `int` | `tok_embeddings->weight.shape(0)` |
| `vocab_size_padded_` | `int` | `round_up(vocab_size_, tp_size_)` |
| `embedding_size_` | `int` | `tok_embeddings->weight.shape(0)` |
| `num_layer_` | `int` | `layers->size()` |
| `head_dim_` | `int` | `layers[0]->attention->head_dim_` |
| `kv_head_num_` | `int` | `layers[0]->attention->kv_head_num_` |
| `layer_types_` | `vector<int>` | Per-layer: check `linear_attn` child exists → 1, else → 0 |
| `tp_size_` | `int` | From EngineParam at construction |
| `tp_rank_` | `int` | From EngineParam at construction |

New `prepare()` logic (added to existing override):
```cpp
void ModelWeight::prepare()
{
    // Existing: recurse into children
    for_each_child([](const char*, Module* child) {
        if (child) child->prepare();
    });

    // Derive model-level fields from first layer
    auto* layer0 = layer(0);
    TM_CHECK(layer0 && layer0->attention);
    data_type_    = layer0->attention->data_type_;
    hidden_units_ = layer0->attention->hidden_dim_;
    head_dim_     = layer0->attention->head_dim_;
    kv_head_num_  = layer0->attention->kv_head_num_;

    vocab_size_     = tok_embeddings->weight.shape(0);
    embedding_size_ = vocab_size_;
    num_layer_      = layers->size();
    vocab_size_padded_ = round_up(vocab_size_, (size_t)tp_size_);

    // Derive layer_types
    layer_types_.resize(num_layer_);
    for (int i = 0; i < num_layer_; ++i) {
        layer_types_[i] = layer(i)->linear_attn ? 1 : 0;
    }
}
```

### 4. turbomind.cc stops parsing model_config

**Delete** all parsing of `node["model_config"]` fields (~75 lines including MoE fields). The `model_config` YAML section is ignored entirely.

**Field migration:**
- `quant_policy`: already parsed from `engine["quant_policy"]` → store in `EngineParam::quant_policy` instead of `ModelParam::quant_policy`
- `tune_layer_num`: parse from `engine["tune_layer_num"]` (default 1) → `EngineParam::tune_layer_num`
- `data_type`: switch to `engine["data_type"]`. Python already generates `dtype` in engine_config YAML. The C++ key name must match what Python generates -- verify the exact key.
- `session_len`: switch to `engine["session_len"]`. Python already generates this in engine_config. Currently read from `model["session_len"]`.
- `model_name_`: stored but never read after construction. Drop entirely.

**Delete from TurboMind::Impl:**
- `model_param_` member
- `moe_param_` member
- All MoE YAML parsing from model_config (lines 336-365)
- `get_moe_method()` can stay but return the new enum type from MoeWeight header

**Update callsites:**
- `CreateWeights()`: `ModelWeight(engine_params_[index])` -- no DataType, no ModelParam, no MoeParam
- `CreateEngine()`: `LanguageModel(param, ctx, *weights_[index], phases_)` -- no ModelParam, no MoeParam. DataType read from `weights_[index]->data_type_` (weights loaded by this point)
- `CreateRequest()`: read `vocab_size`, `hidden_units` from `weights_[index]` (called in WarmUp, after weights loaded)
- `WarmUp()`: read `vocab_size` from `weights_[index]`

**Defer prefix-caching check:** The `HasLinearAttention` check (line 303) runs during YAML parsing before weights exist. Move it to `CreateEngine()` where weights are available: check `weights_[index]->layer_types_` for any entry == 1.

### 5. LanguageModel removes ModelParam and MoeParam

Constructor changes from:
```cpp
LanguageModel(DataType, const ModelParam&, const EngineParam&, const MoeParam&, const Context&, const ModelWeight&, int)
```
to:
```cpp
LanguageModel(const EngineParam&, const Context&, const ModelWeight&, int)
```

Delete `param_` member and `dtype_` member from `Impl`. Replace reads:
- `param_.hidden_units` → `weights_.hidden_units_`
- `model.vocab_size` (local) → `weights_.vocab_size_`
- `dtype_` → `weights_.data_type_`

Delete `model_param()` accessor. Engine reads from `ModelWeight` directly.

### 6. UnifiedDecoder removes ModelParam and MoeParam

Constructor changes from:
```cpp
UnifiedDecoder(const ModelParam&, const EngineParam&, const MoeParam&, const Context&, int, const vector<DecoderLayerWeight*>&)
```
to:
```cpp
UnifiedDecoder(const EngineParam&, const Context&, int, const ModelWeight&)
```

Replaces:
- `model.layer_num` → `model_weight.num_layer_`
- `model.hidden_units` → `model_weight.hidden_units_`
- `model.tune_layer_num` → `engine.tune_layer_num`
- `model.quant_policy` → `engine.quant_policy`
- `model.layer_types` → `model_weight.layer_types_`
- `model.inter_size` accumulation → check `feed_forward` child exists per-layer (already does this)
- `model.linear_state_dtype` → `model_weight.data_type_` (was always set to data_type_)
- `moe.expert_num` accumulation → check `moe_ffn` child exists on any layer

Passes `ModelWeight&` reference to sub-layer constructors so they can extract what they need.

### 7. InputProcessor removes ModelParam

Constructor changes from:
```cpp
InputProcessor(const EngineParam&, const ModelParam&, int)
```
to:
```cpp
InputProcessor(const EngineParam&, int hidden_units, DataType data_type, int)
```

### 8. OutputProcessor removes ModelParam

Constructor changes from:
```cpp
OutputProcessor(const ModelParam&, int, int, int, function<...>)
```
to:
```cpp
OutputProcessor(int vocab_size, int, int, int, function<...>)
```

### 9. SequenceManager removes ModelParam

Constructor changes from taking `const ModelParam&` to individual params. The caller (Engine) extracts these from ModelWeight:

```cpp
SequenceManager(int head_dim, int kv_head_num, int num_layer,
                const vector<int>& layer_types, int quant_policy,
                DataType data_type, DataType runtime_dtype,
                int linear_key_head_dim, int linear_value_head_dim,
                int linear_conv_kernel_dim, int linear_num_key_heads,
                int linear_num_value_heads,
                int cache_block_seq_len, int attn_tp_size, ...)
```

DeltaNet-derived fields come from the DeltaNetWeight child (if any). Engine reads them from the first DecoderLayerWeight that has a `linear_attn` child, defaulting to 0 if none.

### 10. Engine removes model_param() dependency

`Engine::Impl::CreateSequenceManager()` extracts fields from ModelWeight instead of `model_.model_param()`.

`Engine::Impl::Validate()` replaces `HasLinearAttention(model_.model_param())` with a check on `model_weight.layer_types_`.

Engine needs access to `ModelWeight` -- currently it only has `LanguageModel`. Pass `ModelWeight&` directly to Engine constructor.

### 11. Delete MoeParam usage in UnifiedDecoder

Replace `std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)` with checking if any DecoderLayerWeight has a `moe_ffn` child:
```cpp
bool has_moe = false;
for (int i = 0; i < model_weight.num_layer_; ++i) {
    if (model_weight.layer(i)->moe_ffn) {
        has_moe = true;
        break;
    }
}
```

## Files Changed

| File | Change |
|---|---|
| `src/turbomind/models/llama/llama_params.h` | Delete ModelParam, MLAParam, MoeParam, HasLinearAttention. Add quant_policy, tune_layer_num to EngineParam. Keep only EngineParam |
| `src/turbomind/models/attention_weight.h` | Delete `MLAParam mla_{}`. Update `is_mla()` to check `kv_lora_rank_ > 0`. Remove llama_params.h include |
| `src/turbomind/models/attention_weight.cc` | Update MLA field references from `mla_.xxx` to `xxx_` |
| `src/turbomind/models/moe_weight.h` | Add `MoeMethod` enum (moved from MoeParam). Remove llama_params.h include |
| `src/turbomind/models/model_weight.h` | Remove ModelParam/MoeParam from constructor. Remove dead storage members. Make derived fields public. Add layer_types_, head_dim_, kv_head_num_ |
| `src/turbomind/models/model_weight.cc` | Simplify constructor. Add derivation logic in prepare() |
| `src/turbomind/turbomind.cc` | Delete model_config parsing (~75 lines). Switch data_type/session_len to engine_config. Drop model_name_. Update get_moe_method() return type |
| `src/turbomind/models/language_model.h` | Remove ModelParam/MoeParam from constructor signature |
| `src/turbomind/models/language_model.cc` | Replace param_/dtype_ reads with weights_ reads. Remove model_param() accessor |
| `src/turbomind/models/llama/unified_decoder.h` | Remove ModelParam/MoeParam from constructor |
| `src/turbomind/models/llama/unified_decoder.cc` | Read from ModelWeight reference. Replace MoeParam check with child check |
| `src/turbomind/models/input_processor.h` | Remove ModelParam, take hidden_units + data_type |
| `src/turbomind/models/input_processor.cc` | Use individual params |
| `src/turbomind/models/output_processor.h` | Remove ModelParam, take vocab_size |
| `src/turbomind/models/output_processor.cc` | Use int vocab_size |
| `src/turbomind/models/llama/SequenceManager.h` | Remove ModelParam, take individual params |
| `src/turbomind/models/llama/SequenceManager.cc` | Use individual params |
| `src/turbomind/models/llama/moe_ffn_layer.cc` | Update MoeParam::Method references to new enum |
| `src/turbomind/models/llama/unified_attention_layer.h` | Remove llama_params.h include if no longer needed |
| `src/turbomind/models/llama/unified_attention_layer.cc` | Update `weights.mla_.xxx` to `weights.xxx_` |
| `src/turbomind/engine/engine.h` | Add ModelWeight reference to constructor |
| `src/turbomind/engine/engine.cc` | Extract from ModelWeight for SequenceManager and Validate. Remove HasLinearAttention call |

## Python Side (follow-up, not in this change)

The Python side will need to:
1. Stop generating the `model_config` YAML section
2. Move `tune_layer_num` to `engine_config` (new field)
3. Ensure `session_len` is in `engine_config` (already there via TurbomindEngineConfig)
4. Ensure `data_type` key in `engine_config` matches what C++ reads (currently `dtype` in Python, may need to be `data_type`)

C++ will simply ignore `model_config` if present (no code parses it).

## Scope

Pure C++ refactoring. No behavioral changes. `EngineParam` gains two fields but its parsing from `engine_config` is otherwise unchanged.
