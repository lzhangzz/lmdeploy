# Eliminate AttentionParam

## Problem

`AttentionParam` is a config struct threaded through `TurboMind → LanguageModel → UnifiedDecoder → UnifiedAttentionLayer`, but its fields are already available elsewhere — mostly in `AttentionWeight`. This adds coupling and a redundant parameter.

## Current state

`AttentionParam` has 5 fields:

| Field | Current consumer | Already in AttentionWeight? |
|---|---|---|
| `softmax_scale` | `AttentionWeight` at forward time | Yes |
| `use_logn_attn` | `AttentionWeight` at forward time | Yes |
| `max_position_embeddings` | `AttentionWeight::rope_` at forward time | Yes (via `rope_`) |
| `rope` | `UnifiedAttentionLayer` constructor | Yes (`rope_`) |
| `cache_block_seq_len` | `UnifiedAttentionLayer` constructor, `Engine` | No |

`UnifiedDecoder` only uses `rope` and `cache_block_seq_len` from `AttentionParam`. The other 3 fields already flow through `AttentionWeight`. In fact, `softmax_scale`, `use_logn_attn`, and `max_position_embeddings` in `AttentionParam` are dead — parsed from YAML but never consumed by any code path.

## Data flow change

**Before:**
```
YAML → AttentionParam ─┬─ rope ──────────→ UnifiedDecoder → UnifiedAttentionLayer (ctor)
                       ├─ cache_block_seq_len → UnifiedDecoder → UnifiedAttentionLayer (ctor)
                       │                                        → Engine (via model_.attn_param())
                       └─ softmax_scale, use_logn_attn, max_position_embeddings (dead)
```

**After:**
```
AttentionWeight::rope_ ───────→ UnifiedDecoder extracts → UnifiedAttentionLayer (ctor)
EngineParam::cache_block_seq_len → UnifiedAttentionLayer (via engine_param_)
EngineParam::cache_block_seq_len → Engine (via param_)
```

## Design

### 1. Add `cache_block_seq_len` to `EngineParam`

File: `src/turbomind/models/llama/llama_params.h`

Add `int cache_block_seq_len` to `EngineParam` in the "cache params" section (after `cache_chunk_size`).

File: `src/turbomind/turbomind.cc`

Parse `cache_block_seq_len` from `attention["cache_block_seq_len"]` (same YAML node as before) but store into `engine_param_` instead of `attn_param_`. Remove `AttentionParam attn_param_` member.

File: `src/turbomind/engine/engine.cc`

Replace 3 occurrences of `model_.attn_param().cache_block_seq_len` with `param_.cache_block_seq_len`:
- Line 208: `cdiv(session_len_trunc_, param_.cache_block_seq_len)`
- Line 215: `const auto cache_block_seq_len = param_.cache_block_seq_len`
- Line 238: (uses local variable from line 215)

### 2. Remove `AttentionParam` struct

File: `src/turbomind/models/llama/llama_params.h`

Delete the `AttentionParam` struct definition (lines 96-104).

### 3. Remove `AttentionParam` from `LanguageModel`

Files: `src/turbomind/models/language_model.h`, `src/turbomind/models/language_model.cc`

- Remove `const AttentionParam& attn` from `LanguageModel` constructor (line 30) and `Impl` constructor (line 108)
- Remove `const AttentionParam attn_param_` member (line 38) and `attn_param()` accessor (lines 39, 517-519)
- Remove `attn_param_{attn}` from initializer list (line 134)
- Pass `weights_.layers_list()` to `UnifiedDecoder` constructor (line 166) as an additional argument

### 4. Remove `AttentionParam` from `UnifiedDecoder`

Files: `src/turbomind/models/llama/unified_decoder.h`, `src/turbomind/models/llama/unified_decoder.cc`

- Replace `const AttentionParam& attn` constructor parameter with `std::vector<DecoderLayerWeight*> layer_weights`
- Extract `std::vector<AttentionWeight*>` by iterating `layer_weights` and collecting each `.attention` child
- Pass the vector to `UnifiedAttentionLayer` constructor
- Remove `core::to_rope_config(attn.rope)` call — rope comes from `AttentionWeight::rope_` now
- Remove `attn.cache_block_seq_len` — comes from `EngineParam` now

### 5. Rewrite `UnifiedAttentionLayer` constructor

Files: `src/turbomind/models/llama/unified_attention_layer.h`, `src/turbomind/models/llama/unified_attention_layer.cc`

Constructor changes:
- Replace `const core::RopeConfig& rope` + `int cache_block_seq_len` parameters with `std::vector<AttentionWeight*> attn_weights`
- Initialize `rope_` from `attn_weights[0]->rope_` (member stays, used by `init_dynamic_ntk` in `Run()`)
- Read `cache_block_seq_len` from `engine_param_.cache_block_seq_len` (remove `cache_block_seq_len_` member)
- Initialize `rope_param_` via `init_rope_kernel_param(rope_, rope_param_)` — uses the existing `RopeConfig` overload in `attention_weight.cc`
- Move all `Init()` logic into the constructor body:
  - Head dimension validation (`tp_size`, `head_num`, `kv_head_num`, `head_dim` from `attn_weights[0]`)
  - Workspace buffer allocation (`partial_O_`, `partial_ML_`, `split_cnt_`)
  - `tmp_attn_` allocation (conditional on `init_` flag)
  - `Clear(split_cnt_)`

Removals:
- Remove `Init()` method (declaration and implementation)
- Remove `initialized_` flag
- Remove `if (!initialized_) { Init(p); }` check in `Forward()` (line 319-321)
- Remove `cache_block_seq_len_` member (read from `engine_param_` instead)

### 6. Remove `AttentionParam` from `TurboMind::Impl`

File: `src/turbomind/turbomind.cc`

- Remove `AttentionParam attn_param_` member (line 170)
- Remove `attn_param_` from `LanguageModel` constructor call (line 552)
- Remove YAML parsing of `AttentionParam` fields:
  - `attn_param_.softmax_scale` (line 368) — dead code
  - `attn_param_.use_logn_attn` (line 370) — dead code
  - `attn_param_.max_position_embeddings` (line 371) — dead code
  - `parse_rope_param(attention["rope_param"], attn_param_.rope)` (line 373) — rope comes from `AttentionWeight::rope_`
- Move `cache_block_seq_len` parsing: store into `engine_param_` instead of `attn_param_`

### 7. Dead code cleanup

These become unused when `AttentionParam` is removed:

- `RopeParam` struct and related types (`YarnRopeParam`, `Llama3RopeParam`, `MropeRopeParam`) in `llama_rope.h`
- `init_rope_kernel_param(const RopeParam&, ...)` overload in `llama_rope.h` (the `RopeConfig` overload in `attention_weight.cc` remains)
- `to_rope_config()` in `attention_weight.h`
- `parse_rope_param()` and all its helpers (`parse_default_rope_param`, `parse_linear_rope_param`, `parse_dynamic_rope_param`, `parse_yarn_rope_param`, `parse_llama3_rope_param`, `parse_mrope_rope_param`) in `turbomind.cc`

## Not changed

- `AttentionParams<T>` in `kernels/attention/` — unrelated kernel-level struct
- `AttentionWeight` fields — already has everything needed
- Forward-time weight access — `UnifiedAttentionLayer::Forward` still reads `softmax_scale`, `use_logn_attn`, `rope_.max_position_embeddings` from `weights`
- Python `AttentionConfig` in `config.py` — includes `cache_block_seq_len` but that flows through the weight config, not `AttentionParam`
