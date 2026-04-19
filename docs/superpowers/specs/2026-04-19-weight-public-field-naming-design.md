# Weight classes: remove trailing underscore from public fields

## Problem

Several TurboMind weight modules expose **public** data members with a trailing
underscore (e.g. `MoeWeight::hidden_dim_`). That suffix is normally reserved for
**private** implementation details (Google C++ style). The same classes already mix
styles: for example `LinearWeight` exposes `input_dim` and `epilogue` without a
suffix next to `format_` and `policy_`. The inconsistency makes the API harder to
read and suggests the wrong visibility convention.

## Solution

Rename every **public** weight field that currently ends in `_` to the same name
without the suffix. **Private** members keep the trailing underscore unchanged.

This is a C++-only API cleanup: declarations in headers, definitions in `*.cc`,
and all read/write sites in `*.cc` / `*.cu` / `*.h` are updated in one change set.

## Class-by-class mapping

### `MoeWeight` (`moe_weight.h`)

| Current            | New              |
|--------------------|------------------|
| `hidden_dim_`      | `hidden_dim`     |
| `inter_size_`      | `inter_size`     |
| `experts_per_token_` | `experts_per_token` |
| `norm_topk_prob_`  | `norm_topk_prob` |
| `shared_gate_`     | `shared_gate`    |
| `routed_scale_`    | `routed_scale`   |
| `router_bias_`     | `router_bias`    |
| `topk_group_`      | `topk_group`     |
| `topk_method_`     | `topk_method`    |
| `n_group_`         | `n_group`        |
| `scoring_func_`    | `scoring_func`   |
| `router_n_groups_` | `router_n_groups` |
| `expert_num_`      | `expert_num`     |

`num_experts()` and any other inline accessors that read `expert_num_` must use
`expert_num` after the rename. Private members (e.g. `layer_id_`, `method_`,
`mlp_bias_`, `data_type_`, `tp_size_`, `tp_rank_`, `act_type_`,
`fuse_silu_act_`, `block_`) are **not** in scope.

### `AttentionWeight` (`attention_weight.h`)

| Current               | New                 |
|-----------------------|---------------------|
| `hidden_dim_`         | `hidden_dim`        |
| `head_dim_`           | `head_dim`          |
| `head_num_`           | `head_num`          |
| `kv_head_num_`        | `kv_head_num`       |
| `kv_lora_rank_`       | `kv_lora_rank`      |
| `q_lora_rank_`        | `q_lora_rank`       |
| `qk_rope_dim_`        | `qk_rope_dim`       |
| `v_head_dim_`         | `v_head_dim`        |
| `bias_`               | `bias`              |
| `qk_norm_`            | `qk_norm`           |
| `tp_size_`            | `tp_size`           |
| `tp_rank_`            | `tp_rank`           |
| `data_type_`          | `data_type`         |
| `window_size_`        | `window_size`       |
| `sink_`               | `sink`              |
| `attn_output_gate_`   | `attn_output_gate`  |
| `softmax_scale_`      | `softmax_scale`     |
| `use_logn_attn_`      | `use_logn_attn`     |
| `rope_`               | `rope`              |

`is_mla()` (and any other inline methods) that reference `kv_lora_rank_` must
use `kv_lora_rank`.

### `DeltaNetWeight` (`delta_net_weight.h`)

| Current           | New             |
|-------------------|-----------------|
| `hidden_dim_`     | `hidden_dim`    |
| `num_k_heads_`    | `num_k_heads`   |
| `num_v_heads_`    | `num_v_heads`   |
| `key_head_dim_`   | `key_head_dim`  |
| `value_head_dim_` | `value_head_dim`|
| `d_conv_`         | `d_conv`        |
| `bias_`           | `bias`          |
| `tp_size_`        | `tp_size`       |
| `tp_rank_`        | `tp_rank`       |
| `data_type_`      | `data_type`     |

### `LinearWeight` (`linear_weight.h`)

| Current   | New       |
|-----------|-----------|
| `format_` | `format`  |
| `policy_` | `policy`  |

`input_dtype()` / `output_dtype()` continue to read `policy.input_dtype` and
`policy.output_dtype` after renaming `policy_` to `policy`. `copy_metadata_to`
and any kernel code (e.g. `LlamaLinear.cu`) that touches `format_` / `policy_`
must be updated. **Do not** rename private `has_bias_` or `is_grouped_`.

## Representative call sites (non-exhaustive)

Implementation must grep for each old identifier and fix every remaining use.
Known areas include:

- `models/moe_weight.cc` — constructor and `prepare()` assignments.
- `models/attention_weight.cc`, `models/delta_net_weight.cc`,
  `models/linear_weight.cc` — constructors, `prepare()`, `copy_metadata_to`.
- `models/llama/moe_ffn_layer.cc` — reads MoE config fields from `p.weights`.
- `models/model_weight.cc` — reads attention fields into **ModelWeight**’s own
  public derived fields (see errata below).
- `models/llama/unified_attention_layer.cc` — `rope` and related config.
- `models/llama/GatedDeltaNetLayer.cc`, `engine/engine.cc` — DeltaNet fields.
- `models/llama/LlamaLinear.cu` — `policy` quant fields.

New references may appear after rebases; the authoritative check is **zero**
matches for the old public names on the types above (while private `*_` names
elsewhere remain valid).

## Errata — `ModelWeight` public fields

An earlier draft listed all of `ModelWeight` as out of scope. In the current
codebase, `ModelWeight` exposes **public** derived/runtime fields with a trailing
underscore (same convention issue):

`data_type_`, `hidden_units_`, `vocab_size_`, `vocab_size_padded_`,
`embedding_size_`, `num_layer_`, `head_dim_`, `kv_head_num_`, `tp_size_`,
`tp_rank_`.

(`layer_types_` is already suffix-free.) These public members should be renamed
the same way. **Private** `ModelWeight` members (`stream_`, `alloca_`,
`layers_cache_`) stay unchanged. Call sites include `unified_decoder.cc`,
`engine.cc`, `language_model.cc`, and `turbomind.cc`. See **Task 6** in
`docs/superpowers/plans/2026-04-19-weight-public-field-naming.md`.

## Out of scope

- **Private** fields on any weight class (including `FfnWeight`, `NormWeight`,
  `ModelWeight`’s `stream_` / `alloca_` / `layers_cache_`, and private sections of
  the classes above).
- Python deploy / binding code unless it directly references these C++ member
  names (unlikely).
- Broader style refactors (e.g. replacing public data with getters) beyond this
  rename.

## Validation

- Full C++ build for the project’s normal CMake / Ninja (or CI-equivalent)
  configuration until the tree is clean.
- If the project maintains C++ unit tests for affected layers, run them.
- After edits, confirm by search that the **old public** names listed in the
  tables no longer appear in the codebase for these types.

## Risks and notes

- **Name collisions:** Renaming to `format` / `policy` / `bias` / `rope` are
  ordinary member names; if any translation unit defines macros that clash, fix
  by local include order or undef as needed (rare in this codebase).
- **Mechanical errors:** Prefer identifier-specific search over blind replace so
  unrelated `*_` private members are not changed.

## Approval

Design agreed in session: single change set (Approach 1), including
`LinearWeight::format` / `policy` (option A).
