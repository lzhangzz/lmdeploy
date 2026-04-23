# Eliminate Early Padded Attributes from Model Specs

## Problem

Model specs eagerly compute padded attributes (`_kv_head_num_padded`,
`_expert_inter_size_padded`, `_inter_sizes_padded`) during `__init__`.
Padding is a TP concern that should live closer to where it's consumed,
not in the spec layer.

## Design

All dimension padding moves to C++ `prepare()`. Python builders pad
weight tensors for TP sharding but pass raw config values. C++ weight
classes derive padded dimensions from actual weight tensors in
`prepare()`.

### Python Changes

**Specs** — Remove padded attributes, store raw values only:

- Delete `_kv_head_num_padded`, `_expert_inter_size_padded`,
  `_inter_sizes_padded` from all 4 spec files
- Set configs with raw values: `config.kv_head_num = raw`,
  `config.inter_size = raw`
- Remove `_pad_kv_head()` and `_pad_inter_size()` calls from specs
  (the helpers stay in `utils.py` for FfnBuilder use)

**FfnBuilder** — Pad weight tensors for TP sharding:

- w1/w3 split along output dim → constrained by `block_out * tp`
- w2 splits along input dim → constrained by `block_in * tp`
- Padded inter_size = `_pad_inter_size(raw, lcm(block_in, block_out), tp)`
  — satisfies both constraints
- Zero-pad w1/w3 output dim and w2 input dim to the padded size
- Reads `block_in`/`block_out` from weight format (not `_group_size`)

**AttentionBuilder** — No new changes:

- Already pads KV weights via `repeat_kv_for_tp()`
- Config gets raw `kv_head_num` (previously was pre-padded by spec)

### C++ Changes

**FfnWeight** (`ffn_weight.cc`):

- Constructor: store raw `inter_size_` — remove
  `TM_CHECK(inter_size_ % tp_size_ == 0)` and `inter_size_ /= tp_size_`
- `prepare()`: read `w1->output_dim` (per-shard from Python TP), set
  `inter_size_` accordingly

**MoeWeight** (`moe_weight.cc`):

- Constructor: store raw `inter_size` — remove
  `inter_size = cfg.inter_size / cfg.tp_size`
- `prepare()`: derive per-rank inter_size from first expert child's
  weight dimensions; update `inter_size` and `block_cfg.inter_size`
  before block FfnWeight creation

**AttentionWeight** (`attention_weight.cc`):

- `prepare()`: derive `kv_head_num` from w_qkv weight dimensions.
  Per-shard output_dim = `(head_num + 2 * padded_kv) / tp * head_dim`,
  so `padded_kv = (local_total * tp - head_num) / 2`.

### Files Changed

| File | Change |
|------|--------|
| `spec.py` | Remove `_kv_head_num_padded`, remove `_pad_kv_head` call |
| `qwen3_spec.py` | Remove `_expert_inter_size_padded`, `_inter_sizes_padded`, use raw values |
| `qwen3_5_spec.py` | Same as qwen3 |
| `glm4_moe_lite_spec.py` | Same as qwen3 |
| `gpt_oss_spec.py` | Same as qwen3 |
| `builder/ffn.py` | Add weight padding before TP sharding in `add_ffn()` |
| `ffn_weight.cc` | Defer inter_size division to `prepare()` |
| `moe_weight.cc` | Defer inter_size computation to `prepare()` |
| `attention_weight.cc` | Derive kv_head_num from weight dims in `prepare()` |

### What Gets Removed

- `_kv_head_num_padded` attribute from `TextModelSpec`
- `_expert_inter_size_padded` attribute from all spec subclasses
- `_inter_sizes_padded` attribute from all spec subclasses
- `_pad_kv_head()` / `_pad_inter_size()` calls from all specs
- `_group_size` dependency for padding (replaced by `block_out` from
  weight format)
