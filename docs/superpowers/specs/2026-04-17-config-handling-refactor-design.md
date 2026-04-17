# TurboMind Deploy Config Handling Refactor

Date: 2026-04-17

## Problem

HuggingFace-config parsing is scattered across four locations per model in
`lmdeploy/turbomind/deploy/`:

1. `BaseInputModel.model_info()` (e.g. `qwen3_5_spec.py:_qwen35_model_info_base`)
   translates the HF dict to a flat turbomind-style info dict.
2. `TextModelSpec.__init__` re-parses HF into private attributes
   (`self._layer_types`, `self._linear_qkv_split`, `self._n_experts`, layer-prefix
   detection, etc.).
3. `TextModelSpec` factory methods go back to the raw HF dict via `self.cfg[...]`
   when `ModelConfig` doesn't carry the field they need (e.g.
   `glm4_moe_lite_spec.attn` reads `cfg['qk_nope_head_dim']`).
4. `BaseOutputModel.finalize_config` merges the flat info dict into
   `ModelConfig`, then mutates it in place (pads `kv_head_num`, `inter_size`,
   `expert_inter_size`).

Downstream of that, each builder file defines a bespoke
`make_*_config(mc, ...)` adapter (`attention.py:24`, `mla.py:16`, `ffn.py:28`,
`moe.py:14`, `deltanet.py:23`) that reads a subset of `ModelConfig` plus
explicit kwargs and emits a C++ module config struct (`_tm.AttentionConfig`,
`_tm.FfnConfig`, etc.). The subset is inconsistent — `window_size` is per-layer
but passed explicit, `attn_output_gate` is read from `mc`, MLA's `qk_nope_dim`
is explicit because it isn't on `mc`.

The resulting pain clusters:

- **Duplicated HF parsing.** Every new spec re-implements head-dim inference,
  RoPE parsing, `tie_word_embeddings`, MoE-field extraction.
- **Two sources of truth per spec.** `self.cfg` (HF dict) and `self._mc`
  (`ModelConfig`) are used inconsistently.
- **`ModelConfig` is a god dataclass.** ~40 fields covering MLA, MoE,
  linear-attention, sliding window, attention sinks, etc., with `None`/`0`
  defaults that silently mask missing data.
- **Mutation in `finalize_config`.** `pad_for_tp` in `attention.py:96-105`
  has to derive head counts from *tensor shapes* rather than from `ModelConfig`
  because the config was mutated under it.
- **Per-layer lists mixed with scalars.** `inter_size`, `expert_num`,
  `window_size`, `layer_types` are lists; every consumer has defensive
  `list[i] if list and i < len(list) else 0` reads.

## Goals

- One HF parse per spec. Parsed state lives on the spec; `self.cfg` is used
  only during `__init__`.
- No Python-side dataclass that mirrors the shape of a C++ module config.
  Spec holds C++ configs directly; per-layer variants are `clone()`d.
- Delete `make_*_config(mc, ...)` adapter factories; spec writes C++ configs
  itself.
- Narrow `ModelConfig` to the YAML wire format consumed by
  `src/turbomind/turbomind.cc` (plus a handful of Python-only metadata fields
  pending a later cleanup).
- Replace `finalize_config`'s dict-merge/mutate dance with a direct
  `spec.to_legacy_config()` call.
- Merge `BaseInputModel` and `TextModelSpec` into one class per architecture.

## Non-goals

- Changing the C++ typed `ModuleConfig` structs (already refactored in the
  April 9 round).
- Changing the YAML schema read by `turbomind.cc`. We narrow what Python
  emits, but every field `turbomind.cc` still reads keeps its name and type.
- Changing `TurbomindEngineConfig` (user-facing runtime config).
- Extracting `RuntimeConfig` out of `ModelConfig`. The Python-only metadata
  fields (`model_arch`, `chat_template`, `attn_tp_size`, etc.) stay on
  `ModelConfig` for this refactor; a future refactor can move them.
- Removing `ModelConfig` entirely. It stays as the YAML wire format between
  the Python deploy layer and `turbomind.cc`; a future refactor can eliminate
  the execution layer's dependency on it.

## Design

### Target pipeline

```
turbomind.py._from_hf(model_path, engine_config)
│
├─ get_tm_config(model_path, model_name, chat_template_name, engine_config)
│    ├─ Load HF config.json → hf_cfg: dict
│    ├─ Resolve 'auto' dtype → concrete 'float16'/'bfloat16';
│    │  resolve model_format, group_size; assign attn/mlp TP sizes.
│    │  Mutations land on engine_config (or a resolved copy of it) BEFORE
│    │  the spec is instantiated, so engine_cfg.dtype is concrete when the
│    │  spec reads it.
│    ├─ spec_cls = INPUT_MODELS.get(arch_name)
│    ├─ spec = spec_cls(hf_cfg, engine_config)    # parses HF once into templates
│    ├─ tm_cfg = spec.to_legacy_config()          # YAML wire format
│    └─ return (spec, tm_cfg, model_path)
│
├─ self._postprocess_config(tm_cfg, engine_config)
├─ model_comm = _tm.TurboMind.create(config=yaml.safe_dump(tm_cfg.to_dict()))
└─ self._tm_model = OUTPUT_MODELS.get('tm')(
       spec=spec, cfg=tm_cfg, model_comm=model_comm,
       gpu_count=self.gpu_count, model_path=model_path)
```

Inside `TurbomindModel`:

```
TurbomindModel.__init__
├─ store spec, tm_cfg, model_comm, gpu_count, model_path
└─ TextModelLoader(self)
     └─ spec.bind_runtime(contexts, root_handles, attn_ranks, mlp_ranks)

TurbomindModel.export()
├─ params = create_loader(model_path, spec._layer_pattern,
│                         spec._loader_mappings).all_items()
├─ spec.set_params(params)
└─ spec.model()          # spec builds per-layer C++ configs + commits weights
```

### `TextModelSpec` — merged class

`BaseInputModel` and `TextModelSpec` collapse into one class per architecture.
`INPUT_MODELS` registry maps arch name → `TextModelSpec` subclass.

```python
# lmdeploy/turbomind/deploy/spec.py
class TextModelSpec(ABC):
    """Text model spec: HF config → C++ configs + weight commits.

    Subclass contract:
      - __init__ parses HF into self attributes (call super().__init__ first).
      - Override factory methods that apply; factory method NAMES are a
        convention for readability, NOT a protocol. Signatures may differ
        across specs.
    """

    # Class-level loader config (for checkpoint parameter renaming)
    _layer_pattern: str = ''
    _loader_mappings: list = []

    # -------- init / parsing --------------------------------------------
    def __init__(self, hf_cfg: dict, engine_cfg: TurbomindEngineConfig):
        self.hf_cfg = hf_cfg
        self.engine_cfg = engine_cfg
        self._parse_base(hf_cfg)

    def _parse_base(self, cfg):
        """Populate canonical scalars from standard HF keys.
        Fills self._num_layer / _vocab_size / _norm_eps / _head_num /
        _kv_head_num / _kv_head_num_padded / _head_dim / _hidden_units /
        _rope / _max_position_embeddings / _tie_embeddings /
        _layer_prefix / _embed_key / _norm_key / _model_name /
        _tune_layer_num / _embedding_size.
        Subclass overrides this wholesale or supplements in its own __init__.
        `_softmax_scale` and `_group_size` are subclass-responsibility
        (MLA/YaRN for softmax_scale; quant config for group_size).
        """

    # -------- runtime binding (called by TextModelLoader) ---------------
    def bind_runtime(self, *, contexts, root_handles, attn_ranks, mlp_ranks):
        self._contexts = contexts
        self._root_handles = root_handles
        self._attn_ranks = attn_ranks
        self._mlp_ranks = mlp_ranks

    def set_params(self, params: dict):
        self.params = params

    # -------- YAML export (called by finalize_config) -------------------
    def to_legacy_config(self) -> TurbomindModelConfig:
        """Assemble YAML wire-format config by copying fields from the C++
        configs stored on self, plus orchestration scalars."""
        mc = ModelConfig()
        self._copy_template_fields(mc)       # from self._attn_cfg, _ffn_cfg, _dn_cfg
        self._copy_orchestration_fields(mc)  # from scalars + engine_cfg
        self._copy_perlayer_fields(mc)       # subclass-specific
        ac = self._build_attention_config()
        return TurbomindModelConfig(model_config=mc, attention_config=ac,
                                    lora_config=LoraConfig())

    # -------- checkpoint access -----------------------------------------
    def _get(self, key: str) -> torch.Tensor | None:
        return self.params.get(key)

    def _linear(self, pfx: str):
        from .kind_map import build_linear
        return build_linear(self.params, pfx)

    # -------- default factory methods (text-model universals) -----------
    def token_embeds(self, key): ...    # pad vocab, LinearBuilder
    def lm_head(self, key): ...         # pad vocab, transpose, LinearBuilder

    # No default attn/ffn/moe/linear_attn/mla/norm/output_norm/layers/
    # model/num_experts. Subclasses define these as needed with whatever
    # signatures make sense.
```

### C++ configs as canonical parsed state

During `__init__`, each spec constructs **template** C++ configs and stores
them on `self`. Static fields are populated at template-build time;
per-layer fields are set on a `clone()` inside the factory method.

```python
class Qwen3_5Spec(TextModelSpec):
    def __init__(self, hf_cfg, engine_cfg):
        super().__init__(hf_cfg, engine_cfg)

        # Attention template (static fields only)
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim       = self._hidden_units
        self._attn_cfg.head_dim         = self._head_dim
        self._attn_cfg.head_num         = self._head_num
        self._attn_cfg.kv_head_num      = self._kv_head_num_padded
        self._attn_cfg.has_bias         = bool(hf_cfg.get('attention_bias', 0))
        self._attn_cfg.qk_norm          = True
        self._attn_cfg.attn_output_gate = True
        self._attn_cfg.rope_dim         = self._rope.dim
        self._attn_cfg.tp_size          = engine_cfg.attn_tp_size
        self._attn_cfg.data_type        = _cpp_dtype(engine_cfg.dtype)

        # DeltaNet template for linear-attn layers
        self._dn_cfg = _tm.DeltaNetConfig()
        self._dn_cfg.hidden_dim      = self._hidden_units
        self._dn_cfg.num_k_heads     = hf_cfg['linear_num_key_heads']
        # ... etc

        # FFN template
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = _cpp_dtype(engine_cfg.dtype)
        self._ffn_cfg.act_type   = _act_type_id('silu')

        # MoE template (if present)
        if self._n_experts > 0:
            self._moe_cfg = _tm.MoeConfig()
            self._moe_cfg.method            = 1   # kFused
            self._moe_cfg.experts_per_token = hf_cfg['num_experts_per_tok']
            # ... all static MoE fields

        # Per-layer scalars that don't fit in a single template
        self._window_sizes = self._parse_window_sizes(hf_cfg)
        self._inter_sizes_padded = self._parse_inter_sizes_padded(hf_cfg)
        self._expert_nums = self._parse_expert_nums(hf_cfg)

    def attn(self, pfx, layer):
        cfg = self._attn_cfg.clone()
        cfg.window_size = self._window_sizes[layer]
        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)
        ...

    def moe(self, pfx, layer):
        cfg = self._moe_cfg.clone()
        cfg.layer_id    = layer
        cfg.expert_num  = self._expert_nums[layer]
        cfg.inter_size  = self._expert_inter_sizes_padded[layer]
        moe = MoeBuilder(cfg, self._contexts,
                         tp=self.engine_cfg.mlp_tp_size,
                         ranks=self._mlp_ranks)
        ...
```

Fields that remain as simple `self._*` attributes (not in any C++ config):

- Orchestration scalars: `_num_layer`, `_vocab_size`, `_embedding_size`,
  `_norm_eps`, `_tune_layer_num`, `_model_name`, `_group_size`,
  `_tie_embeddings`, `_layer_prefix`, `_embed_key`, `_norm_key`,
  `_softmax_scale`, `_max_position_embeddings`, `_rope` (RopeParam).
- Per-layer lists: `_window_sizes`, `_inter_sizes_padded`, `_expert_nums`,
  `_expert_inter_sizes_padded`, `_layer_types`.

`tp_rank` stays handled by `Builder._ensure_handles` (existing mechanism —
the Builder clones the config and sets `tp_rank` per GPU). Spec does not
touch `tp_rank`.

### `to_legacy_config()` — mechanical copy

```python
def to_legacy_config(self) -> TurbomindModelConfig:
    mc = ModelConfig()

    # Copy from _attn_cfg (covers both standard attn and MLA)
    a = self._attn_cfg
    mc.hidden_units     = a.hidden_dim
    mc.head_num         = a.head_num
    mc.kv_head_num      = a.kv_head_num
    mc.size_per_head    = a.head_dim
    mc.q_lora_rank      = a.q_lora_rank
    mc.kv_lora_rank     = a.kv_lora_rank
    mc.qk_rope_dim      = a.qk_rope_dim
    mc.v_head_dim       = a.v_head_dim
    mc.attn_bias        = int(a.has_bias)
    mc.qk_norm          = a.qk_norm
    mc.attn_sink        = a.attn_sink
    mc.attn_output_gate = a.attn_output_gate

    # Copy from _ffn_cfg
    f = self._ffn_cfg
    mc.mlp_bias        = f.has_bias
    mc.activation_type = _act_type_str(f.act_type)

    # Copy from _dn_cfg if present
    if hasattr(self, '_dn_cfg'):
        dn = self._dn_cfg
        mc.linear_num_key_heads    = dn.num_k_heads
        mc.linear_num_value_heads  = dn.num_v_heads
        mc.linear_key_head_dim     = dn.key_head_dim
        mc.linear_value_head_dim   = dn.value_head_dim
        mc.linear_conv_kernel_dim  = dn.d_conv

    # Per-layer lists
    mc.inter_size  = self._inter_sizes_padded
    mc.window_size = self._window_sizes
    mc.layer_types = self._layer_types

    # Orchestration scalars
    mc.num_layer       = self._num_layer
    mc.vocab_size      = self._vocab_size
    mc.embedding_size  = self._embedding_size
    mc.norm_eps        = self._norm_eps
    mc.tune_layer_num  = self._tune_layer_num
    mc.model_name      = self._model_name
    mc.data_type       = self.engine_cfg.dtype
    mc.session_len     = self.engine_cfg.session_len
    mc.group_size      = self._group_size

    # Engine-level fields mirrored onto ModelConfig (legacy)
    mc.attn_tp_size    = self.engine_cfg.attn_tp_size
    mc.attn_cp_size    = self.engine_cfg.attn_cp_size
    mc.mlp_tp_size     = self.engine_cfg.mlp_tp_size
    mc.model_format    = self.engine_cfg.model_format

    ac = AttentionConfig(
        rope_param=self._rope,
        max_position_embeddings=self._max_position_embeddings,
        softmax_scale=self._softmax_scale,
    )
    return TurbomindModelConfig(model_config=mc, attention_config=ac,
                                lora_config=LoraConfig())
```

The base class splits this into `_copy_template_fields`,
`_copy_orchestration_fields`, and `_copy_perlayer_fields`. Subclasses
override only `_copy_perlayer_fields` when adding arch-specific
per-layer lists (e.g. Qwen3.5 adds `layer_types`; gpt-oss adds
`window_size`; Llama doesn't need to override).

### `ModelConfig` narrowing

Twelve MoE fields removed — all dead after `make_moe_config(mc, ...)` is
deleted, since the spec now writes MoE fields directly into
`_tm.MoeConfig`:

```
expert_num, expert_router_bias, expert_inter_size, experts_per_token,
moe_shared_gate, norm_topk_prob, routed_scale, topk_group, topk_method,
moe_group_num, scoring_func, router_n_groups
```

Fields kept fall into two groups:

- **Read by turbomind.cc from YAML:** `model_name`, `data_type`, `head_num`,
  `kv_head_num`, `size_per_head`, `hidden_units`, `inter_size`, `num_layer`,
  `vocab_size`, `embedding_size`, `norm_eps`, `tune_layer_num`,
  `session_len`, `group_size`, `attn_bias`, `mlp_bias`, `qk_norm`,
  `attn_sink`, `attn_output_gate`, `window_size`, `activation_type`,
  `q_lora_rank`, `kv_lora_rank`, `qk_rope_dim`, `v_head_dim`, `layer_types`,
  `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim`,
  `linear_num_key_heads`, `linear_num_value_heads`.
- **Python-only metadata used by `converter.py` / `turbomind.py`:**
  `model_arch`, `chat_template`, `model_format`, `attn_tp_size`,
  `attn_cp_size`, `mlp_tp_size`. These live on `ModelConfig` for legacy
  reasons; moving them to a `RuntimeConfig` is a separate refactor.

`AttentionConfig` and `LoraConfig` are unchanged.

### TP-aware padding moves onto the spec

Three mutations `finalize_config` does today:

1. Pad each `inter_size[i]` to `group_size * mlp_tp`.
2. Pad `expert_inter_size` to `group_size * mlp_tp`.
3. Pad `kv_head_num` up to `attn_tp` when `attn_tp % kv_head_num == 0`.

All three become derived fields computed once during `_parse_base` (or its
subclass extensions) using `engine_cfg.attn_tp_size` / `engine_cfg.mlp_tp_size`.
Stored as `self._kv_head_num_padded`, `self._inter_sizes_padded`,
`self._expert_inter_sizes_padded`. Both `to_legacy_config()` and factory
methods read the padded values; no runtime mutation of `ModelConfig`.

The shared helper `_pad_inter_size` moves to `source_model/utils.py`
alongside the existing RoPE helpers.

### `finalize_config` after the cut

```python
@classmethod
def finalize_config(cls, spec, cfg):
    produced = spec.to_legacy_config()
    # cfg was pre-seeded by converter with data_type, model_format,
    # model_arch, group_size, session_len, chat_template, model_name.
    # Preserve those on the produced config.
    for name in ('model_arch', 'data_type', 'model_format', 'group_size',
                 'session_len', 'chat_template', 'model_name'):
        val = getattr(cfg.model_config, name, None)
        if val not in (None, '', 0):
            setattr(produced.model_config, name, val)
    produced.model_config.verify()
    cfg.model_config     = produced.model_config
    cfg.attention_config = produced.attention_config
    cfg.lora_config      = produced.lora_config
```

No `config_to_dict` / `config_from_dict` round-trip; no MoE-field merge; no
in-place padding.

### Builder-side changes

Delete `make_*_config(mc, ...)` factories from every builder file:

| File | Factory removed |
|---|---|
| `builder/attention.py` | `make_attention_config(mc, ...)` |
| `builder/mla.py` | `make_mla_config(mc, ...)` |
| `builder/ffn.py` | `make_ffn_config(mc, ...)` |
| `builder/moe.py` | `make_moe_config(mc, ...)` |
| `builder/deltanet.py` | `make_deltanet_config(mc, ...)` |

Builder `__init__.py` drops those exports. Builder class bodies, pipeline
helpers (`pad_for_tp`, `fuse_qkv`, `fuse_gdn`, `fold_kv_b`, etc.), and
primitive wrappers (`make_linear_config(input_dim, output_dim, data_type)`,
`make_norm_config(dim, data_type)`) are unchanged.

Dtype/enum helpers (`_cpp_dtype`, `_act_type_id`, `_torch_dtype_to_cpp`)
stay in `builder/_base.py`; specs import them to set C++ config fields.

## Files touched

| File | Change |
|---|---|
| `lmdeploy/turbomind/deploy/config.py` | Narrow `ModelConfig` (remove 12 MoE fields). |
| `lmdeploy/turbomind/deploy/spec.py` | Become the new `TextModelSpec` base with merged responsibilities. |
| `lmdeploy/turbomind/deploy/source_model/base.py` | Delete `BaseInputModel`. Keep `INPUT_MODELS` registry. |
| `lmdeploy/turbomind/deploy/source_model/utils.py` | Add `_pad_inter_size`, `_pad_kv_head`, `_detect_layer_prefix`. |
| `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Rewrite to new pattern; delete `Qwen3InputModel`. |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Rewrite to new pattern; delete `Qwen3_5InputModel`. |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Rewrite; delete `GptOssInputModel`. |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Rewrite; delete `Glm4MoeLiteInputModel`. |
| `lmdeploy/turbomind/deploy/builder/attention.py` | Remove `make_attention_config`. |
| `lmdeploy/turbomind/deploy/builder/mla.py` | Remove `make_mla_config`. |
| `lmdeploy/turbomind/deploy/builder/ffn.py` | Remove `make_ffn_config`. |
| `lmdeploy/turbomind/deploy/builder/moe.py` | Remove `make_moe_config`. |
| `lmdeploy/turbomind/deploy/builder/deltanet.py` | Remove `make_deltanet_config`. |
| `lmdeploy/turbomind/deploy/builder/__init__.py` | Drop removed exports. |
| `lmdeploy/turbomind/deploy/target_model/base.py` | Shrink `finalize_config`; `__init__` takes `spec` and `model_path`; update `export()`. |
| `lmdeploy/turbomind/deploy/text_model_loader.py` | Shrink to ~10 lines: gather handles/ranks, call `spec.bind_runtime`. |
| `lmdeploy/turbomind/deploy/converter.py` | `get_tm_config` creates spec directly; returns `(spec, tm_cfg, model_path)`. |
| `lmdeploy/turbomind/turbomind.py` | `_from_hf` unpacks `(spec, tm_cfg, model_path)` and passes to `TurbomindModel`. |

No changes to C++ code. No changes to `TurbomindEngineConfig`.
No changes to the YAML schema consumed by `turbomind.cc`.

## Migration strategy

Single atomic refactor on a feature branch. Suggested ordering:

1. Narrow `ModelConfig` in `config.py` (drop the 12 MoE fields).
2. Add shared helpers in `source_model/utils.py`.
3. Write the new `TextModelSpec` base in `spec.py`.
4. Migrate `qwen3_spec.py` as pilot (simplest arch). Validate end-to-end.
5. Migrate remaining specs: `qwen3_5_spec.py`, `gpt_oss_spec.py`,
   `glm4_moe_lite_spec.py`, and any others.
6. Delete `BaseInputModel` and all `XxxInputModel` classes.
7. Update orchestration (`converter.py`, `target_model/base.py`,
   `text_model_loader.py`, `turbomind.py`).
8. Remove `make_*_config` factories from builder files.
9. Run full test matrix (below) before merge.

Steps 1–3 are additive. Step 4 is the first point where the new pipeline
runs end-to-end; design issues surface here before they multiply. Steps 5–8
are mechanical cleanup. Step 9 is the merge gate.

## Testing

Validation via `scripts/test_turbomind_model.py` driven by the
`turbomind-tester` agent (no unit tests exist for this module).

Test matrix — one row per spec file, covering every architectural feature:

| Spec | Model | Coverage |
|---|---|---|
| qwen3_spec | Qwen3 dense | baseline self-attn + dense FFN |
| qwen3_spec | Qwen3-MoE | standard attn + MoE |
| qwen3_5_spec | Qwen3.5 dense | GDN linear-attn + full attn + zero-centered norm |
| qwen3_5_spec | Qwen3.5-MoE | linear-attn + MoE + packed experts |
| gpt_oss_spec | gpt-oss | sliding window + attn sinks + packed experts + MXFP4 |
| glm4_moe_lite_spec | glm4-moe-lite | MLA + MoE + YaRN |

Per model: `tp=1` unquantized, `tp=2` unquantized, plus one quantized format
(`awq` / `fp8` / `mxfp4`) when available.

Validation criteria per run:
1. Return code = 0.
2. Response contains meaningful words relevant to the prompt (per tester spec).
3. YAML wire-format diff matches expectation — `tm_cfg.to_dict()` before and
   after the refactor differs only by the 12 dropped MoE fields. Everything
   else is byte-for-byte identical for fields `turbomind.cc` reads.

### Risk areas

| Risk | Mitigation |
|---|---|
| Field-name mismatch in `to_legacy_config` (e.g., Python `attn_bias` vs C++ `has_bias`) | Golden YAML diff per model catches these silently before they cause correctness drift. |
| TP-padding moved from `finalize_config` to spec `_parse_base` — potential behavior drift | Explicit assert in the spec's `_parse_base`: padded values match what `finalize_config` produced pre-refactor on the same checkpoint. |
| `hf_overrides` → `rope_scaling` round-trip breaks because of narrowed `ModelConfig` | `config_from_dict` filters by dataclass fields, so removed fields in input YAML are ignored. Verify on a model with user-supplied `hf_overrides`. |
| `RopeParam` ownership | Stays as-is — narrow Python dataclass on `self._rope`; exported to `AttentionConfig.rope_param` in YAML; `_attn_cfg.rope_dim` set from `self._rope.dim` at template-build time. |
| `update_params` path (iterator reload) | `spec.bind_runtime` is called once; `export_iter` only re-runs `set_params` + `model()`. Verify no state leakage between iterations. |

## Future cleanups out of scope

- Extract Python-only fields (`model_arch`, `chat_template`,
  `attn_tp_size`, etc.) off `ModelConfig` into a `RuntimeConfig`.
- Eliminate `ModelConfig` entirely — requires `turbomind.cc` to stop reading
  the YAML wire format and read the typed C++ configs directly.
- Simplify `pad_for_tp` in `attention.py` to trust `ModelConfig.kv_head_num`
  again now that it isn't mutated (currently derives from tensor shapes).
