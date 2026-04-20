# Eliminate `TurbomindModelConfig` Design

## Goal

Delete `TurbomindModelConfig` from `lmdeploy/turbomind/deploy/config.py` entirely.
After a prior refactor ([eliminate Python ModelConfig](2026-04-20-eliminate-model-config-python-design.md))
reduced it to a thin field bag, every remaining field is either a duplicate of
`TurbomindEngineConfig`, a dead write, or a YAML container whose one live section
(`AttentionConfig`) can be produced directly from the spec. This design removes the
class and its two methods (`update_from_engine_config`, `to_dict`), plus the now-dead
`LoraConfig` dataclass. After the change, `config.py` holds only `RopeParam` and
`AttentionConfig` — the two genuine live dataclasses.

## Current State

`TurbomindModelConfig` currently has 12 fields (plus two methods). Classifying by
actual downstream reads:

| Field | Status |
|---|---|
| `data_type`, `model_format`, `session_len`, `attn_tp_size`, `attn_cp_size`, `mlp_tp_size` | Duplicated onto `engine_config` at `converter.py:195-210`. No reader reads the `tm_cfg` copy after that point. |
| `model_arch`, `chat_template`, `model_name`, `group_size` | Written in `converter.py` but never read anywhere. Pure dead state. |
| `lora_config` | Serialized to YAML. C++ (`src/turbomind/turbomind.cc:192-193`) reads only `attention_config` and `engine_config`; the `lora_config` section is discarded. |
| `attention_config` | Live. C++ reads `cache_block_seq_len`, `rope_param`, `max_position_embeddings`, `softmax_scale`, `use_logn_attn` from the YAML. |

`update_from_engine_config()` does exactly three non-mechanical things that survive
the refactor — patching `attention_config` with engine_config values:

1. `cache_block_seq_len` ← `engine_config.cache_block_seq_len`
2. `use_logn_attn` ← `engine_config.use_logn_attn`
3. If `engine_config.rope_scaling_factor` is nonzero: replace `rope_param` with a
   `dynamic`-type one carrying the scaling factor.

`_postprocess_config()` in `turbomind.py` consists of a deep-copy hedge (no longer
needed once patching moves to the spec), the `update_from_engine_config` call
(moved), and a 3-line YAML dict build (inlined into `_from_hf`). `self.config` and
`self.config_dict` attributes on `TurboMind` are dead after the refactor — the only
external read of `config_dict` is an ugly indirection at `turbomind.py:497`
(`self.tm_model.config_dict['engine_config'].get('empty_init', False)`) that should
already have been reading `engine_config.empty_init` directly.

`TurboMindInstance.__init__` accepts a `config: TurbomindModelConfig` parameter and
stores it as `self.config`, but no code reads it.

## Design Decisions

1. **Move the `AttentionConfig` patching onto the spec.** The three engine-config-
   sourced fields become part of `spec.to_attention_config()`, which already reads
   `self.engine_cfg` and owns every other `AttentionConfig` input (`_rope`,
   `_max_position_embeddings`, `_softmax_scale`). This keeps all `AttentionConfig`
   construction in one place. Alternative (free function in `turbomind.py`) was
   rejected as pointless scattering.

2. **Drop `lora_config` from the YAML; delete the `LoraConfig` dataclass.** C++
   ignores the section. No Python reader exists. `config.py` shrinks to just the two
   meaningful dataclasses. If LoRA support returns to turbomind, a future PR adds it
   back with real values.

3. **Fold `get_output_model_registered_name_and_config` into `get_tm_config`; delete
   the function and its direct tests.** The function has one production caller and
   no independent reason to exist once its `TurbomindModelConfig` return type is
   gone. The tests that hit it (`test_registered_models`,
   `test_torch_dtype_fallback`, `test_update_from_engine_config`, `test_dtype`,
   `test_compressed_tensors_support_matrix`) are already stale against HEAD — they
   reference the removed `config.model_config.*` attribute. Coverage for dtype
   fallback, group-size validation, and YAML building moves to the end-to-end
   `scripts/test_turbomind_model.py` smoke runs.

## Design

### 1. `spec.to_attention_config()` absorbs engine-config patching

New method body (replaces both current `to_attention_config` and
`_build_attention_config` on `TextModelSpec`):

```python
def to_attention_config(self) -> AttentionConfig:
    """Produce the AttentionConfig for YAML serialization.

    Reads from self.engine_cfg for runtime-patched fields
    (cache_block_seq_len, use_logn_attn, rope_scaling_factor).
    """
    cfg = AttentionConfig(
        rope_param=self._rope,
        max_position_embeddings=self._max_position_embeddings,
        softmax_scale=self._softmax_scale,
    )
    ec = self.engine_cfg
    if ec.cache_block_seq_len:
        cfg.cache_block_seq_len = ec.cache_block_seq_len
    if ec.use_logn_attn:
        cfg.use_logn_attn = int(ec.use_logn_attn)
    if ec.rope_scaling_factor:
        rope = cfg.rope_param or RopeParam(type='', base=0, dim=0)
        rope.type = 'dynamic'
        rope.factor = ec.rope_scaling_factor
        rope.max_position_embeddings = cfg.max_position_embeddings
        cfg.rope_param = rope
        logger.warning(
            '`--rope-scaling-factor` will be removed in a future release. '
            'Please instead use `--hf-overrides`.')
    return cfg
```

The `if ec.X:` guards preserve today's semantics: `update_from_engine_config` skipped
`None`/falsy values, and `AttentionConfig` defaults (e.g., `cache_block_seq_len=64`)
must not be overwritten by `None`. `_build_attention_config` is inlined — it had one
caller and no longer adds value.

### 2. YAML building inlines into `_from_hf`; `_postprocess_config` is deleted

New `_from_hf`:

```python
def _from_hf(self, model_path: str, engine_config: TurbomindEngineConfig):
    from .deploy.converter import get_tm_config
    from .deploy.target_model.base import OUTPUT_MODELS

    assert is_supported(model_path), (
        f'turbomind does not support {model_path}. '
        'Plz try pytorch engine instead.')

    spec, model_path = get_tm_config(model_path, engine_config)

    self._vocab_size = spec._vocab_size
    self.engine_config = engine_config

    config_dict = {
        'attention_config': asdict(spec.to_attention_config()),
        'engine_config': asdict(engine_config),
    }
    logger.info(f'turbomind model config:\n\n'
                f'{json.dumps(config_dict, indent=2)}')

    model_comm = _tm.TurboMind.create(
        model_dir='', config=yaml.safe_dump(config_dict))
    self._create_weight(model_comm)

    self._tm_model = OUTPUT_MODELS.get('tm')(
        spec=spec, model_comm=model_comm, gpu_count=self.gpu_count,
        model_path=model_path)
    return model_comm
```

`self.config` and `self.config_dict` attributes are deleted. `config_dict` is a local
variable — once passed to `yaml.safe_dump`, it goes out of scope.

### 3. `converter.get_tm_config()` returns `(spec, model_path)`; mutates engine_config

`get_output_model_registered_name_and_config` is folded in. Its dtype-inference block
becomes a private `_resolve_dtype(requested, hf_cfg)` helper (the `dtype == 'auto'` →
torch_dtype detection + bf16 fallback at today's lines 86-99).

```python
def get_tm_config(model_path, engine_config: TurbomindEngineConfig,
                  group_size: int = None):
    """Resolve dtype/model_format/group_size/session_len, mutate engine_config
    in place, build the spec.

    Returns:
        tuple: (spec, model_path)
    """
    # 1. Load HF config once; reused for quant_config, dtype, and session_len.
    _, hf_model_cfg = get_model_arch(model_path)

    # 2. Reconcile quant_config (unchanged block, today's 131-166)
    quant_config = search_nested_config(
        hf_model_cfg.to_dict(), 'quantization_config')
    if quant_config:
        # ... existing quant_method / group_size / desc_act / sym reconciliation ...
        engine_config.model_format = quant_method
        group_size = _group_size

    group_size = _validate_quant_group_size(engine_config.model_format, group_size)
    if engine_config.model_format is None:
        engine_config.model_format = 'hf'

    # 3. Resolve dtype and format overrides
    dtype = _resolve_dtype(engine_config.dtype, hf_model_cfg)
    if engine_config.model_format in ('awq', 'gptq', 'compressed-tensors'):
        dtype = 'float16'
        if engine_config.model_format == 'compressed-tensors':
            engine_config.model_format = 'awq'

    # 4. Resolve session_len default
    session_len_default = _get_and_verify_max_len(hf_model_cfg, None)

    # 5. Mutate engine_config with resolved values
    engine_config.dtype = dtype
    if engine_config.session_len is None:
        engine_config.session_len = session_len_default
    engine_config.attn_tp_size = engine_config.attn_tp_size or 1
    engine_config.attn_cp_size = engine_config.attn_cp_size or 1
    engine_config.mlp_tp_size = engine_config.mlp_tp_size or 1

    # 6. Build spec (hf_overrides handling unchanged)
    hf_cfg = load_model_config(model_path)
    if engine_config.hf_overrides:
        logger.warning(f'Overriding HF config with {engine_config.hf_overrides}')
        _deep_merge(hf_cfg, engine_config.hf_overrides)
    spec_name = get_spec_registered_name(model_path, engine_config.model_format)
    spec_cls = INPUT_MODELS.get(spec_name)
    spec = spec_cls(hf_cfg, engine_config, group_size=group_size or 0)

    return spec, model_path
```

`model_name`, `chat_template_name`, `model_arch` are no longer recorded anywhere —
they were dead writes on `TurbomindModelConfig`. `model_name` and
`chat_template_name` are dropped from `get_tm_config()`'s signature entirely.
Verified via grep: `get_tm_config` has exactly one caller (`turbomind._from_hf` at
`turbomind.py:236`) and zero test callers, so the signature change is safe.

### 4. `BaseOutputModel` takes only the runtime handles

```python
class BaseOutputModel(ABC):
    """Base output model. Drives a TextModelSpec through loading + commit."""

    def __init__(self, spec, model_comm, gpu_count, model_path):
        from ..text_model_loader import TextModelLoader
        self.spec = spec
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        self.model_path = model_path
        self.model = TextModelLoader(self)

    # root(), context(), tp_ranks(), export(), export_iter(): unchanged
```

- `finalize_config(cls, spec, cfg)` classmethod is deleted. The `AttentionConfig` it
  installed now lives only in the YAML dict, built in `_from_hf`. No persistent
  `attention_config` attribute is needed on `BaseOutputModel`.
- `self.tm_config`, `self.attn_tp_size`, `self.attn_cp_size`, `self.mlp_tp_size`
  attributes are deleted. Verified with grep: all four are set in today's `__init__`
  but never read anywhere else in the codebase.
- `engine_config` is intentionally NOT accepted as a parameter either. Nothing
  downstream reads `output_model.engine_config` — `TurboMindInstance` already
  reaches engine_config via `self.tm_model.engine_config` on the `TurboMind` instance.
  Adding an `engine_config` attribute to `BaseOutputModel` would reintroduce the
  exact "set but never read" anti-pattern this refactor eliminates.

### 5. `TurboMindInstance` drops the dead `config` parameter

```python
def __init__(self, tm_model: 'TurboMind', cuda_stream_id: int = 0):
    self.tm_model = tm_model
    self.cuda_stream_id = cuda_stream_id

    lazy_init = self.tm_model.engine_config.empty_init
    self._model_inst = None if lazy_init else self._create_model_instance()

    self.lock = None
    self.errcode_map = {...}
```

Call site in `TurboMind.create_instance` simplifies:

```python
def create_instance(self, cuda_stream_id=0):
    return TurboMindInstance(self, cuda_stream_id)
```

The ugly `self.tm_model.config_dict['engine_config'].get('empty_init', False)`
indirection is replaced by `self.tm_model.engine_config.empty_init`, reading the
pydantic dataclass field directly.

### 6. `config.py` shrinks to `RopeParam` + `AttentionConfig`

Deleted from `config.py`:
- `TurbomindModelConfig` dataclass (including `update_from_engine_config`, `to_dict`)
- `LoraConfig` dataclass
- `config_to_dict` free function (only callers are `TurbomindModelConfig.to_dict`
  itself)

Retained unchanged:
- `RopeParam`
- `AttentionConfig`

File shrinks from 112 lines to roughly 40.

## Files Changed

| File | Change |
|---|---|
| `lmdeploy/turbomind/deploy/config.py` | Delete `TurbomindModelConfig`, `LoraConfig`, `config_to_dict`. Drop unused imports (`asdict`, `TurbomindEngineConfig`, `logger`). Keep `RopeParam`, `AttentionConfig`. |
| `lmdeploy/turbomind/deploy/spec.py` | `to_attention_config()` reads `self.engine_cfg` and applies the three engine-config patches. Inline `_build_attention_config`. Import `RopeParam` alongside `AttentionConfig` (needed by the `rope_scaling_factor` branch). |
| `lmdeploy/turbomind/deploy/converter.py` | Fold `get_output_model_registered_name_and_config` into `get_tm_config`. Delete the function. Extract `_resolve_dtype(requested, hf_cfg)` private helper. `get_tm_config(model_path, engine_config, group_size=None)` returns `(spec, model_path)` and mutates `engine_config` in place. Drop `TurbomindModelConfig` import. |
| `lmdeploy/turbomind/deploy/target_model/base.py` | `BaseOutputModel.__init__(spec, model_comm, gpu_count, model_path)`. Delete `finalize_config()`, `self.tm_config`, `self.attn_tp_size`, `self.attn_cp_size`, `self.mlp_tp_size`. Drop `TurbomindModelConfig` import. `engine_config` is not stored — no downstream reader. |
| `lmdeploy/turbomind/turbomind.py` | Delete `_postprocess_config`. Inline YAML build in `_from_hf`. Delete `self.config`, `self.config_dict`. Update `TurboMindInstance.__init__` to drop `config` param. Update `create_instance` call site. Fix `lazy_init` read. Drop `TurbomindModelConfig` import. (`copy` import stays — still used at line 139 for `engine_config` deepcopy.) |
| `tests/test_lmdeploy/test_turbomind/test_converter.py` | Delete `test_torch_dtype_fallback`, `test_registered_models`, `test_update_from_engine_config`, `test_dtype`. Keep `test_ffn_reader_kind_none` (unrelated). |
| `tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py` | Delete `test_compressed_tensors_support_matrix`. Keep the three tests that don't touch `converter.get_output_model_registered_name_and_config`. |

## Verification

Run the smoke script unmodified against two cases:

```bash
# Unquantized path — dtype inference, session_len default, YAML building
python scripts/test_turbomind_model.py Qwen/Qwen3-8B <cache_dir> 1 0

# Quantized path — group_size validation, dtype='float16' override
python scripts/test_turbomind_model.py Qwen/Qwen3-8B-AWQ <cache_dir> 1 0
```

Each run must produce a coherent text response. Dynamic-NTK
(`rope_scaling_factor`) is deprecated in `TurbomindEngineConfig` and not exercised
by the smoke script; the five-line patching branch it controls in
`to_attention_config()` is a direct transcription of the current
`update_from_engine_config` block, so code review is sufficient for that path.

## Non-Goals

- **No C++ changes.** YAML schema stays `{attention_config, engine_config}`, which is
  already what C++ reads (`src/turbomind/turbomind.cc:192-193`). The disappearance of
  the `lora_config` key is invisible — C++ never read it.
- **No new unit tests.** The deleted tests were testing `TurbomindModelConfig`
  internals, not external behavior. End-to-end coverage via the smoke script is
  sufficient. A future follow-up can add focused unit tests for
  `spec.to_attention_config()` patching if desired.
- **`RopeParam` and `AttentionConfig` stay in `config.py`.** They're genuine live
  dataclasses with no cleanup backlog.
- **No signature changes to `TextModelSpec.__init__`.** Subclasses remain
  source-compatible; only `to_attention_config()` changes, and its caller is the
  single YAML-building site in `_from_hf`.
- **No LoRA revival.** If turbomind regains LoRA support, a separate PR reintroduces
  the YAML section with real values.

## Scope

Python-side only. Single logical refactor; landable as one PR or split into:

1. Move patching to `spec.to_attention_config()` + delete `LoraConfig` + drop
   `lora_config` from YAML.
2. Eliminate `_postprocess_config`, inline YAML build in `_from_hf`, drop
   `TurboMindInstance.config`.
3. `get_tm_config` returns `(spec, model_path)`; `BaseOutputModel.__init__` accepts
   only runtime handles; delete `finalize_config`; delete
   `get_output_model_registered_name_and_config`.
4. Delete `TurbomindModelConfig` class; clean up test files.

Each step leaves the codebase in a working, testable state. The plan doc (separate
artifact from this spec) will decide final task granularity.
