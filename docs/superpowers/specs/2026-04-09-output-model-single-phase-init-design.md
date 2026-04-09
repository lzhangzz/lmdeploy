# Output Model Single-Phase Init Design

Date: 2026-04-09

## Summary

Eliminate the two-phase initialization of `BaseOutputModel`. Today the object is born incomplete (`model_comm=None`, `gpu_count=0`), externally mutated by `turbomind.py`, and only then usable. After this change, `BaseOutputModel.__init__` receives `model_comm` and `gpu_count` at construction time and creates `TextModelLoader` — which is itself fully initialized in its own `__init__` (no separate `prepare()` call). The to-file export path is removed (dead code).

## Motivation

Three lines in `turbomind.py._from_hf` encapsulate the evil:

```python
tm_model.model_comm = model_comm      # mutation 1
tm_model.gpu_count = self.gpu_count   # mutation 2
tm_model.model.prepare()              # deferred init
```

Problems:

1. **Two-phase init.** `BaseOutputModel` is constructed with `model_comm=None` and `gpu_count=0`, then mutated externally. The object exists in an unusable state between construction and mutation.
2. **Coupling.** `turbomind.py` reaches into `BaseOutputModel`'s internal fields. It knows about `model_comm`, `gpu_count`, and `model.prepare()` — implementation details that should be hidden.
3. **Fragile ordering.** `model_comm` must be set before `gpu_count`, `prepare()` must follow both, and `export()` must follow `prepare()`. None of this is enforced.

## Circular dependency and resolution

The reason model_comm can't be passed to `__init__` today:

```
BaseOutputModel.__init__  →  computes config (needs input_model)
         ↓
_postprocess_config       →  merges config with engine_config
         ↓
_tm.TurboMind.create()    →  creates model_comm (needs final config)
         ↓
model_comm needed by BaseOutputModel
```

**Resolution:** Extract config computation from `__init__` into a classmethod `finalize_config`. Call it *before* creating `model_comm`. Then pass `model_comm` to `__init__`.

## Approach

### 1. `BaseOutputModel.finalize_config` classmethod

Moves config computation (currently `__init__` lines 58-84) into a classmethod. Takes `input_model` and `cfg`, mutates `cfg` in-place, returns extra computed values.

```python
@classmethod
def finalize_config(cls, input_model, cfg):
    """Finalize cfg by merging input model info. Mutates cfg in-place.

    Returns dict with 'repeat_kv' and 'input_model_info'.
    """
    mc = cfg.model_config
    attn_tp = mc.attn_tp_size
    mlp_tp = mc.mlp_tp_size

    # Get and normalize input model info
    info = input_model.model_info()
    num_layer = int(info['num_layer'])
    for k in ['inter_size', 'expert_num']:
        v = info.get(k)
        if v is not None and not isinstance(v, Sequence):
            info[k] = [v] * num_layer

    # Merge into model_config
    final_cfg = config_to_dict(mc)
    final_cfg.update(info)
    if 'embedding_size' not in info:
        final_cfg['embedding_size'] = info['vocab_size']
    cfg.model_config = config_from_dict(ModelConfig, final_cfg)
    mc = cfg.model_config

    # Pad inter_size / expert_inter_size
    for i, v in enumerate(mc.inter_size):
        mc.inter_size[i] = _pad_inter_size(v, mc.group_size, mlp_tp)
    if mc.expert_num:
        mc.expert_inter_size = _pad_inter_size(
            mc.expert_inter_size, mc.group_size, mlp_tp)

    # Handle repeat_kv
    assert mc.head_num % attn_tp == 0
    repeat_kv = 0
    if attn_tp > mc.kv_head_num and attn_tp % mc.kv_head_num == 0:
        repeat_kv = attn_tp // mc.kv_head_num
        mc.kv_head_num = attn_tp
    mc.verify()
    assert mc.kv_head_num % attn_tp == 0

    # Merge into attention_config and lora_config
    for config_attr in ('attention_config', 'lora_config'):
        orig = getattr(cfg, config_attr)
        cls_type = type(orig)
        merged = config_to_dict(orig)
        merged.update(info)
        setattr(cfg, config_attr, config_from_dict(cls_type, merged))

    return dict(repeat_kv=repeat_kv)
```

### 2. New `BaseOutputModel.__init__`

Receives all dependencies at construction time. No two-phase init.

```python
class BaseOutputModel(ABC):
    def __init__(self, input_model, cfg, model_cls,
                 model_comm, gpu_count, *, repeat_kv):
        super().__init__()
        self.input_model = input_model
        self.model_config = cfg.model_config
        self.attention_config = cfg.attention_config
        self.lora_config = cfg.lora_config
        self.attn_tp_size = cfg.model_config.attn_tp_size
        self.attn_cp_size = cfg.model_config.attn_cp_size
        self.mlp_tp_size = cfg.model_config.mlp_tp_size
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        self.repeat_kv = repeat_kv

        self.model = model_cls(self)
```

No `prepare()` call — `TextModelLoader.__init__` now does everything eagerly (see section 2b).

### 2b. `TextModelLoader.__init__` absorbs `prepare()`

With `model_comm` and `gpu_count` available on the parent at construction time, `prepare()` has no reason to be a separate method. Its body moves into `__init__`:

```python
class TextModelLoader:
    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size

        # Eagerly initialize distributors (was prepare())
        self._attn_ranks = [model.tp_ranks(gpu)[0]
                            for gpu in range(model.gpu_count)]
        self._mlp_ranks = [model.tp_ranks(gpu)[1]
                           for gpu in range(model.gpu_count)]
        handles = []
        for gpu in range(model.gpu_count):
            root = model.root(gpu)
            if root is None:
                break
            handles.append(root)
        self._root = Distributor(handles)
        self._layers = self._root.create_child('layers', ModuleListConfig())
```

`prepare()` is deleted.

### 3. Dead code removal

With the to-file path dropped:

| Removed | Reason |
|---------|--------|
| `out_dir` parameter | No file export |
| `to_file` attribute | No file export |
| `export_weight()` | Only used by file export |
| `save_split()` | Only used by file export |
| `export_config()` body | Was no-op for in-memory; remove the call from `export()`/`export_iter()` |
| `tprint()` helper | Only used by file export methods |
| `_compute_dtype()` helper | Only used by `export_weight()` |
| `update_model_config()` | Absorbed into `finalize_config` |
| `update_attention_config()` | Absorbed into `finalize_config` |
| `update_lora_config()` | Absorbed into `finalize_config` |
| `single_to_list()` | Absorbed into `finalize_config` |
| `permute_qk` attribute | Always `True`; `SpecAttnConfig` already defaults to `True` |
| `TextModelLoader.prepare()` | Body absorbed into `TextModelLoader.__init__` |

### 4. `TextModelLoader._load_layer` update

Remove `permute_qk` from the `SpecAttnConfig` construction — it already defaults to `True`:

```python
spec.configure(SpecAttnConfig(
    tp=self.attn_tp,
    repeat_kv=getattr(self.model, 'repeat_kv', 0),
    head_dim=mc.size_per_head,
    rope_dim=rope_param.dim if rope_param else mc.size_per_head,
    output_gate=getattr(mc, 'attn_output_gate', False),
    kv_head_num=mc.kv_head_num,
))
```

### 5. `converter.py`: `get_tm_model` becomes `get_tm_config`

Returns config + input_model instead of a constructed output model:

```python
def get_tm_config(model_path, model_name, chat_template_name,
                  engine_config, group_size=None):
    """Compute finalized TurbomindModelConfig.

    Returns (input_model, tm_cfg, repeat_kv).
    """
    # ... existing quant detection and validation (unchanged) ...

    input_model = INPUT_MODELS.get(input_model_name)(...)
    _, tm_cfg = get_output_model_registered_name_and_config(...)
    tm_cfg.model_config.chat_template = chat_template_name
    tm_cfg.model_config.model_name = model_name

    if engine_config.attn_tp_size is not None:
        tm_cfg.model_config.attn_tp_size = engine_config.attn_tp_size
    if engine_config.attn_cp_size is not None:
        tm_cfg.model_config.attn_cp_size = engine_config.attn_cp_size
    if engine_config.mlp_tp_size is not None:
        tm_cfg.model_config.mlp_tp_size = engine_config.mlp_tp_size

    extras = BaseOutputModel.finalize_config(input_model, tm_cfg)

    return input_model, tm_cfg, extras['repeat_kv']
```

The `out_dir` parameter is removed (dead — only caller never passed it).

### 6. `turbomind.py._from_hf`: single clean construction

```python
def _from_hf(self, model_path, engine_config):
    from .deploy.converter import get_tm_config
    from .deploy.text_model_loader import TextModelLoader
    from .deploy.target_model.base import OUTPUT_MODELS

    input_model, tm_cfg, repeat_kv = get_tm_config(
        model_path, self.model_name, self.chat_template_name, engine_config)

    self._postprocess_config(tm_cfg, engine_config)

    model_comm = _tm.TurboMind.create(
        model_dir='', config=yaml.safe_dump(self.config_dict))
    self._create_weight(model_comm)

    self._tm_model = OUTPUT_MODELS.get('tm')(
        input_model=input_model,
        cfg=tm_cfg,
        model_cls=TextModelLoader,
        model_comm=model_comm,
        gpu_count=self.gpu_count,
        repeat_kv=repeat_kv)

    return model_comm
```

No post-construction mutation. `turbomind.py` no longer touches `model_comm`, `gpu_count`, or `prepare()` on the output model.

### 7. `TurbomindModel` subclass

Unchanged — it's an empty subclass with a registry decorator:

```python
@OUTPUT_MODELS.register_module(name='tm')
class TurbomindModel(BaseOutputModel):
    pass
```

### 8. `update_params` path

No changes needed. `export_iter()` still works — `self._tm_model` is now fully constructed when `update_params` is first called. The queue-replacement pattern (`tm_model.input_model.model_path = que`) is a separate concern.

## Files changed

| File | Change |
|------|--------|
| `deploy/target_model/base.py` | Add `finalize_config` classmethod, rewrite `__init__`, remove dead code (export_weight, save_split, export_config body, tprint, _compute_dtype, update_*_config, single_to_list, permute_qk, out_dir, to_file) |
| `deploy/text_model_loader.py` | Absorb `prepare()` into `__init__`, delete `prepare()`, remove `permute_qk` from `SpecAttnConfig` construction in `_load_layer` |
| `deploy/converter.py` | Rename `get_tm_model` to `get_tm_config`, return (input_model, tm_cfg, repeat_kv), remove `out_dir` param |
| `turbomind/turbomind.py` | Rewrite `_from_hf` to use `get_tm_config` + single construction, remove 3-line mutation |

No changes to: `distributor.py`, `configs.py`, `spec.py`, C++ code.

## Verification

Test with TP=1 and TP=2 models, verifying that `export()` and `export_iter()` (via `update_params`) produce correct results.
