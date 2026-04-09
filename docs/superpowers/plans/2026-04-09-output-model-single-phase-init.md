# Output Model Single-Phase Init Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the two-phase init of `BaseOutputModel` — pass `model_comm` and `gpu_count` at construction time, absorb `TextModelLoader.prepare()` into `__init__`, remove dead to-file export code.

**Architecture:** Extract config finalization into a `BaseOutputModel.finalize_config` classmethod (called before `model_comm` exists). Then construct `BaseOutputModel` with `model_comm` and `gpu_count` so `TextModelLoader` can fully initialize in its own `__init__`.

**Tech Stack:** Python, TurboMind C++ engine (unchanged)

---

### Task 1: Rewrite `base.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py`

This file changes the most: new `finalize_config` classmethod, new `__init__` signature, dead code removal.

- [ ] **Step 1: Replace the entire file**

The new `base.py` replaces all the old code. Here is the complete new file:

```python
# Copyright (c) OpenMMLab. All rights reserved.

from abc import ABC
from collections.abc import Sequence

from mmengine import Registry

from ..config import (AttentionConfig, LoraConfig, ModelConfig, TurbomindModelConfig,
                      config_from_dict, config_to_dict)
from ..source_model.base import BaseInputModel

OUTPUT_MODELS = Registry('target model', locations=['lmdeploy.turbomind.deploy.target_model.base'])


def _pad_inter_size(inter_size: int, group_size: int, tp: int):
    group_size = max(1, group_size)
    group_num = (inter_size + group_size - 1) // group_size
    groups_per_rank = (group_num + tp - 1) // tp
    inter_size_padded = groups_per_rank * group_size * tp
    return inter_size_padded


class BaseOutputModel(ABC):
    """Base output model."""

    @classmethod
    def finalize_config(cls, input_model, cfg):
        """Finalize cfg by merging input model info. Mutates cfg in-place.

        Returns repeat_kv (int).
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

        return repeat_kv

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

    def root(self, index: int):
        """Return the C++ ``Module`` root for GPU *index*."""
        return self.model_comm.root(index)

    def tp_ranks(self, index: int):
        """Return ``(attn_tp_rank, mlp_tp_rank)`` for GPU *index*."""
        return (self.model_comm.attn_tp_rank(index),
                self.model_comm.mlp_tp_rank(index))

    def export(self) -> None:
        """Export to turbomind model format."""
        num_layer = self.model_config.num_layer
        from tqdm import tqdm
        pbar = tqdm(total=num_layer, desc='Convert to turbomind format', leave=False)
        for i, reader in self.input_model.readers():
            if self.model(i, reader):
                pbar.update(1)
        pbar.close()

    def export_iter(self):
        for i, reader in self.input_model.readers():
            self.model(i, reader)
            yield i

    @property
    def tm_config(self):
        return TurbomindModelConfig(model_config=self.model_config,
                                    attention_config=self.attention_config,
                                    lora_config=self.lora_config)
```

Key changes from the old file:
- **Removed imports:** `os.path`, `torch`, `tqdm` (top-level), `yaml`
- **Removed:** `tprint()`, `_compute_dtype()`, `export_weight()`, `save_split()`, `export_config()`, `single_to_list()`, `update_model_config()`, `update_attention_config()`, `update_lora_config()`, `out_dir`, `to_file`, `permute_qk`, `input_model_info`
- **Added:** `finalize_config()` classmethod
- **Changed:** `__init__` signature now takes `model_comm`, `gpu_count`, `repeat_kv`; no longer computes config
- **Changed:** `root()` and `tp_ranks()` removed `None` guards (model_comm is always set)
- **Changed:** `export()` removed `export_config()` call and `leave=self.to_file`
- **Changed:** `export_iter()` removed `export_config()` call

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/target_model/base.py
git commit -m "refactor(output-model): single-phase init — finalize_config classmethod, new __init__, remove dead to-file code"
```

---

### Task 2: Rewrite `converter.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/converter.py`

Rename `get_tm_model` to `get_tm_config`. Remove `out_dir` parameter. Call `finalize_config` instead of constructing the output model. Return `(input_model, tm_cfg, repeat_kv)`.

- [ ] **Step 1: Update imports and the `get_tm_model` function**

Replace lines 10-13 (imports) and lines 121-211 (the `get_tm_model` function). The rest of the file (lines 1-9, 14-120) stays unchanged.

Replace the import block at lines 10-13:
```python
from .text_model_loader import TextModelLoader
from .source_model.base import INPUT_MODELS
from .target_model.base import OUTPUT_MODELS, BaseOutputModel
```
with:
```python
from .source_model.base import INPUT_MODELS
from .target_model.base import BaseOutputModel
```

Then replace the entire `get_tm_model` function (lines 121-211) with:

```python
def get_tm_config(model_path,
                  model_name,
                  chat_template_name,
                  engine_config: TurbomindEngineConfig,
                  group_size: int = None):
    """Compute finalized TurbomindModelConfig.

    Args:
        model_path (str): the path of the input model, which is supposed
            to be a local path, or huggingface hub repo_id, or modelscope
            hub repo_id
        model_name (str): user customized model name
        chat_template_name (str): the name of the chat template of
            the input model
        engine_config(TurbomindEngineConfig): user input engine config
        group_size(int): refers to the group_size if the input model
            is a grouped quantized model

    Returns:
        tuple: (input_model, tm_cfg, repeat_kv)
    """
    _, cfg = get_model_arch(model_path)
    quant_config = search_nested_config(cfg.to_dict(), 'quantization_config')
    if quant_config:
        quant_method = quant_config.get('quant_method')
        _group_size = int(quant_config.get('group_size', 0))
        version = quant_config.get('version')
        assert engine_config.model_format is None or engine_config.model_format == quant_method, (
            f'mismatched quant method: user input "{engine_config.model_format}" '
            f'vs model quant_config "{quant_method}"')
        assert not group_size or group_size == _group_size, (f'mismatched quant group size: user input "{group_size}" '
                                                             f'vs model quant_config "{_group_size}"')

        if quant_method == 'awq':
            assert version == 'gemm', f'unsupported quant config: {quant_config}'
        elif quant_method == 'gptq':
            assert not quant_config.get('desc_act', False) and quant_config.get(
                'sym', True), f'unsupported quant config: {quant_config}'
        elif quant_method == 'fp8':
            pass
        elif quant_method == 'mxfp4':
            _group_size = 32
        elif quant_method == 'compressed-tensors':
            _format = quant_config['config_groups']['group_0']['format']
            assert _format == 'pack-quantized', ('compressed-tennsors only supports pack-quantized format, '
                                                 f'but got {_format}')
            _weights = quant_config['config_groups']['group_0']['weights']
            _group_size = _weights['group_size']
            _num_bits = _weights['num_bits']
            _type = _weights['type']
            assert _num_bits == 4 and _type == 'int', ('pack-quantized requires 4-bit int, '
                                                       f'but got {_num_bits}-bit {_type}')
        else:
            assert 0, f'unsupported quant_config: {quant_config}'

        engine_config.model_format = quant_method
        group_size = _group_size

    group_size = _validate_quant_group_size(engine_config.model_format, group_size)

    input_model_name = get_input_model_registered_name(model_path, engine_config.model_format)

    fp8_quant = (engine_config.model_format == 'fp8' and not quant_config)
    _model_cls = INPUT_MODELS.get(input_model_name)
    input_model = _model_cls(model_path=model_path,
                             tokenizer_path=model_path,
                             fp8_quant=fp8_quant,
                             model_format=engine_config.model_format)

    output_model_name, tm_cfg = get_output_model_registered_name_and_config(model_path=model_path,
                                                                            model_format=engine_config.model_format,
                                                                            dtype=engine_config.dtype,
                                                                            group_size=group_size)

    tm_cfg.model_config.chat_template = chat_template_name
    tm_cfg.model_config.model_name = model_name

    if engine_config.attn_tp_size is not None:
        tm_cfg.model_config.attn_tp_size = engine_config.attn_tp_size
    if engine_config.attn_cp_size is not None:
        tm_cfg.model_config.attn_cp_size = engine_config.attn_cp_size
    if engine_config.mlp_tp_size is not None:
        tm_cfg.model_config.mlp_tp_size = engine_config.mlp_tp_size

    repeat_kv = BaseOutputModel.finalize_config(input_model, tm_cfg)

    return input_model, tm_cfg, repeat_kv
```

The function body is identical to the old `get_tm_model` except:
- **Removed** `out_dir` parameter
- **Removed** the final block that constructed `OUTPUT_MODELS.get(output_model_name)(...)` — replaced with `BaseOutputModel.finalize_config(input_model, tm_cfg)` call
- **Removed** `from .text_model_loader import TextModelLoader` import (no longer needed here)
- **Removed** `from .target_model.base import OUTPUT_MODELS` import (no longer needed here)
- **Returns** `(input_model, tm_cfg, repeat_kv)` instead of `BaseOutputModel`

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/converter.py
git commit -m "refactor(converter): rename get_tm_model to get_tm_config, return config instead of output model"
```

---

### Task 3: Absorb `prepare()` into `TextModelLoader.__init__`, drop `permute_qk`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

- [ ] **Step 1: Replace `__init__` and delete `prepare()`**

Replace lines 31-53 (the `__init__` method and `prepare` method) with:

```python
    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size

        # Eagerly initialize distributors
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

This replaces:
- Old `__init__` (lines 31-38): was just storing tp sizes and setting `_attn_ranks`, `_mlp_ranks`, `_root`, `_layers` to `None`
- Old `prepare()` (lines 40-53): computed the above values from `model_comm`

Now all initialization happens in `__init__` since `model_comm` is always available.

- [ ] **Step 2: Remove `permute_qk` from `_load_layer`**

In `_load_layer` (around line 297-305), change the `SpecAttnConfig` construction from:

```python
        spec.configure(SpecAttnConfig(
            tp=self.attn_tp,
            permute_qk=getattr(self.model, 'permute_qk', True),
            repeat_kv=getattr(self.model, 'repeat_kv', 0),
            head_dim=mc.size_per_head,
            rope_dim=rope_param.dim if rope_param else mc.size_per_head,
            output_gate=getattr(mc, 'attn_output_gate', False),
            kv_head_num=mc.kv_head_num,
        ))
```

to:

```python
        spec.configure(SpecAttnConfig(
            tp=self.attn_tp,
            repeat_kv=self.model.repeat_kv,
            head_dim=mc.size_per_head,
            rope_dim=rope_param.dim if rope_param else mc.size_per_head,
            output_gate=getattr(mc, 'attn_output_gate', False),
            kv_head_num=mc.kv_head_num,
        ))
```

Changes: removed `permute_qk=getattr(self.model, 'permute_qk', True)` line (defaults to `True` in `SpecAttnConfig`), changed `getattr(self.model, 'repeat_kv', 0)` to `self.model.repeat_kv` (attribute always exists now).

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): absorb prepare() into __init__, drop permute_qk"
```

---

### Task 4: Rewrite `turbomind.py._from_hf`

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py` (lines 228-249)

- [ ] **Step 1: Replace `_from_hf` method**

Replace the `_from_hf` method (lines 228-249) with:

```python
    def _from_hf(self, model_path: str, engine_config: TurbomindEngineConfig):
        """Load model which is in hf format."""
        assert is_supported(model_path), (f'turbomind does not support {model_path}. '
                                          'Plz try pytorch engine instead.')

        from .deploy.converter import get_tm_config
        from .deploy.text_model_loader import TextModelLoader
        from .deploy.target_model.base import OUTPUT_MODELS

        input_model, tm_cfg, repeat_kv = get_tm_config(
            model_path, self.model_name, self.chat_template_name, engine_config)

        self._postprocess_config(tm_cfg, engine_config)

        model_comm = _tm.TurboMind.create(model_dir='',
                                          config=yaml.safe_dump(self.config_dict))
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

Key changes from the old method:
- **Import** changed: `get_tm_model` → `get_tm_config`, plus explicit imports for `TextModelLoader` and `OUTPUT_MODELS`
- **Removed** the 3-line mutation block (`tm_model.model_comm = model_comm`, `tm_model.gpu_count = self.gpu_count`, `tm_model.model.prepare()`)
- **Single construction** call: `OUTPUT_MODELS.get('tm')(...)` with all dependencies including `model_comm`, `gpu_count`, `repeat_kv`

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "refactor(turbomind): rewrite _from_hf for single-phase output model construction"
```

---

### Task 5: Build and test

- [ ] **Step 1: Build the project**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

Expected: Build succeeds with no errors.

- [ ] **Step 2: Check GPU availability**

Use the `mcp__gpu-monitor__get_gpu_usage` tool to verify GPUs are free.

- [ ] **Step 3: Test with a model (TP=1)**

Use the `turbomind-tester` agent to test a model with TP=1. Request at least 128 tokens. Verify the response contains meaningful human words.

- [ ] **Step 4: Test with a model (TP=2)**

Use the `turbomind-tester` agent to test a model with TP=2. Request at least 128 tokens. Verify the response contains meaningful human words.

- [ ] **Step 5: Final commit (if any test fixes were needed)**

```bash
git add -u
git commit -m "fix: address test failures from single-phase init refactor"
```
