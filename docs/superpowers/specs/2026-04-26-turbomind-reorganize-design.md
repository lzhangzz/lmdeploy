# Reorganize `lmdeploy/turbomind/` File Structure

## Problem

The directory names `deploy/`, `builder/`, and `source_model/` don't clearly communicate what each module does. `deploy/` is a legacy name for the model construction pipeline. `source_model/` actually contains per-architecture model spec implementations. The extra `deploy/` nesting adds depth without clarity.

## Solution

Remove the `deploy/` layer and rename sub-packages for clarity:
- `deploy/` contents promoted to `lmdeploy/turbomind/`
- `deploy/builder/` → `builders/` (plural to distinguish package from builder classes)
- `deploy/source_model/` → `models/` (contains per-architecture model specs)

### Current Structure

```
lmdeploy/turbomind/
    __init__.py
    turbomind.py
    supported_models.py
    tokenizer_info.py
    deploy/
        __init__.py, converter.py, linear.py, loader.py, model_loader.py, spec.py, weight_format.py
        builder/
            __init__.py, _base.py, attention.py, decoder_layer.py, deltanet.py, ffn.py,
            mla.py, module_list.py, moe.py, norm.py
        source_model/
            __init__.py, base.py, glm4_moe_lite_spec.py, gpt_oss_spec.py,
            qwen3_5_spec.py, qwen3_spec.py, utils.py
```

### Target Structure

```
lmdeploy/turbomind/
    __init__.py
    turbomind.py
    supported_models.py
    tokenizer_info.py
    converter.py
    linear.py
    loader.py
    model_loader.py
    spec.py
    weight_format.py
    builders/
        __init__.py, _base.py, attention.py, decoder_layer.py, deltanet.py, ffn.py,
        mla.py, module_list.py, moe.py, norm.py
    models/
        __init__.py, base.py, glm4_moe_lite_spec.py, gpt_oss_spec.py,
        qwen3_5_spec.py, qwen3_spec.py, utils.py
```

### Changes

1. **Remove `deploy/` layer**: Move `converter.py`, `linear.py`, `loader.py`, `model_loader.py`, `spec.py`, `weight_format.py` up to `lmdeploy/turbomind/`. Delete `deploy/__init__.py`.

2. **Rename `deploy/builder/` → `builders/`**: Promote to `lmdeploy/turbomind/builders/`.

3. **Rename `deploy/source_model/` → `models/`**: Promote to `lmdeploy/turbomind/models/`.

4. **Update all internal imports**:
   - `from lmdeploy.turbomind.deploy.xxx` → `from lmdeploy.turbomind.xxx`
   - `from lmdeploy.turbomind.deploy.builder.xxx` → `from lmdeploy.turbomind.builders.xxx`
   - `from lmdeploy.turbomind.deploy.source_model.xxx` → `from lmdeploy.turbomind.models.xxx`
   - Relative imports within `builders/` and `models/` unchanged.

5. **Update `__init__.py` exports**: `builders/__init__.py` and `models/__init__.py` need import path updates. Top-level `__init__.py` unchanged.

### Invariants

- No code logic changes
- No file splits or merges
- No public API changes
- Internal file names in `builders/` and `models/` unchanged
