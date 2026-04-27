# TurboMind File Structure Reorganization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `deploy/` nesting layer, rename `builder/` to `builders/` and `source_model/` to `models/` for clarity and discoverability.

**Architecture:** Pure file move and import update. No logic changes. Files in `deploy/` are promoted to `lmdeploy/turbomind/`, sub-packages are renamed, and all import paths are updated to match.

**Tech Stack:** Python, git

---

## File Structure

### Moves

| Source | Destination |
|--------|-------------|
| `lmdeploy/turbomind/deploy/converter.py` | `lmdeploy/turbomind/converter.py` |
| `lmdeploy/turbomind/deploy/linear.py` | `lmdeploy/turbomind/linear.py` |
| `lmdeploy/turbomind/deploy/loader.py` | `lmdeploy/turbomind/loader.py` |
| `lmdeploy/turbomind/deploy/model_loader.py` | `lmdeploy/turbomind/model_loader.py` |
| `lmdeploy/turbomind/deploy/spec.py` | `lmdeploy/turbomind/spec.py` |
| `lmdeploy/turbomind/deploy/weight_format.py` | `lmdeploy/turbomind/weight_format.py` |
| `lmdeploy/turbomind/deploy/builder/` (entire dir) | `lmdeploy/turbomind/builders/` |
| `lmdeploy/turbomind/deploy/source_model/` (entire dir) | `lmdeploy/turbomind/models/` |

### Deletes

| File | Reason |
|------|--------|
| `lmdeploy/turbomind/deploy/__init__.py` | Empty; package removed |

### Import Changes Summary

| File | Old Import | New Import |
|------|-----------|------------|
| `turbomind.py:219` | `from .deploy.converter` | `from .converter` |
| `turbomind.py:220` | `from .deploy.model_loader` | `from .model_loader` |
| `converter.py:9` | `from ...utils` (3 dots) | `from ..utils` (2 dots) |
| `converter.py:10` | `from ..supported_models` (2 dots) | `from .supported_models` (1 dot) |
| `converter.py:11` | `from .builder` | `from .builders` |
| `converter.py:12` | `from .source_model.base` | `from .models.base` |
| `converter.py:13` | `from .source_model.utils` | `from .models.utils` |
| `spec.py:12` | `from .builder` | `from .builders` |
| `spec.py:13` | `from .source_model.utils` | `from .models.utils` |
| `weight_format.py:363` | `from .builder._base` | `from .builders._base` |
| `models/base.py:12` | `locations=['lmdeploy.turbomind.deploy.source_model.base']` | `locations=['lmdeploy.turbomind.models.base']` |
| `models/qwen3_spec.py:12-15` | `from ..builder` | `from ..builders` |
| `models/qwen3_spec.py:16` | `from ..spec` | unchanged |
| `models/qwen3_5_spec.py:9-12` | `from ..builder` | `from ..builders` |
| `models/qwen3_5_spec.py:13` | `from ..builder.attention` | `from ..builders.attention` |
| `models/qwen3_5_spec.py:14` | `from ..spec` | unchanged |
| `models/glm4_moe_lite_spec.py:7-10` | `from ..builder` | `from ..builders` |
| `models/glm4_moe_lite_spec.py:11` | `from ..spec` | unchanged |
| `models/gpt_oss_spec.py:9-12` | `from ..builder` | `from ..builders` |
| `models/gpt_oss_spec.py:13` | `from ..spec` | unchanged |
| `models/utils.py:12` | `from ..linear` | unchanged |
| `models/utils.py:13` | `from ..builder._base` | `from ..builders._base` |

---

## Task 1: Move builder/ → builders/

**Files:**
- Move: `lmdeploy/turbomind/deploy/builder/` → `lmdeploy/turbomind/deploy/builders/`

- [ ] **Step 1: Move the directory**

```bash
cd /data/lmdeploy-modeling
git mv lmdeploy/turbomind/deploy/builder lmdeploy/turbomind/deploy/builders
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builders/
git commit -m "refactor: rename deploy/builder/ to deploy/builders/"
```

---

## Task 2: Move source_model/ → models/

**Files:**
- Move: `lmdeploy/turbomind/deploy/source_model/` → `lmdeploy/turbomind/deploy/models/`

- [ ] **Step 1: Move the directory**

```bash
cd /data/lmdeploy-modeling
git mv lmdeploy/turbomind/deploy/source_model lmdeploy/turbomind/deploy/models
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/models/
git commit -m "refactor: rename deploy/source_model/ to deploy/models/"
```

---

## Task 3: Update imports in builders/ files

No changes needed. The `builders/` files only use `from ..weight_format` and `from ..linear` as parent references. The `..` resolves to `deploy/` now and will resolve to `turbomind/` after promotion — both levels contain `weight_format.py` and `linear.py`, so these imports remain valid throughout.

Internal `.` imports within `builders/` (e.g., `from ._base import Builder`) are unchanged.

---

## Task 4: Update imports in models/ files (builder → builders)

All `from ..builder` references become `from ..builders` since the directory was renamed. Also update the registry `locations` string in `base.py`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/models/base.py` line 12
- Modify: `lmdeploy/turbomind/deploy/models/qwen3_spec.py` lines 12-15
- Modify: `lmdeploy/turbomind/deploy/models/qwen3_5_spec.py` lines 9-13
- Modify: `lmdeploy/turbomind/deploy/models/glm4_moe_lite_spec.py` lines 7-10
- Modify: `lmdeploy/turbomind/deploy/models/gpt_oss_spec.py` lines 9-12
- Modify: `lmdeploy/turbomind/deploy/models/utils.py` line 13

- [ ] **Step 1: Update base.py — registry locations string**

Replace:
```python
INPUT_MODELS = Registry('source model',
                        locations=['lmdeploy.turbomind.deploy.source_model.base'])
```
With:
```python
INPUT_MODELS = Registry('source model',
                        locations=['lmdeploy.turbomind.models.base'])
```

- [ ] **Step 2: Update qwen3_spec.py**

Replace:
```python
from ..builder import (AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
                       MoeBuilder, ModuleListBuilder, TextModelBuilder,
                       _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
```
With:
```python
from ..builders import (AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
                        MoeBuilder, ModuleListBuilder, TextModelBuilder,
                        _act_type_id)
from ..builders import DecoderLayerConfig, ModuleListConfig
```

The `from ..spec import TextModelSpec` line stays unchanged.

- [ ] **Step 3: Update qwen3_5_spec.py**

Replace:
```python
from ..builder import (AttentionBuilder, DecoderLayerBuilder, DeltaNetBuilder,
                       FfnBuilder, MoeBuilder, ModuleListBuilder,
                       TextModelBuilder, _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
from ..builder.attention import split_output_gate
```
With:
```python
from ..builders import (AttentionBuilder, DecoderLayerBuilder, DeltaNetBuilder,
                        FfnBuilder, MoeBuilder, ModuleListBuilder,
                        TextModelBuilder, _act_type_id)
from ..builders import DecoderLayerConfig, ModuleListConfig
from ..builders.attention import split_output_gate
```

The `from ..spec import TextModelSpec` line stays unchanged.

- [ ] **Step 4: Update glm4_moe_lite_spec.py**

Replace:
```python
from ..builder import (DecoderLayerBuilder, FfnBuilder, MLABuilder, MoeBuilder,
                       ModuleListBuilder, TextModelBuilder, _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
```
With:
```python
from ..builders import (DecoderLayerBuilder, FfnBuilder, MLABuilder, MoeBuilder,
                        ModuleListBuilder, TextModelBuilder, _act_type_id)
from ..builders import DecoderLayerConfig, ModuleListConfig
```

The `from ..spec import TextModelSpec` line stays unchanged.

- [ ] **Step 5: Update gpt_oss_spec.py**

Replace:
```python
from ..builder import (AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
                       MoeBuilder, ModuleListBuilder, TextModelBuilder,
                       _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
```
With:
```python
from ..builders import (AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
                        MoeBuilder, ModuleListBuilder, TextModelBuilder,
                        _act_type_id)
from ..builders import DecoderLayerConfig, ModuleListConfig
```

The `from ..spec import TextModelSpec` line stays unchanged.

- [ ] **Step 6: Update utils.py**

Replace (line 13):
```python
from ..builder._base import _dequant_linear
```
With:
```python
from ..builders._base import _dequant_linear
```

The `from ..linear import Linear` lines (12 and 193) stay unchanged.

- [ ] **Step 7: Verify imports resolve**

```bash
cd /data/lmdeploy-modeling
python -c "from lmdeploy.turbomind.deploy.models.qwen3_spec import Qwen3TextSpec; print('OK')"
python -c "from lmdeploy.turbomind.deploy.models.qwen3_5_spec import Qwen3_5Spec; print('OK')"
python -c "from lmdeploy.turbomind.deploy.models.glm4_moe_lite_spec import Glm4MoeLiteSpec; print('OK')"
python -c "from lmdeploy.turbomind.deploy.models.gpt_oss_spec import GptOssSpec; print('OK')"
python -c "from lmdeploy.turbomind.deploy.models.base import INPUT_MODELS; print('OK')"
```
Expected: all print `OK`

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/models/
git commit -m "refactor: update models/ imports from ..builder to ..builders"
```

---

## Task 5: Move deploy/ files up to turbomind/

Move the six top-level files out of `deploy/`, keeping `builders/` and `models/` in place temporarily.

**Files:**
- Move: 6 files from `lmdeploy/turbomind/deploy/` to `lmdeploy/turbomind/`

- [ ] **Step 1: Move files**

```bash
cd /data/lmdeploy-modeling
git mv lmdeploy/turbomind/deploy/converter.py lmdeploy/turbomind/converter.py
git mv lmdeploy/turbomind/deploy/linear.py lmdeploy/turbomind/linear.py
git mv lmdeploy/turbomind/deploy/loader.py lmdeploy/turbomind/loader.py
git mv lmdeploy/turbomind/deploy/model_loader.py lmdeploy/turbomind/model_loader.py
git mv lmdeploy/turbomind/deploy/spec.py lmdeploy/turbomind/spec.py
git mv lmdeploy/turbomind/deploy/weight_format.py lmdeploy/turbomind/weight_format.py
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/converter.py lmdeploy/turbomind/linear.py lmdeploy/turbomind/loader.py lmdeploy/turbomind/model_loader.py lmdeploy/turbomind/spec.py lmdeploy/turbomind/weight_format.py
git commit -m "refactor: promote deploy/ files to turbomind/ level"
```

---

## Task 6: Move builders/ and models/ up to turbomind/

**Files:**
- Move: `lmdeploy/turbomind/deploy/builders/` → `lmdeploy/turbomind/builders/`
- Move: `lmdeploy/turbomind/deploy/models/` → `lmdeploy/turbomind/models/`

- [ ] **Step 1: Move directories**

```bash
cd /data/lmdeploy-modeling
git mv lmdeploy/turbomind/deploy/builders lmdeploy/turbomind/builders
git mv lmdeploy/turbomind/deploy/models lmdeploy/turbomind/models
```

- [ ] **Step 2: Delete deploy/__init__.py and remove empty deploy/ dir**

```bash
git rm lmdeploy/turbomind/deploy/__init__.py
rmdir lmdeploy/turbomind/deploy
```

- [ ] **Step 3: Commit**

```bash
git add -A lmdeploy/turbomind/deploy/ lmdeploy/turbomind/builders/ lmdeploy/turbomind/models/
git commit -m "refactor: promote builders/ and models/ to turbomind/ level, remove deploy/"
```

---

## Task 7: Update all import paths

Now that all files are in their final locations, update every import.

### A. turbomind.py — remove .deploy prefix

**File:** `lmdeploy/turbomind/turbomind.py`

- [ ] **Step 1: Update turbomind.py imports**

Replace (line 219):
```python
from .deploy.converter import get_tm_config
```
With:
```python
from .converter import get_tm_config
```

Replace (line 220):
```python
from .deploy.model_loader import ModelLoader
```
With:
```python
from .model_loader import ModelLoader
```

### B. converter.py — adjust relative import levels + rename references

`converter.py` was moved from `deploy/` (depth 3: `lmdeploy.turbomind.deploy`) to `turbomind/` (depth 2: `lmdeploy.turbomind`). All relative import levels decrease by one, and `builder`/`source_model` names change.

**File:** `lmdeploy/turbomind/converter.py`

- [ ] **Step 2: Update converter.py**

Replace (lines 9-13):
```python
from ...utils import _get_and_verify_max_len, is_bf16_supported
from ..supported_models import SUPPORTED_ARCHS
from .builder import _cpp_dtype
from .source_model.base import INPUT_MODELS
from .source_model.utils import load_model_config
```
With:
```python
from ..utils import _get_and_verify_max_len, is_bf16_supported
from .supported_models import SUPPORTED_ARCHS
from .builders import _cpp_dtype
from .models.base import INPUT_MODELS
from .models.utils import load_model_config
```

The `from .weight_format import (...)` line stays unchanged.

### C. spec.py — rename builder/source_model references

**File:** `lmdeploy/turbomind/spec.py`

- [ ] **Step 3: Update spec.py**

Replace (lines 12-14):
```python
from .builder import NormBuilder, make_norm_config
from .source_model.utils import (parse_rope_param, rope_type_to_int,
                                 get_yarn_params, reorder_rotary_emb)
```
With:
```python
from .builders import NormBuilder, make_norm_config
from .models.utils import (parse_rope_param, rope_type_to_int,
                           get_yarn_params, reorder_rotary_emb)
```

### D. weight_format.py — rename builder reference in lazy import

**File:** `lmdeploy/turbomind/weight_format.py`

- [ ] **Step 4: Update weight_format.py**

The `from .linear import Linear` (line 34) stays unchanged.

Replace the lazy import inside the `FP8Format.dequant()` method (line 363):
```python
from .builder._base import _CPP_TO_TORCH
```
With:
```python
from .builders._base import _CPP_TO_TORCH
```

### E. Files that need no changes

- `model_loader.py` — imports `from .loader import create_loader` (sibling, unchanged)
- `linear.py` — imports `from .weight_format import WeightFormat` inside `if TYPE_CHECKING:` (sibling, unchanged)
- `loader.py` — no relative imports from the package

### F. builders/ files — no changes needed

All `builders/` files use `from ..xxx` to reference the parent (`turbomind/`). Since `weight_format.py` and `linear.py` are now direct siblings of `builders/`, the `..weight_format` and `..linear` references work without changes. Internal `.` references within `builders/` are also unchanged.

### G. models/ files — already updated in Task 4

The `..builder` → `..builders` rename was done in Task 4. The `..spec`, `..linear` references remain valid since `spec.py` and `linear.py` are now direct siblings of `models/`.

- [ ] **Step 5: Verify all imports resolve**

```bash
cd /data/lmdeploy-modeling
python -c "from lmdeploy.turbomind import TurboMind; print('TurboMind OK')"
python -c "from lmdeploy.turbomind.converter import get_tm_config; print('converter OK')"
python -c "from lmdeploy.turbomind.builders import TextModelBuilder; print('builders OK')"
python -c "from lmdeploy.turbomind.models.qwen3_spec import Qwen3TextSpec; print('models OK')"
python -c "from lmdeploy.turbomind.weight_format import WeightFormatResolver; print('weight_format OK')"
python -c "from lmdeploy.turbomind.models.base import INPUT_MODELS; print('registry OK')"
```
Expected: all print `OK`

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/
git commit -m "refactor: update all import paths after deploy/ removal"
```

---

## Task 8: Update test files

Test files reference old paths. Some test files import from modules that no longer exist (pre-existing issue from earlier refactoring) — only update the string references and imports for modules that exist.

**Files:**
- Modify: `tests/test_lmdeploy/test_converter.py`
- Modify: `tests/test_lmdeploy/test_turbomind/test_weight_format_resolver.py`
- Modify: `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`
- Note (no changes possible): `tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py` imports from `lmdeploy.turbomind.deploy.parameter` and `lmdeploy.turbomind.deploy.source_model.qwen` — both modules don't exist, so this test is already broken before our changes.
- Note (no changes possible): `tests/test_lmdeploy/test_turbomind/test_converter.py` imports from `lmdeploy.turbomind.deploy.source_model.internlm2` and `lmdeploy.turbomind.deploy.source_model.llama` — both modules don't exist, already broken.

- [ ] **Step 1: Update test_converter.py**

Replace:
```python
from lmdeploy.turbomind.deploy.converter import _deep_merge
```
With:
```python
from lmdeploy.turbomind.converter import _deep_merge
```

Note: This file also imports from `lmdeploy.turbomind.deploy.source_model.internlm2` and `lmdeploy.turbomind.deploy.source_model.llama` — these modules don't exist (pre-existing breakage). Do NOT update those lines.

- [ ] **Step 2: Update test_weight_format_resolver.py**

This file uses `importlib.util.spec_from_file_location` with string module names and a `sys.modules` registration loop.

Replace the loop iteration string (line 63):
```python
for _pkg in ('lmdeploy.turbomind', 'lmdeploy.turbomind.deploy'):
```
With:
```python
for _pkg in ('lmdeploy.turbomind',):
```

Replace all string occurrences of `'lmdeploy.turbomind.deploy.linear'` with `'lmdeploy.turbomind.linear'`.

Replace all string occurrences of `'lmdeploy.turbomind.deploy.weight_format'` with `'lmdeploy.turbomind.weight_format'`.

- [ ] **Step 3: Update test_transform_tensors.py**

This file creates fake `sys.modules` entries with string module paths and uses `importlib` to load modules by string name.

Replace all string occurrences of:
- `'lmdeploy.turbomind.deploy'` → `'lmdeploy.turbomind'` (in sys.modules operations, lines 72-77)
- `'lmdeploy.turbomind.deploy.linear'` → `'lmdeploy.turbomind.linear'` (line 91)
- `'lmdeploy.turbomind.deploy.weight_format'` → `'lmdeploy.turbomind.weight_format'` (line 96)
- `'lmdeploy.turbomind.deploy.builder._base'` → `'lmdeploy.turbomind.builders._base'` (line 100)
- `'lmdeploy.turbomind.deploy.builder'` → `'lmdeploy.turbomind.builders'` (lines 105-110)
- Also update the comment on line 16 that mentions `lmdeploy.turbomind.deploy.linear`

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "refactor: update test import paths for turbomind reorganization"
```

---

## Task 9: Build and smoke test

Verify the reorganization works end-to-end.

**Files:**
- No file changes

- [ ] **Step 1: Build**

```bash
cd /data/lmdeploy-modeling/build && ninja
```
Expected: Build succeeds with no errors.

- [ ] **Step 2: Check GPU availability**

Use the `get_gpu_usage` MCP tool to find an empty GPU.

- [ ] **Step 3: Run model smoke test**

Pick one model from `list_models` and run:
```bash
cd /data/lmdeploy-modeling
python scripts/test_turbomind_model.py <model_id> --gpu <gpu_id> --tp 1 --max_new_tokens 128
```

Expected: Model loads successfully and produces meaningful text output.

- [ ] **Step 4: Commit final state if needed**

If any fixup commits were needed, combine them:
```bash
git add -A && git commit -m "fix: resolve remaining import issues from reorganization"
```
