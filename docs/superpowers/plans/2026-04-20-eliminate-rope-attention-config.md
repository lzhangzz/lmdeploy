# Eliminate `RopeParam` and `AttentionConfig` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete `lmdeploy/turbomind/deploy/config.py` and both remaining dataclasses (`RopeParam`, `AttentionConfig`). Move `self._rope` to `types.SimpleNamespace`; drop the `attention_config` YAML section; patch C++ to read `cache_block_seq_len` from `engine_config`.

**Architecture:** Six incremental tasks. Tasks 1–2 are localized cleanups that don't change any external shape. Task 3 moves YAML construction of `attention_config` to an inline one-key dict (still the same YAML shape C++ reads today). Tasks 4–5 swap `RopeParam` → `SimpleNamespace` and delete `config.py`. Task 6 patches C++ and drops the `attention_config` YAML section entirely.

**Tech Stack:** Python (`types.SimpleNamespace`, pydantic dataclasses, YAML), C++ (`yaml-cpp`, CMake/ninja build).

---

## File Structure

| File | Responsibility after change |
|---|---|
| `lmdeploy/turbomind/deploy/config.py` | **Deleted.** |
| `lmdeploy/turbomind/deploy/source_model/utils.py` | `parse_rope_param` constructs `SimpleNamespace` instead of `RopeParam` |
| `lmdeploy/turbomind/deploy/spec.py` | `_parse_base()` owns the `rope_scaling_factor` patch; no `to_attention_config()` method; no `AttentionConfig`/`RopeParam` imports |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Uses `self._apply_rope(self._attn_cfg.rope)` like the three other specs |
| `lmdeploy/turbomind/turbomind.py` | `_from_hf` builds `{'engine_config': asdict(engine_config)}` directly (no `attention_config` section) |
| `src/turbomind/turbomind.cc` | Reads `cache_block_seq_len` from `engine` instead of `attention`; no `attention_config` YAML reads at all |

---

## Task 1: Move `rope_scaling_factor` patch into `_parse_base`

**Rationale:** The patch currently lives inside `to_attention_config()`, where it rebuilds a dynamic `RopeParam` on every call. Moving it to `_parse_base()` makes `self._rope` correct immediately after parsing, so every downstream consumer (including `_apply_rope` and subclass reads of `self._rope.dim`) sees the override. This unblocks the later deletion of `to_attention_config()` without risking double-log or stale rope state.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py:103-112` (`_parse_base` tail)
- Modify: `lmdeploy/turbomind/deploy/spec.py:152-160` (`to_attention_config` rope block)

### Steps

- [ ] **Step 1: Add the patch to `_parse_base`**

In `lmdeploy/turbomind/deploy/spec.py`, inside `_parse_base`, add the patch block right after the `parse_rope_param` call (around line 104). The current block is:

```python
        self._rope, self._max_position_embeddings = parse_rope_param(
            cfg, head_dim)

        # Layer-prefix detection deferred until weights loaded; default now.
```

Insert between these two statements:

```python
        self._rope, self._max_position_embeddings = parse_rope_param(
            cfg, head_dim)

        # Apply the deprecated --rope-scaling-factor override here so every
        # downstream consumer (subclass __init__, _apply_rope) sees the
        # patched rope before building C++ templates.
        if self.engine_cfg.rope_scaling_factor:
            self._rope.type = 'dynamic'
            self._rope.factor = self.engine_cfg.rope_scaling_factor
            self._rope.max_position_embeddings = self._max_position_embeddings
            logger.warning(
                '`--rope-scaling-factor` will be removed in a future release. '
                'Please instead use `--hf-overrides`.')

        # Layer-prefix detection deferred until weights loaded; default now.
```

- [ ] **Step 2: Delete the rope_scaling_factor block from `to_attention_config`**

In the same file, `to_attention_config()` currently has:

```python
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

Delete the `if ec.rope_scaling_factor:` block (8 lines). The method tail becomes:

```python
        if ec.use_logn_attn:
            cfg.use_logn_attn = int(ec.use_logn_attn)
        return cfg
```

Do NOT remove the `RopeParam` import yet — it's still referenced elsewhere (notably the `cfg.rope_param=self._rope` pass-through that still exists in this method).

- [ ] **Step 3: Smoke test**

Run:

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3-4B /nvme4/huggingface_hub/hub 1 0
```

Expected: model loads, produces a coherent paragraph, exit 0. No `--rope-scaling-factor` deprecation warning (the smoke run doesn't set `rope_scaling_factor`).

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "$(cat <<'EOF'
refactor(turbomind): move rope_scaling_factor override into _parse_base

The patch now runs immediately after parse_rope_param, so self._rope is
correct for every downstream consumer (subclass __init__, _apply_rope,
rope-dim reads). Removing the duplicate inside to_attention_config
eliminates the double-apply hazard that blocked deleting that method.
EOF
)"
```

---

## Task 2: DRY `glm4_moe_lite_spec.py` rope copy

**Rationale:** Three other specs (`qwen3_spec.py`, `qwen3_5_spec.py`, `gpt_oss_spec.py`) already use `self._apply_rope(self._attn_cfg.rope)`. `glm4_moe_lite_spec.py` still has the 15-line inline block. Since upcoming tasks change `_rope`'s representation, we'd have to touch this block anyway — DRY'ing it now avoids revisiting.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:88-102`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:14` (import)

### Steps

- [ ] **Step 1: Replace the inline rope copy**

In `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`, delete lines 88-102 (the entire rope-copy block):

```python
        self._attn_cfg.rope.type = rope_type_to_int(self._rope.type)
        self._attn_cfg.rope.base = self._rope.base
        self._attn_cfg.rope.dim  = self._rope.dim
        self._attn_cfg.rope.factor = self._rope.factor
        self._attn_cfg.rope.max_position_embeddings = self._max_position_embeddings
        if self._rope.type == 'yarn':
            self._attn_cfg.rope.yarn_attention_factor = self._rope.attention_factor
            self._attn_cfg.rope.yarn_beta_fast = self._rope.beta_fast
            self._attn_cfg.rope.yarn_beta_slow = self._rope.beta_slow
        elif self._rope.type == 'llama3':
            self._attn_cfg.rope.llama3_low_freq_factor = self._rope.low_freq_factor
            self._attn_cfg.rope.llama3_high_freq_factor = self._rope.high_freq_factor
            self._attn_cfg.rope.llama3_original_max_position_embeddings = self._rope.original_max_position_embeddings
        elif self._rope.type == 'mrope':
            self._attn_cfg.rope.mrope_section = self._rope.mrope_section
```

Replace with a single line:

```python
        self._apply_rope(self._attn_cfg.rope)
```

- [ ] **Step 2: Drop the unused `rope_type_to_int` import**

In `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`, line 14 currently reads:

```python
from .utils import _pad_inter_size, _pad_kv_head, get_yarn_params, layer_progress, parse_rope_param, rope_type_to_int
```

Remove `rope_type_to_int` from the list (it's no longer used in this file after Step 1):

```python
from .utils import _pad_inter_size, _pad_kv_head, get_yarn_params, layer_progress, parse_rope_param
```

Verify:

```bash
rg -n "rope_type_to_int" lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
```

Expected: no matches.

- [ ] **Step 3: Smoke test with GLM4**

GLM-4.7-Flash is the direct consumer of the changed block. Run:

```bash
python scripts/test_turbomind_model.py zai-org/GLM-4.7-Flash /nvme2/huggingface_hub/hub 1 0
```

Expected: coherent paragraph, exit 0. (GLM exercises `_apply_rope` through the newly-inserted single call, verifying the DRY'd path produces identical behavior to the previous inline block.)

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "$(cat <<'EOF'
refactor(deploy): DRY rope copy in glm4_moe_lite_spec

Replace the 15-line inline rope-copy block with self._apply_rope,
matching the three other specs. Drop the now-unused rope_type_to_int
import. Behavior-preserving; the helper does field-for-field the same
mapping.
EOF
)"
```

---

## Task 3: Delete `to_attention_config()`; inline `attention_config` YAML as one-key dict

**Rationale:** After Task 1, `to_attention_config()` does nothing that can't be expressed as a one-line dict literal in `_from_hf`. Inlining it removes the method, drops the `cache_block_seq_len` / `use_logn_attn` patching (which was adding nothing C++ ever reads), and leaves the YAML shape (`attention_config.cache_block_seq_len`) compatible with the current C++ reader.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py:136-163` (delete `to_attention_config`)
- Modify: `lmdeploy/turbomind/deploy/spec.py:14` (drop `AttentionConfig`/`RopeParam` imports)
- Modify: `lmdeploy/turbomind/turbomind.py:222-225` (inline YAML dict build)

### Steps

- [ ] **Step 1: Delete `to_attention_config()` from `spec.py`**

In `lmdeploy/turbomind/deploy/spec.py`, delete the entire `to_attention_config` method (roughly lines 136-163, including the blank line before it and its block-comment header "YAML export — produce AttentionConfig for C++ consumption"). The surrounding structure becomes:

```python
    def set_params(self, params: dict):
        self.params = params
        # Re-detect layer prefix ...
        if not self._pin_layer_prefix:
            self._layer_prefix, self._embed_key, self._norm_key = \
                detect_layer_prefix(params, self.hf_cfg)

    # ------------------------------------------------------------------
    # Checkpoint access helpers
    # ------------------------------------------------------------------

    def _get(self, key: str) -> torch.Tensor | None:
        return self.params.get(key)
```

Run `rg -n "to_attention_config" lmdeploy/` — only hits should be in docstrings or older plan docs. No live code should call it.

- [ ] **Step 2: Drop `AttentionConfig`/`RopeParam` imports from `spec.py`**

In `lmdeploy/turbomind/deploy/spec.py`, line 14 currently reads:

```python
from .config import AttentionConfig, RopeParam
```

Delete this line entirely. `_apply_rope` uses `rope_type_to_int` (from `.source_model.utils`, already imported) and does not need either class.

Verify:

```bash
rg -n "AttentionConfig|RopeParam" lmdeploy/turbomind/deploy/spec.py
```

Expected: no matches.

- [ ] **Step 3: Inline the YAML `attention_config` dict in `_from_hf`**

In `lmdeploy/turbomind/turbomind.py`, `_from_hf` currently has:

```python
        config_dict = {
            'attention_config': asdict(spec.to_attention_config()),
            'engine_config': asdict(engine_config),
        }
```

Replace with:

```python
        config_dict = {
            'attention_config': {
                'cache_block_seq_len': engine_config.cache_block_seq_len,
            },
            'engine_config': asdict(engine_config),
        }
```

C++ still reads `attention["cache_block_seq_len"]` from this shape — no schema change for the C++ side.

- [ ] **Step 4: Smoke test**

Run:

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3-4B /nvme4/huggingface_hub/hub 1 0
```

Expected: coherent paragraph, exit 0. The logged `turbomind model config` JSON should show `attention_config: {cache_block_seq_len: 64}` only (no `rope_param`, `softmax_scale`, `use_logn_attn`, `max_position_embeddings`).

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py lmdeploy/turbomind/turbomind.py
git commit -m "$(cat <<'EOF'
refactor(turbomind): delete to_attention_config; inline YAML dict

spec.to_attention_config was producing four YAML fields that C++ never
reads (softmax_scale, use_logn_attn, max_position_embeddings, rope_param).
Replace it with an inline one-key dict literal in _from_hf carrying just
cache_block_seq_len, which is the sole field C++ actually reads from the
attention_config YAML section.
EOF
)"
```

---

## Task 4: Replace `RopeParam` with `SimpleNamespace`

**Rationale:** After Task 3, `RopeParam` is no longer referenced anywhere except `parse_rope_param` (return value) and the `AttentionConfig.rope_param` field annotation. Swapping `parse_rope_param` to return a `SimpleNamespace` preserves `self._rope.dim`-style attribute access at all call sites (~30 of them) while dropping the dataclass.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py` (imports + `parse_rope_param` body)

### Steps

- [ ] **Step 1: Add `SimpleNamespace` import, drop `RopeParam` import**

In `lmdeploy/turbomind/deploy/source_model/utils.py`, update the imports near the top. The current state:

```python
import math

import torch

from lmdeploy.archs import get_model_arch

from ..config import RopeParam
from ..kind_map import TRIVIAL_FORMAT
```

Change to:

```python
import math
from types import SimpleNamespace

import torch

from lmdeploy.archs import get_model_arch

from ..kind_map import TRIVIAL_FORMAT
```

(Removed `from ..config import RopeParam`; added `from types import SimpleNamespace`.)

- [ ] **Step 2: Change `parse_rope_param` to construct a `SimpleNamespace`**

In the same file, `parse_rope_param` currently has (around line 61):

```python
    rope_param = RopeParam(type='default', base=rope_theta, dim=head_dim)
```

Replace with a `SimpleNamespace` that mirrors today's full `RopeParam` defaults:

```python
    rope_param = SimpleNamespace(
        type='default',
        base=rope_theta,
        dim=head_dim,
        factor=1.0,
        max_position_embeddings=None,
        attention_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        low_freq_factor=None,
        high_freq_factor=None,
        original_max_position_embeddings=None,
        mrope_section=None,
    )
```

Every `rope_param.X = Y` assignment later in the function (for dynamic/linear/llama3/yarn/mrope branches) continues to work because `SimpleNamespace` is dynamic.

Also update the return-type annotation on the function signature:

```python
def parse_rope_param(cfg: dict, head_dim: int) -> tuple[SimpleNamespace, int]:
```

(previously `tuple[RopeParam, int]`).

- [ ] **Step 3: Smoke test**

Run:

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3-4B /nvme4/huggingface_hub/hub 1 0
```

Expected: coherent paragraph, exit 0. `_apply_rope` in `spec.py` continues to read `self._rope.type`, `.base`, `.dim`, etc. unchanged because `SimpleNamespace` supports attribute access.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/utils.py
git commit -m "$(cat <<'EOF'
refactor(deploy): parse_rope_param returns SimpleNamespace

Replaces the RopeParam pydantic dataclass with a types.SimpleNamespace
carrying the same 12 fields and the same defaults. Attribute access
at every call site (self._rope.dim, etc.) is preserved. This untethers
us from config.py for rope parsing.
EOF
)"
```

---

## Task 5: Delete `config.py`

**Rationale:** After Task 4, nothing imports anything from `lmdeploy/turbomind/deploy/config.py`. Delete the file.

**Files:**
- Delete: `lmdeploy/turbomind/deploy/config.py`

### Steps

- [ ] **Step 1: Verify no live imports remain**

Run:

```bash
rg -n "from.*deploy\.config|from \.config\b|from \.\.config\b" lmdeploy/ tests/ scripts/ 2>/dev/null
```

Expected: no matches. If any remain, they're leftover references from earlier tasks — fix them.

Also check for bare-name imports:

```bash
rg -n "AttentionConfig|RopeParam|config_to_dict" lmdeploy/ tests/ scripts/ 2>/dev/null
```

Expected: only hits should be inside `lmdeploy/turbomind/deploy/config.py` itself (about to be deleted). Any external hit is a bug to fix.

- [ ] **Step 2: Delete the file**

```bash
git rm lmdeploy/turbomind/deploy/config.py
```

- [ ] **Step 3: Smoke test**

Run:

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3-4B /nvme4/huggingface_hub/hub 1 0
```

Expected: coherent paragraph, exit 0. Specifically, no `ImportError` or `ModuleNotFoundError` for anything in `config`.

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor(turbomind): delete deploy/config.py

With RopeParam and AttentionConfig replaced by SimpleNamespace + inline
YAML dict, config.py has no remaining users. Delete the file.
EOF
)"
```

---

## Task 6: Patch C++; drop `attention_config` from YAML

**Rationale:** `cache_block_seq_len` already lives on `TurbomindEngineConfig` and therefore on the YAML's `engine_config` section. Patching C++ to read it from `engine` instead of `attention` lets us drop the `attention_config` YAML section entirely. End state: the YAML handed to C++ is `{'engine_config': ...}` with exactly one top-level key.

**Files:**
- Modify: `src/turbomind/turbomind.cc:192` (delete unused attention local)
- Modify: `src/turbomind/turbomind.cc:198` (read cache_block_seq_len from engine)
- Modify: `lmdeploy/turbomind/turbomind.py` (drop attention_config from config_dict)

### Steps

- [ ] **Step 1: Patch `turbomind.cc:198`**

In `src/turbomind/turbomind.cc`, the current line 198 reads:

```cpp
    engine_param_.cache_block_seq_len = attention["cache_block_seq_len"].as<int>(0);
```

Change to:

```cpp
    engine_param_.cache_block_seq_len = engine["cache_block_seq_len"].as<int>(0);
```

- [ ] **Step 2: Delete the unused `attention` local**

After Step 1, line 192 (`const auto attention = node["attention_config"];`) has no remaining reader. Delete that line.

The surrounding block should read:

```cpp
    /// TODO: move config parsing to suitable place
    const auto engine = node["engine_config"];

    data_type_ = data_type_from_string(engine["dtype"].as<std::string>());
    TM_CHECK(data_type_ == kBfloat16 || data_type_ == kHalf);

    engine_param_.cache_block_seq_len = engine["cache_block_seq_len"].as<int>(0);
```

Verify:

```bash
rg -n "attention\[|attention_config" src/turbomind/turbomind.cc
```

Expected: no matches.

- [ ] **Step 3: Rebuild C++**

```bash
cd build && ninja
```

Expected: build succeeds, no errors or warnings about `attention`. The incremental build recompiles only `turbomind.cc` and re-links.

- [ ] **Step 4: Drop `attention_config` from YAML in `_from_hf`**

In `lmdeploy/turbomind/turbomind.py`, `_from_hf` currently builds:

```python
        config_dict = {
            'attention_config': {
                'cache_block_seq_len': engine_config.cache_block_seq_len,
            },
            'engine_config': asdict(engine_config),
        }
```

Replace with:

```python
        config_dict = {'engine_config': asdict(engine_config)}
```

- [ ] **Step 5: Smoke test**

Run:

```bash
cd ..   # back to repo root if you cd'd into build
python scripts/test_turbomind_model.py Qwen/Qwen3-4B /nvme4/huggingface_hub/hub 1 0
python scripts/test_turbomind_model.py Qwen/Qwen3-4B-AWQ /nvme4/huggingface_hub/hub 1 0
```

Expected: both runs produce coherent paragraphs, exit 0. The logged `turbomind model config` JSON now has exactly one top-level key: `engine_config`. No `attention_config` section at all.

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/turbomind.cc lmdeploy/turbomind/turbomind.py
git commit -m "$(cat <<'EOF'
refactor(turbomind): C++ reads cache_block_seq_len from engine_config

src/turbomind/turbomind.cc now reads cache_block_seq_len from the
engine_config YAML section, where the value already lives on
TurbomindEngineConfig. The attention_config YAML section is dropped
entirely since it held no other live fields. The Python-side YAML
build shrinks to {'engine_config': asdict(engine_config)}.
EOF
)"
```

---

## Self-Review

### Spec Coverage

| Spec section | Task(s) |
|---|---|
| Decision 1: Aggressive Python + C++ elimination | Task 6 (C++ + final YAML); Task 3 + Task 5 (Python side) |
| Decision 2: `SimpleNamespace` for `_rope` | Task 4 |
| Decision 3: DRY `glm4_moe_lite_spec.py` | Task 2 |
| Design §1: `parse_rope_param` returns `SimpleNamespace` | Task 4 |
| Design §2: `_parse_base` absorbs patch; `to_attention_config` deleted | Task 1 (move); Task 3 (delete method) |
| Design §3: Spec subclasses (only glm4_moe_lite changes) | Task 2 |
| Design §4: `turbomind.py` drops `attention_config` from YAML | Task 3 (interim one-key); Task 6 (final drop) |
| Design §5: Delete `config.py` | Task 5 |
| Design §6: C++ reads from `engine` | Task 6 |
| Files Changed table rows | All 6 files touched across Tasks 1–6 |
| Verification: unquantized smoke run | Tasks 1, 2, 3, 4, 5, 6 (each step) |
| Verification: quantized smoke run | Task 6 Step 5 |
| Rebuild C++ | Task 6 Step 3 (`cd build && ninja`) |

### Placeholder Scan

- No "TBD", "TODO", "implement later", or "fill in details".
- Every code step has complete code.
- Every shell step has exact commands and expected output.
- Build command is concrete: `cd build && ninja`.

### Type Consistency

- `SimpleNamespace` is the type of `self._rope` from Task 4 onward. `_apply_rope` reads it via attribute access, which `SimpleNamespace` supports identically to the previous `RopeParam`.
- `AttentionConfig` is referenced in Tasks 1–3 (method body, imports) but gone by Task 5; reference count is consistent.
- `RopeParam` is referenced up through Task 3 (via `to_attention_config` and the `.config` import); gone from Task 4 onward. Consistent.
- `engine_config.cache_block_seq_len` is `int` (default 64 per `messages.py:279`); the YAML dict literal in Task 3 carries the same type; C++ reads it as `int` (Task 6). Consistent end-to-end.
- `_from_hf` call signature unchanged across all tasks; only the body of `config_dict` changes.

No inconsistencies found.
