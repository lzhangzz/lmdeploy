# Eliminate `RopeParam` and `AttentionConfig` Design

## Goal

Delete `lmdeploy/turbomind/deploy/config.py` entirely, removing both remaining
dataclasses (`RopeParam`, `AttentionConfig`). The Python `AttentionConfig` is
effectively dead on the C++ side; `RopeParam` is a Python-internal data bag that
can be replaced by a `types.SimpleNamespace`. The YAML payload handed to C++
shrinks to a single `engine_config` section, and C++ reads `cache_block_seq_len`
from `engine_config` (where it already lives on `TurbomindEngineConfig`).

## Current State

After the prior refactor ([eliminate TurbomindModelConfig]
(2026-04-20-eliminate-turbomind-model-config-design.md)), `config.py` is down to
29 lines holding two pydantic dataclasses:

| Class | Fields (count) | Purpose today |
|---|---|---|
| `RopeParam` | 12 (type, base, dim, factor, max_position_embeddings, attention_factor, beta_fast, beta_slow, low_freq_factor, high_freq_factor, original_max_position_embeddings, mrope_section) | Python-internal bag populated by `parse_rope_param`, stored as `self._rope`, consumed by `_apply_rope` (→ pybind) and by subclasses reading `self._rope.dim` for rope permutation |
| `AttentionConfig` | 5 (softmax_scale, cache_block_seq_len, use_logn_attn, max_position_embeddings, rope_param) | Produced by `spec.to_attention_config()`; serialized into the YAML's `attention_config` section |

**Discovery — most of `AttentionConfig` is dead in YAML.** `src/turbomind/turbomind.cc:185-241`
loads the YAML once and reads from `node["attention_config"]` for **exactly one
field**: `attention["cache_block_seq_len"]` (line 198). Every other field
(`rope_param`, `softmax_scale`, `use_logn_attn`, `max_position_embeddings`)
is serialized but never read by any C++ consumer.

**Discovery — C++ rope data flows through pybind, not YAML.** Each spec
subclass builds `self._attn_cfg = _tm.AttentionConfig()` (the pybind-bound C++
struct — distinct from Python's `AttentionConfig`) and calls
`self._apply_rope(self._attn_cfg.rope)` to copy `self._rope` field-by-field
onto the C++ struct. This is the live path for rope parameters. The YAML
`rope_param` dict is never consulted.

**Discovery — `cache_block_seq_len` already exists on `engine_config`.**
`TurbomindEngineConfig.cache_block_seq_len` is defined at `messages.py:279`
with default 64. The value on the YAML's `attention_config` section is
produced by patching `engine_config.cache_block_seq_len` into
`AttentionConfig.cache_block_seq_len` inside `spec.to_attention_config()`. If
C++ reads the field from `engine_config` directly, the patching and the
entire `attention_config` YAML section become unnecessary.

**Duplicate rope-copy block.** `glm4_moe_lite_spec.py:88-102` still contains
an inline copy of the rope fields onto `_attn_cfg.rope` that the other three
specs already DRY'd into `self._apply_rope(self._attn_cfg.rope)`. Any refactor
touching rope representation has to touch that block anyway; DRY'ing it is the
natural completion.

## Design Decisions

1. **Aggressive Python + C++ elimination.** Patch `turbomind.cc:198` to read
   `cache_block_seq_len` from `engine_config` instead of `attention_config`.
   Drop the entire `attention_config` YAML section. Delete both Python
   dataclasses. Delete `config.py`. Alternative (Python-only, keep a one-key
   `attention_config` dict in YAML) was rejected: the C++ change is a one-line
   edit to a line we own, and it eliminates the last reason `attention_config`
   exists anywhere.

2. **Replace `RopeParam` with `types.SimpleNamespace`.** `self._rope.dim`
   attribute access is preserved at every call site (~30 of them across
   `spec.py`, `source_model/utils.py`, and four spec subclasses).
   `parse_rope_param` constructs a `SimpleNamespace` with all 12 fields set to
   today's `RopeParam` defaults. Alternative (plain dict) was rejected: the
   diff was ~30 attribute-to-subscript rewrites for no semantic gain.

3. **DRY `glm4_moe_lite_spec.py` inline rope copy.** Replace lines 88-102 with
   `self._apply_rope(self._attn_cfg.rope)` and drop the `rope_type_to_int`
   import. Matches the other three specs and the intent of the pre-existing
   dedup design at
   `docs/superpowers/specs/2026-04-20-rope-param-dedup-design.md`.

## Design

### 1. `parse_rope_param` returns a `SimpleNamespace`

In `lmdeploy/turbomind/deploy/source_model/utils.py`, replace the
`RopeParam(type='default', base=rope_theta, dim=head_dim)` construction
(line 61) with:

```python
from types import SimpleNamespace  # add import at top

def parse_rope_param(cfg: dict, head_dim: int) -> tuple[SimpleNamespace, int]:
    """Parse RoPE configuration from a model config dict.

    Returns:
        rope_param: SimpleNamespace with the full set of rope fields.
          Unused-by-type fields keep default values (matching today's
          RopeParam dataclass defaults).
        max_position_embeddings: int (0 if not present in config)
    """
    # ... rope_theta / rope_scaling / max_position_embeddings resolution
    # unchanged from today ...

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

    # ... type-specific branches (dynamic / linear / llama3 / yarn / mrope)
    # unchanged — every `rope_param.X = Y` assignment still works because
    # SimpleNamespace is dynamic ...

    return rope_param, max_position_embeddings
```

Drop `from ..config import RopeParam` from the import block.

All other `parse_rope_param` assignments (`rope_param.type = 'dynamic'`,
`rope_param.factor = scaling_factor`, etc.) are byte-identical.

### 2. `_parse_base` absorbs the `rope_scaling_factor` patch; `to_attention_config` is deleted

Today, `spec.to_attention_config()` does three things: build an
`AttentionConfig`, patch `cache_block_seq_len` / `use_logn_attn` from
`engine_config`, and for `engine_config.rope_scaling_factor` replace
`rope_param` with a dynamic-type one. The first two go away with the
dataclass. The third needs a new home.

Put it at the end of `TextModelSpec._parse_base()`, right after
`parse_rope_param`:

```python
# In TextModelSpec._parse_base, after `self._rope, self._max_position_embeddings = parse_rope_param(cfg, head_dim)`
if self.engine_cfg.rope_scaling_factor:
    self._rope.type = 'dynamic'
    self._rope.factor = self.engine_cfg.rope_scaling_factor
    self._rope.max_position_embeddings = self._max_position_embeddings
    logger.warning(
        '`--rope-scaling-factor` will be removed in a future release. '
        'Please instead use `--hf-overrides`.')
```

This runs before any spec subclass accesses `self._rope`, so the override is
visible everywhere (including `_apply_rope`'s pybind copy).

Delete the entire `to_attention_config()` method from `TextModelSpec`. Drop
`from .config import AttentionConfig` (and any `RopeParam` import) from
`spec.py`.

`_apply_rope()` body is unchanged. `SimpleNamespace` supports the same
attribute access as the previous dataclass, so `self._rope.type`,
`self._rope.base`, etc. all continue to work.

### 3. Spec subclasses

- `qwen3_spec.py`, `qwen3_5_spec.py`, `gpt_oss_spec.py`: **No changes.**
  They already use `self._apply_rope(self._attn_cfg.rope)` and access
  `self._rope.dim` via attribute syntax, which `SimpleNamespace` supports.

- `glm4_moe_lite_spec.py` (the odd one out): delete the inline rope-copy
  block at lines 88-102:

  ```python
  # DELETE:
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

  Replace with a single call:

  ```python
  self._apply_rope(self._attn_cfg.rope)
  ```

  Drop `rope_type_to_int` from the `from .utils import ...` line (still
  needed for `parse_rope_param`, `_pad_kv_head`, etc.).

  The post-hoc `self._rope.max_position_embeddings` and
  `self._rope.attention_factor` mutations earlier in `__init__`
  (roughly lines 68-70) are unchanged — `SimpleNamespace` supports both
  mutation and subsequent attribute reads via `_apply_rope`.

### 4. `turbomind.py` — drop `attention_config` from YAML

In `_from_hf`, the current YAML build is:

```python
config_dict = {
    'attention_config': asdict(spec.to_attention_config()),
    'engine_config': asdict(engine_config),
}
```

Replace with:

```python
config_dict = {'engine_config': asdict(engine_config)}
```

Run `rg -n "to_attention_config|AttentionConfig|RopeParam" lmdeploy/` to
confirm no references remain in Python sources.

### 5. Delete `config.py`

After the above, no module imports anything from
`lmdeploy/turbomind/deploy/config.py`. Delete the file.

### 6. C++ reads `cache_block_seq_len` from `engine`

In `src/turbomind/turbomind.cc`, change line 198 from:

```cpp
engine_param_.cache_block_seq_len = attention["cache_block_seq_len"].as<int>(0);
```

to:

```cpp
engine_param_.cache_block_seq_len = engine["cache_block_seq_len"].as<int>(0);
```

The surrounding declaration `const auto attention = node["attention_config"];`
on line 192 becomes unused after this edit. Delete it too.

## Files Changed

| File | Change |
|---|---|
| `lmdeploy/turbomind/deploy/config.py` | **Deleted** (file removed). |
| `lmdeploy/turbomind/deploy/source_model/utils.py` | `parse_rope_param` returns a `SimpleNamespace` with all 12 fields preinitialized to today's `RopeParam` defaults. Drop `from ..config import RopeParam`. Add `from types import SimpleNamespace`. |
| `lmdeploy/turbomind/deploy/spec.py` | Delete `to_attention_config()` method. Move the `rope_scaling_factor → dynamic rope` override into `_parse_base()` immediately after `parse_rope_param`. Drop `from .config import AttentionConfig` (and `RopeParam` if imported). `_apply_rope()` body unchanged. |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Replace inline rope-copy block (lines 88-102) with `self._apply_rope(self._attn_cfg.rope)`. Remove `rope_type_to_int` from the `.utils` import. |
| `lmdeploy/turbomind/turbomind.py` | In `_from_hf`, change `config_dict` to `{'engine_config': asdict(engine_config)}`. Remove any `spec.to_attention_config()` call. |
| `src/turbomind/turbomind.cc` | Line 198: read `cache_block_seq_len` from `engine` instead of `attention`. Delete the unused `const auto attention = node["attention_config"];` declaration on line 192. |

## Verification

Rebuild the C++ extension before testing (incremental build via the
pre-configured `build/` directory):

```bash
cd build && ninja
```

Then run the smoke script unmodified against two cases (concrete models
resolved from the local registry):

```bash
# Unquantized — exercises parse_rope_param defaults and _apply_rope
python scripts/test_turbomind_model.py Qwen/Qwen3-4B /nvme4/huggingface_hub/hub 1 0

# Quantized — exercises the full pipeline under AWQ
python scripts/test_turbomind_model.py Qwen/Qwen3-4B-AWQ /nvme4/huggingface_hub/hub 1 0
```

Each run must produce a coherent text response. The logged
`turbomind model config` JSON should show exactly one top-level key
(`engine_config`); no `attention_config` section.

The `rope_scaling_factor` dynamic-NTK path is not exposed by the smoke
script; it's a deprecated `TurbomindEngineConfig` field whose handler is a
five-line transcription from the current `to_attention_config()` code. Code
review covers it.

## Non-Goals

- **No other C++ changes.** Only the single YAML key swap (one-line edit
  plus deletion of the now-unused `attention` local variable) is needed.
- **No new Python tests.** No surviving tests assert on `RopeParam` or
  `AttentionConfig` shape — the converter-level tests were deleted in the
  prior refactor.
- **No relocation of `parse_rope_param` or `_apply_rope`.** Both stay where
  they are.
- **No changes to the three already-DRY'd spec subclasses.** Only
  `glm4_moe_lite_spec.py` changes — and that's the stale duplicate, not a
  new inconsistency.
- **No signature or behavioral changes to `TurbomindEngineConfig`.**
  `cache_block_seq_len` already exists on it; we only change where C++
  reads the value from in the YAML.
- **No removal of unused `RopeParam` fields** (e.g., `mrope_section` for
  models that don't use mrope). `SimpleNamespace` carries all 12 fields
  the same way `RopeParam` did, with the same defaults.

### Incidental behavior notes

- **`--rope-scaling-factor` now takes effect in the C++ kernel for
  non-GLM specs.** Before this refactor, the override was applied inside
  `to_attention_config()` — which ran *after* each spec subclass had
  already called `_apply_rope(self._attn_cfg.rope)`. The pybind-bound
  `_tm.AttentionConfig.rope` carried the unpatched rope to C++, so the
  deprecation warning fired but the kernel used `type='default'`.
  Moving the patch into `_parse_base()` (Task 1) reorders it ahead of
  `_apply_rope`, so the patched rope reaches C++. This is the behavior
  the deprecation warning was always warning toward; no user-visible
  regression. The smoke script does not exercise `--rope-scaling-factor`
  directly — code review + the surrounding `dynamic` rope test coverage
  in existing C++ kernels is the verification.

- **GLM4 MoE Lite continues to silently ignore `--rope-scaling-factor`.**
  `Glm4MoeLiteSpec.__init__` re-invokes `parse_rope_param(hf_cfg, qk_rope_dim)`
  after `super().__init__()`, overwriting the patched `self._rope` with
  a fresh parse. This is unchanged behavior — the old code path had the
  same problem — but it does mean the "every downstream consumer sees
  the override" framing has one asterisk. Re-applying the patch inside
  GLM4 is out of scope for this refactor (the flag is deprecated and
  the users we care about have migrated to `--hf-overrides`).

## Scope

Cross-language (Python + 1-line C++). Landable as a single PR or split into
two:

1. **Python-only:** Introduce `SimpleNamespace` for `_rope` (via
   `parse_rope_param`), move `rope_scaling_factor` override into
   `_parse_base`, DRY `glm4_moe_lite_spec.py`, delete `to_attention_config()`,
   delete `AttentionConfig` and `RopeParam` dataclasses, delete `config.py`.
   YAML builds the `attention_config` section inline as a one-key dict
   literal:

   ```python
   config_dict = {
       'attention_config': {'cache_block_seq_len': engine_config.cache_block_seq_len},
       'engine_config': asdict(engine_config),
   }
   ```

   This is a working, testable state — C++ still reads
   `attention["cache_block_seq_len"]` as today.

2. **Cross-language final:** Patch `turbomind.cc` to read
   `cache_block_seq_len` from `engine`. Drop the `attention_config` dict
   from the YAML in `_from_hf`. Config dict becomes
   `{'engine_config': asdict(engine_config)}`.

Splitting is optional; the refactor is small enough to land as a single
PR. The plan doc will decide final task granularity.
