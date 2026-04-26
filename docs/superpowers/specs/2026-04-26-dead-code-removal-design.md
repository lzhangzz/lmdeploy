# Dead Code Removal in `lmdeploy/turbomind/deploy/`

## Scope

Remove unreferenced definitions and an unused file from `lmdeploy/turbomind/deploy/`, plus stale `__pycache__` artifacts.

## Changes

### 1. `converter.py` — `SUPPORTED_FORMATS` (line 18)

Module-level constant, never imported or referenced anywhere.

### 2. `format_ops.py` — Delete entire file

Three functions (`can_permute`, `can_split`, `param_shape`) with zero references in any file type (Python, markdown, configs). Not exported from any `__init__.py`.

### 3. `linear.py` — Remove 8 dead definitions

Standalone functions (never called):
- `split_out_dim` (line 81) — only caller is the dead `Linear.split_out_dim` method
- `concat_out_dim` (line 87)
- `permute_out_dim` (line 92)
- `permute_in_dim` (line 97)
- `transpose` (line 112)

Linear class methods (never called on any instance):
- `Linear.split_out_dim` (line 140)
- `Linear.split_in_dim` (line 149)
- `Linear.concat_in_dim` (classmethod, line 179)

### 4. `__pycache__` — Delete stale `.pyc` files

For modules that no longer have corresponding `.py` files (e.g., `parameter.py`, `commit.py`, `config.py`, `configs.py`, `distributor.py`, `kind_map.py`, `policy.py`, `builder/_old.py`, `builder/linear.py`).

### Alive (not touched)

`Linear` dataclass, `pad_out_dim`, `pad_in_dim`, `Linear.concat_out_dim` (classmethod, used by `builder/deltanet.py`), private helpers (`_norm`, `_has_input_dim`, `_pad_1d`, `_permute_along`).

## Risk

None. No imports, no callers, no dynamic/string-based references.

## Verification

- Build: `ninja` from `build/`
- Test: `scripts/test_turbomind_model.py` with any model

## Commit

Single commit for all changes.
