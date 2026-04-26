# Dead Code Removal in `lmdeploy/turbomind/deploy/`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all unreferenced definitions, an unused file, and stale `__pycache__` artifacts from `lmdeploy/turbomind/deploy/`.

**Architecture:** Straightforward deletion — remove dead constants, functions, methods, and files with no callers or importers anywhere in the repo.

**Tech Stack:** Python, git

---

### Task 1: Remove `SUPPORTED_FORMATS` from `converter.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/converter.py:18`

- [ ] **Step 1: Delete the constant**

Delete line 18 from `converter.py`:
```python
SUPPORTED_FORMATS = ['hf', 'awq', 'gptq', 'compressed-tensors', 'fp8', 'mxfp4', None]
```

The edit removes the line and the blank line immediately after it, so lines 18-19 (the constant + blank separator) collapse.

- [ ] **Step 2: Verify no remaining references**

```bash
grep -rn "SUPPORTED_FORMATS" --include="*.py" .
```
Expected: no output.

---

### Task 2: Delete `format_ops.py`

**Files:**
- Delete: `lmdeploy/turbomind/deploy/format_ops.py`

- [ ] **Step 1: Delete the file**

```bash
rm lmdeploy/turbomind/deploy/format_ops.py
```

- [ ] **Step 2: Delete its `__pycache__` entry**

```bash
rm lmdeploy/turbomind/deploy/__pycache__/format_ops.cpython-312.pyc
```

- [ ] **Step 3: Verify no references remain**

```bash
grep -rn "format_ops" . --include="*.py" --include="*.md"
```
Expected: no output.

---

### Task 3: Remove dead code from `linear.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py`

Dead items to remove:
- `split_out_dim` standalone (lines 81-84) — only caller (`Linear.split_out_dim`) is also being removed
- `concat_out_dim` standalone (lines 87-89) — never called
- `permute_out_dim` (lines 92-94) — never called
- `permute_in_dim` (lines 97-99) — never called
- `transpose` (lines 112-114) — never called
- `Linear.split_out_dim` method (lines 140-147)
- `Linear.split_in_dim` method (lines 149-160)
- `Linear.concat_in_dim` classmethod (lines 179-197)

- [ ] **Step 1: Remove the 5 dead standalone functions**

Remove lines 81-99 (`split_out_dim`, `concat_out_dim`, `permute_out_dim`, `permute_in_dim`) and lines 112-114 (`transpose`).

The `# Tensor functions` section should look like this after removal:

```python
# ---------------------------------------------------------------------------
# Tensor functions
# ---------------------------------------------------------------------------


def pad_out_dim(t: Tensor, target: int, dim: int) -> Tensor:
    """Pad *dim* to *target* size with zeros."""
    return _pad_1d(t, _norm(dim, t.dim()), target)


def pad_in_dim(t: Tensor, target: int, dim: int) -> Tensor:
    """Pad *dim* to *target* size with zeros."""
    return _pad_1d(t, _norm(dim, t.dim()), target)
```

- [ ] **Step 2: Remove the 3 dead `Linear` methods**

Remove `split_out_dim` (lines 140-147), `split_in_dim` (lines 149-160), and `concat_in_dim` (lines 179-197).

The `Linear` class should retain only:
- `tensors`, `weight_format`, `data_format` fields
- `concat_out_dim` classmethod (the one that's actually used)

- [ ] **Step 3: Verify the remaining file**

Confirm the file still has:
- Internal helpers: `_norm`, `_has_input_dim`, `_pad_1d`, `_permute_along`
- Tensor functions: `pad_out_dim`, `pad_in_dim`
- `Linear` dataclass with fields + `concat_out_dim` classmethod

- [ ] **Step 4: Verify no references to removed symbols**

```bash
grep -rn "split_out_dim\|split_in_dim\|concat_in_dim\|permute_out_dim\|permute_in_dim" --include="*.py" . | grep -v __pycache__ | grep -v "lmdeploy/turbomind/deploy/linear.py"
```
Expected: only `weight_format.py:120` comment referencing `concat_in_dim` (will be fixed in Task 4).

---

### Task 4: Update comment in `weight_format.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/weight_format.py:120`

- [ ] **Step 1: Update the comment**

Change line 120 from:
```
    ``Linear.concat_out_dim`` / ``Linear.concat_in_dim``.
```
to:
```
    ``Linear.concat_out_dim``.
```

---

### Task 5: Clean stale `__pycache__` files

**Files:**
- Delete: 13 stale `.pyc` files (gitignored, not tracked — local cleanup only)

- [ ] **Step 1: Delete stale `.pyc` files**

```bash
rm lmdeploy/turbomind/deploy/__pycache__/config.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/module.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/kind_map.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/policy.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/parameter.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/load_context.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/transforms.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/commit.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/configs.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/distributor.cpython-312.pyc
rm lmdeploy/turbomind/deploy/__pycache__/builder.cpython-312.pyc
rm lmdeploy/turbomind/deploy/builder/__pycache__/_old.cpython-312.pyc
rm lmdeploy/turbomind/deploy/builder/__pycache__/linear.cpython-312.pyc
```

- [ ] **Step 2: Verify no stale `.pyc` remain**

```bash
find lmdeploy/turbomind/deploy -name "*.pyc" | while read f; do
  base=$(echo "$f" | sed 's|/__pycache__/|/|' | sed 's|\.cpython-312\.pyc|.py|')
  [ ! -f "$base" ] && echo "STALE: $f"
done
```
Expected: no output.

> `.pyc` files are gitignored — no commit needed for this step.

---

### Task 6: Build, test, and commit

**No file changes.** Verify the cleanup doesn't break anything, then commit all changes.

- [ ] **Step 1: Build**

```bash
cd build && ninja
```
Expected: build succeeds.

- [ ] **Step 2: Run turbomind model test**

```bash
python scripts/test_turbomind_model.py --model <model_id> --tp 1
```
Expected: model responds with meaningful human words, at least 128 tokens.

> **Note:** Choose any available model from `list_models` MCP tool. Verify the response is coherent.

- [ ] **Step 3: Single commit for all changes**

```bash
git add lmdeploy/turbomind/deploy/converter.py \
        lmdeploy/turbomind/deploy/format_ops.py \
        lmdeploy/turbomind/deploy/linear.py \
        lmdeploy/turbomind/deploy/weight_format.py
git commit -m "$(cat <<'EOF'
chore: remove dead code from deploy directory

Remove SUPPORTED_FORMATS constant, format_ops.py module, dead
standalone functions and Linear methods in linear.py, and update
a stale comment in weight_format.py.
EOF
)"
```
Expected: clean commit with all changes.
