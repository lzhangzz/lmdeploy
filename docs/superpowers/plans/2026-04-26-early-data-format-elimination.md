# Eliminate `data_format` Threading from `Linear` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive `data_format` at commit time from `weight_format` + `data_type` instead of threading it through ~15 `Linear` construction sites.

**Architecture:** One-line behavioral change in `Builder._add_linear()` — call `linear.weight_format.make_data_format(compute_dtype)` instead of reading `linear.data_format`. Everything else is deletion.

**Tech Stack:** Python, `_turbomind` C++ extension (`DataFormat`, `ResolveLinearWeightFormat`)

---

### Task 1: Derive `data_format` at commit time + drop assertion

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:409-411,429`

- [ ] **Step 1: Replace `linear.data_format` with derived value**

In `_add_linear`, change line 429:

```python
# Before
lin_cfg.format     = linear.data_format

# After
lin_cfg.format     = linear.weight_format.make_data_format(compute_dtype)
```

- [ ] **Step 2: Drop the assertion that `data_format is not None`**

Remove lines 409-411:

```python
# Remove this block:
        assert linear.data_format is not None, (
            f"{name}: Linear.data_format must be populated by "
            f"WeightFormatResolver.resolve or a fusion helper.")
```

- [ ] **Step 3: Verify the module loads**

```bash
python -c "from lmdeploy.turbomind.deploy.linear import Linear; from lmdeploy.turbomind.deploy.builder import Builder"
```

Expected: clean import, no errors. At this point `data_format` is still on `Linear` and still set by all construction sites, but no longer consumed.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/_base.py
git commit -m "$(cat <<'EOF'
refactor: derive data_format from weight_format at commit time

Replace linear.data_format with linear.weight_format.make_data_format(compute_dtype)
in _add_linear. Drop the now-redundant assertion.
EOF
)"
```

---

### Task 2: Remove `data_format` from `Linear` and all call sites

**Why one commit:** `data_format` is a required field on the `Linear` dataclass (no default). You cannot remove it from construction sites without removing the field, and vice versa. Both must change atomically.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py:89,99-106`
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:107,111-128,180,241,644`
- Modify: `lmdeploy/turbomind/deploy/builder/mla.py:46-50,62-64`
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py:225-226,315-318`
- Modify: `lmdeploy/turbomind/deploy/builder/deltanet.py:40-42,92-93`
- Modify: `lmdeploy/turbomind/deploy/weight_format.py:492-494`

- [ ] **Step 1: Remove field from `Linear` dataclass (`linear.py:89`)**

```python
# Before
    tensors: dict[str, Tensor]
    weight_format: "WeightFormat" = field(compare=False, repr=False)
    data_format: "_tm.DataFormat" = field(compare=False, repr=False)

# After
    tensors: dict[str, Tensor]
    weight_format: "WeightFormat" = field(compare=False, repr=False)
```

- [ ] **Step 2: Simplify `concat_out_dim` — remove data_format uniformity check (`linear.py:99-106`)**

```python
# Before
        wfmts = {x.weight_format for x in xs}
        dfmts = {x.data_format  for x in xs}
        assert len(wfmts) == 1 and len(dfmts) == 1, (
            "concat_out_dim requires uniform weight_format and data_format; "
            "call dequant_mixed first if formats differ.")
        return Linear(tensors=result,
                      weight_format=next(iter(wfmts)),
                      data_format=next(iter(dfmts)))

# After
        wfmts = {x.weight_format for x in xs}
        assert len(wfmts) == 1, (
            "concat_out_dim requires uniform weight_format; "
            "call dequant_mixed first if formats differ.")
        return Linear(tensors=result,
                      weight_format=next(iter(wfmts)))
```

- [ ] **Step 3: Strip `data_format=` from `_dequant_linear` (`_base.py:104-108`)**

```python
# Before
    return Linear(
        tensors=new_tensors,
        weight_format=trivial,
        data_format=trivial.make_data_format(data_type),
    )

# After
    return Linear(
        tensors=new_tensors,
        weight_format=trivial,
    )
```

- [ ] **Step 4: Simplify `_ensure_compatible_formats` — drop data_format normalization (`_base.py:111-128`)**

Replace the entire function body:

```python
# Before
def _ensure_compatible_formats(linears: dict[str, Linear], *, data_type) -> dict[str, Linear]:
    """Dequant linears to a common trivial format if a fusion group has mixed formats."""
    formats = {name: lin.weight_format.name for name, lin in linears.items()}
    if len(set(formats.values())) <= 1:
        # Weight formats agree; normalize data_format to a single shared object.
        target_df = next(iter(linears.values())).data_format
        return {name: (Linear(lin.tensors, weight_format=lin.weight_format,
                              data_format=target_df)
                       if lin.data_format is not target_df else lin)
                for name, lin in linears.items()}
    result = {name: _dequant_linear(lin, data_type=data_type) for name, lin in linears.items()}
    # Normalize data_format after dequant — each _dequant_linear may produce
    # a distinct DataFormat object even when they represent the same dtype.
    target_df = next(iter(result.values())).data_format
    return {name: (Linear(lin.tensors, weight_format=lin.weight_format,
                          data_format=target_df)
                   if lin.data_format is not target_df else lin)
            for name, lin in result.items()}

# After
def _ensure_compatible_formats(linears: dict[str, Linear], *, data_type) -> dict[str, Linear]:
    """Dequant linears to a common trivial format if a fusion group has mixed formats."""
    formats = {name: lin.weight_format.name for name, lin in linears.items()}
    if len(set(formats.values())) <= 1:
        return linears
    return {name: _dequant_linear(lin, data_type=data_type) for name, lin in linears.items()}
```

- [ ] **Step 5: Strip `data_format=` from `transform_output_dim` (`_base.py:180`)**

```python
# Before
            Linear(ts, weight_format=first.weight_format,
                   data_format=first.data_format)

# After
            Linear(ts, weight_format=first.weight_format)
```

- [ ] **Step 6: Strip `data_format=` from `transform_input_dim` (`_base.py:241`)**

```python
# Before
            Linear(ts, weight_format=first.weight_format,
                   data_format=first.data_format)

# After
            Linear(ts, weight_format=first.weight_format)
```

- [ ] **Step 7: Strip `data_format=` from `add_lm_head` (`_base.py:644`)**

```python
# Before
        padded = Linear(
            tensors={k: pad_out_dim(t, padded_vocab, dim=-1)
                     for k, t in linear.tensors.items()},
            weight_format=linear.weight_format,
            data_format=linear.data_format)

# After
        padded = Linear(
            tensors={k: pad_out_dim(t, padded_vocab, dim=-1)
                     for k, t in linear.tensors.items()},
            weight_format=linear.weight_format)
```

- [ ] **Step 8: Strip `data_format=` from `fold_kv_b` return (`mla.py:46-50`)**

```python
# Before
    return (Linear(tensors={"weight": q_folded.contiguous()},
                   weight_format=q_b.weight_format,
                   data_format=q_b.data_format),
            Linear(tensors={"weight": o_folded.contiguous()},
                   weight_format=wo.weight_format,
                   data_format=wo.data_format))

# After
    return (Linear(tensors={"weight": q_folded.contiguous()},
                   weight_format=q_b.weight_format),
            Linear(tensors={"weight": o_folded.contiguous()},
                   weight_format=wo.weight_format))
```

- [ ] **Step 9: Strip `data_format=` from `pad_wo_input` return (`mla.py:62-64`)**

```python
# Before
    return Linear(tensors={"weight": w.contiguous()},
                  weight_format=wo.weight_format,
                  data_format=wo.data_format)

# After
    return Linear(tensors={"weight": w.contiguous()},
                  weight_format=wo.weight_format)
```

- [ ] **Step 10: Strip `data_format=` from `split_qkv` (`deltanet.py:40-42`)**

```python
# Before
        new_linears.append(Linear(tensors=tensors,
                                  weight_format=linear.weight_format,
                                  data_format=linear.data_format))

# After
        new_linears.append(Linear(tensors=tensors,
                                  weight_format=linear.weight_format))
```

- [ ] **Step 11: Strip `data_format=` from `fuse_gdn` return (`deltanet.py:92-93`)**

```python
# Before
    return Linear(tensors=fused_tensors, weight_format=first.weight_format,
                  data_format=first.data_format)

# After
    return Linear(tensors=fused_tensors, weight_format=first.weight_format)
```

- [ ] **Step 12: Strip `data_format=` from `reorder_rotary_emb` (`utils.py:225-226`)**

```python
# Before
        return Linear(tensors=new_tensors, weight_format=x.weight_format,
                      data_format=x.data_format)

# After
        return Linear(tensors=new_tensors, weight_format=x.weight_format)
```

- [ ] **Step 13: Strip `data_format=` from `read_packed_moe_expert` (`utils.py:315-318`)**

```python
# Before
    w1 = Linear(tensors=w1_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)
    w3 = Linear(tensors=w3_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)

# After
    w1 = Linear(tensors=w1_t, weight_format=gate_up.weight_format)
    w3 = Linear(tensors=w3_t, weight_format=gate_up.weight_format)
```

- [ ] **Step 14: Strip `data_format=` from `_build_linear` return (`weight_format.py:492-494`)**

```python
# Before
        return Linear(tensors=tensors,
                      weight_format=fmt,
                      data_format=fmt.make_data_format(self._data_type))

# After
        return Linear(tensors=tensors,
                      weight_format=fmt)
```

- [ ] **Step 15: Verify all modules load**

```bash
python -c "
from lmdeploy.turbomind.deploy.linear import Linear
from lmdeploy.turbomind.deploy.weight_format import TrivialFormat, WeightFormatResolver
from lmdeploy.turbomind.deploy.builder._base import Builder, transform_output_dim, transform_input_dim
from lmdeploy.turbomind.deploy.builder.mla import MLABuilder, fold_kv_b, pad_wo_input
from lmdeploy.turbomind.deploy.builder.deltanet import DeltaNetBuilder, split_qkv, fuse_gdn
from lmdeploy.turbomind.deploy.source_model.utils import reorder_rotary_emb, read_packed_moe_expert
print('All modules loaded')
"
```

Expected: clean import, no errors.

- [ ] **Step 16: Commit**

```bash
git add lmdeploy/turbomind/deploy/linear.py \
        lmdeploy/turbomind/deploy/builder/_base.py \
        lmdeploy/turbomind/deploy/builder/mla.py \
        lmdeploy/turbomind/deploy/source_model/utils.py \
        lmdeploy/turbomind/deploy/builder/deltanet.py \
        lmdeploy/turbomind/deploy/weight_format.py
git commit -m "$(cat <<'EOF'
refactor: remove data_format from Linear dataclass

data_format is now derived at commit time from weight_format + data_type.
Remove the field from the dataclass, strip it from all construction sites,
and simplify concat_out_dim / _ensure_compatible_formats.
EOF
)"
```

---

### Task 3: Update tests

**Files:**
- Modify: `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py:134,280-293`

- [ ] **Step 1: Remove `data_format='placeholder'` from `_make_linear` helper (line 134)**

```python
# Before
    return Linear(tensors=tensors,
                  weight_format='placeholder',
                  data_format='placeholder')

# After
    return Linear(tensors=tensors,
                  weight_format='placeholder')
```

- [ ] **Step 2: Update `test_format_propagation` — only test weight_format (lines 280-293)**

```python
# Before
    def test_format_propagation(self):
        """Output inherits weight_format and data_format from first input."""

        @transform_output_dim
        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        lin = _make_linear(out_dim=4, in_dim=3)
        # Set dummy formats
        object.__setattr__(lin, 'weight_format', 'fake_fmt')
        object.__setattr__(lin, 'data_format', 'fake_data')
        result = identity(lin)
        assert result.weight_format == 'fake_fmt'
        assert result.data_format == 'fake_data'

# After
    def test_format_propagation(self):
        """Output inherits weight_format from first input."""

        @transform_output_dim
        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        lin = _make_linear(out_dim=4, in_dim=3)
        object.__setattr__(lin, 'weight_format', 'fake_fmt')
        result = identity(lin)
        assert result.weight_format == 'fake_fmt'
```

- [ ] **Step 3: Run the transform tensor tests**

```bash
python -m pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Run the weight format resolver tests**

```bash
python -m pytest tests/test_lmdeploy/test_turbomind/test_weight_format_resolver.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Verify a minimal construction**

```bash
python -c "
from lmdeploy.turbomind.deploy.linear import Linear
lin = Linear(tensors={'weight': None}, weight_format='test')
print('Linear() OK:', lin)
"
```

Expected: clean output, no errors.

- [ ] **Step 6: Commit**

```bash
git add tests/test_lmdeploy/test_turbomind/test_transform_tensors.py
git commit -m "$(cat <<'EOF'
test: remove data_format references from transform tensor tests

data_format field no longer exists on Linear. Update fixtures and
test_format_propagation to only verify weight_format.
EOF
)"
```

---

### Task 4: End-to-end model verification

- [ ] **Step 1: Check GPU availability**

```bash
python -c "
from lmdeploy.turbomind.deploy.builder import _tm
# If _turbomind loads, we can test
print('_turbomind loaded')
"
```

- [ ] **Step 2: Run a model test**

Use `scripts/test_turbomind_model.py` to verify a model loads and produces coherent output (at least 128 tokens of meaningful response).

- [ ] **Step 3: Commit any remaining changes if needed**

---

### Summary of Changes

| File | What changed |
|---|---|
| `lmdeploy/turbomind/deploy/linear.py` | Remove `data_format` field; simplify `concat_out_dim` |
| `lmdeploy/turbomind/deploy/builder/_base.py` | 1 behavioral line + drop assertion + strip from 5 call sites + simplify `_ensure_compatible_formats` |
| `lmdeploy/turbomind/deploy/builder/mla.py` | Strip `data_format=` from 2 functions |
| `lmdeploy/turbomind/deploy/source_model/utils.py` | Strip `data_format=` from 2 functions |
| `lmdeploy/turbomind/deploy/builder/deltanet.py` | Strip `data_format=` from 2 functions |
| `lmdeploy/turbomind/deploy/weight_format.py` | Strip `data_format=` from `_build_linear` |
| `tests/.../test_transform_tensors.py` | Remove `data_format` fixture and assertion |

Net: ~1 line changed, ~35 lines deleted across 7 files.
