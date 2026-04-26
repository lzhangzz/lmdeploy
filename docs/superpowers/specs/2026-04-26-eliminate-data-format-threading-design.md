# Eliminate `data_format` threading from `Linear`

## Problem

`data_format` is a `_tm.DataFormat` C++ value struct that describes quantization parameters
(storage dtype, block sizes, scale/zero dtypes) for the TurboMind runtime. It is:

- **Created** in exactly one place: `WeightFormat.make_data_format(data_type)`
- **Consumed** in exactly one place: `Builder._add_linear()` at `_base.py:429`
- **Threaded** through ~15 intermediate `Linear()` construction sites in between

Every function that constructs a new `Linear` must carry `data_format` forward, even though
the field is fully derivable at the consumption point from `weight_format` + `data_type`,
both of which are already present.

Additionally, `_ensure_compatible_formats` contains identity-check normalization logic
(`_base.py:115-128`) whose sole purpose is to deduplicate `DataFormat` objects that were
independently created by separate `make_data_format` calls — a symptom of the threading.

## Design

**One line changes at the consumption point** (`_base.py:429`):

```python
# Before
lin_cfg.format = linear.data_format

# After
lin_cfg.format = linear.weight_format.make_data_format(compute_dtype)
```

`compute_dtype` (`self.config.data_type`) is the same value that `WeightFormatResolver`
originally passed to `make_data_format`. Same input, same C++ call, same result.

## Deletions

### `linear.py`
- Remove `data_format` field from `Linear` dataclass (line 89)
- Remove `data_format` uniformity check in `concat_out_dim` (lines 100-106 simplified to check only `weight_format`)

### `builder/_base.py`
- Remove `data_format=` from `_dequant_linear` (line 107)
- Remove data_format identity-normalization blocks from `_ensure_compatible_formats` (lines 115-128)
- Remove `data_format=` from `transform_output_dim` wrapper (line 180)
- Remove `data_format=` from `transform_input_dim` wrapper (line 241)
- Remove `data_format is not None` assertion from `_add_linear` (lines 409-411)
- Remove `data_format=` from `add_lm_head` (line 644)

### `builder/mla.py`
- Remove `data_format=` from `fold_kv_b` return (lines 47, 50)
- Remove `data_format=` from `pad_wo_input` return (line 64)

### `source_model/utils.py`
- Remove `data_format=` from `_reorder_rotary_emb` (line 226)
- Remove `data_format=` from `split_gate_up` (lines 316, 318)

### `builder/deltanet.py`
- Remove `data_format=` from two construction sites (lines 42, 93)

### `weight_format.py`
- Remove `data_format=` from `_build_linear` return (line 494)

### Tests
- `test_transform_tensors.py`: remove `data_format='placeholder'` and `data_format='fake_data'` fixtures (lines 134, 290, 293)

## Impact

- ~25 lines removed, 1 changed
- No behavioral change — `make_data_format` receives the same arguments, calls the same
  pure C++ function, produces the same `DataFormat` value
- `weight_format` already threads correctly through all transformations (verified at all
  ~12 construction sites)
