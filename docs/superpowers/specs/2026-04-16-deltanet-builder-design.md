# DeltaNetBuilder Design

## Summary

Add a `DeltaNetBuilder(Builder)` subclass to encapsulate DeltaNet (Gated Delta Net)
linear attention weight loading logic. Currently this logic lives inline in
`qwen3_5_spec.py:linear_attn()` as ~60 lines of raw `_commit_linear` /
`_commit_tensor` / `_add_norm_child` calls with complex fusing, transposition,
and TP reshaping. The new builder gives DeltaNet the same first-class treatment
that `AttentionBuilder`, `FfnBuilder`, and `MoeBuilder` already have.

## Motivation

- **Consistency**: Every other module type has a dedicated Builder subclass with
  convenience methods. DeltaNet uses the generic `Builder` base directly.
- **Encapsulation**: The 60-line `linear_attn()` method mixes checkpoint
  reading with fusing logic, TP rule lookup, conv1d transposition, and TP
  reshaping. The builder absorbs the commit-side complexity.

## API

```python
class DeltaNetBuilder(Builder):
    """DeltaNet (Gated Delta Net) weight loading builder."""

    def add_input_projections(self, *, in_proj_qkv=None, in_proj_z=None,
                              in_proj_b=None, in_proj_a=None, out_proj=None,
                              qkv_split=None):
        """Fuse GDN input projections into in_proj_all, commit all linears.

        Internally calls fuse_gdn_in_proj() to merge qkv/z/b/a into a single
        in_proj_all with TP interleaving. Commits each resulting linear using
        _LINEAR_ATTN_TP_RULES for split side lookup.
        """

    def add_scalar_params(self, a_log=None, dt_bias=None):
        """Commit A_log and dt_bias as OUTPUT-split tensors."""

    def add_conv1d(self, conv1d, qkv_split=None):
        """Transpose HF layout [conv_dim, d_conv] to TM layout [d_conv, conv_dim],
        squeeze leading dim if needed, apply TP Q/K/V interleaving when tp > 1,
        then commit."""

    def add_norm(self, norm_weight, data_type):
        """Add inline norm child."""
```

## What moves into the builder

| Logic | Current location | New location |
|-------|-----------------|--------------|
| `fuse_gdn_in_proj()` call | spec `linear_attn()` | `add_input_projections()` |
| TP rule lookup from `_LINEAR_ATTN_TP_RULES` | spec loop | `add_input_projections()` |
| conv1d: HF->TM transpose (`conv1d.t()`) | spec inline | `add_conv1d()` |
| conv1d: squeeze `[d,1,out]` to `[d,out]` | spec inline | `add_conv1d()` |
| conv1d: TP Q/K/V interleaving | spec inline (~15 lines) | `add_conv1d()` |
| A_log, dt_bias commits | spec loop | `add_scalar_params()` |
| norm child | spec inline | `add_norm()` |

## What stays in the spec

The spec owns **reading weights from checkpoint** — calling `self._get()` and
`self._linear()`. The builder only receives prepared tensors and linears and
handles commit logic. This matches the `AttentionBuilder` pattern: the spec
reads q/k/v/o, the builder fuses and commits.

## Dead code removal

The existing D parameter code (`self._get(f"{pfx}.D")` + `_commit_tensor("D", ...)`)
is dead: no D key exists in Qwen3.5 checkpoints, and the C++ side
(`DeltaNetWeight`, `GatedDeltaNetLayer`) has no D parameter. This code will be
removed from `qwen3_5_spec.py`.

## Simplified spec code

`qwen3_5_spec.py:linear_attn()` shrinks from ~60 lines to ~12:

```python
def linear_attn(self, pfx, layer):
    dn_cfg = DeltaNetConfig.from_model_config(
        self._mc, tp_size=tp, tp_rank=0, dtype=dtype)
    builder = DeltaNetBuilder(dn_cfg, self._contexts,
                              tp=tp, ranks=self._attn_ranks)

    builder.add_input_projections(
        in_proj_qkv=self._linear(f"{pfx}.in_proj_qkv"),
        in_proj_z=self._linear(f"{pfx}.in_proj_z"),
        in_proj_b=self._linear(f"{pfx}.in_proj_b"),
        in_proj_a=self._linear(f"{pfx}.in_proj_a"),
        out_proj=self._linear(f"{pfx}.out_proj"),
        qkv_split=self._linear_qkv_split)
    builder.add_scalar_params(
        a_log=self._get(f"{pfx}.A_log"),
        dt_bias=self._get(f"{pfx}.dt_bias"))
    builder.add_conv1d(
        self._get(f"{pfx}.conv1d.weight"),
        qkv_split=self._linear_qkv_split)
    builder.add_norm(
        self._get(f"{pfx}.norm.weight"), data_type=dtype)
    return builder
```

## `fuse_gdn_in_proj` location

The `fuse_gdn_in_proj` function stays as a module-level helper in `builder.py`
(pure tensor operation, no `self`). `DeltaNetBuilder.add_input_projections()`
calls it internally. The public export remains available.

## No changes needed

- **C++ side**: `DeltaNetBuilder` produces the same C++ module handles via the
  same `Builder` base class mechanisms.
- **Other specs** (`qwen3_spec`, `gpt_oss_spec`, `glm4_moe_lite_spec`): no
  changes.
- **`_LINEAR_ATTN_TP_RULES`**: stays as module-level dict in `builder.py`.
