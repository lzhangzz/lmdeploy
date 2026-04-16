# add_input_projections Refactor Design

## Goal

Decompose `fuse_gdn_in_proj` (80-line monolith) into a pipeline of named, standalone functions, mirroring the pattern established by the `add_qkv_proj` refactoring.

## Background

After the `add_qkv_proj` refactoring, attention QKV fusion follows a clean pipeline: `dequant_mixed` -> `pad_for_tp` -> `split_output_gate` -> `fuse_qkv`. Each step is a standalone function with a clear contract.

`DeltaNetBuilder.add_input_projections` currently passes all input projections to `fuse_gdn_in_proj`, which handles format compatibility, QKV sub-splitting, and TP interleaving in a single function.

## Changes

### 1. Split QKV early

New function `split_qkv(linear, qkv_split)` splits the combined `in_proj_qkv` linear into separate Q, K, V linears along the output dimension. This eliminates the complex QKV sub-splitting logic inside `fuse_gdn_in_proj`.

```python
def split_qkv(linear: Linear, qkv_split: tuple[int, int, int]) -> tuple[Linear, Linear, Linear]:
    """Split combined QKV linear into Q, K, V linears along output dim."""
```

After splitting, Q, K, V are independent linears with uniform dimensions, so TP interleaving is straightforward — no special per-sub-projection handling needed.

### 2. Format compatibility via existing `_ensure_compatible_formats`

Reuse `_ensure_compatible_formats` from `_base.py` for all 6 linears (Q, K, V, Z, B, A). No new function needed — this is already the correct tool for the job.

### 3. New `fuse_gdn` function

New function `fuse_gdn(q, k, v, z, b, a, *, tp)` replaces the 80-line `fuse_gdn_in_proj`. With QKV already split, the TP interleaving logic is straightforward: reshape each linear to `[batch, tp, per_shard]`, concatenate along the last dim, flatten.

```python
def fuse_gdn(q, k, v, z, b, a, *, tp) -> Linear:
    """Fuse GDN input projections with TP interleaving.

    Layout per tp-shard: [Q | K | V | Z | B | A].
    For tp=1 reduces to simple concat along output dim.
    """
```

`qkv_split` is **required** (not optional). The defensive `None` fallback is removed — if a model claims to have linear attention layers, it must provide the QKV split dimensions. This lets us delete the "naive interleave" code path from `fuse_gdn_in_proj`.

### 4. Remove defensive `qkv_split=None` path

In `qwen3_5_spec.py`, remove the `else: self._linear_qkv_split = None` branch. The type changes from `tuple[int, int, int] | None` to `tuple[int, int, int]`. If the config is missing required fields, the code crashes — which is correct behavior.

In `spec.py`, the base class attribute changes to `_linear_qkv_split: tuple[int, int, int]`.

### 5. `pad_for_tp` skipped

GDN head counts (key_heads=16, value_heads=48/32) are already TP-divisible for TP=2 and TP=4. Skipping for now. If a future model needs it, `pad_for_tp` can be added to the pipeline.

### Revised `add_input_projections`

```python
def add_input_projections(self, *, in_proj_qkv=None, in_proj_z=None,
                          in_proj_b=None, in_proj_a=None, out_proj=None,
                          qkv_split=None):
    q, k, v = split_qkv(in_proj_qkv, qkv_split)
    group = _ensure_compatible_formats(
        {"q": q, "k": k, "v": v, "z": in_proj_z, "b": in_proj_b, "a": in_proj_a})
    fused = fuse_gdn(group["q"], group["k"], group["v"],
                     group["z"], group["b"], group["a"],
                     tp=self._tp)
    self._commit_linear("in_proj_all", fused, SplitSide.OUTPUT,
                        model_dtype=self.config.data_type)
    if out_proj is not None:
        self._commit_linear("out_proj", out_proj, SplitSide.INPUT,
                            model_dtype=self.config.data_type)
```

Spec interface unchanged — same kwargs.

## Code to Delete

| What | File | Why |
|------|------|-----|
| `fuse_gdn_in_proj` | `deltanet.py` | Replaced by pipeline |
| `_GDN_IN_PROJ_KEYS` | `deltanet.py` | Only used by `fuse_gdn_in_proj` |

## Scope

Internal refactoring of `DeltaNetBuilder` pipeline only. No spec interface changes. No behavioral changes. Two new standalone functions (`split_qkv`, `fuse_gdn`) replace one monolith (`fuse_gdn_in_proj`).
