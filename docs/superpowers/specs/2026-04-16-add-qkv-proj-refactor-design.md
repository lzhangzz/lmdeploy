# add_qkv_proj Refactor Design

## Goal

Decompose `merge_qkv_linear` (75-line monolith) and `add_qkv_proj` (thin wrapper) into a pipeline of named, standalone functions. Each logical step becomes its own function with a clear contract.

## Current State

`AttentionBuilder.add_qkv_proj` calls `merge_qkv_linear` which handles:
- Format compatibility (dequant mixed formats)
- Block-boundary safety checks + dequant
- KV head repetition (GQA)
- Output gate splitting (Qwen3.5)
- RoPE permutation
- TP interleaving + QKV merge

All interleaved in a single loop over tensor kinds.

## New Architecture

### Pipeline

```
Spec:  reorder_rotary_emb_linear(q) + reorder_rotary_emb_linear(k)
       |
Builder add_qkv_proj:
       dequant_mixed(q, k, v)
       -> pad_for_tp(q, k, v)       [pads q, repeats kv to tp-divisible]
       -> split_output_gate(q)       [conditional: Qwen3.5 gate]
       -> fuse_qkv(q, k, v, gate)
       -> _commit_linear('w_qkv')    [existing]
```

RoPE permutation moves from builder to specs (like norms already do). Each remaining step is a standalone function.

### Function Signatures

```python
# --- New helpers (standalone functions in attention.py) ---

def dequant_mixed(q: Linear, k: Linear, v: Linear) -> tuple[Linear, Linear, Linear]:
    """Dequantize to trivial if formats are mixed or block ops are unsafe.

    Handles two cases:
    1. q, k, v have different weight formats -> dequant all to trivial
    2. q, k were already dequantized to trivial by reorder_rotary_emb_linear
       -> dequant v too so all three match for fusion
    """

def pad_for_tp(q: Linear, k: Linear, v: Linear, *,
               tp: int, head_dim: int,
               q_heads: int, kv_heads: int) -> tuple[Linear, Linear, Linear]:
    """Make head counts tp-divisible.

    q: pad with zero heads to reach tp-divisible count.
       e.g. q_heads=30, tp=4 -> pad to 32 (adds 2 zero heads).
    kv: repeat heads to reach tp-divisible count (preserves real data).
       e.g. kv_heads=2, tp=4 -> repeat to 4, GQA group size halves.

    For quantized formats:
    - weight: pad with zeros / repeat_interleave at head granularity
    - scales: pad with 1.0 / repeat_interleave at block granularity
    - zeros: pad with 0.0 / repeat_interleave at block granularity
    - Also pads quantization blocks so total blocks are tp-divisible
    """

def split_output_gate(q: Linear, *, head_dim: int) -> tuple[Linear, Linear]:
    """Split output gate from Q projection (Qwen3.5).

    Q's output dim is 2 * head_num * head_dim. Reshape to
    [batch, head_num, 2, head_dim], split into q_real [:,:,0,:] and
    gate [:,:,1,:].
    Returns (q_real, gate).
    """

def fuse_qkv(q: Linear, k: Linear, v: Linear, *,
             tp: int, gate: Linear | None = None) -> Linear:
    """Fuse Q, K, V (and optionally gate) into a single w_qkv Linear.

    Concatenates output channels with TP interleaving:
    Reshape each to [batch, tp, per_shard_out], cat along last dim,
    reshape to [batch, total_out * tp].
    """
```

```python
# --- New helper in source_model/utils.py (or attention.py) ---

def reorder_rotary_emb_linear(linear: Linear, head_dim: int, rope_dim: int) -> Linear:
    """Apply RoPE permutation to all tensors in a Linear.

    Quantization-aware:
    - If quantized and block_out % head_dim != 0, dequantizes first
      (permuting within a head would cross block boundaries).
    - For weight/bias: element-level RoPE permutation.
    - For scales/zeros when block_out % head_dim == 0: block-level channel
      shuffling. Each head maps to (block_out / head_dim) complete blocks,
      so we apply the same interleave pattern at block granularity.
    - For scales/zeros when dequantized: skipped (trivial format has none).
    """
```

### add_qkv_proj (revised)

```python
def add_qkv_proj(self, q, k, v):
    q, k, v = dequant_mixed(q, k, v)
    q, k, v = pad_for_tp(q, k, v, tp=self._tp,
                          head_dim=self.config.head_dim,
                          q_heads=self.config.head_num,
                          kv_heads=self.config.kv_head_num)
    gate = None
    if self.config.attn_output_gate:
        q, gate = split_output_gate(q, head_dim=self.config.head_dim)
    merged = fuse_qkv(q, k, v, tp=self._tp, gate=gate)
    self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                        model_dtype=self.config.data_type)
```

### Spec Changes

Each spec adds `reorder_rotary_emb_linear` calls on q and k before `add_qkv_proj`.

Example (`qwen3_spec.py`):
```python
q, k, v, o = [self._linear(f'{pfx}.{x}_proj') for x in 'qkvo']
q = reorder_rotary_emb_linear(q, mc.size_per_head, self._rope_dim)
k = reorder_rotary_emb_linear(k, mc.size_per_head, self._rope_dim)
attn.add_qkv_proj(q, k, v)
```

Affected specs: `qwen3_spec.py`, `qwen3_5_spec.py`, `gpt_oss_spec.py`.

## Code to Delete

| What | File | Why |
|------|------|-----|
| `merge_qkv_linear` | `attention.py` | Replaced by pipeline |
| `_merge_qkv` | `attention.py` | Absorbed into `fuse_qkv` |
| `_merge_qkvg` | `attention.py` | Absorbed into `fuse_qkv` |
| `_reorder_rotary_emb` | `attention.py` | Replaced by spec calling `reorder_rotary_emb_linear` |
| `_block_ops_need_dequant` | `_base.py` | Absorbed into `dequant_mixed` and `reorder_rotary_emb_linear` |
| `permute_qk` parameter | `merge_qkv_linear`, `AttentionConfig` | RoPE moved to specs |
| `repeat_kv` parameter | `merge_qkv_linear`, `AttentionConfig` | Absorbed into `pad_for_tp` |

## Quantization-Aware Interactions

The pipeline has two dequant triggers:

1. **`reorder_rotary_emb_linear`** (in spec): Dequantizes q/k if `block_out % head_dim != 0`.
2. **`dequant_mixed`** (in builder): Detects mixed formats (e.g., q/k trivial from step 1, v still quantized) and dequantizes remaining linears.

This is correct: after both steps, all three linears are guaranteed to be in the same format.

## Block-Level Scale Permutation (detail)

When `block_out % head_dim == 0`, scales/zeros can be shuffled at block granularity:

```
# Each head = (block_out / head_dim) blocks
blocks_per_head = block_out // head_dim
n_heads = tensor.size(-1) // blocks_per_head

# Reshape to [batch, n_heads, blocks_per_head]
t = tensor.view(*tensor.shape[:-1], n_heads, blocks_per_head)

# Apply same interleave pattern at block scale
t = reorder_rotary_emb(t, blocks_per_head, rope_dim * blocks_per_head // head_dim)
new_tensors[kind] = t.reshape(tensor.shape)
```
