# TP Loading Refactor Design

Date: 2026-04-08
Scope: `text_model_loader.py`, `load_context.py`
Goal: Eliminate redundant per-rank spec reads by restructuring `_load_layer` into component-major processing with a GPU-transparent writer abstraction.

## Problem Statement

In `TextModelLoader._load_layer`, the outer loop is GPU-major:

```python
for gpu in range(gpu_count):
    attn_rank, mlp_rank = self.model.tp_ranks(gpu)
    self._load_attention(handle, spec, layer, mc, dtype, attn_rank)
    self._load_ffn(handle, spec, layer, mc, dtype, mlp_rank)
    ...
```

Each `_load_*` method calls spec methods like `spec.attn_linears(layer)` internally. With TP=N, the same weight is read from checkpoint and merged/transformed N times, but only 1/N is committed each time.

## Design Decisions

1. **Component-major structure** -- Invert the loop: for each component, read once, then distribute to all GPUs. The GPU loop moves inside a writer abstraction.
2. **`LayerWriter` abstraction** -- Wraps all GPU handles for one logical layer. Operations (`create_child`, `commit_linear`, `commit_tensor`) iterate over GPUs internally. The `_process_*` methods see no GPU loop.
3. **Bound `(tp, ranks)` at `create_child`** -- `create_child` binds `tp` (TP size) and `ranks` (per-GPU rank list). Child writers inherit bound values. `commit_linear`/`commit_tensor` use them when `split_side` is given. No string-based dispatch.
4. **GPU-0 processing** -- Weights move to GPU 0 once, transforms/fuses happen there, shards are sent to target GPUs via the C++ `copy_from` cross-device path.
5. **Rename `_load_*` to `_process_*`** -- The methods no longer read from spec (reads are hoisted). They transform, shard, and commit. `_process_*` reflects this.
6. **Per-expert MoE iteration** -- MoE reads and distributes one expert at a time within `_process_moe`, keeping GPU 0 memory bounded.
7. **Single `create_child` path** -- Add trivial typed configs for structural modules (`ModuleListConfig`, `NormConfig`, `DecoderLayerConfig`) so all module creation goes through `create_child(name, config)`. No `create_child_raw` / dict-based path on the writer.

## `LayerWriter` API

```python
class LayerWriter:
    """Wraps all GPU handles for one logical layer.

    The GPU loop is internal.  Outside callers see single-layer semantics.
    """

    def __init__(self, handles, tp=1, ranks=None):
        """
        Args:
            handles: [c++ module handle, ...]  One per GPU.
            tp:    Bound TP size (inherited by children).
            ranks: Bound per-GPU rank list [rank_gpu0, rank_gpu1, ...].
                   None means broadcast (rank=0 for all GPUs).
        """
        self._handles = handles
        self._tp = tp
        self._ranks = ranks

    @property
    def tp_size(self):
        return self._tp

    def _rank_for(self, gpu_idx):
        if self._ranks and self._tp > 1:
            return self._ranks[gpu_idx]
        return 0

    # -- Module creation --------------------------------------------------

    def create_child(self, name, config, tp=None, ranks=None):
        """Create a typed module child on ALL GPUs.

        Calls ``config.for_rank(rank).to_cpp()`` per GPU.
        Returns a new LayerWriter scoped to the created children,
        with tp/ranks rebound if provided (otherwise inherited).
        """
        new_tp = tp if tp is not None else self._tp
        new_ranks = ranks if ranks is not None else self._ranks
        children = []
        for i, handle in enumerate(self._handles):
            rank = new_ranks[i] if new_ranks and new_tp > 1 else 0
            child = handle.create_child(name, config.for_rank(rank).to_cpp())
            children.append(child)
        return LayerWriter(children, tp=new_tp, ranks=new_ranks)

    # -- Weight commit ----------------------------------------------------

    def commit_linear(self, name, linear, split_side=None, model_dtype=None):
        """Commit a Linear bundle to all GPUs.

        If split_side is given, uses bound tp/ranks for sharding.
        If split_side is None, broadcasts (tp=1).
        """
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            rank = self._rank_for(i) if tp > 1 else 0
            commit_linear(handle, linear, name,
                          split_side=split_side, split_num=tp,
                          rank=rank, model_dtype=model_dtype)

    def commit_tensor(self, name, tensor, split_side=None):
        """Commit a raw tensor to all GPUs.

        Same split_side semantics as commit_linear.
        """
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            rank = self._rank_for(i) if tp > 1 else 0
            commit_tensor(handle, tensor, name,
                          split_side=split_side, split_num=tp,
                          rank=rank)
```

## Trivial typed configs for structural modules

Structural modules (ModuleList, NormWeight, DecoderLayerWeight) get trivial typed configs so all creation goes through the single `create_child(name, config)` path. Each config implements `for_rank(rank)` (returns `self` — no rank-dependent fields) and `to_cpp()`.

```python
@dataclass
class ModuleListConfig:
    """Config for ModuleList (pure container, no parameters)."""
    def for_rank(self, rank): return self
    def to_cpp(self):
        return _tm.ModuleListConfig()

@dataclass
class NormConfig:
    """Config for NormWeight."""
    dim: int = 0
    data_type: int = 0
    def for_rank(self, rank): return self
    def to_cpp(self):
        cfg = _tm.NormConfig()
        cfg.dim = self.dim
        cfg.data_type = self.data_type
        return cfg

@dataclass
class DecoderLayerConfig:
    """Config for DecoderLayerWeight (pure container)."""
    def for_rank(self, rank): return self
    def to_cpp(self):
        return _tm.DecoderLayerConfig()
```

C++ side: add corresponding empty/trivial config structs in `module_config.h`, config-based constructors, and pybind11 bindings. The constructors just delegate to the existing default/dict-based constructors.

## Restructured `_load_layer`

```python
def _load_layer(self, layer, spec):
    mc = self.model.model_config
    spec.configure(SpecAttnConfig(
        tp=self.attn_tp,
        permute_qk=getattr(self.model, 'permute_qk', True),
        repeat_kv=getattr(self.model, 'repeat_kv', 0),
        head_dim=mc.size_per_head,
        rope_dim=...,
        output_gate=getattr(mc, 'attn_output_gate', False),
        kv_head_num=mc.kv_head_num,
    ))

    writer = self._layer_writer(layer)

    self._process_norms(writer, spec, layer)
    self._process_attention(writer, spec, layer)
    self._process_ffn(writer, spec, layer)
    self._process_moe(writer, spec, layer)
    self._process_linear_attn(writer, spec, layer)
    self._process_raw_tensors(writer, spec, layer)
```

### `_layer_writer` factory

```python
def _layer_writer(self, layer):
    """Create a LayerWriter for the given layer across all GPUs."""
    handles = []
    for gpu in range(self.model.gpu_count):
        root = self.model.root(gpu)
        if root is None:
            break
        layers = root.child('layers') or \
            root.create_child('layers', ModuleListConfig().to_cpp())
        layer_mod = layers.child(str(layer)) or \
            layers.create_child(str(layer), DecoderLayerConfig().to_cpp())
        handles.append(layer_mod)
    return LayerWriter(handles)
```

### Precomputed rank lists

```python
# In __init__:
self._attn_ranks = [self.model.tp_ranks(gpu)[0]
                    for gpu in range(self.model.gpu_count)]
self._mlp_ranks  = [self.model.tp_ranks(gpu)[1]
                    for gpu in range(self.model.gpu_count)]
```

## Example: `_process_attention`

```python
def _process_attention(self, writer, spec, layer):
    mc = self.model.model_config
    dtype = _cpp_dtype(mc.data_type)

    attn_linears = spec.attn_linears(layer)      # read once
    if not attn_linears:
        return

    window_size = 0
    ws_list = mc.window_size
    if ws_list and layer < len(ws_list):
        window_size = ws_list[layer]

    attn_cfg = AttentionConfig.from_model_config(
        mc, tp_size=self.attn_tp, tp_rank=0,
        dtype=dtype, window_size=window_size)
    attn = writer.create_child('attention', attn_cfg,
                               tp=self.attn_tp, ranks=self._attn_ranks)

    for name, lin in attn_linears.items():
        rule = _ATTN_TP_RULES.get(name, {})
        attn.commit_linear(name, lin, model_dtype=dtype, **rule)
```

Key points:
- `spec.attn_linears(layer)` called once (not N times).
- `create_child` binds `tp=self.attn_tp, ranks=self._attn_ranks`.
- `commit_linear` uses bound tp/ranks when `split_side` is in `rule`; broadcasts otherwise.

## Example: `_process_moe` (per-expert iteration)

```python
def _process_moe(self, writer, spec, layer):
    if spec.num_experts(layer) <= 0:
        return
    mc = self.model.model_config
    dtype = _cpp_dtype(mc.data_type)

    # Read gate/shared_gate once
    gate_linear = getattr(spec, 'moe_gate_linear', lambda l: None)(layer)
    shared_gate_linear = getattr(spec, 'moe_shared_gate_linear', lambda l: None)(layer)

    moe_cfg = MoeConfig.from_model_config(mc, layer_id=layer, ...)
    moe = writer.create_child('moe_ffn', moe_cfg,
                              tp=self.mlp_tp, ranks=self._mlp_ranks)

    # Gate (broadcast, no split)
    if gate_linear is not None:
        moe.commit_linear('gate', gate_linear, model_dtype=dtype)
    else:
        gate_cfg = LinearConfig(input_dim=..., output_dim=spec.num_experts(layer), ...)
        moe.create_child('gate', gate_cfg)

    # Shared gate
    if shared_gate_linear is not None:
        moe.commit_linear('shared_gate', shared_gate_linear, model_dtype=dtype)
    elif mc.moe_shared_gate:
        shared_gate_cfg = LinearConfig(input_dim=..., output_dim=1, ...)
        moe.create_child('shared_gate', shared_gate_cfg)

    # Experts: one at a time to bound GPU 0 memory
    experts = moe.create_child('experts', ModuleListConfig())
    for e in range(spec.num_experts(layer)):
        expert_cfg = FfnConfig.from_model_config(mc, ...)
        expert = experts.create_child(str(e), expert_cfg)

        expert_linears = spec.moe_ffn_linears(layer, e)  # read one expert
        w1 = expert_linears.get('w1')
        w3 = expert_linears.get('w3')
        w2 = expert_linears.get('w2')
        if w1 is not None and w3 is not None:
            # Fuse and commit (transform on GPU 0, distribute)
            _fuse_and_commit_ffn(expert, w1, w3, w2, ...)
        else:
            for name, lin in expert_linears.items():
                rule = _FFN_TP_RULES.get(name, {})
                expert.commit_linear(name, lin, model_dtype=dtype, **rule)
```

## `commit_ffn` adaptation

`commit_ffn` (aliased as `_fuse_and_commit_ffn`) in `load_context.py` currently takes a single C++ module handle and calls `commit_linear` on it. With the writer pattern, it receives a `LayerWriter` instead. The function signature changes from:

```python
def commit_ffn(ffn_mod, w1, w3, w2, tp, rank, act_type, is_moe, model_dtype):
```

to:

```python
def commit_ffn(writer, w1, w3, w2, act_type, is_moe, model_dtype):
```

The `tp` and `rank` parameters are dropped -- the writer already has them bound. Internally, `commit_ffn` calls `writer.commit_linear(...)` instead of `commit_linear(ffn_mod, ...)`.

## `_commit_tensors` adjustment

In `load_context.py`, `_commit_tensors` currently moves every shard to GPU unconditionally:

```python
shard = shard.cuda().contiguous()
```

Change to handle tensors already on GPU (from GPU-0 processing):

```python
if not shard.is_cuda:
    shard = shard.cuda(0).contiguous()
elif not shard.is_contiguous():
    shard = shard.contiguous()
```

The C++ `copy_from` handles the cross-GPU transfer when the shard is on GPU 0 and the allocated tensor is on the target GPU.

## `_load_global`

`_load_global` reads only 3 tensors from spec (`tok_embeddings`, `norm_weight`, `output_weight`). Apply the same pattern for consistency: read once, use a writer to distribute. Low-priority since the payoff is small.

## Unchanged

- `spec.py`, `transforms.py`, `linear.py` -- no changes
- All source model specs -- no changes

## Files Changed

| File | Change |
|------|--------|
| `text_model_loader.py` | Add `LayerWriter`, restructure `_load_layer`, rename `_load_*` to `_process_*`, precompute rank lists |
| `load_context.py` | Adjust `_commit_tensors` for GPU-resident tensors; adapt `commit_ffn` to accept `LayerWriter` |
| `configs.py` | Add `ModuleListConfig`, `NormConfig`, `DecoderLayerConfig` |
| `src/turbomind/core/module_config.h` | Add trivial C++ config structs for ModuleList, NormWeight, DecoderLayerWeight |
| `src/turbomind/python/bind.cpp` | Bind trivial config structs, add `create_child` overloads |
