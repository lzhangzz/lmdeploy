# Builder-Driven Model Loading Design

Date: 2026-04-15
Status: Draft

## Problem

The current model loading pipeline in `TextModelLoader` uses hardcoded `_process_*` methods that know about every module type (attention, FFN, MoE, linear attention). Adding a new module type or changing how a module is loaded requires modifying `TextModelLoader`, which violates the separation between "what the model has" (Spec) and "how to load it into TurboMind" (Builder).

The `_process_*` methods mix three concerns: reading weights from the spec, transforming them (fusion, padding), and committing them to C++. TP split rules live in external tables (`_ATTN_TP_RULES`, `_FFN_TP_RULES`) disconnected from the modules they serve.

## Design

Introduce a **Builder** pattern where:

- **Builders** (AttentionBuilder, FfnBuilder, MoeBuilder, etc.) encapsulate the weight transforms + TP sharding + C++ commit for each module type. Builders create their own C++ module handles independently and are bound to parents later.
- **Specs** drive the loading flow by creating builders directly and calling their methods. Each model architecture implements `model()` which builds the full hierarchy in one call.
- **All weights** are loaded upfront via safetensors memory mapping (no real memory cost), enabling the spec to build everything in a single pass.

### Builder Base Class

The `Builder` base class absorbs `Distributor`. It holds GPU handles, creates C++ modules, and auto-binds children via `__setattr__`.

```python
class Builder:
    """Base class for all module builders."""

    def __init__(self, config, contexts):
        self.config = config
        self._contexts = contexts
        self._tp = 1
        self._ranks = None
        # Create C++ module instances on each GPU via registry
        self.handles = []
        for ctx in self._contexts:
            with ctx:
                handle = _tm.create_module(config.to_cpp())
                self.handles.append(handle)

    def __setattr__(self, name, value):
        if isinstance(value, Builder):
            # Bind: parent C++ module takes ownership of child.
            # C++ side finds the x-macro registered field,
            # assigns the unique_ptr, sets parent pointer.
            for parent_h, child_h in zip(self.handles, value.handles):
                parent_h.add_child(name, child_h)
            return
        super().__setattr__(name, value)
```

**Handle creation mechanism**:
- Builders create C++ modules independently in `__init__` via `_tm.create_module(config)`. No parent reference needed.
- When a builder is assigned to a parent's attribute (e.g., `d.attention = attn_builder`), the parent's `__setattr__` calls `parent_handle.add_child(name, child_handle)` for each GPU. The C++ side finds the x-macro registered field, assigns the unique_ptr, and sets the parent pointer.
- GPU contexts are passed from the Spec to builder constructors.

**`TextModelBuilder`** (root builder) is a special case - it wraps pre-existing C++ `ModelWeight` handles from the runtime rather than creating new modules:

```python
class TextModelBuilder(Builder):
    """Wraps the existing C++ root ModelWeight."""
    def __init__(self, handles, contexts):
        self.handles = handles  # pre-existing
        self._contexts = contexts
        self._tp = 1
        self._ranks = None
```

### TP Split Knowledge

Builders own TP split knowledge. The `SplitSide` enum becomes internal to the builder layer. Specs never reference it.

- `AttentionBuilder.add_qkv_proj()` knows it splits along OUTPUT
- `AttentionBuilder.add_o_proj()` knows it splits along INPUT
- `FfnBuilder.add_ffn()` knows w1/w3 = OUTPUT, w2 = INPUT
- Direct parameters have per-name TP rules built into the builder

The spec interface simplifies:
```python
# Before: spec provides split_side
def attn_params(self, layer) -> dict[str, tuple[Tensor, SplitSide | None]]:

# After: spec just provides tensors
def attn_params(self, layer) -> dict[str, Tensor]:
```

### Concrete Builders

#### AttentionBuilder

Encapsulates attention weight loading. Absorbs `_process_attention` logic and `_ATTN_TP_RULES`.

```python
class AttentionBuilder(Builder):
    def __init__(self, config, contexts, tp=1, ranks=None):
        super().__init__(config, contexts)
        self._tp = tp
        self._ranks = ranks

    def add_qkv_proj(self, q, k, v):
        """Fuse QKV via merge_qkv_linear, shard along output dim, commit."""
        merged = merge_qkv_linear(q, k, v, tp=self._tp,
                                  head_dim=self.config.head_dim,
                                  rope_dim=self.config.rope_dim,
                                  permute_qk=self.config.permute_qk,
                                  attn_output_gate=self.config.output_gate,
                                  repeat_kv=self.config.repeat_kv,
                                  kv_head_num=self.config.kv_head_num)
        self._commit_linear('w_qkv', merged, SplitSide.OUTPUT)

    def add_o_proj(self, o):
        """Shard along input dim, commit."""
        self._commit_linear('wo', o, SplitSide.INPUT)

    def add_qk_norm(self, q, k):
        """Create NormConfig children for q_norm, k_norm, commit tensors."""
        self._add_norm_child('q_norm', q)
        self._add_norm_child('k_norm', k)

    def add_param(self, name, tensor):
        """Commit a direct parameter. Builder determines split side."""
        split_side = self._PARAM_TP_RULES.get(name)
        self._commit_tensor(name, tensor, split_side)
```

#### FfnBuilder

Encapsulates FFN weight loading with w1+w3 fusion. Absorbs `_process_ffn` logic and `_FFN_TP_RULES`.

```python
class FfnBuilder(Builder):
    def add_ffn(self, w1, w2, w3):
        """Fuse w1+w3 if possible, shard, commit.

        Fuse_silu is determined by weight format inspection (via
        _should_fuse_silu). The C++ config is updated after fusion
        is determined, before prepare() runs.
        """
        fused, fused_silu = None, False
        if w1 is not None and w3 is not None:
            fused, fused_silu = fuse_ffn_linears(
                w1, w3, self._tp, self.config.act_type, is_moe=self.config.is_moe)

        # Update C++ config with actual fuse_silu
        for handle in self.handles:
            handle.set_config_field('fuse_silu', fused_silu)

        if fused is not None:
            self._commit_linear('w1w3', fused, SplitSide.OUTPUT)
        else:
            if w1: self._commit_linear('w1', w1, SplitSide.OUTPUT)
            if w3: self._commit_linear('w3', w3, SplitSide.OUTPUT)
        if w2:
            self._commit_linear('w2', w2, SplitSide.INPUT)
```

Config dependency: `fuse_silu` is only known after weight format inspection inside `add_ffn`. The C++ module is created with a preliminary config, then updated. This works because `fuse_silu` is only consumed during `prepare()` which runs after all weights are committed.

#### MoeBuilder

Encapsulates MoE weight loading. The spec drives expert iteration.

```python
class MoeBuilder(Builder):
    def add_gate(self, name, linear, model_dtype=None):
        """Commit a gate linear."""
        self._commit_linear(name, linear, split_side=None, model_dtype=model_dtype)

    def add_param(self, name, tensor, split_side=None):
        """Commit a non-expert MoE parameter."""
        self._commit_tensor(name, tensor, split_side)
```

The spec creates a `ModuleListBuilder` for experts and assigns it to `m.experts`:

```python
# In spec:
def moe(self, pfx, layer):
    m = MoeBuilder(MoeConfig.from_model_config(self.config, layer_id=layer, ...),
                   contexts=self._contexts, tp=self._tp)
    for name, lin in self.moe_gate(layer).items():
        m.add_gate(name, lin, model_dtype=self._dtype)
    for name, tensor in self.moe_params(layer).items():
        m.add_param(name, tensor)
    experts = ModuleListBuilder(ModuleListConfig(), contexts=self._contexts)
    for i in range(self.num_experts(layer)):
        expert = self.ffn(f'{pfx}.{i}', layer)
        experts[str(i)] = expert
    m.experts = experts
    return m
```

#### NormBuilder

Lightweight builder for norm weights.

```python
class NormBuilder(Builder):
    def set_weight(self, tensor):
        self._commit_tensor('weight', tensor)
```

#### ModuleListBuilder

Indexed children via `__setitem__`.

```python
class ModuleListBuilder(Builder):
    def __setitem__(self, index, builder):
        for parent_h, child_h in zip(self.handles, builder.handles):
            parent_h.add_child(str(index), child_h)
```

### Weight Loading

All weights are loaded upfront into one dict. Safetensors uses memory mapping, so the actual data is only read when tensors are accessed (during builder commit). This enables a single `model()` call.

The `readers()` flow changes:

```
Before:
  readers() -> [(layer=-1, params), (layer=0, params), (layer=1, params), ...]
  For each: create spec -> TextModelLoader(layer, spec) -> _process_*

After:
  readers() -> all_params (one mmap-backed dict)
  Create spec once -> spec.model() -> specs create builders -> builders commit
```

Changes to `loader.py`:
- `SafetensorsLoader` returns all params in one batch
- `PytorchLoader` similarly returns all params at once

Changes to `source_model/base.py`:
- `readers()` yields a single `(layer=ALL, params=all_params)` entry
- The spec is created once with all weights

Changes to `target_model/base.py`:
- `export()` creates one spec and calls `spec.model()` instead of iterating

### Spec Integration

The `TextModelSpec` base class:

```python
class TextModelSpec(ABC):
    params: dict[str, torch.Tensor]  # ALL weights, mmap-backed
    _contexts: list  # GPU contexts, injected by loader
    _root_handles: list  # Root C++ ModelWeight handles, injected by loader
    _config: ModelConfig  # resolved model config

    # Weight reading helpers (same as current)
    def _get(self, key) -> torch.Tensor | None: ...
    def _linear(self, prefix) -> Linear | None: ...
    def _read_ffn_linears(self, pfx) -> dict[str, Linear]: ...

    # Attention config fields (set once in model(), used by attn() and sub-methods)
    _tp: int = 1
    _head_dim: int = 0
    _rope_dim: int = 0
    _permute_qk: bool = True
    _attn_output_gate: bool = False
    _repeat_kv: int = 0
    _kv_head_num: int = 0

    @abstractmethod
    def model(self):
        """Build the full model hierarchy. Called once by the loader."""

    @abstractmethod
    def model_info(self) -> dict: ...

    def num_experts(self, layer) -> int:
        return 0
```

A model-specific spec (e.g., Qwen3):

```python
class Qwen3Spec(TextModelSpec):
    def _configure(self):
        """Set attention config fields from model config. Called at start of model()."""
        mc = self._config
        self._tp = ...  # from TP config
        self._head_dim = mc.size_per_head
        self._rope_dim = mc.rope_param.dim if mc.rope_param else mc.size_per_head
        self._permute_qk = True
        self._attn_output_gate = mc.attn_output_gate
        self._repeat_kv = ...  # from repeat_kv config
        self._kv_head_num = mc.kv_head_num

    def token_embeds(self):
        emb = self._get('model.embed_tokens.weight')
        if emb is None:
            return None
        mc = self._config
        tp = self._tp * self._attn_cp_size
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
        cfg = LinearConfig(input_dim=padded_vocab, output_dim=mc.hidden_units // tp,
                           data_type=...)
        m = LinearBuilder(cfg, contexts=self._contexts, tp=tp)
        m.set_weight(emb_padded, split_side=SplitSide.OUTPUT)
        return m

    def lm_head(self):
        output = self._get('lm_head.weight')
        if output is None:
            return None
        mc = self._config
        tp = self._tp * self._attn_cp_size
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        output_padded = pad_out_dim(output, padded_vocab, dim=0)
        output_t = output_padded.t()
        cfg = LinearConfig(input_dim=mc.hidden_units, output_dim=padded_vocab // tp,
                           data_type=...)
        m = LinearBuilder(cfg, contexts=self._contexts, tp=tp)
        m.set_weight(output_t, split_side=SplitSide.OUTPUT)
        return m

    def norm(self, pfx):
        tensor = self._get(f'{pfx}.weight')
        if tensor is None:
            return None
        m = NormBuilder(NormConfig(dim=tensor.shape[-1], ...), contexts=self._contexts)
        m.set_weight(tensor)
        return m

    def attn(self, pfx, layer):
        m = AttentionBuilder(AttentionConfig.from_model_config(
                                 self.config, tp_size=self._tp, ...),
                             contexts=self._contexts, tp=self._tp)
        q, k, v, o = [self._linear(f'{pfx}.{x}_proj') for x in 'qkvo']
        q, k = reorder_rotary_emb(q, k, m.config.head_dim, m.config.rope_dim)
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(o)
        q, k = [self._get(f'{pfx}.{x}_norm.weight') for x in 'qk']
        q, k = self._permute_qk_tensors(q, k)
        m.add_qk_norm(q, k)
        for name, tensor in self.attn_params(layer).items():
            m.add_param(name, tensor)
        return m

    def ffn(self, pfx, layer=None):
        linears = self._read_ffn_linears(pfx)
        cfg = FfnConfig.from_model_config(self.config, tp_size=self._tp,
                                          inter_size=self._inter_size(layer), ...)
        m = FfnBuilder(cfg, contexts=self._contexts, tp=self._tp)
        m.add_ffn(linears.get('w1'), linears.get('w2'), linears.get('w3'))
        return m

    def moe(self, pfx):
        m = MoeBuilder(MoeConfig(...), contexts=self._contexts, tp=self._tp)
        for name, lin in self.moe_gate(...).items():
            m.add_gate(name, lin)
        experts = ModuleListBuilder(ModuleListConfig(), contexts=self._contexts)
        for i in range(self.num_experts):
            linears = self.moe_ffn_linears(layer, i)
            expert = FfnBuilder(FfnConfig(...), contexts=self._contexts, tp=self._tp)
            expert.add_ffn(linears.get('w1'), linears.get('w2'), linears.get('w3'))
            experts[str(i)] = expert
        m.experts = experts
        return m

    def layers(self, pfx):
        m = ModuleListBuilder(ModuleListConfig(), contexts=self._contexts)
        for i in range(self.num_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), contexts=self._contexts)
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm')
            d.attention = self.attn(f'{pfx}.{i}.self_attn', layer=i)
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', layer=i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', layer=i)
            m[str(i)] = d
        return m

    def model(self):
        self._configure()
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds()
        root.norm = self.norm('model.norm')
        root.output = self.lm_head()
        root.layers = self.layers('model.layers')
```

### TextModelLoader Simplification

`TextModelLoader` shrinks from 336 lines to ~20 lines:

```python
class TextModelLoader:
    def __init__(self, model):
        handles = [model.root(gpu) for gpu in range(model.gpu_count)
                   if model.root(gpu) is not None]
        contexts = [model.context(gpu) for gpu in range(model.gpu_count)]
        self._root_handles = handles
        self._contexts = contexts

    def __call__(self, spec):
        spec._contexts = self._contexts
        spec._root_handles = self._root_handles
        spec.model()
```

All `_process_*` methods are eliminated. The loader no longer knows about module types or TP rules.

## C++ Changes

| Change | Description |
|--------|-------------|
| `create_module(config)` | New free function. Extracts module creation from `Module::create_child`. Returns `unique_ptr<Module>` from registry. Bound as `_tm.create_module(config)`. |
| `add_child(name, child)` | Expose existing `Module::add_child` to Python. Takes a module handle, finds x-macro field, assigns unique_ptr, sets parent. |
| No changes to x-macro system, registry, or config structs | The existing infrastructure is sufficient. |

The split from `create_child` into `create_module` + `add_child` enables the builder pattern where module creation and parent binding are separate steps.

## File Changes

### Files deleted

| File | Reason |
|------|--------|
| `distributor.py` | Absorbed into `builder.py` |

### Files added

| File | Contents |
|------|----------|
| `builder.py` | `Builder` base class, `TextModelBuilder`, `DecoderLayerBuilder`, `AttentionBuilder`, `FfnBuilder`, `MoeBuilder`, `NormBuilder`, `ModuleListBuilder`. Absorbs `commit_linear`/`commit_tensor`/`_commit_tensors` from `commit.py` and `fuse_ffn_linears` from `transforms.py`. |

### Files modified

| File | Change |
|------|--------|
| `text_model_loader.py` | Gutted to ~20 lines. Only injects handles/contexts into spec and calls `spec.model()`. |
| `commit.py` | Core commit logic (`_commit_tensors`, `commit_linear`, `commit_tensor`) moves into `Builder._commit_*` methods. File may be deleted or kept as utility. |
| `transforms.py` | `fuse_ffn_linears` and helpers move into `FfnBuilder`. File deleted if empty. |
| `spec.py` | `TextModelSpec` gains `model()`. Methods like `attn_linears()`, `ffn_linears()` are replaced by builder-returning methods (`attn()`, `ffn()`). `SplitSide` moves to builder layer. |
| `source_model/qwen3_spec.py` | Implements `model()`, `attn()`, `ffn()`, `moe()`, `layers()`. |
| `source_model/gpt_oss_spec.py` | Same pattern. |
| `source_model/qwen3_5_spec.py` | Same pattern (including linear attention builder). |
| `source_model/glm4_moe_lite_spec.py` | Same pattern (including MLA-specific builder usage). |
| `source_model/base.py` | `readers()` returns all weights in one batch (mmap-backed). |
| `loader.py` | Returns all params in one batch instead of per-layer. |
| `target_model/base.py` | `export()` creates spec once and calls `spec.model()`. |
| `src/turbomind/python/bind.cpp` | Add `create_module(config)` and `add_child(name, module)` bindings. |
| `src/turbomind/core/module.h` | Add `static create()` factory method. |

## Verification

1. All 13 existing test models load correctly and produce valid outputs
2. Both TP=1 and TP=2 configurations work
3. No regressions in quantized format handling (AWQ, GPTQ, FP8, MXFP4, compressed-tensors)
4. Memory usage during loading is comparable to current (mmap should make this equivalent)
5. Loading time is comparable to current
