# Module Hierarchy Refactor Design

Date: 2026-04-03

## Problem

The turbomind model loading pipeline has tight coupling between C++ and Python.
Adding a new model architecture requires modifying both C++ `ensure_child` methods
and Python spec classes. The C++ module tree structure is hardcoded in composite
modules like `DecoderLayerWeight`, `AttentionWeight`, and `FfnWeight`.

## Goal

Reduce coupling so that new model architectures can be supported with **Python-only
changes**. The C++ side provides reusable module building blocks; the Python side
determines the module tree structure and drives creation.

## Design Principles

1. **Systematic over ad-hoc** — repeated patterns get a systematic solution
2. **Design the pipeline as a whole** — not item by item
3. **Composable pieces** — reusable across architectures, quantization, etc.
4. **Functional composition** over declarative data structures
5. **Build as we load** — modules created just before weights are committed

## Architecture

### C++ Module Factory System

#### Module Registry

A static registry maps type name strings to factory functions:

```cpp
using ConfigValue = std::variant<int64_t, std::string, double>;
using ModuleConfig = std::map<std::string, ConfigValue>;

class ModuleRegistry {
public:
    using Factory = std::function<std::unique_ptr<Module>(const ModuleConfig&)>;
    static ModuleRegistry& instance();
    void register_type(const std::string& name, Factory factory);
    std::unique_ptr<Module> create(const std::string& type,
                                    const ModuleConfig& config) const;
};
```

Each existing module type self-registers:

```cpp
// In linear_weight.cc
static bool _registered = ModuleRegistry::instance().register_type(
    "LinearWeight",
    [](const ModuleConfig& cfg) -> std::unique_ptr<Module> {
        auto m = std::make_unique<LinearWeight>();
        m->configure(cfg.at("input_dim"), cfg.at("output_dim"),
                     static_cast<DataType>(cfg.at("data_type")),
                     cfg.count("has_bias") && std::get<int64_t>(cfg.at("has_bias")));
        return m;
    });
```

#### create_child on Module

New method that uses the registry to create and attach children:

```cpp
class Module {
public:
    Module* create_child(const std::string& name,
                         const std::string& type_name,
                         const ModuleConfig& config);
};
```

#### Public Child Access

Replace cumbersome `child("name") + static_cast` with a template accessor:

```cpp
class Module {
public:
    template<typename T = Module>
    T* get(const std::string& name) const {
        return static_cast<T*>(child(name));
    }

    const auto& children() const { return children_; }
    Module* operator[](const std::string& name) const { return child(name); }
};
```

Existing typed accessors in subclasses (e.g., `AttentionWeight::w_qkv()`) can be
simplified to use `get<T>()` internally or removed if direct `get<T>()` access
is sufficient for the execution side.

#### Config Value Type

Uses `std::variant<int64_t, std::string, double>` to handle:
- Integer dimensions, flags, enum values
- Floating-point values (e.g., norm epsilon)
- String values (e.g., activation type names)

### Python Side

#### Naming

| Current | New | Purpose |
|---------|-----|---------|
| `ModelWeightSpec` | `TextModelSpec` | Per-architecture checkpoint-to-module mapping |
| `TransformerV2` | `TextModelLoader` | Drives loading for text models |
| `commit_linear_module` | `commit_linear` | Copy Linear bundle to C++ |
| `commit_tensor_module` | `commit_tensor` | Copy raw tensor to C++ |

**Spec method names:**

| Current | New | Notes |
|---------|-----|-------|
| `attn_norm(layer)` | (dropped) | Inline as one-liner in `load_layer` |
| `ffn_norm(layer)` | (dropped) | Inline as one-liner in `load_layer` |
| (new) | `load_attn(ctx, layer)` | Loads attention modules + weights |
| (new) | `load_ffn(ctx, layer)` | Loads dense FFN modules + weights |
| (new) | `load_experts(ctx, layer)` | Loads MoE experts + weights |
| (new) | `load_linear_attn(ctx, layer)` | Loads linear attention (GDN) |
| `load_misc(ctx)` | `load_global(ctx)` | Loads non-layer weights (embeddings, output, final norm) |

#### LoadContext

Wraps a C++ Module handle and provides composable loading primitives:

```python
class LoadContext:
    """Wraps a C++ Module handle. Provides composable loading primitives."""

    def create(self, name: str, module_type: str, **config) -> 'LoadContext':
        """Create a child module via the C++ registry.
        Returns a new LoadContext rooted at the created module."""
        handle = self._handle.create_child(name, module_type, config)
        return LoadContext(handle, self._tp_config)

    def load_linear(self, name: str, linear: Linear, tp_rule: SplitSide | None = None):
        """Create a LinearWeight child and commit weight data."""
        child = self.create(name, "LinearWeight",
                            input_dim=linear.input_dim,
                            output_dim=linear.output_dim, ...)
        commit_linear(child._handle, linear, tp_rule=tp_rule,
                      rank=self._rank, tp=self._tp_size)

    def load_tensor(self, name: str, tensor: Tensor,
                    module_type: str,
                    module_config: dict | None = None,
                    tp_rule: SplitSide | None = None):
        """Create a module child and commit tensor data.

        module_type is required — the caller must specify what C++ module type
        to create (e.g., "NormWeight", "LinearWeight", "TensorParameter").
        """
        config = module_config or {}
        child = self.create(name, module_type, **config)
        commit_tensor(child._handle, tensor, tp_rule=tp_rule,
                      rank=self._rank, tp=self._tp_size)

    def child(self, name: str) -> 'LoadContext':
        """Return a LoadContext for an existing child."""
        return LoadContext(self._handle.get(name), self._tp_config)
```

#### TextModelSpec Evolution

The spec gains `load_layer` and `load_global` methods that use `LoadContext` for
functional composition. The existing weight-mapping methods
(`_read_attn_linears`, `ffn_linears`, etc.) stay for subclasses to implement.
The `attn_norm` and `ffn_norm` methods are dropped — norms are loaded inline
as one-liners in `load_layer`.

```python
class TextModelSpec(ABC):
    # -- Abstract: per-architecture weight reading (existing) --
    def _read_attn_linears(self, layer: int) -> dict[str, Linear]:
        return {}

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        return {}

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        return {}

    def raw_layer_tensors(self, layer: int):
        return []

    # ... existing abstract methods ...

    # -- NEW: Composable loading sub-methods --

    def load_attn(self, ctx: LoadContext, layer: int):
        """Create attention modules and load weights."""
        attn_linears = self.attn_linears(layer)
        if not attn_linears:
            return
        attn = ctx.create("attention", "AttentionWeight",
                          hidden_dim=..., head_dim=..., ...)
        for name, lin in attn_linears.items():
            attn.load_linear(name, lin, tp_rule=_ATTN_TP_RULES.get(name))
        for path, tensor, side in self.raw_layer_tensors(layer):
            if path.startswith("attention."):
                attn.load_tensor(path.split(".")[-1], tensor, side)

    def load_ffn(self, ctx: LoadContext, layer: int):
        """Create dense FFN modules and load weights (with w1/w3 fusion)."""
        ffn_linears = self.ffn_linears(layer)
        if not ffn_linears:
            return
        ffn = ctx.create("feed_forward", "FfnWeight",
                         hidden_dim=..., inter_size=..., ...)
        load_fused_ffn(ffn, ffn_linears, self._tp_config)

    def load_experts(self, ctx: LoadContext, layer: int):
        """Create MoE expert modules and load weights."""
        moe = ctx.create("moe_ffn", "MoeWeight",
                         layer_id=layer, ...)
        for e in range(self.num_experts(layer)):
            expert = moe.create(str(e), "FfnWeight",
                                hidden_dim=..., inter_size=..., ...)
            expert_linears = self.moe_ffn_linears(layer, e)
            load_fused_ffn(expert, expert_linears, self._tp_config)
        # Gate
        gate_tensor = self.moe_gate(layer)
        if gate_tensor is not None:
            moe.load_tensor("gate", gate_tensor, module_type="LinearWeight", ...)

    def load_linear_attn(self, ctx: LoadContext, layer: int):
        """Create linear attention (GDN/DeltaNet) modules and load weights."""
        la_linears = self.linear_attn_linears(layer)
        if not la_linears:
            return
        la = ctx.create("linear_attn", "DeltaNetWeight",
                        hidden_dim=..., ...)
        for name, lin in la_linears.items():
            la.load_linear(name, lin, tp_rule=_LINEAR_ATTN_TP_RULES.get(name))

    def load_global(self, ctx: LoadContext):
        """Load non-layer modules (tok_embeddings, final norm, output head)."""
        # Token embeddings
        emb = self.tok_embeddings()
        if emb is not None:
            ctx.load_tensor("tok_embeddings", emb_padded,
                            module_type="LinearWeight", ...)

        # Final norm
        norm = self.norm_weight()
        if norm is not None:
            ctx.load_tensor("norm", norm,
                            module_type="NormWeight",
                            module_config={"dim": self.hidden_dim, "dtype": self.data_type})

        # Output head
        output = self.output_weight()
        if output is not None:
            ctx.load_tensor("output", output_t,
                            module_type="LinearWeight", ...)

    # -- NEW: Functional loading (default implementation) --

    def load_layer(self, ctx: LoadContext, layer: int):
        """Create modules and load weights for one layer.

        Default implementation composes load_attn, load_ffn, etc.
        Subclasses may override for full control.
        """
        # Configure TP params for merge/fusion (idempotent).
        # Sets _attn_tp, _head_dim, _rope_dim, etc. so that
        # merge_qkv_linear, fuse_gdn_in_proj, etc. work correctly.
        self.configure(
            attn_tp=ctx.tp_size,
            permute_qk=self.permute_qk,
            repeat_kv=ctx.repeat_kv,
            head_dim=ctx.head_dim,
            rope_dim=ctx.rope_dim,
            attn_output_gate=ctx.attn_output_gate,
            kv_head_num=ctx.kv_head_num,
        )

        # Norms (inline one-liners — no separate method needed)
        ctx.load_tensor("attention_norm", self._get(f"model.layers.{layer}.input_layernorm.weight"),
                        module_type="NormWeight",
                        module_config={"dim": self.hidden_dim, "dtype": self.data_type})
        ctx.load_tensor("ffn_norm", self._get(f"model.layers.{layer}.post_attention_layernorm.weight"),
                        module_type="NormWeight",
                        module_config={"dim": self.hidden_dim, "dtype": self.data_type})

        # Composable sub-components
        self.load_attn(ctx, layer)
        if self.num_experts(layer) > 0:
            self.load_experts(ctx, layer)
        else:
            self.load_ffn(ctx, layer)
        self.load_linear_attn(ctx, layer)
```

#### TextModelLoader

Replaces `TransformerV2`. A generic driver with zero hardcoded module paths:

```python
class TextModelLoader:
    """Drives the model loading pipeline for text models."""

    def __init__(self, model: BaseOutputModel):
        self.model = model
        # ... extract config ...

    def __call__(self, layer: int, spec: TextModelSpec):
        if layer < 0:
            self._load_global(spec)
        else:
            self._load_layer(layer, spec)

    def _load_layer(self, layer: int, spec: TextModelSpec):
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            rank = self.model.tp_ranks(gpu)
            # Ensure layers list and layer entry exist
            layers = root.get("layers")
            if layers is None:
                layers = root.create_child("layers", "ModuleList")
            ctx = LoadContext(layers[str(layer)], rank, ...)
            spec.load_layer(ctx, layer)

    def _load_global(self, spec: TextModelSpec):
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            rank = self.model.tp_ranks(gpu)
            ctx = LoadContext(root, rank, ...)
            spec.load_global(ctx)
```

#### Reusable Functional Helpers

Common patterns extracted as composable functions:

```python
def load_fused_ffn(ctx: LoadContext, ffn_linears: dict, tp_config):
    """Fuse w1+w3 (if applicable) and commit FFN weights."""
    w1 = ffn_linears.get("w1")
    w3 = ffn_linears.get("w3")
    w2 = ffn_linears.get("w2")
    if w1 is not None and w3 is not None:
        fused_silu = _should_fuse_silu(w1, tp_config.act_type)
        # TP-shard before fusion (handles block-scale alignment for FP8 etc.)
        w1_shard = _shard_linear_for_tp(w1, tp_config.tp, tp_config.rank)
        w3_shard = _shard_linear_for_tp(w3, tp_config.tp, tp_config.rank)
        if fused_silu:
            fused = interleave_linears(w1_shard, w3_shard)
        else:
            fused = chunk_linears(w1_shard, w3_shard)
        ctx.load_linear("w1w3", fused, tp_rule=OUTPUT)
        # ... handle fused_silu flag ...
    if w2 is not None:
        ctx.load_linear("w2", w2, tp_rule=INPUT)
```

### What Stays the Same

- **C++ module classes** (LinearWeight, NormWeight, FfnWeight, AttentionWeight,
  MoeWeight, DeltaNetWeight, DecoderLayerWeight, ModelWeight) — their internal
  logic and typed accessors remain
- **WeightFormat** and format detection (kind_map.py)
- **Linear** tensor bundle class
- **Python source model architecture** (BaseInputModel, registry, per-architecture
  input models)
- **Loader** (safetensors, pytorch checkpoint loading)
- **TP split rules** (_ATTN_TP_RULES, _FFN_TP_RULES, etc.)

### What Changes

| Component | Change |
|-----------|--------|
| C++ `Module` | Add `create_child(name, type, config)` + registry + `get<T>()` |
| C++ `ensure_child` | Removed from composite modules (creation driven by Python) |
| Python `TransformerV2` | Replaced by `TextModelLoader` |
| Python `ModelWeightSpec` | Renamed to `TextModelSpec`, gains `load_layer`/`load_global` + composable sub-methods |
| Python `LoadContext` | New class — composable loading primitives |
| Python `commit_*_module` | Renamed to `commit_*` (drop "_module" suffix) |
| Python `attn_norm`/`ffn_norm` | Dropped — inlined as one-liners in `load_layer` |

## Migration Path

### Step 1 — C++ Module Registry (independent)

Add the registry, `create_child`, `get<T>()`, and config types to the C++ Module
system. Register all existing module types. This is purely additive — no existing
code breaks.

### Step 2 — Python LoadContext + TextModelLoader (independent)

Create `LoadContext`, `TextModelLoader`, and updated `TextModelSpec` base class
with `load_layer`/`load_global` default implementations. These can coexist with
the existing `TransformerV2` pipeline.

### Step 3 — Migrate One Architecture (integration)

Convert one spec (e.g., Qwen3Spec) to work with the new pipeline. Test end-to-end
with a model load. This validates the full design.

### Step 4 — Migrate Remaining Architectures

Convert remaining specs (GPT-OSS, GLM4-MoE-Lite, Qwen3.5).

### Step 5 — Remove Old Code

Remove `TransformerV2`, `ensure_child` from composite modules, and old
`commit_*_module` functions.

Steps 1 and 2 are independent and can be done in parallel. Step 3 is the critical
integration point.
