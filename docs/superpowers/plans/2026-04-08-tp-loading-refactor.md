# TP Loading Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate redundant per-rank spec reads in `TextModelLoader._load_layer` by introducing a `LayerWriter` abstraction that hides the multi-GPU loop, making each `_process_*` method read from spec once and distribute to all GPUs transparently.

**Architecture:** Add `LayerWriter` class wrapping all GPU handles for a layer. Each `_process_*` method owns the full read → transform → commit lifecycle for its component. FFN fusion is refactored to work on full (unsharded) tensors. Trivial typed configs (`ModuleListConfig`, `NormConfig`, `DecoderLayerConfig`) enable a single `create_child` path.

**Tech Stack:** Python 3.10+, C++17, pybind11, PyTorch

**Spec:** `docs/superpowers/specs/2026-04-08-tp-loading-refactor-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/turbomind/core/module_config.h` | Modify | Add `ModuleListConfig`, `NormConfig`, `DecoderLayerConfig` structs |
| `src/turbomind/core/module.h` | Modify | Add `ModuleList(const core::ModuleListConfig&)` constructor |
| `src/turbomind/core/module.cc` | Modify | Implement config constructor, keep registrar |
| `src/turbomind/models/norm_weight.h` | Modify | Add `NormWeight(const core::NormConfig&)` constructor |
| `src/turbomind/models/norm_weight.cc` | Modify | Implement config constructor |
| `src/turbomind/models/decoder_layer_weight.h` | Modify | Add `DecoderLayerWeight(const core::DecoderLayerConfig&)` constructor |
| `src/turbomind/models/decoder_layer_weight.cc` | Modify | Implement config constructor |
| `src/turbomind/python/bind.cpp` | Modify | Bind trivial config structs + create_child overloads |
| `lmdeploy/turbomind/deploy/configs.py` | Modify | Add `ModuleListConfig`, `NormConfig`, `DecoderLayerConfig` |
| `lmdeploy/turbomind/deploy/linear.py` | Modify | Add `fused_count` field to `Linear` dataclass |
| `lmdeploy/turbomind/deploy/transforms.py` | Modify | Refactor `fuse_ffn_linears` to full-tensor |
| `lmdeploy/turbomind/deploy/load_context.py` | Modify | Update `_commit_tensors` for fused sharding + GPU-resident tensors; remove `commit_ffn` |
| `lmdeploy/turbomind/deploy/text_model_loader.py` | Modify | Add `LayerWriter`, rewrite `_load_layer`, convert `_load_*` → `_process_*` |

---

## Task 1: Add C++ trivial config structs

**Files:**
- Modify: `src/turbomind/core/module_config.h` (after line 81)

- [ ] **Step 1: Add structs to module_config.h**

Append before the closing `}  // namespace turbomind::core`:

```cpp
struct ModuleListConfig {};

struct NormConfig {
    int      dim{};
    DataType data_type{};
};

struct DecoderLayerConfig {};
```

- [ ] **Step 2: Build to verify**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds (header-only change, nothing includes it yet beyond existing users)

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/module_config.h
git commit -m "feat(core): add trivial config structs for ModuleList, NormWeight, DecoderLayerWeight"
```

---

## Task 2: Add config-based constructors to structural modules

**Files:**
- Modify: `src/turbomind/core/module.h` (line 245)
- Modify: `src/turbomind/core/module.cc` (after ModuleList implementation)
- Modify: `src/turbomind/models/norm_weight.h` (line 13)
- Modify: `src/turbomind/models/norm_weight.cc`
- Modify: `src/turbomind/models/decoder_layer_weight.h` (line 19)
- Modify: `src/turbomind/models/decoder_layer_weight.cc`

- [ ] **Step 1: Add ModuleList config constructor**

In `src/turbomind/core/module.h`, after the existing `ModuleList() = default;` (line 245), add:

```cpp
    explicit ModuleList(const core::ModuleListConfig&) {}  // empty config, no-op
```

Add include at top of module.h:
```cpp
#include "src/turbomind/core/module_config.h"
```

- [ ] **Step 2: Add NormWeight config constructor**

In `src/turbomind/models/norm_weight.h`, after line 13 (`NormWeight() = default;`), add:

```cpp
    explicit NormWeight(const core::NormConfig& cfg);
```

In `src/turbomind/models/norm_weight.cc`, add implementation (after existing constructors):

```cpp
NormWeight::NormWeight(const core::NormConfig& cfg)
{
    configure(cfg.dim, cfg.data_type);
}
```

Add include at top of norm_weight.h:
```cpp
#include "src/turbomind/core/module_config.h"
```

- [ ] **Step 3: Add DecoderLayerWeight config constructor**

In `src/turbomind/models/decoder_layer_weight.h`, after line 19 (`DecoderLayerWeight() = default;`), add:

```cpp
    explicit DecoderLayerWeight(const core::DecoderLayerConfig&) {}  // no-op
```

Add include at top of decoder_layer_weight.h:
```cpp
#include "src/turbomind/core/module_config.h"
```

- [ ] **Step 4: Build to verify**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/core/module.h src/turbomind/models/norm_weight.h src/turbomind/models/norm_weight.cc src/turbomind/models/decoder_layer_weight.h src/turbomind/models/decoder_layer_weight.cc
git commit -m "feat(models): add config-based constructors for ModuleList, NormWeight, DecoderLayerWeight"
```

---

## Task 3: Bind trivial configs in pybind11

**Files:**
- Modify: `src/turbomind/python/bind.cpp` (after DeltaNetConfig binding ~line 513, and in config-based create_child ~line 688)

- [ ] **Step 1: Add config struct Python bindings**

After the DeltaNetConfig binding (around line 513), add:

```cpp
    py::class_<turbomind::core::ModuleListConfig>(m, "ModuleListConfig")
        .def(py::init<>());

    py::class_<turbomind::core::NormConfig>(m, "NormConfig")
        .def(py::init<>())
        .def_readwrite("dim", &turbomind::core::NormConfig::dim)
        .def_readwrite("data_type", &turbomind::core::NormConfig::data_type);

    py::class_<turbomind::core::DecoderLayerConfig>(m, "DecoderLayerConfig")
        .def(py::init<>());
```

- [ ] **Step 2: Add create_child overloads for new configs**

In the config-based `create_child` lambda (around line 688-737), add three new `try/catch` blocks **before** the final `throw`:

```cpp
            try {
                auto cfg = config_obj.cast<turbomind::core::ModuleListConfig>();
                auto child = std::make_unique<turbomind::core::ModuleList>(cfg);
                auto* raw = child.get();
                m.add_child(name, std::move(child));
                return raw;
            } catch (py::cast_error&) {}

            try {
                auto cfg = config_obj.cast<turbomind::core::NormConfig>();
                auto child = std::make_unique<turbomind::NormWeight>(cfg);
                auto* raw = child.get();
                m.add_child(name, std::move(child));
                return raw;
            } catch (py::cast_error&) {}

            try {
                auto cfg = config_obj.cast<turbomind::core::DecoderLayerConfig>();
                auto child = std::make_unique<turbomind::DecoderLayerWeight>(cfg);
                auto* raw = child.get();
                m.add_child(name, std::move(child));
                return raw;
            } catch (py::cast_error&) {}
```

Also add the necessary includes at the top of bind.cpp if not already present:
```cpp
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
```

- [ ] **Step 3: Build**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Quick smoke test**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "import _turbomind as tm; c = tm.ModuleListConfig(); print('OK')" `
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/python/bind.cpp
git commit -m "feat(bind): expose ModuleListConfig, NormConfig, DecoderLayerConfig"
```

---

## Task 4: Add Python trivial config dataclasses

**Files:**
- Modify: `lmdeploy/turbomind/deploy/configs.py` (append after SpecAttnConfig)

- [ ] **Step 1: Add dataclasses to configs.py**

Append at end of file:

```python

@dataclass
class ModuleListConfig:
    """Config for ModuleList (pure container, no parameters)."""
    def for_rank(self, rank: int) -> ModuleListConfig:
        return self
    def to_cpp(self) -> _tm.ModuleListConfig:
        return _tm.ModuleListConfig()


@dataclass
class NormConfig:
    """Config for NormWeight."""
    dim: int = 0
    data_type: int = 0
    def for_rank(self, rank: int) -> NormConfig:
        return self
    def to_cpp(self) -> _tm.NormConfig:
        cfg = _tm.NormConfig()
        cfg.dim = self.dim
        cfg.data_type = _tm.DataType(self.data_type) if self.data_type else _tm.DataType(0)
        return cfg


@dataclass
class DecoderLayerConfig:
    """Config for DecoderLayerWeight (pure container)."""
    def for_rank(self, rank: int) -> DecoderLayerConfig:
        return self
    def to_cpp(self) -> _tm.DecoderLayerConfig:
        return _tm.DecoderLayerConfig()
```

- [ ] **Step 2: Verify import**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.configs import ModuleListConfig, NormConfig, DecoderLayerConfig; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/configs.py
git commit -m "feat(deploy): add ModuleListConfig, NormConfig, DecoderLayerConfig dataclasses"
```

---

## Task 5: Refactor fuse_ffn_linears to full-tensor

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py` (line 138, Linear dataclass)
- Modify: `lmdeploy/turbomind/deploy/transforms.py` (lines 108-141, fuse_ffn_linears)
- Modify: `lmdeploy/turbomind/deploy/load_context.py` (line 170, _commit_tensors; lines 348-377, commit_ffn)

This task refactors FFN fusion to work on full (unsharded) tensors. The `rank` parameter is dropped. `tp` is kept because `_can_fuse_w1w3` needs it for block-scale alignment checks. A `fused_count` field is added to `Linear` so the commit path can shard chunked layouts correctly.

- [ ] **Step 1: Add `fused_count` field to Linear dataclass**

In `lmdeploy/turbomind/deploy/linear.py`, after line 141 (`data_format` field), add:

```python
    fused_count: int = field(default=1, compare=False, repr=False)
```

- [ ] **Step 2: Set `fused_count=2` in `chunk_linears`**

In `lmdeploy/turbomind/deploy/linear.py`, change the return statement of `chunk_linears` (line 250-252) from:

```python
    return Linear(tensors={k: v.contiguous() for k, v in fused.items()},
                  weight_format=w1.weight_format,
                  data_format=w1.data_format)
```

to:

```python
    return Linear(tensors={k: v.contiguous() for k, v in fused.items()},
                  weight_format=w1.weight_format,
                  data_format=w1.data_format,
                  fused_count=2)
```

Note: `interleave_linears` does NOT need `fused_count` — interleaved layout shards correctly with standard output-dim splitting because alternating elements distribute evenly across ranks.

- [ ] **Step 3: Refactor `fuse_ffn_linears` in transforms.py**

Replace the function (lines 108-141) with:

```python
def fuse_ffn_linears(
    w1: Linear,
    w3: Linear,
    tp: int,
    act_type: str,
    is_moe: bool = False,
) -> tuple[Linear | None, bool]:
    """Optionally fuse w1/w3 on full (unsharded) tensors for FFN.

    Returns (fused_w1w3_or_none, fused_silu).
    When fusion is possible, fused_w1w3 is set.
    When block-scale boundaries prevent fusion, returns (None, fused_silu).

    TP sharding is NOT done here — the caller's commit path handles it
    via split_side=SplitSide.OUTPUT.  ``tp`` is only used for the
    block-scale alignment check in ``_can_fuse_w1w3``.
    """
    fused_silu = _should_fuse_silu(w1, act_type, is_moe)
    can_fuse = _can_fuse_w1w3(w1, tp)

    if can_fuse:
        if fused_silu:
            w1w3 = interleave_linears(w1, w3)
        else:
            w1w3 = chunk_linears(w1, w3)
        return (w1w3, fused_silu)
    else:
        return (None, fused_silu)
```

- [ ] **Step 4: Update `_commit_tensors` for fused_count sharding**

In `lmdeploy/turbomind/deploy/load_context.py`, change `_commit_tensors` (line 138) to accept the linear's `fused_count`:

Replace the function signature (line 138-139):

```python
def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int):
```

with:

```python
def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int,
                    fused_count: int = 1):
```

Then replace the sharding block (lines 164-168):

```python
        if tensor_split_dim is not None and split_num > 1:
            split_size = tensor.shape[tensor_split_dim] // split_num
            shard = tensor.split(split_size, dim=tensor_split_dim)[rank]
        else:
            shard = tensor
```

with:

```python
        if tensor_split_dim is not None and split_num > 1:
            if fused_count > 1 and tensor_split_dim == (tensor.dim() - 1):
                # Chunked fused layout: split each chunk's output dim equally.
                # [*, 2*N] -> reshape [*, 2, N] -> shard N -> [*, 2, N/tp] -> reshape [*, 2*N/tp]
                orig_shape = tensor.shape
                n_chunks = fused_count
                chunk_size = orig_shape[tensor_split_dim] // n_chunks
                new_shape = orig_shape[:tensor_split_dim] + (n_chunks, chunk_size)
                tensor_3d = tensor.reshape(new_shape)
                shard_size = chunk_size // split_num
                shard_3d = tensor_3d[..., rank * shard_size:(rank + 1) * shard_size]
                final_shape = orig_shape[:tensor_split_dim] + (n_chunks * shard_size,)
                shard = shard_3d.reshape(final_shape)
            else:
                split_size = tensor.shape[tensor_split_dim] // split_num
                shard = tensor.split(split_size, dim=tensor_split_dim)[rank]
        else:
            shard = tensor
```

- [ ] **Step 5: Pass `fused_count` through `commit_linear`**

In `lmdeploy/turbomind/deploy/load_context.py`, update the `_commit_tensors` call inside `commit_linear` (line 276-277):

```python
    _commit_tensors(linear_mod, linear, cpp_dtype, group_size,
                    split_side, split_num, rank,
                    fused_count=linear.fused_count)
```

- [ ] **Step 6: Update `commit_ffn` to work with new fuse_ffn_linears**

Replace `commit_ffn` (lines 348-377) with:

```python
def commit_ffn(ffn_mod, w1: Linear, w3: Linear, w2: Linear | None,
               tp: int, rank: int, act_type: str, is_moe: bool = False,
               model_dtype=None):
    """Preprocess, fuse (interleave or chunk) and commit FFN weights.

    DEPRECATED: Will be removed once text_model_loader uses LayerWriter.
    """
    from .transforms import fuse_ffn_linears

    fused, fused_silu = fuse_ffn_linears(w1, w3, tp, act_type, is_moe)

    if fused is not None:
        commit_linear(ffn_mod, fused, "w1w3",
                           split_side=SplitSide.OUTPUT, split_num=tp,
                           rank=rank, model_dtype=model_dtype)
        ffn_mod.set_fused_silu(fused_silu)
    else:
        commit_linear(ffn_mod, w1, "w1",
                           split_side=SplitSide.OUTPUT, split_num=tp,
                           rank=rank, model_dtype=model_dtype)
        commit_linear(ffn_mod, w3, "w3",
                           split_side=SplitSide.OUTPUT, split_num=tp,
                           rank=rank, model_dtype=model_dtype)

    if w2 is not None:
        commit_linear(ffn_mod, w2, "w2",
                           split_side=SplitSide.INPUT, split_num=tp,
                           rank=rank, model_dtype=model_dtype)
```

- [ ] **Step 7: Build and verify existing tests still pass**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds. Run a model test via the turbomind-tester agent to verify the existing loading path (which still uses the old `_load_layer`) produces correct output with 128+ tokens.

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/linear.py lmdeploy/turbomind/deploy/transforms.py lmdeploy/turbomind/deploy/load_context.py
git commit -m "refactor(transforms): fuse_ffn_linears works on full tensors, add fused_count sharding"
```

---

## Task 6: Add LayerWriter class + precomputed rank lists

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py` (add class, update `__init__`)

- [ ] **Step 1: Add LayerWriter class**

In `text_model_loader.py`, after the imports (line 14) and before `class TextModelLoader` (line 21), add:

```python

class LayerWriter:
    """Wraps all GPU handles for one logical layer.

    The GPU loop is internal.  Outside callers see single-layer semantics.
    """

    def __init__(self, handles, tp=1, ranks=None):
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
        """Commit a raw tensor to all GPUs."""
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            rank = self._rank_for(i) if tp > 1 else 0
            commit_tensor(handle, tensor, name,
                          split_side=split_side, split_num=tp,
                          rank=rank)

```

- [ ] **Step 2: Add precomputed rank lists to `__init__`**

Replace `__init__` (lines 28-31) with:

```python
    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size
        self._attn_ranks = [model.tp_ranks(gpu)[0]
                            for gpu in range(model.gpu_count)]
        self._mlp_ranks = [model.tp_ranks(gpu)[1]
                           for gpu in range(model.gpu_count)]
```

- [ ] **Step 3: Add `_layer_writer` factory method**

Add after `__init__`, before `__call__`:

```python
    def _layer_writer(self, layer: int) -> LayerWriter:
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

- [ ] **Step 4: Update imports**

Replace the imports at the top of the file with:

```python
from .configs import (
    AttentionConfig, FfnConfig, MoeConfig, DeltaNetConfig, LinearConfig,
    SpecAttnConfig, ModuleListConfig, NormConfig, DecoderLayerConfig,
)
from .load_context import (
    LoadContext, _cpp_dtype, _act_type_id,
    commit_linear, commit_tensor,
    _ATTN_TP_RULES, _FFN_TP_RULES, _LINEAR_ATTN_TP_RULES,
)
from .spec import SplitSide
from .transforms import fuse_ffn_linears
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "feat(loader): add LayerWriter class and precomputed rank lists"
```

---

## Task 7: Rewrite _load_layer and convert _load_* to _process_*

This is the core refactoring task. Replace the GPU-major `_load_layer` with component-major `_process_*` methods that each own the full read → transform → commit lifecycle.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

- [ ] **Step 1: Replace `_load_layer` and `_make_tp_config`**

Replace the `_make_tp_config` method (lines 42-55) and `_load_layer` method (lines 259-301) with:

```python
    def _load_layer(self, layer: int, spec: 'TextModelSpec'):
        mc = self.model.model_config
        rope_param = self.model.attention_config.rope_param
        spec.configure(SpecAttnConfig(
            tp=self.attn_tp,
            permute_qk=getattr(self.model, 'permute_qk', True),
            repeat_kv=getattr(self.model, 'repeat_kv', 0),
            head_dim=mc.size_per_head,
            rope_dim=rope_param.dim if rope_param else mc.size_per_head,
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

- [ ] **Step 2: Replace `_load_norms` with `_process_norms`**

Replace `_load_norms` (lines 61-68) with:

```python
    def _process_norms(self, writer: LayerWriter, spec: 'TextModelSpec',
                       layer: int):
        """Read, transform, commit norm weights."""
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)
        hidden = mc.hidden_units

        # --- READ ---
        attn_norm = spec.attn_norm(layer)
        ffn_norm = spec.ffn_norm(layer)

        # --- COMMIT ---
        norm_cfg = NormConfig(dim=hidden, data_type=dtype)
        attention_norm = writer.create_child('attention_norm', norm_cfg)
        ffn_norm_w = writer.create_child('ffn_norm', norm_cfg)
        attention_norm.commit_tensor('weight', attn_norm)
        ffn_norm_w.commit_tensor('weight', ffn_norm)
```

- [ ] **Step 3: Replace `_load_attention` with `_process_attention`**

Replace `_load_attention` (lines 70-91) with:

```python
    def _process_attention(self, writer: LayerWriter, spec: 'TextModelSpec',
                           layer: int):
        """Read, transform, commit attention weights."""
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)

        # --- READ ---
        attn_linears = spec.attn_linears(layer)
        if not attn_linears:
            return

        # --- COMMIT ---
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

- [ ] **Step 4: Replace `_load_ffn` with `_process_ffn`**

Replace `_load_ffn` (lines 93-124) with:

```python
    def _process_ffn(self, writer: LayerWriter, spec: 'TextModelSpec',
                     layer: int):
        """Read, transform (fuse w1+w3), commit FFN weights."""
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)

        # --- READ ---
        ffn_linears = spec.ffn_linears(layer)
        if not ffn_linears:
            return
        w1 = ffn_linears.get('w1')
        w3 = ffn_linears.get('w3')
        w2 = ffn_linears.get('w2')

        # --- TRANSFORM ---
        fused, fused_silu = (None, False)
        if w1 is not None and w3 is not None:
            fused, fused_silu = fuse_ffn_linears(
                w1, w3, self.mlp_tp, mc.activation_type, is_moe=False)

        # --- COMMIT ---
        inter_size = 0
        is_list = mc.inter_size
        if is_list and layer < len(is_list):
            inter_size = is_list[layer]

        ffn_cfg = FfnConfig.from_model_config(
            mc, tp_size=self.mlp_tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=fused_silu, inter_size=inter_size)
        ffn = writer.create_child('feed_forward', ffn_cfg,
                                  tp=self.mlp_tp, ranks=self._mlp_ranks)

        if fused is not None:
            ffn.commit_linear('w1w3', fused,
                              split_side=SplitSide.OUTPUT, model_dtype=dtype)
        else:
            if w1 is not None:
                ffn.commit_linear('w1', w1,
                                  split_side=SplitSide.OUTPUT, model_dtype=dtype)
            if w3 is not None:
                ffn.commit_linear('w3', w3,
                                  split_side=SplitSide.OUTPUT, model_dtype=dtype)

        if w2 is not None:
            ffn.commit_linear('w2', w2,
                              split_side=SplitSide.INPUT, model_dtype=dtype)
```

- [ ] **Step 5: Replace `_load_moe` with `_process_moe`**

Replace `_load_moe` (lines 126-200) with:

```python
    def _process_moe(self, writer: LayerWriter, spec: 'TextModelSpec',
                     layer: int):
        """Read, transform, commit MoE weights (per-expert iteration)."""
        if spec.num_experts(layer) <= 0:
            return
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)
        hidden = mc.hidden_units

        expert_num = 0
        en_list = mc.expert_num
        if en_list and layer < len(en_list):
            expert_num = en_list[layer]

        moe_cfg = MoeConfig.from_model_config(
            mc, layer_id=layer, tp_size=self.mlp_tp, tp_rank=0,
            dtype=dtype, act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
        moe = writer.create_child('moe_ffn', moe_cfg,
                                  tp=self.mlp_tp, ranks=self._mlp_ranks)

        # --- gate (broadcast) ---
        gate_linear = getattr(spec, 'moe_gate_linear', lambda l: None)(layer)
        if gate_linear is not None:
            moe.commit_linear('gate', gate_linear, model_dtype=dtype)
        else:
            gate_cfg = LinearConfig(
                input_dim=hidden,
                output_dim=spec.num_experts(layer),
                data_type=dtype,
                has_bias=getattr(mc, 'expert_router_bias', False))
            moe.create_child('gate', gate_cfg)

        # --- shared_gate (broadcast) ---
        shared_gate_linear = getattr(spec, 'moe_shared_gate_linear', lambda l: None)(layer)
        if shared_gate_linear is not None:
            moe.commit_linear('shared_gate', shared_gate_linear, model_dtype=dtype)
        elif mc.moe_shared_gate:
            shared_gate_cfg = LinearConfig(
                input_dim=hidden, output_dim=1, data_type=dtype, has_bias=False)
            moe.create_child('shared_gate', shared_gate_cfg)

        # --- experts: per-expert READ → TRANSFORM → COMMIT ---
        expert_inter = mc.expert_inter_size or 0
        experts = moe.create_child('experts', ModuleListConfig())
        for e in range(spec.num_experts(layer)):
            expert_cfg = FfnConfig.from_model_config(
                mc, tp_size=self.mlp_tp, tp_rank=0, dtype=dtype,
                act_type=_act_type_id(mc.activation_type),
                fuse_silu=True, inter_size=expert_inter, fused_moe=True)
            expert = experts.create_child(str(e), expert_cfg)

            # READ
            expert_linears = spec.moe_ffn_linears(layer, e)
            w1 = expert_linears.get('w1')
            w3 = expert_linears.get('w3')
            w2 = expert_linears.get('w2')

            # TRANSFORM
            fused, fused_silu = (None, False)
            if w1 is not None and w3 is not None:
                fused, fused_silu = fuse_ffn_linears(
                    w1, w3, self.mlp_tp, mc.activation_type, is_moe=True)

            # COMMIT
            if fused is not None:
                expert.commit_linear('w1w3', fused,
                                     split_side=SplitSide.OUTPUT, model_dtype=dtype)
            else:
                if w1 is not None:
                    expert.commit_linear('w1', w1,
                                         split_side=SplitSide.OUTPUT, model_dtype=dtype)
                if w3 is not None:
                    expert.commit_linear('w3', w3,
                                         split_side=SplitSide.OUTPUT, model_dtype=dtype)

            if w2 is not None:
                expert.commit_linear('w2', w2,
                                     split_side=SplitSide.INPUT, model_dtype=dtype)
```

- [ ] **Step 6: Replace `_load_linear_attn` with `_process_linear_attn`**

Replace `_load_linear_attn` (lines 202-217) with:

```python
    def _process_linear_attn(self, writer: LayerWriter, spec: 'TextModelSpec',
                             layer: int):
        """Read, transform, commit linear-attention (DeltaNet) weights."""
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)

        # --- READ ---
        la_linears = spec.linear_attn_linears(layer)
        if not la_linears:
            return

        # --- COMMIT ---
        dn_cfg = DeltaNetConfig.from_model_config(
            mc, tp_size=self.attn_tp, tp_rank=0, dtype=dtype)
        linear_attn = writer.create_child('linear_attn', dn_cfg,
                                          tp=self.attn_tp, ranks=self._attn_ranks)

        for name, lin in la_linears.items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            linear_attn.commit_linear(name, lin, model_dtype=dtype, **rule)
```

- [ ] **Step 7: Replace `_load_raw_tensors` with `_process_raw_tensors`**

Replace `_load_raw_tensors` (lines 219-253) with:

```python
    def _process_raw_tensors(self, writer: LayerWriter, spec: 'TextModelSpec',
                             layer: int):
        """Read and commit raw per-layer tensors."""
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)

        # --- READ ---
        raw_items = list(spec.raw_layer_tensors(layer))
        if not raw_items:
            return

        # --- COMMIT ---
        for tm_path, tensor, split_side in raw_items:
            parts = tm_path.split('.')
            # Navigate / auto-create intermediate modules on all GPUs
            mod = writer
            for seg in parts[:-1]:
                # Try to navigate to existing child on first GPU handle
                first_child = mod._handles[0].child(seg) if mod._handles else None
                if first_child is not None:
                    # Child exists on first GPU — wrap all existing children
                    children = [h.child(seg) for h in mod._handles]
                    mod = LayerWriter(children, tp=mod._tp, ranks=mod._ranks)
                else:
                    # Auto-create as NormWeight on all GPUs
                    if tensor.dim() > 1:
                        norm_dim = tensor.shape[-1]
                    else:
                        norm_dim = tensor.shape[-1] if tensor.dim() >= 1 else 0
                    cfg = NormConfig(dim=norm_dim, data_type=dtype)
                    mod = mod.create_child(seg, cfg)

            mod.commit_tensor(parts[-1], tensor, split_side=split_side)
```

- [ ] **Step 8: Update `_load_global` to use `_cpp_dtype`**

In `_load_global` (line 303+), replace the LoadContext usage. Change:

```python
            tp_config = self._make_tp_config(attn_rank)
            ctx = LoadContext(root, tp_config, mc)
            dtype = ctx.cpp_dtype
```

to:

```python
            dtype = _cpp_dtype(mc.data_type)
```

(Keep the rest of `_load_global` as-is for now — it already reads spec once per component.)

- [ ] **Step 9: Remove `_make_tp_config` method**

Delete the `_make_tp_config` method entirely (lines 42-55). It's no longer used — `_load_layer` builds `SpecAttnConfig` directly, and `_load_global` uses `_cpp_dtype` directly.

- [ ] **Step 10: Remove old `_load_*` methods**

Delete the old methods (they've been replaced by `_process_*`):
- `_load_norms`
- `_load_attention`
- `_load_ffn`
- `_load_moe`
- `_load_linear_attn`
- `_load_raw_tensors`

- [ ] **Step 11: Clean up unused imports**

Remove unused imports from the import block. `LoadContext` is no longer needed in this file (only used in `_load_global`, which now uses `_cpp_dtype` directly). Remove `_fuse_and_commit_ffn` import.

- [ ] **Step 12: Build and test**

Run a model test via the turbomind-tester agent. The model must respond with meaningful output at 128+ tokens. Test at least one dense model.

- [ ] **Step 13: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): rewrite _load_layer with LayerWriter, convert _load_* to _process_*"
```

---

## Task 8: Clean up load_context.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/load_context.py`

- [ ] **Step 1: Adjust `_commit_tensors` for GPU-resident tensors**

In `_commit_tensors`, replace line 170:

```python
        shard = shard.cuda().contiguous()
```

with:

```python
        if not shard.is_cuda:
            shard = shard.cuda(0).contiguous()
        elif not shard.is_contiguous():
            shard = shard.contiguous()
```

Do the same for `commit_tensor` at line 307:

```python
    shard = shard.cuda().contiguous()
```

→

```python
    if not shard.is_cuda:
        shard = shard.cuda(0).contiguous()
    elif not shard.is_contiguous():
        shard = shard.contiguous()
```

- [ ] **Step 2: Mark `commit_ffn` as deprecated**

Add a deprecation comment to `commit_ffn` (it's still used by the `_load_global` path's potential future refactor, and may have external callers). For now, keep it but mark:

```python
def commit_ffn(ffn_mod, w1: Linear, w3: Linear, w2: Linear | None,
               tp: int, rank: int, act_type: str, is_moe: bool = False,
               model_dtype=None):
    """DEPRECATED: Use LayerWriter + fuse_ffn_linears directly."""
```

- [ ] **Step 3: Build and test**

Run a model test via the turbomind-tester agent to verify the GPU-resident tensor path works correctly.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/load_context.py
git commit -m "refactor(load_context): GPU-resident tensor support in _commit_tensors, deprecate commit_ffn"
```

---

## Task 9: Integration test

- [ ] **Step 1: Test dense BF16/FP16 model (TP=1)**

Use the turbomind-tester agent. Verify 128+ token coherent output.

- [ ] **Step 2: Test quantized model (AWQ or GPTQ, TP=1)**

If available. Verify 128+ token coherent output.

- [ ] **Step 3: Test dense model (TP=2) if multi-GPU available**

This exercises the actual LayerWriter distribution path. Check GPU usage first.

- [ ] **Step 4: Test MoE model if available**

This exercises the per-expert iteration path.

- [ ] **Step 5: Fix any regressions found**

Iterate until all tests produce meaningful output.

---

## Self-Review

**1. Spec coverage check:**
- Component-major structure → Task 7 ✓
- LayerWriter abstraction → Task 6 ✓
- Bound (tp, ranks) at create_child → Task 6 (LayerWriter.create_child) ✓
- GPU-0 processing → Task 8 Step 1 (_commit_tensors adjustment) ✓
- Rename _load_* to _process_* → Task 7 Steps 2-7 ✓
- Per-expert MoE iteration → Task 7 Step 5 ✓
- Single create_child path → Tasks 1-4 (trivial configs) ✓
- Refactor FFN fusion to full-tensor → Task 5 ✓

**2. Placeholder scan:** No TBD/TODO/vague steps found. All code blocks are complete.

**3. Type consistency:**
- `LayerWriter.create_child` returns `LayerWriter` — consistent across all `_process_*` methods
- `fuse_ffn_linears(w1, w3, tp, act_type, is_moe)` → returns `(fused_or_None, fused_silu)` — matches Task 5 Step 3 signature and Task 7 Steps 4/5 call sites
- `commit_linear(handle, linear, name, split_side=..., split_num=..., rank=..., model_dtype=...)` — matches existing function signature in load_context.py
- `NormConfig.for_rank(rank)` returns `self` — used in _process_norms and _process_raw_tensors
- `_cpp_dtype` imported and used consistently (replaces `ctx.cpp_dtype`)

**4. Spec deviation:** `fuse_ffn_linears` keeps the `tp` parameter (spec said drop both tp and rank). This is necessary because `_can_fuse_w1w3(w1, tp)` needs tp for block-scale alignment checks. Only `rank` is dropped (no internal sharding).
