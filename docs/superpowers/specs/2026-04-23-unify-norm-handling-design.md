# Unify Norm Handling Across Spec + Builder

Date: 2026-04-23

## Problem

Norm is a first-class C++ module (`NormConfig` + `NormBuilder`) but its
Python loading code is wired up two different ways depending on where the
norm sits in the model:

- **Top-level / decoder-layer norms** (`root.norm`, `d.attention_norm`,
  `d.ffn_norm`) use the standard module pattern: the spec constructs a
  `NormBuilder`, calls `set_weight`, and binds it via
  `parent.name = norm_builder`. Binding flows through
  `Builder.__setattr__` → `add_child_raw`, identical to every other
  child module.
- **Inline norms inside attention / MLA / DeltaNet** (`q_norm`, `k_norm`,
  `q_a_layernorm`, `kv_a_layernorm`, deltanet `norm`) go through a
  special `Builder._add_norm_child` method on the **base** `Builder`
  class. That method calls `handle.create_child(name, norm_cfg)`
  directly, bypassing `NormBuilder` and the standard binding path.

Four concrete smells:

1. `Builder._add_norm_child` lives on the base class and imports
   `make_norm_config` at call time to dodge a circular import. The base
   class should know nothing about specific module types.
2. Two different code paths commit a norm weight tensor:
   `NormBuilder.set_weight` has its own hand-rolled copy logic, while
   `_add_norm_child` uses `_copy_shard_to_param` directly.
3. Binding happens inconsistently: `__setattr__` → `add_child_raw` for
   standalone norms, `handle.create_child` for inline norms.
4. Parent builders expose norm-specific helpers with inconsistent names
   (`add_qk_norm`, `add_norm`, `add_norms`) and inconsistent signatures.

## Goal

One concept, one code path for every norm in every spec:

> A norm is a `NormBuilder` child, constructed via
> `self.norm(weight)` on the spec and bound via direct attribute
> assignment on the parent.

After the change, the base `Builder` has zero norm knowledge; parent
builders have zero norm-specific methods; and every norm in every spec
is built the same way.

## Design

### Builder side

**Delete**:

- `Builder._add_norm_child` in `lmdeploy/turbomind/deploy/builder/_base.py`
  (and the circular-import comment near the top of the file).
- `AttentionBuilder.add_qk_norm` in `attention.py`.
- `MLABuilder.add_norms` in `mla.py`.
- `DeltaNetBuilder.add_norm` in `deltanet.py`.

**Simplify** `NormBuilder.set_weight` in `norm.py` to reuse the shared
`_commit_tensor` path:

```python
class NormBuilder(Builder):
    def set_weight(self, tensor: torch.Tensor):
        self._commit_tensor('weight', tensor)
```

The existing inline implementation (GPU move + dtype cast + alloc + copy)
is already equivalent to `_commit_tensor('weight', tensor,
split_side=None)` — no TP split for norm weights, same default
`alloc_dtype` (the tensor's own torch dtype via `_torch_dtype_to_cpp`),
same `_cast_shard_for_tm` cast before copy. The None guard in
`set_weight` is dropped — `_commit_tensor` already short-circuits on
`None`. Also drop the now-unused `_torch_dtype_to_cpp` /
`_cast_shard_for_tm` imports from `norm.py`.

### Spec side

Add two helpers to `TextModelSpec` in
`lmdeploy/turbomind/deploy/spec.py`:

```python
def norm(self, weight: torch.Tensor, *, dim: int | None = None,
         data_type=None) -> NormBuilder:
    """Build a NormBuilder for *weight* under this spec's contexts.

    ``dim`` defaults to ``weight.shape[-1]``. ``data_type`` defaults to
    the spec's compute dtype. Both overridable for the rare case where
    the C++ ``NormConfig`` needs a different value.
    """
    cfg = make_norm_config(
        dim=dim if dim is not None else weight.shape[-1],
        data_type=data_type if data_type is not None else self._cpp_dtype(),
        norm_eps=self._norm_eps,
    )
    m = NormBuilder(cfg, self._contexts)
    m.set_weight(weight)
    return m

def qk_norm(self, weight: torch.Tensor, *,
            head_dim: int, rope_dim: int) -> NormBuilder:
    """Build a per-head NormBuilder that follows the Q/K RoPE layout.

    ``head_dim`` and ``rope_dim`` specify the attention head geometry
    the norm belongs to; they match the values passed to
    ``reorder_rotary_emb`` on the corresponding Q/K projections.
    """
    return self.norm(reorder_rotary_emb(weight, head_dim, rope_dim))
```

Notes:

- `norm(weight)` takes a tensor, not a key. Callers do `self._get(key)`
  explicitly. This makes the method uniform across every norm site,
  including ones where the tensor is pre-processed (zero-centering,
  RoPE reordering).
- `qk_norm` takes `head_dim` and `rope_dim` as explicit kwargs to match
  the `reorder_rotary_emb` call on the sibling Q/K projections. No
  hidden dependence on `self._head_dim` / `self._rope.dim`.

The existing per-spec `output_norm(key)` and `norm(key)` wrappers (4
copies, one each in `qwen3_spec.py`, `qwen3_5_spec.py`,
`glm4_moe_lite_spec.py`, `gpt_oss_spec.py`) are **deleted**. The single
tensor-taking `norm` on the base replaces them.

### Zero-centering (Qwen3.5)

Every norm in Qwen3.5 is zero-centered. Today `_zero_centered(...)` is
called at three sites in `qwen3_5_spec.py`: once inside `output_norm`
(covering the root norm and, via `norm(key)` delegation, every decoder
layer norm) and twice inline at the `attn()` qk-norm call site. After
the change it lives in one override:

```python
# qwen3_5_spec.py
def norm(self, weight, *, dim=None, data_type=None):
    return super().norm(self._zero_centered(weight),
                        dim=dim, data_type=data_type)
```

`qk_norm` is **not** overridden — it inherits correctly because it
internally delegates to `self.norm`, which the override shadows.

### Call-site rewrites

All norm bindings become `parent.name = self.norm(weight)` (or
`self.qk_norm(weight, head_dim=..., rope_dim=...)` for per-head
RoPE-reordered norms).

**`qwen3_spec.py`**:

```python
# model()
root.norm = self.norm(self._get(self._norm_key))

# layers()
d.attention_norm = self.norm(self._get(f'{pfx}.{i}.input_layernorm.weight'))
d.ffn_norm       = self.norm(self._get(f'{pfx}.{i}.post_attention_layernorm.weight'))

# attn() — replaces add_qk_norm + the two manual reorder_rotary_emb calls
attn.q_norm = self.qk_norm(self._get(f'{pfx}.q_norm.weight'),
                           head_dim=self._head_dim, rope_dim=self._rope.dim)
attn.k_norm = self.qk_norm(self._get(f'{pfx}.k_norm.weight'),
                           head_dim=self._head_dim, rope_dim=self._rope.dim)
```

**`qwen3_5_spec.py`** — identical to qwen3 for norms (zero-centering is
now in the `norm` override, not at each call site):

```python
# model(), layers() — same shape as qwen3
root.norm        = self.norm(self._get(self._norm_key))
d.attention_norm = self.norm(self._get(f'{pfx}.{i}.input_layernorm.weight'))
d.ffn_norm       = self.norm(self._get(f'{pfx}.{i}.post_attention_layernorm.weight'))

# attn() — same shape as qwen3
attn.q_norm = self.qk_norm(self._get(f'{pfx}.q_norm.weight'),
                           head_dim=self._head_dim, rope_dim=self._rope.dim)
attn.k_norm = self.qk_norm(self._get(f'{pfx}.k_norm.weight'),
                           head_dim=self._head_dim, rope_dim=self._rope.dim)

# linear_attn() — replaces builder.add_norm(...)
builder.norm = self.norm(self._get(f'{pfx}.norm.weight'))
```

**`glm4_moe_lite_spec.py`** — replaces the MLA `builder.add_norms(...)`
call. Also incidentally corrects the kwarg-name / C++-child-name
mismatch: the old `add_norms` took `q_a_norm=` / `kv_a_norm=` while the
C++ children are `q_a_layernorm` / `kv_a_layernorm`:

```python
builder.q_a_layernorm  = self.norm(self._get(f'{pfx}.q_a_layernorm.weight'))
builder.kv_a_layernorm = self.norm(self._get(f'{pfx}.kv_a_layernorm.weight'))
```

**`gpt_oss_spec.py`** — same layer-norm / output-norm pattern as qwen3
(no qk-norm in gpt-oss).

## What disappears

- `Builder._add_norm_child` and its circular-import workaround.
- `AttentionBuilder.add_qk_norm`, `MLABuilder.add_norms`,
  `DeltaNetBuilder.add_norm`.
- Per-spec `output_norm(key)` / `norm(key)` wrappers (4 copies).
- `NormBuilder.set_weight`'s hand-rolled copy path.
- `_zero_centered` calls sprinkled at every Qwen3.5 norm call site.

## What appears

- `TextModelSpec.norm(weight, *, dim=None, data_type=None)` — one base
  helper.
- `TextModelSpec.qk_norm(weight, *, head_dim, rope_dim)` — per-head
  RoPE-layout variant.
- One `Qwen3_5Spec.norm` override.

## Non-goals

- Not touching `reorder_rotary_emb` itself (separate design already
  landed).
- Not collapsing the `self._get(key)` → `self.norm(weight)` pattern into
  a key-taking convenience. The two-step form keeps tensor-level
  transforms (zero-centering, RoPE reorder) visible and composable.
- Not touching `NormConfig` / C++ side.
- Not extracting a shared `_rope_reorder` helper on the spec.
  `reorder_rotary_emb` call sites for Q/K projections are orthogonal to
  this change; a later cleanup can deduplicate them if desired.

## Risk / verification

The C++-visible state the kernel consumes must be identical before and
after:

1. **Parent-child binding.** Old path: `parent.create_child(name, cfg)`
   = `Module::create(cfg)` + `parent.add_child(name, ...)`
   (`core/module.cc`). New path: `_tm.create_module(cfg)` + `__setattr__`
   → `parent.add_child_raw(name, child)` = `parent.add_child(name,
   std::move(child))`. Same C++ call at the end; `add_child_raw` is
   literally the "deferred parent binding" form of `create_child`
   (`bind.cpp:652-660`).

2. **Weight slot contents.** Both old `_add_norm_child` and new
   `NormBuilder.set_weight → _commit_tensor('weight', tensor)` feed
   `_copy_shard_to_param(handle, 'weight', tensor)` with default
   `alloc_shape` (= `shard.shape`) and default `alloc_dtype` (=
   `_torch_dtype_to_cpp(shard.dtype)`). Byte contents of the allocated
   slot are identical.

3. **`NormConfig.data_type`.** The C++ kernel consumes `NormConfig.
   data_type` via `NormWeight::prepare()` → `EnsureFloatDtype(weight,
   dtype_)` (`norm_weight.cc:44-47`), which casts the stored weight to
   that dtype at prepare time. Every current inline-norm call site
   passes `data_type=self.config.data_type` where
   `self.config.data_type == self._cpp_dtype()` (verified in all four
   specs: qwen3, qwen3_5, glm4_moe_lite, gpt_oss). The new base
   `self.norm` defaults to `self._cpp_dtype()`. Unchanged.

4. **`norm_eps`.** Spec already passes `self._norm_eps` at every call
   site. The new `self.norm` reads `self._norm_eps` from the instance.
   Unchanged.

5. **Tensor pre-processing.** The only tensor transforms currently
   applied before norm commit are `_zero_centered` (Qwen3.5 only) and
   `reorder_rotary_emb` (qk-norm in Qwen3/Qwen3.5). `_zero_centered`
   relocates to the `Qwen3_5Spec.norm` override (covers all three
   current sites). `reorder_rotary_emb` relocates into `qk_norm` on the
   base spec (covers both qwen3 and qwen3_5 qk-norm sites). Bytes
   produced are identical to today's inline transforms.

Verification: run `scripts/test_turbomind_model.py` against one
representative checkpoint per affected spec (qwen3 dense + qwen3-moe,
qwen3.5 dense + qwen3.5 with linear-attn layers, glm4-moe-lite for MLA,
gpt-oss) and confirm each produces coherent human text of ≥128 tokens,
per AGENTS.md.
