# Deferred Module Creation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move C++ `_tm.create_module` off the first-commit side effect onto an explicit `build()` step, so Python sharding/padding/fusion math can update the module config before construction. Delete the C++-side shape-based recovery in `FfnWeight` / `AttentionWeight` / `ModelWeight` in favour of authoritative Python values.

**Architecture:** `Builder` accumulates staged commits and staged child attachments, plus a mutable `self.config`. `build()` creates C++ modules with the final config, drains all stages, and returns a `BuiltModule` wrapper whose handle list is the only legitimate RHS for `parent.x = ...` assignments. After `build()`, every Builder operation raises. Authoritative values flow Python → C++: `FfnConfig.inter_size` (padded global), `AttentionConfig.kv_head_num` (post-repeat), and `ModelWeight.vocab_size_padded` (read from the sharded `output` child's dim).

**Tech Stack:** Python (lmdeploy/turbomind/deploy), C++ (src/turbomind/models), TurboMind in-tree extension (`_turbomind`), CUDA build via `ninja` from `build/`.

**Spec:** `docs/superpowers/specs/2026-04-23-deferred-module-creation-design.md`

---

## File Structure

### Python (`lmdeploy/turbomind/deploy/`)

- **Modified:** `builder/_base.py` — rewritten Builder lifecycle (BuiltModule, build(), staging dicts, _apply_*).
- **Modified:** `builder/ffn.py` — `FfnBuilder.add_ffn` pushes `cfg.inter_size` after `_pad_ffn_for_tp`.
- **Modified:** `builder/attention.py` — `AttentionBuilder.add_qkv_proj` pushes `cfg.kv_head_num` after `repeat_kv_for_tp`.
- **Modified:** `spec.py` — `TextModelSpec.norm` returns `m.build()`.
- **Modified:** `source_model/qwen3_spec.py` — add `.build()` to every factory return; `root.build()` at end of `model()`.
- **Modified:** `source_model/qwen3_5_spec.py` — same pattern.
- **Modified:** `source_model/gpt_oss_spec.py` — same pattern.
- **Modified:** `source_model/glm4_moe_lite_spec.py` — same pattern + MLA `kv_head_num` nudge on cloned cfg.

### C++ (`src/turbomind/models/`)

- **Modified:** `ffn_weight.cc` — delete shape-derived `inter_size_` recovery in `prepare()`; change constructor to `inter_size_{cfg.inter_size / cfg.tp_size}`.
- **Modified:** `attention_weight.cc` — collapse `prepare()` to `Module::prepare();`.
- **Modified:** `model_weight.cc` — replace `round_up(vocab_size, tp_size)` with `TM_CHECK_NOTNULL(output)->output_dim * tp_size`.

### No new files. No test files to create (no unit tests for this subsystem per `AGENTS.md`).

---

## Task 1: Python lifecycle refactor (atomic)

This task rewrites every Python file in one committable state. The tree is broken mid-task (between step 1 and step 8 inclusive), so steps 1-8 must all land before the first smoke test. Commit once at the end (step 11) after end-to-end smoke tests pass.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py`
- Modify: `lmdeploy/turbomind/deploy/spec.py`
- Modify: `lmdeploy/turbomind/deploy/builder/ffn.py`
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

---

- [ ] **Step 1: Rewrite `builder/_base.py` — BuiltModule + new Builder lifecycle**

Replace the section from line 328 (`class Builder:`) to the end of file (line 619) with the code below. Everything before line 328 (imports, enums, dtype helpers, `_dequant_linear`, `_ensure_compatible_formats`, `transform_output_dim`, `transform_input_dim`, `_copy_shard_to_param`, `_shard`) stays byte-for-byte unchanged.

The rewrite is a single coherent change; don't split this step across multiple edits.

```python
# ---------------------------------------------------------------------------
# BuiltModule — opaque handle bundle returned by Builder.build()
# ---------------------------------------------------------------------------


class BuiltModule:
    """Opaque handle bundle returned by Builder.build().

    The only legitimate RHS for ``parent.child = ...`` assignments.
    Holds one C++ module handle per GPU context.
    """
    __slots__ = ('handles',)

    def __init__(self, handles: list):
        self.handles = list(handles)

    def __iter__(self):
        return iter(self.handles)

    def __len__(self):
        return len(self.handles)


# ---------------------------------------------------------------------------
# Builder base class
# ---------------------------------------------------------------------------


class Builder:
    """Wraps N GPU handles for a single logical module.

    Lifecycle:
      1. ``__init__(config, contexts, tp, ranks)`` — stage-ready. No C++
         module exists yet. ``config`` is mutable.
      2. Subclass ``add_*`` helpers call ``_commit_linear`` /
         ``_commit_tensor`` / assign child ``BuiltModule``s via
         ``__setattr__``/``__setitem__``.  All three route into pending
         dicts; none touch C++.
      3. ``build()`` freezes ``config``, creates C++ modules via
         ``_tm.create_module`` on each GPU context, drains pending
         commits/attachments, and returns a ``BuiltModule`` wrapping the
         new handles.
      4. Post-build the Builder is inert: any further commit / assign /
         index raises.
    """

    def __init__(self, config, contexts, tp=1, ranks=None):
        # Use object.__setattr__ to bypass our own __setattr__ invariants.
        object.__setattr__(self, '_contexts', contexts)
        object.__setattr__(self, '_tp', tp)
        object.__setattr__(self, '_ranks', ranks)
        object.__setattr__(self, 'config', config)

        object.__setattr__(self, '_pending_linears', {})   # name -> (Linear, split_side, model_dtype)
        object.__setattr__(self, '_pending_tensors', {})   # name -> (tensor, split_side, model_dtype)
        object.__setattr__(self, '_pending_children', {})  # name -> list[_tm.Module]

        object.__setattr__(self, '_handles', None)
        object.__setattr__(self, '_built', False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tp_size(self):
        return self._tp

    def _rank_for(self, gpu_idx: int) -> int:
        if self._ranks and self._tp > 1:
            return self._ranks[gpu_idx]
        return 0

    # ------------------------------------------------------------------
    # Attribute / item assignment — freezes post-build, strict on RHS
    # ------------------------------------------------------------------

    def __setattr__(self, name, value):
        if self._built:
            raise RuntimeError(
                f"{type(self).__name__} is built; cannot assign {name!r}")
        if isinstance(value, Builder):
            raise TypeError(
                f"{type(self).__name__}.{name}: assign .build() output "
                f"(BuiltModule), not the Builder itself")
        if isinstance(value, BuiltModule):
            self._pending_children[name] = value.handles
            return
        object.__setattr__(self, name, value)

    def __setitem__(self, index, value):
        if self._built:
            raise RuntimeError(
                f"{type(self).__name__} is built; cannot set index {index}")
        if isinstance(value, Builder):
            raise TypeError(
                f"{type(self).__name__}[{index}]: call .build() first")
        assert isinstance(value, BuiltModule), (
            f"{type(self).__name__}[{index}] requires a BuiltModule")
        self._pending_children[str(index)] = value.handles

    # ------------------------------------------------------------------
    # Commit methods — stage only (block post-build via assert)
    # ------------------------------------------------------------------

    def _commit_linear(self, name: str, linear: Linear,
                       split_side: SplitSide | None = None,
                       model_dtype=None):
        """Stage a ``Linear`` commit under ``name``.  Applied during
        ``build()`` in ``_apply_linear``.
        """
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")
        self._pending_linears[name] = (linear, split_side, model_dtype)

    def _commit_tensor(self, name: str, tensor: torch.Tensor | None,
                       split_side: SplitSide | None = None, *,
                       model_dtype=None):
        """Stage a raw-tensor commit under ``name``.  Applied during
        ``build()`` in ``_apply_tensor``.
        """
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")
        self._pending_tensors[name] = (tensor, split_side, model_dtype)

    # ------------------------------------------------------------------
    # Build — one-way transition
    # ------------------------------------------------------------------

    def build(self) -> BuiltModule:
        """Freeze config, create C++ modules, drain staged state.

        Idempotent: a second call returns a fresh ``BuiltModule``
        wrapping the same handles with no side effects.
        """
        if self._built:
            return BuiltModule(self._handles)
        self._create_handles()
        object.__setattr__(self, '_built', True)     # bypass frozen guard
        for name, (lin, side, mdt) in self._pending_linears.items():
            self._apply_linear(name, lin, side, mdt)
        for name, (t, side, mdt) in self._pending_tensors.items():
            self._apply_tensor(name, t, side, mdt)
        for name, child_handles in self._pending_children.items():
            self._attach_handles(name, child_handles)
        return BuiltModule(self._handles)

    def _create_handles(self):
        """Default: create one C++ module per context using
        ``_cfg_for_rank``.  ``TextModelBuilder`` overrides to no-op.
        """
        handles = []
        for i, ctx in enumerate(self._contexts):
            with ctx:
                cfg = self._cfg_for_rank(i)
                handles.append(_tm.create_module(cfg))
        object.__setattr__(self, '_handles', handles)

    def _cfg_for_rank(self, gpu_idx: int):
        """Clone ``self.config`` and set per-rank ``tp_rank`` if applicable."""
        if self._tp > 1 and hasattr(self.config, 'tp_rank'):
            cfg = self.config.clone()
            cfg.tp_rank = self._ranks[gpu_idx]
            return cfg
        return self.config

    def _attach_handles(self, name: str, child_handles: list):
        for i, (parent_h, child_h) in enumerate(
                zip(self._handles, child_handles)):
            with self._contexts[i]:
                parent_h.add_child_raw(name, child_h)

    # ------------------------------------------------------------------
    # Apply — drained during build (logic moved from old _commit_*)
    # ------------------------------------------------------------------

    def _apply_linear(self, name: str, linear: Linear,
                      split_side: SplitSide | None,
                      model_dtype):
        """Create the LinearWeight child on each GPU handle and copy
        sharded tensors into its param slots.  Body relocated from the
        old ``_commit_linear`` (lines 453-528 in the pre-refactor file).
        """
        w = linear.tensors.get('weight')
        if w is None:
            return

        assert linear.data_format is not None, (
            f"{name}: Linear.data_format must be populated by "
            f"WeightFormatResolver.resolve or a fusion helper.")
        weight_cpp_dtype = linear.data_format.dtype
        fmt = linear.weight_format

        tp = self._tp if split_side else 1
        split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

        in_dim, out_dim = w.shape[0], w.shape[-1]
        if split_side == SplitSide.OUTPUT:
            out_dim //= tp
        elif split_side == SplitSide.INPUT:
            in_dim //= tp

        compute_dtype = (model_dtype if model_dtype is not None
                         else _infer_compute_dtype(linear))
        lin_cfg = _tm.LinearConfig()
        lin_cfg.input_dim  = in_dim
        lin_cfg.output_dim = out_dim
        lin_cfg.data_type  = compute_dtype or _tm.DataType.TYPE_INVALID
        lin_cfg.format     = linear.data_format
        lin_cfg.has_bias   = 'bias' in linear.tensors

        tensors = {k: fmt.pack(t, k) for k, t in linear.tensors.items()}
        is_quantized = linear.data_format.is_quantized()

        kind_split_dims = {
            kind: None if (kind == 'bias' and split_side == SplitSide.INPUT)
                  else split_dim
            for kind in tensors
        }

        if tp > 1 and split_dim is not None:
            for kind, tensor in tensors.items():
                kind_split_dim = kind_split_dims[kind]
                if kind_split_dim is not None:
                    d = tensor.shape[kind_split_dim]
                    assert d % tp == 0, (
                        f"TP split: {name}.{kind} dim {kind_split_dim} "
                        f"has size {d}, not divisible by tp={tp}.")

        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0

                linear_mod = (handle.child(name)
                              or handle.create_child(name, lin_cfg))

                for kind, tensor in tensors.items():
                    shard = _shard(tensor, kind_split_dims[kind], tp, rank)

                    if kind == 'weight' and is_quantized:
                        alloc_shape, alloc_dtype = ([in_dim, out_dim],
                                                    weight_cpp_dtype)
                    elif kind == 'weight' and model_dtype is not None:
                        alloc_shape, alloc_dtype = None, model_dtype
                    else:
                        alloc_shape, alloc_dtype = None, None

                    _copy_shard_to_param(linear_mod, kind, shard,
                                         alloc_shape=alloc_shape,
                                         alloc_dtype=alloc_dtype)

    def _apply_tensor(self, name: str, tensor: torch.Tensor | None,
                      split_side: SplitSide | None,
                      model_dtype):
        """Shard tensor per rank and copy into the named root param slot.
        Body relocated from the old ``_commit_tensor`` (lines 545-556).
        """
        if tensor is None:
            return

        tp = self._tp if split_side else 1
        split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0
                shard = _shard(tensor, split_dim, tp, rank)
                _copy_shard_to_param(handle, name, shard,
                                     alloc_dtype=model_dtype)


# ---------------------------------------------------------------------------
# TextModelBuilder -- wraps pre-existing root handles
# ---------------------------------------------------------------------------


class TextModelBuilder(Builder):
    """Root Builder that wraps handles created by the TurboMind runtime.

    The root's C++ module already exists before Python sees it, so
    ``_create_handles`` is overridden to a no-op.  Commits and child
    attachments otherwise follow the base lifecycle.
    """

    def __init__(self, handles, contexts, *,
                 tp, ranks, vocab_size, data_type):
        super().__init__(config=None, contexts=contexts, tp=tp, ranks=ranks)
        object.__setattr__(self, '_vocab_size', vocab_size)
        object.__setattr__(self, '_data_type', data_type)
        object.__setattr__(self, '_handles', list(handles))

    def _create_handles(self):
        # Root handles were pre-populated by the runtime; nothing to create.
        assert self._handles is not None

    def add_token_embeds(self, tensor):
        """Commit the raw embedding lookup as the ``tok_embeddings`` root param.

        Shards along hidden (output) dim by ``self._tp``.  No vocab
        padding -- embedding lookup never indexes past ``vocab - 1``.
        """
        self._commit_tensor('tok_embeddings', tensor,
                            split_side=SplitSide.OUTPUT,
                            model_dtype=self._data_type)

    def add_lm_head(self, linear):
        """Pad output dim to ``round_up(vocab_size, tp)`` and commit to the
        ``output`` LinearWeight root child.

        Works for every checkpoint format in use today -- trivial / AWQ /
        GPTQ / compressed-tensors / MXFP4 all have ``block_out is None``,
        so padding every tensor in the bundle along ``dim=-1`` keeps the
        format-specific block structure intact.  FP8 ``lm_head``
        (``block_out == 128``) would misalign scales under naive padding
        but is not a configuration used by any released checkpoint.
        """
        padded_vocab = ((self._vocab_size + self._tp - 1)
                        // self._tp) * self._tp
        padded = Linear(
            tensors={k: pad_out_dim(t, padded_vocab, dim=-1)
                     for k, t in linear.tensors.items()},
            weight_format=linear.weight_format,
            data_format=linear.data_format)
        self._commit_linear('output', padded,
                            split_side=SplitSide.OUTPUT,
                            model_dtype=self._data_type)
```

Verify the file parses (syntactic check only — can't yet run anything because specs still use the old API):

```bash
python -c "import ast; ast.parse(open('lmdeploy/turbomind/deploy/builder/_base.py').read())"
```

Expected: no output, exit 0.

- [ ] **Step 2: Update `spec.py::TextModelSpec.norm` to return `m.build()`**

Replace the body of `TextModelSpec.norm` at `lmdeploy/turbomind/deploy/spec.py:159-172` with:

```python
    def norm(self, weight, *, dim=None, data_type=None):
        """Build a NormBuilder for *weight* under this spec's contexts.

        ``dim`` defaults to ``weight.shape[-1]``. ``data_type`` defaults to
        the spec's compute dtype.
        """
        cfg = make_norm_config(
            dim=dim if dim is not None else weight.shape[-1],
            data_type=data_type if data_type is not None else self._cpp_dtype(),
            norm_eps=self._norm_eps,
        )
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(weight)
        return m.build()
```

`qk_norm` (lines 174-180) is unchanged — it forwards to `norm` and inherits the new return type.

- [ ] **Step 3: Update `builder/ffn.py::FfnBuilder.add_ffn` — push `inter_size`**

In `lmdeploy/turbomind/deploy/builder/ffn.py`, the current `FfnBuilder.add_ffn` (lines 178-207) reads:

```python
    def add_ffn(self, w1, w2, w3):
        """Pad weights for TP alignment, fuse w1+w3 if possible, then shard and commit.

        The fusion result determines ``fuse_silu`` on the C++ module config.
        Updating ``self.config.fuse_silu`` **before** any ``_commit_linear``
        call ensures the C++ module is lazily created with the correct flag.
        """
        # Pad weights for TP alignment before any fusion or sharding
        w1, w2, w3 = _pad_ffn_for_tp(w1, w2, w3, self._tp)

        act_type = getattr(self.config, 'act_type', 0)
        if isinstance(act_type, int):
            act_type = {0: 'silu', 1: 'gpt-oss'}.get(act_type, 'silu')
        fused, fused_silu = fuse_w1w3(
            w1, w3, self._tp, act_type,
            is_moe=getattr(self.config, 'fused_moe', False))

        self.config.fuse_silu = fused_silu
        # ... commits ...
```

Add a single line just after `_pad_ffn_for_tp`, pushing the padded-global `inter_size` onto the config before any commit or fusion fires:

```python
    def add_ffn(self, w1, w2, w3):
        """Pad weights for TP alignment, fuse w1+w3 if possible, then shard and commit.

        Updates ``self.config.inter_size`` to the padded-global value and
        ``self.config.fuse_silu`` from the fusion decision, both before any
        ``_commit_linear`` call.  These become the authoritative values
        ``_tm.create_module`` sees when ``build()`` fires.
        """
        w1, w2, w3 = _pad_ffn_for_tp(w1, w2, w3, self._tp)
        self.config.inter_size = w1.tensors['weight'].size(-1)

        act_type = getattr(self.config, 'act_type', 0)
        if isinstance(act_type, int):
            act_type = {0: 'silu', 1: 'gpt-oss'}.get(act_type, 'silu')
        fused, fused_silu = fuse_w1w3(
            w1, w3, self._tp, act_type,
            is_moe=getattr(self.config, 'fused_moe', False))

        self.config.fuse_silu = fused_silu
        # ... commits unchanged ...
```

Only the docstring and the added `self.config.inter_size = ...` line change; keep the rest of the method body byte-for-byte.

- [ ] **Step 4: Update `builder/attention.py::AttentionBuilder.add_qkv_proj` — push `kv_head_num`**

In `lmdeploy/turbomind/deploy/builder/attention.py`, `AttentionBuilder.add_qkv_proj` at lines 111-121:

```python
    def add_qkv_proj(self, q, k, v, *, gate=None):
        """Fuse Q/K/V into a single w_qkv with TP interleave, commit.

        Pipeline: dequant_mixed -> repeat_kv_for_tp -> fuse_qkv -> commit.
        """
        q, k, v, gate = dequant_mixed(q, k, v, gate, data_type=self.config.data_type)
        k, v = repeat_kv_for_tp(k, v, tp=self._tp,
                                head_dim=self.config.head_dim)
        merged = fuse_qkv(q, k, v, tp=self._tp, gate=gate)
        self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                            model_dtype=self.config.data_type)
```

Add the `cfg.kv_head_num` push right after `repeat_kv_for_tp` — the authoritative global kv-head count is derived from the (possibly padded) K tensor:

```python
    def add_qkv_proj(self, q, k, v, *, gate=None):
        """Fuse Q/K/V into a single w_qkv with TP interleave, commit.

        Pipeline: dequant_mixed -> repeat_kv_for_tp -> fuse_qkv -> commit.
        Updates ``self.config.kv_head_num`` to the padded global count
        produced by ``repeat_kv_for_tp`` before any commit fires.
        """
        q, k, v, gate = dequant_mixed(q, k, v, gate, data_type=self.config.data_type)
        k, v = repeat_kv_for_tp(k, v, tp=self._tp,
                                head_dim=self.config.head_dim)
        self.config.kv_head_num = _infer_heads(k, self.config.head_dim)
        merged = fuse_qkv(q, k, v, tp=self._tp, gate=gate)
        self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                            model_dtype=self.config.data_type)
```

`_infer_heads` is already defined at lines 41-46 of the same file. No new import needed.

- [ ] **Step 5: Update `source_model/qwen3_spec.py` — `.build()` at every return, `root.build()` at end of `model()`**

Every factory that constructs a Builder now ends with `return m.build()`. The `layers()` method wraps each decoder-layer with `.build()` when inserting into the module list, and wraps the container with `.build()` at return. Finally, `model()` ends with `root.build()`.

Change `attn`, `ffn`, `moe`, `layers`, `model` bodies as follows. Unchanged lines are marked with `...` for brevity; the actual edit keeps those lines exactly as they are today.

In `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`:

`model()` (lines 100-112) — add `root.build()`:

```python
    def model(self):
        ec = self.engine_cfg
        root = TextModelBuilder(
            self._root_handles, self._contexts,
            tp=ec.attn_tp_size * ec.attn_cp_size,
            ranks=self._model_tp_ranks,
            vocab_size=self._vocab_size,
            data_type=self._cpp_dtype())
        root.add_token_embeds(self._get(self._embed_key))
        root.norm = self.norm(self._get(self._norm_key))
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers(self._layer_prefix)
        root.build()
```

`attn()` (lines 118-143) — final `return attn` → `return attn.build()`:

```python
    def attn(self, pfx, layer):
        q = self._linear(f'{pfx}.q_proj')
        k = self._linear(f'{pfx}.k_proj')
        v = self._linear(f'{pfx}.v_proj')
        o = self._linear(f'{pfx}.o_proj')

        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               resolver=self._resolver)
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               resolver=self._resolver)

        cfg = self._attn_cfg.clone()
        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)

        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        attn.q_norm = self.qk_norm(self._get(f'{pfx}.q_norm.weight'),
                                   head_dim=self._head_dim, rope_dim=self._rope.dim)
        attn.k_norm = self.qk_norm(self._get(f'{pfx}.k_norm.weight'),
                                   head_dim=self._head_dim, rope_dim=self._rope.dim)

        return attn.build()
```

`ffn()` (lines 145-160) — `return m` → `return m.build()`:

```python
    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes[layer])
        cfg.fuse_silu  = False
        cfg.fused_moe  = fused_moe

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m.build()
```

`moe()` (lines 162-184) — `m.experts = experts.build()` and `return m.build()`:

```python
    def moe(self, pfx, layer):
        if self.num_experts(layer) <= 0:
            return None

        cfg = self._moe_cfg.clone()
        cfg.layer_id   = layer
        cfg.expert_num = self._expert_nums[layer]
        cfg.inter_size = self._expert_inter_size

        m = MoeBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)

        m.add_gate('gate', self._linear(f'{pfx}.gate'),
                   model_dtype=self._cpp_dtype())

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self.ffn(
                f'{pfx}.experts.{e}', layer,
                inter_size=self._expert_inter_size, fused_moe=True)
        m.experts = experts.build()
        return m.build()
```

`layers()` (lines 186-200) — `layers[str(i)] = d.build()` per layer; `return layers.build()`:

```python
    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in layer_progress(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(
                self._get(f'{pfx}.{i}.input_layernorm.weight'))
            d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(
                self._get(f'{pfx}.{i}.post_attention_layernorm.weight'))
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d.build()
        return layers.build()
```

- [ ] **Step 6: Update `source_model/qwen3_5_spec.py`**

Apply the same pattern as Step 5:

`model()` (lines 141-153): add `root.build()` as the last line.

`attn()` (lines 172-196): final `return attn` → `return attn.build()`.

`linear_attn()` (lines 198-218): final `return builder` → `return builder.build()`.

`ffn()` (lines 224-248): final `return m` → `return m.build()`. The `None` short-circuit when all three projections are absent stays; only the successful path gains `.build()`.

`moe()` (lines 250-275): `m.experts = experts.build()` before `return m.build()`.

`_packed_moe_ffn()` (lines 277-293): final `return m` → `return m.build()`.

`_moe_expert_ffn()` (lines 295-299) is unchanged — it just delegates to `self.ffn(...)` or `self._packed_moe_ffn(...)`, both of which now return `BuiltModule`.

`layers()` (lines 305-321): add `layers[str(i)] = d.build()` per iteration and `return layers.build()`.

`Qwen3_5Spec.norm` override (lines 159-161) is unchanged — `super().norm` already returns the new `BuiltModule`.

- [ ] **Step 7: Update `source_model/gpt_oss_spec.py`**

Same pattern, files at `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`:

`model()` (lines 107-119): add `root.build()` last line.

`attn()` (lines 126-147): `return attn` → `return attn.build()`.

`ffn()` (lines 153-168): `return m` → `return m.build()`.

`moe()` (lines 170-191): `m.experts = experts.build()` before `return m.build()`.

`_packed_moe_ffn()` (lines 205-223): `return m` → `return m.build()`.

`layers()` (lines 193-203): `layers[str(i)] = d.build()` per iteration; `return layers.build()`.

- [ ] **Step 8: Update `source_model/glm4_moe_lite_spec.py` — factories + MLA `kv_head_num` nudge**

Same `.build()` pattern plus the Section-2 Python nudge for MLA.

In `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`:

`model()` (lines 148-159): add `root.build()` last line.

`attn()` (lines 165-182) — new `kv_head_num` nudge on cloned cfg plus final `.build()`:

```python
    def attn(self, pfx, layer):
        cfg = self._attn_cfg.clone()
        # MLA: the compressed KV latent is not sharded across TP ranks, so
        # pad kv_head_num up to tp_size before the C++ module is created.
        if cfg.kv_lora_rank > 0 and cfg.kv_head_num < self.engine_cfg.attn_tp_size:
            cfg.kv_head_num = self.engine_cfg.attn_tp_size
        builder = MLABuilder(cfg, self._contexts,
                             tp=self.engine_cfg.attn_tp_size,
                             ranks=self._attn_ranks)

        q_b = (self._linear(f'{pfx}.q_b_proj', optional=True) or
               self._linear(f'{pfx}.q_proj'))
        builder.add_projections(
            q_a_proj=self._linear(f'{pfx}.q_a_proj'),
            q_b_proj=q_b,
            kv_a_proj=self._linear(f'{pfx}.kv_a_proj_with_mqa'),
            kv_b_proj=self._linear(f'{pfx}.kv_b_proj'),
            wo=self._linear(f'{pfx}.o_proj'),
        )
        builder.q_a_layernorm  = self.norm(self._get(f'{pfx}.q_a_layernorm.weight'))
        builder.kv_a_layernorm = self.norm(self._get(f'{pfx}.kv_a_layernorm.weight'))
        return builder.build()
```

`ffn()` (lines 188-203): `return m` → `return m.build()`.

`moe()` (lines 205-230): `m.experts = experts.build()` before `return m.build()`.

`layers()` (lines 232-245): `layers[str(i)] = d.build()` per iteration; `return layers.build()`.

- [ ] **Step 9: Smoke test — Qwen3 dense, `tp=1`**

Pick an available Qwen3 dense model via the `user-model-server` MCP:
- Call the MCP tool `list_models` to see what's locally cached.
- Call `get_model_cache_path` for the chosen model to get `cache_dir`.

Run the smoke test:

```bash
python scripts/test_turbomind_model.py <model_path> <cache_dir> 1 0
```

Expected output: the `--- response begin ---` … `--- response end ---` block contains at least 128 tokens of coherent text relevant to the script's prompt. Gibberish response indicates a bug — DO NOT commit until resolved.

- [ ] **Step 10: Smoke test — Qwen3 dense, `tp=2`**

Check GPU availability via the `user-gpu-monitor` MCP (`get_gpu_usage`) first. Pick two empty GPUs.

```bash
python scripts/test_turbomind_model.py <model_path> <cache_dir> 2 0,1
```

Expected: ≥128 tokens of coherent response. This exercises the `repeat_kv_for_tp` and `_pad_ffn_for_tp` paths under the new staged-commit lifecycle — the Section 2 config pushes (`kv_head_num`, `inter_size`) are now flowing into C++, and C++'s existing recovery blocks agree with them (not yet deleted at this point).

If the tp=2 run gibberish or crashes with a missing-config error, the Section 2 Python updates in Steps 3–4 are the first place to look.

- [ ] **Step 11: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/_base.py \
        lmdeploy/turbomind/deploy/builder/ffn.py \
        lmdeploy/turbomind/deploy/builder/attention.py \
        lmdeploy/turbomind/deploy/spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py

git commit -m "$(cat <<'EOF'
deploy: defer module creation via explicit build() lifecycle

Builder no longer creates its C++ module as a side effect of the first
commit. Commits and child attachments stage into pending dicts; build()
freezes the config, calls _tm.create_module, drains staged state, and
returns a BuiltModule wrapping the handles. After build() the Builder
is inert.

FfnBuilder.add_ffn and AttentionBuilder.add_qkv_proj now push the
padded-global inter_size and kv_head_num onto config before the first
commit. Glm4MoeLiteSpec.attn applies the MLA kv_head_num = tp_size
nudge on the cloned cfg before constructing MLABuilder.

All specs (qwen3, qwen3.5, gpt-oss, glm4-moe-lite) end every factory
with .build() and call root.build() as the last line of model().

No C++ changes yet; existing prepare() recoveries still run and agree
with the Python values. Follow-up commits remove those redundant
recoveries.
EOF
)"
```

---

## Task 2: Delete FFN C++ recovery; derive per-rank inter_size in constructor

Now that `FfnConfig.inter_size` is authoritative (padded global, pushed by `FfnBuilder.add_ffn`), delete the shape-derived recovery in `FfnWeight::prepare()` and change the constructor to compute per-rank once.

**Files:**
- Modify: `src/turbomind/models/ffn_weight.cc`

- [ ] **Step 1: Edit `ffn_weight.cc` constructor and `prepare()`**

Change the constructor at `src/turbomind/models/ffn_weight.cc:10-21` from:

```cpp
FfnWeight::FfnWeight(const core::FfnConfig& cfg)
    : hidden_dim_{cfg.hidden_dim}
    , inter_size_{cfg.inter_size}
    , bias_{cfg.has_bias}
    , tp_size_{cfg.tp_size}
    , tp_rank_{cfg.tp_rank}
    , data_type_{cfg.data_type}
    , act_type_{static_cast<ActivationType>(cfg.act_type)}
    , is_fused_silu_{cfg.fuse_silu && static_cast<ActivationType>(cfg.act_type) == ActivationType::kSilu}
    , is_fused_moe_{cfg.fused_moe}
{
}
```

To:

```cpp
FfnWeight::FfnWeight(const core::FfnConfig& cfg)
    : hidden_dim_{cfg.hidden_dim}
    , inter_size_{cfg.inter_size / cfg.tp_size}
    , bias_{cfg.has_bias}
    , tp_size_{cfg.tp_size}
    , tp_rank_{cfg.tp_rank}
    , data_type_{cfg.data_type}
    , act_type_{static_cast<ActivationType>(cfg.act_type)}
    , is_fused_silu_{cfg.fuse_silu && static_cast<ActivationType>(cfg.act_type) == ActivationType::kSilu}
    , is_fused_moe_{cfg.fused_moe}
{
}
```

Change `prepare()` at lines 23-54 from:

```cpp
void FfnWeight::prepare()
{
    // Derive per-rank inter_size from actual weight dimensions.
    // Weight tensors are already TP-sharded by the Python builder,
    // so w1 output_dim equals per-rank inter_size.  For fused w1w3,
    // output_dim = 2 * inter_size (gate + up).
    if (w1w3) {
        inter_size_ = w1w3->output_dim / 2;
    } else if (w1) {
        inter_size_ = w1->output_dim;
    }

    // Set epilogue on existing w1w3 child if fused silu is active.
    if (w1w3) {
        auto* fused = static_cast<LinearWeight*>(w1w3.get());
        if (is_fused_silu_) {
            fused->epilogue = gemm::Epilogue::kGatedSilu;
        }
    }

    // Propagate grouped-GEMM flag for MoE expert weights
    if (is_fused_moe_) {
        auto set_grouped = [](const char*, Module* m) {
            if (auto* lw = dynamic_cast<LinearWeight*>(m)) {
                lw->set_grouped(true);
            }
        };
        for_each_child(set_grouped);
    }

    Module::prepare();  // recurse into children
}
```

To:

```cpp
void FfnWeight::prepare()
{
    // Set epilogue on existing w1w3 child if fused silu is active.
    if (w1w3) {
        auto* fused = static_cast<LinearWeight*>(w1w3.get());
        if (is_fused_silu_) {
            fused->epilogue = gemm::Epilogue::kGatedSilu;
        }
    }

    // Propagate grouped-GEMM flag for MoE expert weights
    if (is_fused_moe_) {
        auto set_grouped = [](const char*, Module* m) {
            if (auto* lw = dynamic_cast<LinearWeight*>(m)) {
                lw->set_grouped(true);
            }
        };
        for_each_child(set_grouped);
    }

    Module::prepare();  // recurse into children
}
```

The `inter_size_` recovery block is gone; the rest is unchanged.

- [ ] **Step 2: Rebuild**

```bash
cd build && ninja
```

Expected: build succeeds with no errors. Warnings about unused headers (`"src/turbomind/kernels/gemm/types.h"` — still used for `Epilogue`) should not appear.

- [ ] **Step 3: Smoke test — Qwen3 dense `tp=2` (FFN coverage)**

```bash
python scripts/test_turbomind_model.py <qwen3_model_path> <cache_dir> 2 0,1
```

Expected: ≥128 tokens of coherent output, identical quality to the Task 1 Step 10 result. The C++ constructor now derives per-rank `inter_size_` from Python's authoritative padded global.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/ffn_weight.cc
git commit -m "$(cat <<'EOF'
models: delete FfnWeight inter_size recovery; derive per-rank in ctor

FfnBuilder.add_ffn now pushes the padded-global inter_size onto cfg
before the first commit, so the C++ shape-based recovery is redundant.
Compute per-rank inter_size_ once in the constructor from the
authoritative cfg.inter_size / cfg.tp_size.
EOF
)"
```

---

## Task 3: Delete AttentionWeight C++ recovery

With `AttentionBuilder.add_qkv_proj` pushing the repeated-global `kv_head_num` and `Glm4MoeLiteSpec.attn` handling the MLA nudge, `AttentionWeight::prepare()` no longer needs to reconstruct either value.

**Files:**
- Modify: `src/turbomind/models/attention_weight.cc`

- [ ] **Step 1: Collapse `AttentionWeight::prepare()` to `Module::prepare()`**

At `src/turbomind/models/attention_weight.cc:34-54`, replace:

```cpp
void AttentionWeight::prepare()
{
    Module::prepare();

    if (!w_qkv) {
        // MLA models use separate q_a/q_b/kv_a projections.
        // The compressed KV latent is not sharded across TP ranks, so
        // pad kv_head_num to tp_size to survive the engine's division.
        if (kv_lora_rank > 0 && kv_head_num < tp_size) {
            kv_head_num = tp_size;
        }
        return;
    }

    // Derive kv_head_num from actual weight tensor dimensions.
    // Python's repeat_kv_for_tp() physically pads KV heads, and w_qkv
    // is TP-sharded, so output_dim is per-shard.
    int local_total = w_qkv->output_dim / head_dim;
    int q_parts     = attn_output_gate ? 2 : 1;  // [Q|K|V] vs [Q|K|V|Gate]
    kv_head_num = (local_total * tp_size - q_parts * head_num) / 2;
}
```

With:

```cpp
void AttentionWeight::prepare()
{
    Module::prepare();
}
```

The header `<math.h>` / `kernels/core/math.h` include (already present for `init_rope_kernel_param`) stays — do not remove it.

- [ ] **Step 2: Rebuild**

```bash
cd build && ninja
```

Expected: build succeeds.

- [ ] **Step 3: Smoke test — Qwen3 dense `tp=2` (non-MLA attention)**

```bash
python scripts/test_turbomind_model.py <qwen3_model_path> <cache_dir> 2 0,1
```

Expected: ≥128 tokens of coherent output. The C++ side now trusts `cfg.kv_head_num` as-is.

- [ ] **Step 4: Smoke test — GLM-4 MoE Lite `tp=2` (MLA attention path)**

Pick a GLM-4 MoE Lite model via the MCP (`list_models`, `get_model_config` to confirm `model_type == 'glm4_moe'` or equivalent MLA-flavoured config).

```bash
python scripts/test_turbomind_model.py <glm4_model_path> <cache_dir> 2 0,1
```

Expected: ≥128 tokens of coherent output. This is the only path that exercises the `kv_head_num < tp_size` nudge (moved to Python in Step 8 of Task 1); if this fails with an MLA-shape assertion, the nudge isn't firing.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/attention_weight.cc
git commit -m "$(cat <<'EOF'
models: delete AttentionWeight::prepare kv_head_num recovery

AttentionBuilder.add_qkv_proj pushes the padded-global kv_head_num
onto cfg before the first commit, and Glm4MoeLiteSpec.attn applies the
MLA kv_head_num = tp_size nudge on the cloned cfg. prepare() therefore
no longer needs to reconstruct either value from the w_qkv shape.
EOF
)"
```

---

## Task 4: Update ModelWeight vocab_size_padded derivation

Replace the `round_up` duplication with a read of the `output` LinearWeight child's `output_dim * tp_size`. The Python side already populates `output` in `TextModelBuilder.add_lm_head` with the padded vocab tensor, and `root.build()` materializes the child before the engine calls `ModelWeight::prepare()`.

**Files:**
- Modify: `src/turbomind/models/model_weight.cc`

- [ ] **Step 1: Edit `ModelWeight::prepare()`**

At `src/turbomind/models/model_weight.cc:42-45`, replace:

```cpp
    vocab_size        = tok_embeddings.shape(0);
    embedding_size    = vocab_size;
    num_layer         = layers->size();
    vocab_size_padded = round_up((size_t)vocab_size, (size_t)tp_size);
```

With:

```cpp
    vocab_size        = tok_embeddings.shape(0);
    embedding_size    = vocab_size;
    num_layer         = layers->size();
    vocab_size_padded = TM_CHECK_NOTNULL(output)->output_dim * tp_size;
```

`TM_CHECK_NOTNULL` lives at `src/turbomind/core/check.h:141`; it is already available via the transitive include of `src/turbomind/core/module.h` from `model_weight.h`. If the build complains about unresolved `TM_CHECK_NOTNULL`, add:

```cpp
#include "src/turbomind/core/check.h"
```

just below the existing `#include "src/turbomind/kernels/core/math.h"` at the top of `model_weight.cc`. Verify the build first; only add if needed.

The `round_up` helper call from `src/turbomind/kernels/core/math.h` is no longer used in this file by `prepare()`. Check whether any other function in this file still uses `round_up` — if not, remove the include. (Current state: no other use, so the include can be dropped. Check with a grep on the edited file to confirm.)

- [ ] **Step 2: Rebuild**

```bash
cd build && ninja
```

Expected: build succeeds.

- [ ] **Step 3: Smoke test — any model (vocab_size_padded exercised on every lm_head commit)**

Pick any previously-tested model (e.g., Qwen3 dense `tp=2`):

```bash
python scripts/test_turbomind_model.py <qwen3_model_path> <cache_dir> 2 0,1
```

Expected: ≥128 tokens of coherent output. If `ModelWeight::prepare()` aborts with "TM_CHECK_NOTNULL(output)", the `output` child wasn't created — most likely the spec didn't call `add_lm_head` or `root.build()` isn't running.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/model_weight.cc
git commit -m "$(cat <<'EOF'
models: derive vocab_size_padded from output child dim

Python pads lm_head to round_up(vocab_size, tp_size) in
TextModelBuilder.add_lm_head; reading back via output->output_dim *
tp_size eliminates the C++-side round_up duplication. TM_CHECK_NOTNULL
fails loudly if a spec skips add_lm_head.
EOF
)"
```

---

## Task 5: Full verification matrix

Run the complete model matrix from the spec's Verification section to catch any regression the earlier single-model smoke tests missed.

**Files:** None modified; this task only runs smoke tests.

Before running each row, always:
1. Call MCP `user-gpu-monitor::get_gpu_usage` to pick empty GPUs.
2. Call MCP `user-model-server::list_models` + `get_model_cache_path` to locate the model.

- [ ] **Step 1: Qwen3-MoE**

Pick a Qwen3-MoE model (config with `num_experts > 0`). Run at `tp=2`:

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 2 0,1
```

Expected: ≥128 tokens of coherent text. Exercises `MoeBuilder` + staged experts `ModuleListBuilder`.

- [ ] **Step 2: Qwen3.5 linear-attention variant**

Pick a Qwen3.5 model whose HF config has `layer_types` containing `linear_attention`. Run at `tp=2`:

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 2 0,1
```

Expected: ≥128 tokens. Exercises `DeltaNetBuilder`, zero-centered norm override, MoE shared expert path.

- [ ] **Step 3: GPT-OSS (mxfp4 packed experts)**

Pick a GPT-OSS model. Run at `tp=2`:

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 2 0,1
```

Expected: ≥128 tokens. Exercises MXFP4 + packed experts + sliding-window attention.

- [ ] **Step 4: Qwen3 + AWQ**

Pick a Qwen3 checkpoint with `--model-format awq` (identifiable by `quantization_config.quant_method == "awq"` in the HF config). Run at `tp=1`:

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 1 0
```

Expected: ≥128 tokens. Exercises quantized w1w3 fusion; regression-critical because `FfnBuilder.add_ffn`'s new `cfg.inter_size` push must match the quantized block-scale boundary logic in `_pad_ffn_for_tp`.

- [ ] **Step 5: Qwen3 + FP8**

Pick a Qwen3-FP8 checkpoint. Run at `tp=1`:

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 1 0
```

Expected: ≥128 tokens. Exercises FP8 format + `_ensure_compatible_formats` dequant path.

- [ ] **Step 6: Qwen3 + GPTQ**

Pick a Qwen3-GPTQ checkpoint. Run at `tp=1`:

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 1 0
```

Expected: ≥128 tokens. Exercises GPTQ + `synthesize_zeros` path.

- [ ] **Step 7: Qwen3 + compressed-tensors**

Pick a Qwen3 compressed-tensors checkpoint. Run at `tp=1`:

```bash
python scripts/test_turbomind_model.py <model> <cache_dir> 1 0
```

Expected: ≥128 tokens.

- [ ] **Step 8: Manual smoke — strictness errors**

Confirm the new strictness produces clear errors. These are NOT committed; use a scratch branch or revert after:

1. In any spec (e.g., `qwen3_spec.py::attn`), temporarily change `return attn.build()` to `return attn`. Run any smoke test. Expected traceback:

   ```
   TypeError: DecoderLayerBuilder.attention: assign .build() output (BuiltModule), not the Builder itself
   ```

   Revert the change.

2. In a Python REPL (after the Task 1 commit lands):

   ```python
   from lmdeploy.turbomind.deploy.builder import AttentionBuilder
   import _turbomind as _tm
   cfg = _tm.AttentionConfig()
   # ... populate a minimal cfg ...
   b = AttentionBuilder(cfg, contexts=[...], tp=1)
   b.build()
   b.add_o_proj(some_linear)
   ```

   Expected:

   ```
   AssertionError: AttentionBuilder is built; commit 'wo' rejected
   ```

   Skip this sub-step if rigging a minimal context isn't convenient; the `TypeError` above already exercises the `_built` block path reachable during build.

- [ ] **Step 9: No commit — verification only**

Task 5 makes no code changes. If any of Steps 1-7 produce gibberish or errors:
- Gibberish → bug in the relocated `_apply_*` bodies or in the Section 2 config pushes; investigate and add a fix task.
- Python `TypeError: ... BuiltModule` → a factory missing `.build()` in Tasks 1 Step 5-8; audit each spec against the Step 5 pattern.
- C++ check abort → missing `output`, or config field read stale; verify the Tasks 2-4 edits.

Resolve with an additional fix task, not by editing the commits from Tasks 1-4.

---

## Summary

| Task | Files | Commit message |
|------|-------|----------------|
| 1    | 8 Python files (`_base.py`, `ffn.py`, `attention.py`, `spec.py`, 4 spec subclasses) | `deploy: defer module creation via explicit build() lifecycle` |
| 2    | `ffn_weight.cc`                            | `models: delete FfnWeight inter_size recovery; derive per-rank in ctor` |
| 3    | `attention_weight.cc`                      | `models: delete AttentionWeight::prepare kv_head_num recovery` |
| 4    | `model_weight.cc`                          | `models: derive vocab_size_padded from output child dim` |
| 5    | None                                       | (no commit; verification only)                                          |

Total: 4 commits.
