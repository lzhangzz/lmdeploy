# Loading Pipeline Layer Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 bugs from 6 bugfix commits by enforcing inter-layer contracts across the 5-layer loading pipeline — eliminating `_process_raw_tensors`, `_MOD_TP_ATTR`, and `fused_count` entirely.

**Architecture:** Three independent workstreams: (A) sharding-aware `chunk_linears` + `fused_count` removal, (B) spec interface change from `raw_layer_tensors` to per-component `attn_params`/`moe_params`/`linear_attn_params`, (C) loader rewrite to use new spec methods and eliminate navigation. All three converge at integration testing.

**Tech Stack:** Python 3.10+, C++17, pybind11, PyTorch

**Spec:** `docs/superpowers/specs/2026-04-08-loading-layer-contracts-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `lmdeploy/turbomind/deploy/linear.py` | Modify | Remove `fused_count` field, update `chunk_linears` for sharding-aware fusion |
| `lmdeploy/turbomind/deploy/transforms.py` | Modify | Pass `tp` to `chunk_linears` in `fuse_ffn_linears` |
| `lmdeploy/turbomind/deploy/load_context.py` | Modify | Remove `fused_count` from `_commit_tensors` and `commit_linear` |
| `lmdeploy/turbomind/deploy/spec.py` | Modify | Add `attn_params()`, `moe_params()`, `linear_attn_params()`; remove `raw_layer_tensors()` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Modify | Migrate `raw_layer_tensors` → `attn_params` + `moe_params` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Modify | Migrate `raw_layer_tensors` → `attn_params` + `moe_params` + `linear_attn_params` |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Modify | Migrate `raw_layer_tensors` → `attn_params` + `moe_params` |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Modify | Migrate `raw_layer_tensors` → `attn_params` + `moe_params` |
| `lmdeploy/turbomind/deploy/text_model_loader.py` | Modify | Delete `_process_raw_tensors` + `_MOD_TP_ATTR`; add param handling to `_process_attention`, `_process_moe`, `_process_linear_attn` |
| `lmdeploy/turbomind/deploy/module.py` | Modify | Update re-exports |

---

## Task 1: Sharding-aware `chunk_linears`

Make `chunk_linears` TP-aware so the result is naively shardable, matching the pattern used by `merge_qkv_v2` and `fuse_gdn_in_proj`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py` (lines 241-254)
- Modify: `lmdeploy/turbomind/deploy/transforms.py` (lines 131-132)

- [ ] **Step 1: Update `chunk_linears` to take `tp` and produce TP-interleaved output**

In `lmdeploy/turbomind/deploy/linear.py`, replace the `chunk_linears` function (lines 241-254) with:

```python
def chunk_linears(w1: Linear, w3: Linear, tp: int = 1) -> Linear:
    """Concatenate w1 and w3 along the output dim (chunk layout).

    When ``tp > 1``, the result is TP-interleaved so that a naive
    output-dim split gives each rank ``[w1_shard | w3_shard]``.
    This matches the pattern used by ``merge_qkv_v2`` and
    ``fuse_gdn_in_proj``.
    """
    fused: dict[str, Tensor] = {}
    for kind in w1.tensors:
        t1 = w1.tensors[kind]
        t3 = w3.tensors[kind]
        if tp <= 1 or not _has_input_dim(t1):
            # No TP or 1-D (bias): simple concatenation
            dim = -1 if _has_input_dim(t1) else 0
            fused[kind] = torch.cat([t1, t3], dim=dim)
        else:
            # TP-aware: reshape [K, N] -> [K, tp, N/tp], cat on inner dim,
            # then flatten back to [K, 2*N].
            d = t1.dim() - 1  # output dim (last)
            shape = list(t1.shape)
            r1 = t1.reshape(shape[:d] + [tp, shape[d] // tp])
            r3 = t3.reshape(shape[:d] + [tp, shape[d] // tp])
            combined = torch.cat([r1, r3], dim=d + 1)
            c_shape = list(combined.shape)
            fused[kind] = combined.reshape(
                c_shape[:d] + [c_shape[d] * c_shape[d + 1]])
    return Linear(tensors={k: v.contiguous() for k, v in fused.items()},
                  weight_format=w1.weight_format,
                  data_format=w1.data_format)
```

Note: `fused_count=2` is removed from the return. The `fused_count` field will be deleted in Task 2.

- [ ] **Step 2: Update `fuse_ffn_linears` to pass `tp` to `chunk_linears`**

In `lmdeploy/turbomind/deploy/transforms.py`, replace lines 131-132:

```python
        else:
            w1w3 = chunk_linears(w1, w3)
```

with:

```python
        else:
            w1w3 = chunk_linears(w1, w3, tp)
```

- [ ] **Step 3: Build**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds (Python-only changes)

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/linear.py lmdeploy/turbomind/deploy/transforms.py
git commit -m "refactor(linear): sharding-aware chunk_linears with TP interleaving"
```

---

## Task 2: Remove `fused_count` from the pipeline

With sharding-aware fusion, `fused_count` is no longer needed anywhere.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py` (line 142)
- Modify: `lmdeploy/turbomind/deploy/load_context.py` (lines 138-140, 165-181, 245-252, 293-297)

- [ ] **Step 1: Remove `fused_count` field from `Linear` dataclass**

In `lmdeploy/turbomind/deploy/linear.py`, delete line 142:

```python
    fused_count: int = field(default=1, compare=False, repr=False)
```

- [ ] **Step 2: Remove `fused_count` parameter from `_commit_tensors`**

In `lmdeploy/turbomind/deploy/load_context.py`, change the signature (lines 138-140) from:

```python
def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int,
                    fused_count: int = 1):
```

to:

```python
def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int):
```

- [ ] **Step 3: Remove `fused_count` sharding logic from `_commit_tensors`**

In `lmdeploy/turbomind/deploy/load_context.py`, replace the sharding block (lines 165-181) with:

```python
        if tensor_split_dim is not None and split_num > 1:
            split_size = tensor.shape[tensor_split_dim] // split_num
            shard = tensor.split(split_size, dim=tensor_split_dim)[rank]
        else:
            shard = tensor
```

This removes the `pos_split_dim` normalization, the `fused_count > 1` branch, and the reshape-shard-reshape logic entirely.

- [ ] **Step 4: Remove `fused_count` preservation from `commit_linear`**

In `lmdeploy/turbomind/deploy/load_context.py`, in the `Linear` reconstruction around line 247-252, change:

```python
        linear = Linear(tensors=linear.tensors,
                        weight_format=linear.weight_format,
                        data_format=linear.weight_format.to_data_format(
                            cpp_dtype.value if cpp_dtype else 0,
                            group_size),
                        fused_count=linear.fused_count)
```

to:

```python
        linear = Linear(tensors=linear.tensors,
                        weight_format=linear.weight_format,
                        data_format=linear.weight_format.to_data_format(
                            cpp_dtype.value if cpp_dtype else 0,
                            group_size))
```

- [ ] **Step 5: Remove `fused_count` argument from the `_commit_tensors` call in `commit_linear`**

In `lmdeploy/turbomind/deploy/load_context.py`, find the call to `_commit_tensors` inside `commit_linear` (around line 293-297) and remove the `fused_count=linear.fused_count` keyword argument.

- [ ] **Step 6: Build**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/linear.py lmdeploy/turbomind/deploy/load_context.py
git commit -m "refactor(commit): remove fused_count — sharding-aware fusion handles TP"
```

---

## Task 3: Add per-component param methods to TextModelSpec

Add `attn_params()`, `moe_params()`, and `linear_attn_params()` base methods to `TextModelSpec`. Remove `raw_layer_tensors()`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py` (lines 237-265)

- [ ] **Step 1: Replace `raw_layer_tensors` with per-component param methods**

In `lmdeploy/turbomind/deploy/spec.py`, replace the block from `attn_norm` through `raw_layer_tensors` (lines 237-265) with:

```python
    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return None

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return None

    def attn_params(
        self, layer: int
    ) -> dict[str, tuple[torch.Tensor, SplitSide | None]]:
        """Return non-linear attention parameters.

        Each entry is ``{param_name: (tensor, split_side)}``.
        ``param_name`` is the leaf name within the attention subtree
        (e.g. ``"q_norm.weight"``, ``"sinks"``).
        ``split_side`` is ``None`` for broadcast, or a ``SplitSide`` value.
        """
        return {}

    def moe_params(
        self, layer: int
    ) -> dict[str, tuple[torch.Tensor, SplitSide | None]]:
        """Return non-expert MoE parameters.

        Each entry is ``{param_name: (tensor, split_side)}``.
        ``param_name`` is the leaf name within the moe_ffn subtree
        (e.g. ``"gate.weight"``, ``"score_correction_bias"``).
        """
        return {}

    def linear_attn_params(
        self, layer: int
    ) -> dict[str, tuple[torch.Tensor, SplitSide | None]]:
        """Return non-linear linear-attention (GDN) parameters.

        Each entry is ``{param_name: (tensor, split_side)}``.
        ``param_name`` is the leaf name within the linear_attn subtree
        (e.g. ``"A_log"``, ``"dt_bias"``, ``"conv1d"``, ``"norm.weight"``).
        """
        return {}

    def tok_embeddings(self) -> torch.Tensor | None:
        return None

    def output_weight(self) -> torch.Tensor | None:
        return None

    def norm_weight(self) -> torch.Tensor | None:
        return None
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "refactor(spec): replace raw_layer_tensors with attn_params/moe_params/linear_attn_params"
```

---

## Task 4: Migrate Qwen3 spec

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` (lines 71-86)

- [ ] **Step 1: Replace `raw_layer_tensors` with `attn_params` and `moe_params`**

In `qwen3_spec.py`, replace the `raw_layer_tensors` method (lines 71-86) with:

```python
    def attn_params(self, layer):
        params = {}
        q = self._get(f"{self._layer_prefix}.{layer}.self_attn.q_norm.weight")
        k = self._get(f"{self._layer_prefix}.{layer}.self_attn.k_norm.weight")
        if q is not None and k is not None:
            q, k = self._permute_qk_tensors(q, k)
        if q is not None:
            params["q_norm.weight"] = (q, None)
        if k is not None:
            params["k_norm.weight"] = (k, None)
        return params

    def moe_params(self, layer):
        params = {}
        if self._n_experts > 0:
            gate = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                params["gate.weight"] = (gate, None)
        return params
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git commit -m "refactor(qwen3): migrate raw_layer_tensors to attn_params + moe_params"
```

---

## Task 5: Migrate Qwen3.5 spec

This is the most complex migration — includes zero-centered norms, MoE gate+shared_gate, and GDN raw tensors (A_log, dt_bias, conv1d with TP interleaving, norm).

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (lines 213-266)

- [ ] **Step 1: Replace `raw_layer_tensors` with three param methods**

In `qwen3_5_spec.py`, replace the `raw_layer_tensors` method (lines 213-266) with:

```python
    def attn_params(self, layer):
        params = {}
        if not self._is_linear_attn(layer):
            q = self._zero_centered(
                self._get(f"{self._layer_prefix}.{layer}.self_attn.q_norm.weight"))
            k = self._zero_centered(
                self._get(f"{self._layer_prefix}.{layer}.self_attn.k_norm.weight"))
            if q is not None and k is not None:
                q, k = self._permute_qk_tensors(q, k)
            if q is not None:
                params["q_norm.weight"] = (q, None)
            if k is not None:
                params["k_norm.weight"] = (k, None)
        return params

    def moe_params(self, layer):
        params = {}
        if self._n_experts > 0:
            gate = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                params["gate.weight"] = (gate, None)
            sg = self._get(
                f"{self._layer_prefix}.{layer}.mlp.shared_expert_gate.weight")
            if sg is not None:
                sg = sg.t() if sg.dim() > 1 else sg
                params["shared_gate.weight"] = (sg, None)
        return params

    def linear_attn_params(self, layer):
        params = {}
        if not self._is_linear_attn(layer):
            return params
        pfx = f"{self._layer_prefix}.{layer}.linear_attn"
        for key in ["A_log", "dt_bias"]:
            t = self._get(f"{pfx}.{key}")
            if t is not None:
                params[key] = (t, SplitSide.OUTPUT)
        conv1d = self._get(f"{pfx}.conv1d.weight")
        if conv1d is not None and conv1d.ndim == 3 and conv1d.shape[1] == 1:
            conv1d = conv1d.squeeze(1)
        # C++ kernel expects [d_conv, conv_dim]; HF stores [conv_dim, d_conv].
        if conv1d is not None:
            conv1d = conv1d.t().contiguous()
            if self._attn_tp > 1 and self._linear_qkv_split is not None:
                q_dim, k_dim, v_dim = self._linear_qkv_split
                d_conv = conv1d.shape[0]
                tp = self._attn_tp
                q_part = conv1d[:, :q_dim]
                k_part = conv1d[:, q_dim:q_dim + k_dim]
                v_part = conv1d[:, q_dim + k_dim:]
                conv1d = torch.cat([
                    q_part.reshape(d_conv, tp, q_dim // tp),
                    k_part.reshape(d_conv, tp, k_dim // tp),
                    v_part.reshape(d_conv, tp, v_dim // tp),
                ], dim=2).reshape(d_conv, -1).contiguous()
            params["conv1d"] = (conv1d, SplitSide.OUTPUT)
        norm = self._get(f"{pfx}.norm.weight")
        if norm is not None:
            params["norm.weight"] = (norm, None)
        return params
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(qwen3_5): migrate raw_layer_tensors to attn_params + moe_params + linear_attn_params"
```

---

## Task 6: Migrate GptOss spec

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (lines 125-137)

- [ ] **Step 1: Replace `raw_layer_tensors` with `attn_params` and `moe_params`**

In `gpt_oss_spec.py`, replace the `raw_layer_tensors` method (lines 125-137) with:

```python
    def attn_params(self, layer):
        params = {}
        sinks = self._get(
            f"{self._layer_prefix}.{layer}.self_attn.sinks")
        if sinks is not None:
            params["sinks"] = (sinks, SplitSide.OUTPUT)
        return params

    def moe_params(self, layer):
        params = {}
        gate = self._get(
            f"{self._layer_prefix}.{layer}.mlp.router.weight")
        if gate is not None:
            gate = gate.t() if gate.dim() > 1 else gate
            params["gate.weight"] = (gate, None)
        gate_bias = self._get(
            f"{self._layer_prefix}.{layer}.mlp.router.bias")
        if gate_bias is not None:
            params["gate.bias"] = (gate_bias, None)
        return params
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(gpt_oss): migrate raw_layer_tensors to attn_params + moe_params"
```

---

## Task 7: Migrate GLM4 MoE Lite spec

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` (lines 182-204)

- [ ] **Step 1: Replace `raw_layer_tensors` with `attn_params` and `moe_params`**

In `glm4_moe_lite_spec.py`, replace the `raw_layer_tensors` method (lines 182-204) with:

```python
    def attn_params(self, layer):
        params = {}
        q_a = self._get(
            f"{self._layer_prefix}.{layer}.self_attn.q_a_layernorm.weight")
        kv_a = self._get(
            f"{self._layer_prefix}.{layer}.self_attn.kv_a_layernorm.weight")
        if q_a is not None:
            params["q_a_layernorm.weight"] = (q_a, None)
        if kv_a is not None:
            params["kv_a_layernorm.weight"] = (kv_a, None)
        return params

    def moe_params(self, layer):
        params = {}
        if self.num_experts(layer) > 0:
            gate = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                params["gate.weight"] = (gate, None)
            gate_bias = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.bias")
            if gate_bias is not None:
                params["gate.bias"] = (gate_bias, None)
            correction = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.e_score_correction_bias")
            if correction is not None:
                params["score_correction_bias"] = (correction, None)
        return params
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(glm4_moe_lite): migrate raw_layer_tensors to attn_params + moe_params"
```

---

## Task 8: Update loader to use new spec methods

Delete `_process_raw_tensors` and `_MOD_TP_ATTR`. Add parameter handling to `_process_attention`, `_process_moe`, and `_process_linear_attn`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py` (lines 150-175, 225-304, 306-374, 396-401)

- [ ] **Step 1: Add param handling to `_process_attention`**

In `text_model_loader.py`, after the linear commit loop in `_process_attention` (after line 175), add:

```python
        # --- Parameters (q_norm, k_norm, sinks, etc.) ---
        for name, (tensor, split_side) in spec.attn_params(layer).items():
            parts = name.split('.')
            parent = attn
            for seg in parts[:-1]:
                parent = parent.create_child(seg, NormConfig(
                    dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                    data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)
```

- [ ] **Step 2: Add param handling to `_process_moe`**

In `text_model_loader.py`, after the shared_gate block in `_process_moe` (after line 265), add:

```python
        # --- Non-expert MoE parameters (gate/shared_gate raw tensors, etc.) ---
        for name, (tensor, split_side) in spec.moe_params(layer).items():
            parts = name.split('.')
            parent = moe
            for seg in parts[:-1]:
                existing = parent._handles[0].child(seg) if parent._handles else None
                if existing is not None:
                    children = [h.child(seg) for h in parent._handles]
                    parent = LayerWriter(children)
                else:
                    parent = parent.create_child(seg, NormConfig(
                        dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                        data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)
```

Note: `moe_params` returns names like `"gate.weight"` and `"gate.bias"`. The `gate` child module was already created by `_process_moe`'s gate handling above. For `"gate.weight"`, we navigate to the existing `gate` child and commit `weight`. For `"score_correction_bias"` (no dot), we commit directly to the moe writer.

- [ ] **Step 3: Add param handling to `_process_linear_attn`**

In `text_model_loader.py`, after the linear commit loop in `_process_linear_attn` (after line 325), add:

```python
        # --- Parameters (A_log, dt_bias, conv1d, norm.weight) ---
        for name, (tensor, split_side) in spec.linear_attn_params(layer).items():
            parts = name.split('.')
            parent = linear_attn
            for seg in parts[:-1]:
                parent = parent.create_child(seg, NormConfig(
                    dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                    data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)
```

- [ ] **Step 4: Delete `_MOD_TP_ATTR` and `_process_raw_tensors`**

In `text_model_loader.py`, delete the `_MOD_TP_ATTR` dict (lines 327-335) and the entire `_process_raw_tensors` method (lines 337-374).

- [ ] **Step 5: Remove `_process_raw_tensors` call from `_load_layer`**

In `text_model_loader.py`, delete the `self._process_raw_tensors(writer, spec, layer)` call from `_load_layer` (line 401).

- [ ] **Step 6: Remove unused gate/shared_gate inline handling from `_process_moe`**

The gate and shared_gate weight/bias tensors are now delivered by `spec.moe_params()`. The inline `moe_gate_linear` / `moe_shared_gate_linear` getattr-based handling (lines 247-265) and the `gate_cfg`/`shared_gate_cfg` `create_child` fallbacks should be removed.

Replace lines 246-265:

```python
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
```

with:

```python
        # --- gate + shared_gate + score_correction_bias from spec ---
        gate_cfg = LinearConfig(
            input_dim=hidden,
            output_dim=spec.num_experts(layer),
            data_type=dtype,
            has_bias=getattr(mc, 'expert_router_bias', False))
        moe.create_child('gate', gate_cfg)

        if mc.moe_shared_gate:
            shared_gate_cfg = LinearConfig(
                input_dim=hidden, output_dim=1, data_type=dtype, has_bias=False)
            moe.create_child('shared_gate', shared_gate_cfg)

        for name, (tensor, split_side) in spec.moe_params(layer).items():
            parts = name.split('.')
            parent = moe
            for seg in parts[:-1]:
                existing = parent._handles[0].child(seg) if parent._handles else None
                if existing is not None:
                    children = [h.child(seg) for h in parent._handles]
                    parent = LayerWriter(children)
                else:
                    parent = parent.create_child(seg, NormConfig(
                        dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                        data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)
```

This creates the gate/shared_gate modules first (so `commit_tensor` for `"gate.weight"` finds them), then commits tensors from `spec.moe_params()`.

- [ ] **Step 7: Build**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): eliminate _process_raw_tensors, use per-component params"
```

---

## Task 9: Update module.py re-exports

Remove `raw_layer_tensors`-related symbols and add new spec method references if needed.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/module.py` (line 27)

- [ ] **Step 1: Remove `_fuse_and_commit_ffn` if still listed; verify no stale re-exports**

In `lmdeploy/turbomind/deploy/module.py`, check the `__all__` list and the actual imports. Remove any references to:
- `_fuse_and_commit_ffn` (if listed in `__all__` or imported)
- `commit_ffn` (if listed — it's deprecated)

Also remove `_shard_linear_for_tp` from the re-exports if it's no longer used externally.

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/module.py
git commit -m "refactor(module): clean up re-exports after raw_layer_tensors removal"
```

---

## Task 10: Integration test

Verify the full pipeline works across model types.

**Files:** None (testing only)

- [ ] **Step 1: Check GPU availability**

Run `get_gpu_usage` to find empty GPUs.

- [ ] **Step 2: Test a dense BF16 model (TP=1)**

Use the turbomind-tester agent. Verify 128+ token coherent output.

- [ ] **Step 3: Test a quantized model (AWQ or GPTQ, TP=1)**

Use the turbomind-tester agent. Verify 128+ token coherent output.

- [ ] **Step 4: Test a MoE model (TP=1)**

Use the turbomind-tester agent. Verify 128+ token coherent output. This exercises the new `moe_params` path.

- [ ] **Step 5: Test TP=2 (dense or MoE)**

This exercises the sharding-aware `chunk_linears`. Use the turbomind-tester agent. Verify 128+ token coherent output.

- [ ] **Step 6: Test a model with linear attention (Qwen3.5) if available**

This exercises the `linear_attn_params` path. Use the turbomind-tester agent. Verify 128+ token coherent output.

- [ ] **Step 7: Fix any regressions found**

Iterate until all tests produce meaningful output.

---

## Self-Review

**1. Spec coverage check:**
- Layer 1 (Config Protocol): `for_rank` on `LinearConfig` already fixed in code ✓ (no plan task needed — already done in bugfix 3132c646). DataType wrapping already fixed in code ✓. Type fidelity (topk_method/scoring_func) already fixed ✓. These are *already correct in the codebase* — the spec documents the rules for future development.
- Layer 2 (LayerWriter): Eliminate `_MOD_TP_ATTR` → Task 8 Step 4 ✓. Lazy init → already done ✓. No navigation → Tasks 3-8 ✓.
- Layer 3 (Component Processing): Parameter ownership → Tasks 3-8 ✓. Eliminate `raw_layer_tensors` → Tasks 3-7 ✓. READ→TRANSFORM→CREATE→COMMIT → already correct in code ✓.
- Layer 4 (Commit Layer): Sharding-aware `chunk_linears` → Task 1 ✓. Remove `fused_count` → Task 2 ✓.
- Layer 5 (C++ Constructors): `fused_moe` forwarding, `expert_num` direct assignment, string types → all already fixed in bugfix commits ✓. The spec documents rules for future development.

**2. Placeholder scan:** No TBDs, TODOs, or vague steps. All code is complete.

**3. Type consistency:**
- `attn_params` / `moe_params` / `linear_attn_params` return `dict[str, tuple[torch.Tensor, SplitSide | None]]` — consistent in spec.py base class and all 4 model specs.
- `chunk_linears(w1, w3, tp=1)` signature matches the call in `fuse_ffn_linears(..., tp, ...)` which passes `tp` through.
- `_commit_tensors` signature change (remove `fused_count`) is consistent with `commit_linear` call site change.

**4. Spec deviation:** Layer 1 and Layer 5 rules are already implemented in the codebase via the 6 bugfix commits. The plan only implements the *new* changes (Layers 2-4). This is correct — the spec documents both existing rules (for reference) and new changes (for implementation).
