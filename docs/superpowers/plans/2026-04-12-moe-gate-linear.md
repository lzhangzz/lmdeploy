# MoE Gate Linear Weight Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move MoE gate/shared_gate weights from `commit_tensor` to `commit_linear` via `Linear` bundles, consistent with all other linear layers.

**Architecture:** Add `moe_gate()` to `TextModelSpec`, move gate/shared_gate entries there as `Linear` bundles, simplify `_process_moe` to use `commit_linear` for gates.

**Tech Stack:** Python, git

---

### Task 1: Add `moe_gate()` base method and implement in all specs

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py:261-270`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:83-91`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:228-241`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:194-210`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:133-144`

- [ ] **Step 1: Add `moe_gate()` base method to `spec.py`**

Insert after `moe_params()` (after line 270):

```python
    def moe_gate(
        self, layer: int
    ) -> dict[str, Linear]:
        """Return MoE gate and shared_gate as Linear bundles.

        Keys are child module names (e.g. ``"gate"``, ``"shared_gate"``).
        Values are :class:`Linear` bundles with weight (and optional bias).
        """
        return {}
```

- [ ] **Step 2: Implement `moe_gate()` in `qwen3_spec.py`**

Replace the entire `moe_params` method (lines 83-91) with:

```python
    def moe_gate(self, layer):
        gates = {}
        if self._n_experts > 0:
            gate = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                gates["gate"] = Linear({"weight": gate})
        return gates

    def moe_params(self, layer):
        return {}
```

- [ ] **Step 3: Implement `moe_gate()` in `qwen3_5_spec.py`**

Replace the entire `moe_params` method (lines 228-241) with:

```python
    def moe_gate(self, layer):
        gates = {}
        if self._n_experts > 0:
            gate = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                gates["gate"] = Linear({"weight": gate})
            sg = self._get(
                f"{self._layer_prefix}.{layer}.mlp.shared_expert_gate.weight")
            if sg is not None:
                sg = sg.t() if sg.dim() > 1 else sg
                gates["shared_gate"] = Linear({"weight": sg})
        return gates

    def moe_params(self, layer):
        return {}
```

- [ ] **Step 4: Implement `moe_gate()` in `glm4_moe_lite_spec.py`**

Replace the entire `moe_params` method (lines 194-210) with:

```python
    def moe_gate(self, layer):
        gates = {}
        if self.num_experts(layer) > 0:
            gate = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                tensors = {"weight": gate}
                gate_bias = self._get(
                    f"{self._layer_prefix}.{layer}.mlp.gate.bias")
                if gate_bias is not None:
                    tensors["bias"] = gate_bias
                gates["gate"] = Linear(tensors)
        return gates

    def moe_params(self, layer):
        params = {}
        if self.num_experts(layer) > 0:
            correction = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.e_score_correction_bias")
            if correction is not None:
                params["score_correction_bias"] = (correction, None)
        return params
```

- [ ] **Step 5: Implement `moe_gate()` in `gpt_oss_spec.py`**

Replace the entire `moe_params` method (lines 133-144) with:

```python
    def moe_gate(self, layer):
        gates = {}
        gate = self._get(
            f"{self._layer_prefix}.{layer}.mlp.router.weight")
        if gate is not None:
            gate = gate.t() if gate.dim() > 1 else gate
            tensors = {"weight": gate}
            gate_bias = self._get(
                f"{self._layer_prefix}.{layer}.mlp.router.bias")
            if gate_bias is not None:
                tensors["bias"] = gate_bias
            gates["gate"] = Linear(tensors)
        return gates

    def moe_params(self, layer):
        return {}
```

- [ ] **Step 6: Verify imports resolve**

```bash
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "
from lmdeploy.turbomind.deploy.source_model.qwen3_spec import Qwen3TextSpec
from lmdeploy.turbomind.deploy.source_model.qwen3_5_spec import Qwen3_5Spec
from lmdeploy.turbomind.deploy.source_model.glm4_moe_lite_spec import Glm4MoeLiteSpec
from lmdeploy.turbomind.deploy.source_model.gpt_oss_spec import GptOssSpec
print('OK')
"
```

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py lmdeploy/turbomind/deploy/source_model/qwen3_spec.py lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(deploy): add moe_gate() to specs, move gate weights to Linear bundles

Gate and shared_gate weights are now returned as Linear bundles from
moe_gate() instead of raw tensors from moe_params(). This aligns them
with how all other linear layers (attention, FFN) are committed."
```

---

### Task 2: Rewrite `_process_moe` to use `commit_linear` for gates

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py:168-247`

- [ ] **Step 1: Replace the gate pre-creation and moe_params loop**

In `text_model_loader.py`, replace lines 175 and 189-208 (the `hidden` variable assignment and the gate/shared_gate creation + moe_params loop):

Current code (lines 175, 189-208):
```python
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

        # --- gate + shared_gate modules (created first, params committed from spec) ---
        gate_cfg = LinearConfig(
            input_dim=hidden,
            output_dim=spec.num_experts(layer),
            data_type=dtype,
            has_bias=mc.expert_router_bias)
        moe.create_child('gate', gate_cfg)

        if mc.moe_shared_gate:
            shared_gate_cfg = LinearConfig(
                input_dim=hidden, output_dim=1, data_type=dtype, has_bias=False)
            moe.create_child('shared_gate', shared_gate_cfg)

        # --- Non-expert MoE parameters (gate/shared_gate weights, score_correction_bias) ---
        for name, (tensor, split_side) in spec.moe_params(layer).items():
            parts = name.split('.')
            target = moe
            for seg in parts[:-1]:
                target = target.child(seg)
            target.commit_tensor(parts[-1], tensor, split_side=split_side)
```

Replace with:
```python
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

        # --- gate linears ---
        for name, linear in spec.moe_gate(layer).items():
            moe.commit_linear(name, linear, model_dtype=dtype)

        # --- non-expert MoE parameters (score_correction_bias, etc.) ---
        for name, (tensor, split_side) in spec.moe_params(layer).items():
            moe.commit_tensor(name, tensor, split_side=split_side)
```

- [ ] **Step 2: Remove unused `LinearConfig` import check**

Check if `LinearConfig` is still used in `text_model_loader.py` after the change:
```bash
grep -n 'LinearConfig' lmdeploy/turbomind/deploy/text_model_loader.py
```

It should still appear in `_load_global` (tok_cfg, out_cfg). If so, keep the import. If not, remove it.

- [ ] **Step 3: Verify import resolves**

```bash
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "from lmdeploy.turbomind.deploy.text_model_loader import TextModelLoader; print('OK')"
```

- [ ] **Step 4: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(deploy): use commit_linear for MoE gate weights

Replace gate/shared_gate pre-creation + commit_tensor with direct
commit_linear calls. Gate Linear bundles from moe_gate() handle child
creation automatically. The moe_params() loop no longer needs
dot-navigation since gate entries are removed."
```

---

### Task 3: Test all MoE models with TP=1 and TP=2

**Files:** None (testing only)

- [ ] **Step 1: Build**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 2: Test all MoE models with TP=1 and TP=2**

Use the turbomind-tester agent to test all MoE models (Qwen3-MoE, Qwen3.5-MoE, GLM4-MoE-Lite, GptOss variants) with both TP=1 and TP=2. Each test should use a prompt requiring at least 128 tokens. Verify responses contain meaningful human words.

- [ ] **Step 3: Fix any failures**

If any model fails, diagnose and fix before proceeding.
