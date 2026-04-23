# Spec attn() Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove dead/redundant parameters (`repeat_kv`, `tp_rank=0`, `window_size`) from spec `attn()` methods and config factory functions.

**Architecture:** Three independent cleanup tasks. Each removes a parameter that is either dead (`repeat_kv`) or always-defaulted (`tp_rank=0`, `window_size=0`) from factory signatures and call sites.

**Tech Stack:** Python, pybind11 C++ config structs.

---

### Task 1: Remove `repeat_kv` from the new pipeline

The builder's `pad_for_tp` now handles KV repetition. `repeat_kv` is dead — not used by the C++ runtime. Remove it from the new builder pipeline only (factory functions, specs, `TextModelLoader`, `BaseOutputModel`).

**Files:**
- Modify: `lmdeploy/turbomind/deploy/module_configs.py:42-64,67-98`
- Modify: `lmdeploy/turbomind/deploy/spec.py:23`
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py:49`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:75-79`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:176-180`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:93-97`

- [ ] **Step 1: Remove `repeat_kv` from `make_attention_config` and `make_mla_config`**

In `lmdeploy/turbomind/deploy/module_configs.py`:

Change `make_attention_config` signature (line 42-43) from:
```python
def make_attention_config(mc, *, tp_size, tp_rank, dtype, window_size,
                         rope_dim=0, repeat_kv=0):
```
to:
```python
def make_attention_config(mc, *, tp_size, tp_rank=0, dtype, window_size=0,
                         rope_dim=0):
```
(Note: also defaults `tp_rank=0` and `window_size=0` here to prepare for Tasks 2 and 3.)

Delete line 63: `cfg.repeat_kv = repeat_kv`.

In `make_mla_config` (line 67), change signature from:
```python
def make_mla_config(mc, *, tp_size, tp_rank, dtype, window_size,
                    qk_nope_dim=0):
```
to:
```python
def make_mla_config(mc, *, tp_size, tp_rank=0, dtype, window_size=0,
                    qk_nope_dim=0):
```
(Again also defaults `tp_rank` and `window_size` here.)

Delete line 97: `cfg.repeat_kv = 0`.

- [ ] **Step 2: Remove `_repeat_kv` from spec base class**

In `lmdeploy/turbomind/deploy/spec.py`, delete line 23:
```python
_repeat_kv: int = 0
```

- [ ] **Step 3: Remove `spec._repeat_kv` injection in TextModelLoader**

In `lmdeploy/turbomind/deploy/text_model_loader.py`, delete line 49:
```python
spec._repeat_kv = self.model.repeat_kv
```

- [ ] **Step 4: Remove `repeat_kv=self._repeat_kv` from all 3 spec calls**

In `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`, change lines 75-79 from:
```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            window_size=window_size,
            rope_dim=self._rope_dim,
            repeat_kv=self._repeat_kv)
```
to:
```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, dtype=dtype,
            rope_dim=self._rope_dim)
```

In `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`, change lines 176-180 from:
```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            window_size=window_size,
            rope_dim=self._rope_dim,
            repeat_kv=self._repeat_kv)
```
to:
```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, dtype=dtype,
            rope_dim=self._rope_dim)
```

In `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`, change lines 93-97 from:
```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            window_size=window_size,
            rope_dim=self._rope_dim,
            repeat_kv=self._repeat_kv)
```
to:
```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, dtype=dtype,
            window_size=window_size,
            rope_dim=self._rope_dim)
```
(Keep `window_size=window_size` here since gpt_oss passes non-zero values.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: remove repeat_kv from new builder pipeline"
```

---

### Task 2: Default `tp_rank=0` in all factory functions, remove from spec calls

The builder's `_ensure_handles()` overrides `tp_rank` with the correct rank when `tp > 1`. Specs always pass `tp_rank=0`. Default it in the factory signatures and remove from all call sites.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/module_configs.py:106-168`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:110-111,133-134`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:242-243,265-266,363-364`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:124-125,147-148,246-247`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:72-73,108-109,131-132`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:202-203`

- [ ] **Step 1: Default `tp_rank=0` in remaining factory functions**

In `lmdeploy/turbomind/deploy/module_configs.py`:

Change `make_ffn_config` (line 106) from:
```python
def make_ffn_config(mc, *, tp_size, tp_rank, dtype, act_type,
                    fuse_silu, inter_size=None, fused_moe=False):
```
to:
```python
def make_ffn_config(mc, *, tp_size, tp_rank=0, dtype, act_type,
                    fuse_silu, inter_size=None, fused_moe=False):
```

Change `make_moe_config` (line 122) from:
```python
def make_moe_config(mc, *, layer_id, tp_size, tp_rank, dtype,
                    act_type, fuse_silu, expert_num):
```
to:
```python
def make_moe_config(mc, *, layer_id, tp_size, tp_rank=0, dtype,
                    act_type, fuse_silu, expert_num):
```

Change `make_deltanet_config` (line 155) from:
```python
def make_deltanet_config(mc, *, tp_size, tp_rank, dtype):
```
to:
```python
def make_deltanet_config(mc, *, tp_size, tp_rank=0, dtype):
```

(Note: `make_attention_config` and `make_mla_config` already got `tp_rank=0` in Task 1.)

- [ ] **Step 2: Remove `tp_rank=0` from all spec call sites**

In every spec file, remove `tp_rank=0,` from every config factory call. The calls that need changing:

**qwen3_spec.py:**
- Line 110-111 `make_ffn_config(mc, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_ffn_config(mc, tp_size=tp, dtype=dtype,`
- Line 133-134 `make_moe_config(mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_moe_config(mc, layer_id=layer, tp_size=tp, dtype=dtype,`

**qwen3_5_spec.py:**
- Line 202-203 `make_deltanet_config(mc, tp_size=tp, tp_rank=0, dtype=dtype)` -> `make_deltanet_config(mc, tp_size=tp, dtype=dtype)`
- Line 242-243 `make_ffn_config(mc, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_ffn_config(mc, tp_size=tp, dtype=dtype,`
- Line 265-266 `make_moe_config(mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_moe_config(mc, layer_id=layer, tp_size=tp, dtype=dtype,`
- Line 363-364 `make_ffn_config(mc, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_ffn_config(mc, tp_size=tp, dtype=dtype,`

**gpt_oss_spec.py:**
- Line 124-125 `make_ffn_config(mc, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_ffn_config(mc, tp_size=tp, dtype=dtype,`
- Line 147-148 `make_moe_config(mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_moe_config(mc, layer_id=layer, tp_size=tp, dtype=dtype,`
- Line 246-247 `make_ffn_config(mc, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_ffn_config(mc, tp_size=tp, dtype=dtype,`

**glm4_moe_lite_spec.py:**
- Line 72-73 `make_mla_config(mc, tp_size=tp, tp_rank=0, dtype=dtype, window_size=0,` -> `make_mla_config(mc, tp_size=tp, dtype=dtype,`
- Line 108-109 `make_ffn_config(mc, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_ffn_config(mc, tp_size=tp, dtype=dtype,`
- Line 131-132 `make_moe_config(mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,` -> `make_moe_config(mc, layer_id=layer, tp_size=tp, dtype=dtype,`

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: default tp_rank=0 in factory functions, remove from spec calls"
```

---

### Task 3: Remove `window_size` boilerplate from specs that never use it

Only `gpt_oss_spec.py` passes non-zero `window_size` values. Default `window_size=0` in factory functions (already done in Task 1 for `make_attention_config` and `make_mla_config`). Remove the 4-line per-layer lookup from `qwen3_spec.py` and `qwen3_5_spec.py`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:70-73`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:171-174`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:88-97`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:72-73`

- [ ] **Step 1: Remove window_size lookup and arg from `qwen3_spec.py`**

In `qwen3_spec.py`, delete lines 70-73 (the window_size lookup block):
```python
        window_size = 0
        ws_list = mc.window_size
        if ws_list and layer < len(ws_list):
            window_size = ws_list[layer]
```

The `make_attention_config` call (already cleaned in Task 1) should look like:
```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, dtype=dtype,
            rope_dim=self._rope_dim)
```
(No `window_size` arg — it defaults to 0 in the factory.)

- [ ] **Step 2: Remove window_size lookup and arg from `qwen3_5_spec.py`**

In `qwen3_5_spec.py`, delete lines 171-174 (the window_size lookup block):
```python
        window_size = 0
        ws_list = mc.window_size
        if ws_list and layer < len(ws_list):
            window_size = ws_list[layer]
```

The `make_attention_config` call (already cleaned in Task 1) should look like:
```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, dtype=dtype,
            rope_dim=self._rope_dim)
```
(No `window_size` arg.)

- [ ] **Step 3: Remove `window_size=window_size` from `gpt_oss_spec.py` call (keep lookup)**

In `gpt_oss_spec.py`, the per-layer lookup (lines 88-91) stays. The `make_attention_config` call from Task 1 already has `window_size=window_size` — no change needed here.

- [ ] **Step 4: Remove `window_size=0` from `glm4_moe_lite_spec.py` call**

Already cleaned in Task 1 (the `make_mla_config` call lost `window_size=0` since it's the default). Verify no `window_size` arg remains.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: remove window_size boilerplate from specs that never use it"
```

---

### Task 4: Clean up `repeat_kv` from the old pipeline plumbing

Remove `repeat_kv` from `BaseOutputModel.finalize_config`, `BaseOutputModel.__init__`, `converter.get_tm_config`, and `turbomind.py`. Also remove the dead `repeat_kv` property from `load_context.py`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py:27-76,78-90`
- Modify: `lmdeploy/turbomind/deploy/converter.py:137-139,204-206`
- Modify: `lmdeploy/turbomind/turbomind.py:237-252`
- Modify: `lmdeploy/turbomind/deploy/load_context.py:381,418-419`

- [ ] **Step 1: Remove `repeat_kv` from `finalize_config` return**

In `lmdeploy/turbomind/deploy/target_model/base.py`:

Change docstring (line 30) from:
```python
        Returns repeat_kv (int).
```
to:
```python
        Returns None.
```

Delete the `repeat_kv` computation block (lines 59-64):
```python
        # Handle repeat_kv
        assert mc.head_num % attn_tp == 0
        repeat_kv = 0
        if attn_tp > mc.kv_head_num and attn_tp % mc.kv_head_num == 0:
            repeat_kv = attn_tp // mc.kv_head_num
            mc.kv_head_num = attn_tp
```

**Important:** Keep the `mc.kv_head_num = attn_tp` mutation — the config still needs the padded value. Replace the block with:
```python
        # Pad kv_head_num to tp-divisible
        assert mc.head_num % attn_tp == 0
        if attn_tp > mc.kv_head_num and attn_tp % mc.kv_head_num == 0:
            mc.kv_head_num = attn_tp
```

Change the return (line 76) from:
```python
        return repeat_kv
```
to:
```python
        return
```

- [ ] **Step 2: Remove `repeat_kv` from `BaseOutputModel.__init__`**

In `lmdeploy/turbomind/deploy/target_model/base.py`, change `__init__` signature (line 78-79) from:
```python
    def __init__(self, input_model, cfg, model_cls,
                 model_comm, gpu_count, *, repeat_kv):
```
to:
```python
    def __init__(self, input_model, cfg, model_cls,
                 model_comm, gpu_count):
```

Delete line 90: `self.repeat_kv = repeat_kv`.

- [ ] **Step 3: Remove `repeat_kv` from `converter.py`**

In `lmdeploy/turbomind/deploy/converter.py`:

Change `get_tm_config` docstring (line 139) from:
```python
        tuple: (input_model, tm_cfg, repeat_kv)
```
to:
```python
        tuple: (input_model, tm_cfg)
```

Change lines 204-206 from:
```python
    repeat_kv = BaseOutputModel.finalize_config(input_model, tm_cfg)

    return input_model, tm_cfg, repeat_kv
```
to:
```python
    BaseOutputModel.finalize_config(input_model, tm_cfg)

    return input_model, tm_cfg
```

- [ ] **Step 4: Remove `repeat_kv` from `turbomind.py`**

In `lmdeploy/turbomind/turbomind.py`, change lines 237-252 from:
```python
        input_model, tm_cfg, repeat_kv = get_tm_config(
            model_path, self.model_name, self.chat_template_name, engine_config)

        self._postprocess_config(tm_cfg, engine_config)

        model_comm = _tm.TurboMind.create(model_dir='',
                                          config=yaml.safe_dump(self.config_dict))
        self._create_weight(model_comm)

        self._tm_model = OUTPUT_MODELS.get('tm')(
            input_model=input_model,
            cfg=tm_cfg,
            model_cls=TextModelLoader,
            model_comm=model_comm,
            gpu_count=self.gpu_count,
            repeat_kv=repeat_kv)
```
to:
```python
        input_model, tm_cfg = get_tm_config(
            model_path, self.model_name, self.chat_template_name, engine_config)

        self._postprocess_config(tm_cfg, engine_config)

        model_comm = _tm.TurboMind.create(model_dir='',
                                          config=yaml.safe_dump(self.config_dict))
        self._create_weight(model_comm)

        self._tm_model = OUTPUT_MODELS.get('tm')(
            input_model=input_model,
            cfg=tm_cfg,
            model_cls=TextModelLoader,
            model_comm=model_comm,
            gpu_count=self.gpu_count)
```

- [ ] **Step 5: Remove `repeat_kv` property from `load_context.py`**

In `lmdeploy/turbomind/deploy/load_context.py`:

Change the docstring (line 381) from:
```python
                       rope_dim, permute_qk, repeat_kv, attn_output_gate,
```
to:
```python
                       rope_dim, permute_qk, attn_output_gate,
```

Delete lines 417-419:
```python
    @property
    def repeat_kv(self) -> int:
        return self._tp_config.get('repeat_kv', 0)
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor: remove repeat_kv from old pipeline plumbing"
```

---

### Task 5: Verify with model tests

Run model tests to verify the refactoring didn't break anything.

- [ ] **Step 1: Check GPU availability**

Run: check `get_gpu_usage` MCP tool for empty GPUs.

- [ ] **Step 2: Test Qwen3-4B TP=1**

Run:
```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3-4B --tp 1
```
Expected: Model responds with meaningful text.

- [ ] **Step 3: Test Qwen3-4B TP=2**

Run:
```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3-4B --tp 2
```
Expected: Model responds with meaningful text.

- [ ] **Step 4: Test gpt-oss-20b TP=1**

Run:
```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py gpt-oss-20b --tp 1
```
Expected: Model responds with meaningful text.

- [ ] **Step 5: Test Qwen3.5-27B TP=1**

Run:
```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen3.5-27B --tp 1
```
Expected: Model responds with meaningful text.
