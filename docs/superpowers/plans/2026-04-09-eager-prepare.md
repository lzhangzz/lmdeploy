# Eager Prepare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the lazy `_ensure_ranks` / `_layer_writer` / `_root_distributor` pattern with an explicit `prepare()` call, storing root and layers distributors as eager attributes.

**Architecture:** Add `prepare()` to `TextModelLoader` that eagerly computes ranks and creates distributors once. Call it from `turbomind.py` after `model_comm`/`gpu_count` are set. Delete three lazy-init methods.

**Tech Stack:** Python, pybind11 C++ handles, TurboMind model loading pipeline.

---

### Task 1: Add `prepare()` to `TextModelLoader` and update `__init__`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

- [ ] **Step 1: Update `__init__` to initialize `_root` and `_layers`**

Replace the current `__init__` (lines 31-36):

```python
    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size
        self._attn_ranks = None
        self._mlp_ranks = None
```

With:

```python
    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size
        self._attn_ranks = None
        self._mlp_ranks = None
        self._root = None
        self._layers = None
```

- [ ] **Step 2: Add `prepare()` method**

Insert this method right after `__init__` (before `_ensure_ranks`):

```python
    def prepare(self):
        """Eagerly initialize distributors. Called after model_comm is set."""
        self._attn_ranks = [self.model.tp_ranks(gpu)[0]
                            for gpu in range(self.model.gpu_count)]
        self._mlp_ranks = [self.model.tp_ranks(gpu)[1]
                           for gpu in range(self.model.gpu_count)]
        handles = []
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            handles.append(root)
        self._root = Distributor(handles)
        self._layers = self._root.create_child('layers', ModuleListConfig())
```

- [ ] **Step 3: Verify the file parses**

Run:
```
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "from lmdeploy.turbomind.deploy.text_model_loader import TextModelLoader; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "feat(loader): add prepare() for eager distributor initialization"
```

---

### Task 2: Wire `prepare()` call in `turbomind.py`

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py`

- [ ] **Step 1: Add `prepare()` call after `gpu_count` assignment**

In `turbomind.py`, in the `_from_hf` method, change lines 246-248:

```python
        tm_model.model_comm = model_comm
        tm_model.gpu_count = self.gpu_count
        return model_comm
```

To:

```python
        tm_model.model_comm = model_comm
        tm_model.gpu_count = self.gpu_count
        tm_model.model.prepare()
        return model_comm
```

- [ ] **Step 2: Verify the file parses**

Run:
```
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "from lmdeploy.turbomind.turbomind import TurboMind; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "feat(turbomind): call prepare() after model_comm is set"
```

---

### Task 3: Delete `_ensure_ranks`, `_layer_writer`, `_root_distributor` and update callers

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

- [ ] **Step 1: Delete `_ensure_ranks` method**

Delete the entire `_ensure_ranks` method (lines 38-44):

```python
    def _ensure_ranks(self):
        """Compute per-GPU rank lists lazily (gpu_count may be 0 at __init__ time)."""
        if self._attn_ranks is None:
            self._attn_ranks = [self.model.tp_ranks(gpu)[0]
                                for gpu in range(self.model.gpu_count)]
            self._mlp_ranks = [self.model.tp_ranks(gpu)[1]
                               for gpu in range(self.model.gpu_count)]
```

- [ ] **Step 2: Delete `_layer_writer` method**

Delete the entire `_layer_writer` method (lines 46-58):

```python
    def _layer_writer(self, layer: int) -> Distributor:
        """Create a Distributor for the given layer across all GPUs."""
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
        return Distributor(handles)
```

- [ ] **Step 3: Delete `_root_distributor` method**

Delete the entire `_root_distributor` method (lines 60-68):

```python
    def _root_distributor(self) -> Distributor:
        """Create a Distributor wrapping the root handles from all GPUs."""
        handles = []
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            handles.append(root)
        return Distributor(handles)
```

- [ ] **Step 4: Update `_load_layer` to use `self._layers`**

Replace the `_load_layer` method body. Change from:

```python
    def _load_layer(self, layer: int, spec: 'TextModelSpec'):
        self._ensure_ranks()
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
```

To:

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

        writer = self._layers.create_child(str(layer), DecoderLayerConfig())
```

Two changes: removed `self._ensure_ranks()` call, replaced `self._layer_writer(layer)` with `self._layers.create_child(str(layer), DecoderLayerConfig())`.

- [ ] **Step 5: Update `_load_global` to use `self._root`**

Replace the `_load_global` method body. Change from:

```python
    def _load_global(self, spec: 'TextModelSpec'):
        from .linear import pad_out_dim

        mc = self.model.model_config
        tp = self.attn_tp * self.model.attn_cp_size
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        dtype = _cpp_dtype(mc.data_type)
        hidden = mc.hidden_units

        self._ensure_ranks()
        root = self._root_distributor()
```

To:

```python
    def _load_global(self, spec: 'TextModelSpec'):
        from .linear import pad_out_dim

        mc = self.model.model_config
        tp = self.attn_tp * self.model.attn_cp_size
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        dtype = _cpp_dtype(mc.data_type)
        hidden = mc.hidden_units

        root = self._root
```

Two changes: removed `self._ensure_ranks()` call, replaced `root = self._root_distributor()` with `root = self._root`.

- [ ] **Step 6: Verify the file parses**

Run:
```
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "from lmdeploy.turbomind.deploy.text_model_loader import TextModelLoader; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): replace lazy _ensure_ranks/_layer_writer/_root_distributor with eager prepare()"
```

---

### Task 4: Verify with model tests

**Files:** None (testing only)

- [ ] **Step 1: Check GPU availability**

Use `get_gpu_usage` MCP tool to confirm GPUs are available and not occupied.

- [ ] **Step 2: Test a model with TP=1**

Use the turbomind-tester agent to test a model (e.g., a small Llama or Qwen model) with TP=1.

Prompt should request at least 128 tokens and the response must contain meaningful human words.

This exercises both `_load_global` (tok_embeddings, norm, output via `self._root`) and `_load_layer` (all `_process_*` methods via `self._layers`).

- [ ] **Step 3: Test a model with TP=2 (if GPUs available)**

Use the turbomind-tester agent to test with TP=2.

This exercises the TP sharding path in `Distributor.create_child` and `commit_tensor` — the critical path that changed.

- [ ] **Step 4: Verify responses are meaningful**

Every response must contain coherent, relevant text. Gibberish indicates a bug in the weight loading pipeline.
