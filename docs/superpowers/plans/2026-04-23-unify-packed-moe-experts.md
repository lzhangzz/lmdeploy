# Unify Packed MoE Expert Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ad-hoc packed-MoE-expert helper chains in `qwen3_5_spec.py` and `gpt_oss_spec.py` with a single free function `read_packed_moe_expert` in `source_model/utils.py`, parameterized on two bool kwargs (`interleaved`, `trans`) that cover both specs' variation axes.

**Architecture:** A new free function in `utils.py` does `build_linear(index=e)` + optional trivial-layout `.t()` fixup + split (contiguous or stride-2 interleaved) + Linear wrap, returning `(w1, w2, w3)`. Each spec's packed-expert method shrinks to one call plus the standard FfnBuilder ceremony. qwen3_5 keeps its unpacked→packed fallback via a compact `or`-expression.

**Tech Stack:** Python, PyTorch, TurboMind deploy pipeline

---

### Task 1: Add `read_packed_moe_expert` to `source_model/utils.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py:12` (extend kind_map import)
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py:13` (insert linear import)
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py` (append function at EOF; file currently ends at line 260)

- [ ] **Step 1: Extend the `..kind_map` import**

Line 12 before:
```python
from ..kind_map import TRIVIAL_FORMAT
```

After:
```python
from ..kind_map import TRIVIAL_FORMAT, build_linear
```

- [ ] **Step 2: Add a top-level `Linear` import**

Insert a new import line immediately after line 12 (the extended `..kind_map` line), before line 13 (`from ..builder._base import _dequant_linear`):

```python
from ..linear import Linear
```

The resulting block at lines 12-14 should read:
```python
from ..kind_map import TRIVIAL_FORMAT, build_linear
from ..linear import Linear
from ..builder._base import _dequant_linear
```

No circular-import risk: `linear.py` imports only `torch`, `dataclasses`, and `_turbomind` — none of the `source_model` tree.

- [ ] **Step 3: Append the `read_packed_moe_expert` function at end of file**

Append the following function at the end of `utils.py` (after `layer_progress`, which currently ends the file):

```python
def read_packed_moe_expert(
    params: dict,
    gate_up_pfx: str,
    down_pfx: str,
    expert_idx: int,
    *,
    data_type,
    weight_format,
    interleaved: bool = False,
    trans: bool = False,
) -> tuple[Linear, Linear, Linear]:
    """Read one packed MoE expert's fused gate_up + down and split into
    (w1, w2, w3) Linears in TM layout.

    ``gate_up_pfx`` and ``down_pfx`` are the full prefixes to the two
    packed tensors (e.g. ``'model.layers.5.mlp.experts.gate_up_proj'``).
    The caller composes these strings; this helper concatenates nothing.

    Parameters
    ----------
    interleaved : bool
        Split scheme for the fused gate_up output dim.
        ``False`` -> contiguous ``[..., :half]`` / ``[..., half:]`` (qwen3.5).
        ``True``  -> stride-2 interleaved ``[..., ::2]`` / ``[..., 1::2]`` (gpt-oss).
    trans : bool
        For trivial-format checkpoints that store the packed tensor in
        ``[n_experts, in, out]`` layout (gpt-oss), transposes the 2D
        ``weight`` tensor to undo the HF-to-TM transpose applied by
        ``_normalize_trivial``. Only affects the ``weight`` kind on
        trivial-format linears; quantized formats use their own normalizers.
    """
    gate_up = build_linear(params, gate_up_pfx, index=expert_idx,
                           data_type=data_type, weight_format=weight_format)
    down    = build_linear(params, down_pfx, index=expert_idx,
                           data_type=data_type, weight_format=weight_format)

    if trans:
        for lin in (gate_up, down):
            if lin.weight_format.name == 'trivial':
                w = lin.tensors.get('weight')
                if w is not None and w.dim() == 2:
                    lin.tensors['weight'] = w.t().contiguous()

    w1_t: dict[str, torch.Tensor] = {}
    w3_t: dict[str, torch.Tensor] = {}
    for kind, t in gate_up.tensors.items():
        if interleaved:
            w1_t[kind] = t[..., ::2].contiguous()
            w3_t[kind] = t[..., 1::2].contiguous()
        else:
            half = t.shape[-1] // 2
            w1_t[kind] = t[..., :half].contiguous()
            w3_t[kind] = t[..., half:].contiguous()
    w1 = Linear(tensors=w1_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)
    w3 = Linear(tensors=w3_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)
    return w1, down, w3
```

- [ ] **Step 4: Sanity-check the module imports cleanly**

```bash
python -c "from lmdeploy.turbomind.deploy.source_model.utils import read_packed_moe_expert; print('OK')"
```

Expected output: `OK`. A `NameError` or `ImportError` here means Step 1 or 2 was incomplete.

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/utils.py
git commit -m "$(cat <<'EOF'
deploy: add read_packed_moe_expert helper for packed MoE experts

Single free function that reads a packed MoE expert's fused gate_up +
down via build_linear(index=e), optionally applies a trivial-layout
transpose fix, and splits the fused gate_up either contiguously or with
stride-2 interleaving. Returns (w1, w2, w3) Linears. Callers own the
two full prefix strings.
EOF
)"
```

---

### Task 2: Refactor `qwen3_5_spec.py` to use the helper

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:7` (drop `torch` import)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:16` (drop `build_linear` import)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:17` (drop `Linear` import)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:20` (add `read_packed_moe_expert`)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:274-315` (replace `_moe_expert_ffn` + delete `_packed_moe_expert_indexed`)

- [ ] **Step 1: Drop the `torch` import**

Line 7 before:
```python
import torch
```

After: delete the line entirely. After Step 5 (deletion of `_packed_moe_expert_indexed`), `torch` has no remaining uses in this file — its only current use is the `dict[str, torch.Tensor]` type annotations inside the deleted method.

- [ ] **Step 2: Drop the `build_linear` import**

Line 16 before:
```python
from ..kind_map import build_linear
```

After: delete the line entirely. After Step 5 (deletion of `_packed_moe_expert_indexed`), `build_linear` has no remaining uses in this file — all other reads go through `self._linear(...)`, which does the `build_linear` call internally.

- [ ] **Step 3: Drop the `Linear` import**

Line 17 before:
```python
from ..linear import Linear
```

After: delete the line entirely. `Linear` is no longer referenced in this file once `_packed_moe_expert_indexed` is deleted in Step 5.

- [ ] **Step 4: Add `read_packed_moe_expert` to the utils import**

The `from .utils import` line is currently line 20 but shifts upward with each of Steps 1-3's deletions — locate by content, not number. Before:
```python
from .utils import layer_progress, reorder_rotary_emb
```

After:
```python
from .utils import layer_progress, read_packed_moe_expert, reorder_rotary_emb
```

- [ ] **Step 5: Replace `_moe_expert_ffn` and delete `_packed_moe_expert_indexed`**

Lines 274-315 before (the two methods together):
```python
    def _moe_expert_ffn(self, pfx, layer, expert_idx, inter_size):
        expert_pfx = f'{pfx}.experts.{expert_idx}'
        result = self.ffn(expert_pfx, layer,
                          inter_size=inter_size, fused_moe=True)
        if result is not None:
            return result
        packed_pfx = f'{pfx}.experts'
        return self._packed_moe_expert_indexed(packed_pfx, expert_idx, inter_size)

    def _packed_moe_expert_indexed(self, pfx, expert_idx, inter_size):
        gate_up_lin = build_linear(self.params, f'{pfx}.gate_up_proj',
                                   index=expert_idx,
                                   data_type=self._cpp_dtype(),
                                   weight_format=self._weight_format)
        down_lin = build_linear(self.params, f'{pfx}.down_proj',
                                index=expert_idx,
                                data_type=self._cpp_dtype(),
                                weight_format=self._weight_format)
        if gate_up_lin is None or down_lin is None:
            return None

        gate_tensors: dict[str, torch.Tensor] = {}
        up_tensors: dict[str, torch.Tensor] = {}
        for kind, t in gate_up_lin.tensors.items():
            half = t.shape[-1] // 2
            gate_tensors[kind] = t[..., :half].contiguous()
            up_tensors[kind] = t[..., half:].contiguous()

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.fuse_silu  = False
        cfg.fused_moe  = True

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        w1 = Linear(tensors=gate_tensors, weight_format=gate_up_lin.weight_format,
                    data_format=gate_up_lin.data_format)
        w3 = Linear(tensors=up_tensors,   weight_format=gate_up_lin.weight_format,
                    data_format=gate_up_lin.data_format)
        m.add_ffn(w1, down_lin, w3)
        return m
```

After:
```python
    def _packed_moe_ffn(self, mlp_pfx, expert_idx, inter_size):
        w1, w2, w3 = read_packed_moe_expert(
            self.params,
            f'{mlp_pfx}.experts.gate_up_proj',
            f'{mlp_pfx}.experts.down_proj',
            expert_idx,
            data_type=self._cpp_dtype(),
            weight_format=self._weight_format,
        )
        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.fuse_silu  = False
        cfg.fused_moe  = True
        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def _moe_expert_ffn(self, mlp_pfx, layer, expert_idx, inter_size):
        expert_pfx = f'{mlp_pfx}.experts.{expert_idx}'
        return (self.ffn(expert_pfx, layer, inter_size=inter_size, fused_moe=True)
                or self._packed_moe_ffn(mlp_pfx, expert_idx, inter_size))
```

Note: the `pfx` positional arg was renamed to `mlp_pfx` for clarity (it is the `.mlp` prefix, not the experts prefix). The `moe()` call site passes the same value — no change needed to `moe()`.

The `or`-fallback preserves today's exact semantics: `self.ffn(...)` returns a truthy `FfnBuilder` when any of `gate_proj` / `up_proj` / `down_proj` exists at the per-expert prefix, otherwise it returns `None` and the packed path fires.

- [ ] **Step 6: Sanity-check the file imports and has no stale references**

```bash
python -c "from lmdeploy.turbomind.deploy.source_model import qwen3_5_spec; print('OK')"
```
Expected output: `OK`. A `NameError` for `Linear`, `build_linear`, or `torch` means an edit was incomplete.

```bash
rg -n '_packed_moe_expert_indexed' lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
```
Expected: zero matches.

```bash
rg -n '^import torch|^from \.\.kind_map import build_linear|^from \.\.linear import Linear' lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
```
Expected: zero matches — all three dropped imports are gone.

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "$(cat <<'EOF'
deploy: collapse qwen3_5 packed MoE experts via read_packed_moe_expert

Replaces the 33-line _packed_moe_expert_indexed helper with a call to
the shared read_packed_moe_expert utility. The unpacked->packed fallback
is preserved via a compact `or`-expression in place of the explicit None
ladder. Drops now-unused torch, build_linear, and Linear imports.
EOF
)"
```

---

### Task 3: Refactor `gpt_oss_spec.py` to use the helper

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:7` (drop `torch` import)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:15` (drop `build_linear` import)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:16` (drop `Linear` import)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:19` (add `read_packed_moe_expert`)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:191-193` (update expert-loop call)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:209-258` (delete three helpers, add `_packed_moe_ffn`)

- [ ] **Step 1: Drop the `torch` import**

Line 7 before:
```python
import torch
```

After: delete the line entirely. After Step 6 (deletion of `_deinterleave`), `torch` has no remaining uses in this file — its only current use is the `torch.Tensor` type annotation inside `_deinterleave`.

- [ ] **Step 2: Drop the `build_linear` import**

Line 15 before:
```python
from ..kind_map import build_linear
```

After: delete the line entirely. After Step 6 (deletion of `_read_packed_expert`), `build_linear` has no remaining uses in this file — all other reads go through `self._linear(...)`.

- [ ] **Step 3: Drop the `Linear` import**

Line 16 before:
```python
from ..linear import Linear
```

After: delete the line entirely. After Step 6 (deletion of `_deinterleave`), `Linear` has no remaining uses.

- [ ] **Step 4: Add `read_packed_moe_expert` to the utils import**

Line 19 (note: three deletions above will shift this line — locate by content, not number) before:
```python
from .utils import layer_progress, reorder_rotary_emb
```

After:
```python
from .utils import layer_progress, read_packed_moe_expert, reorder_rotary_emb
```

- [ ] **Step 5: Update the expert loop in `moe()`**

Lines 191-193 before (locate by content, line numbers shift with deletions above):
```python
        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self._packed_expert_ffn(
                f'{pfx}.experts.{e}', self._expert_inter_size)
```

After:
```python
        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self._packed_moe_ffn(
                pfx, e, self._expert_inter_size)
```

The old call passed `f'{pfx}.experts.{e}'` as a single composed string which `_packed_expert_ffn` then re-parsed via `rsplit('.', 1)`. The new call passes the MLP prefix and expert id separately — no string parsing.

- [ ] **Step 6: Delete three helpers, add `_packed_moe_ffn`**

Lines 209-258 before (all three methods, including the section-header comment block):
```python
    # ------------------------------------------------------------------
    # Packed-expert decoding (gate_up interleaved, TM layout)
    # ------------------------------------------------------------------

    def _read_packed_expert(self, prefix: str, expert: int):
        lin = build_linear(self.params, prefix, index=expert,
                           data_type=self._cpp_dtype(),
                           weight_format=self._weight_format)
        if lin is None:
            return None
        if lin.weight_format.name == 'trivial':
            w = lin.tensors.get('weight')
            if w is not None and w.dim() == 2:
                lin.tensors['weight'] = w.t().contiguous()
        return lin

    @staticmethod
    def _deinterleave(lin: Linear):
        gate_t: dict[str, torch.Tensor] = {}
        up_t: dict[str, torch.Tensor] = {}
        for kind, t in lin.tensors.items():
            gate_t[kind] = t[..., ::2].contiguous()
            up_t[kind]   = t[..., 1::2].contiguous()
        return (Linear(tensors=gate_t, weight_format=lin.weight_format,
                       data_format=lin.data_format),
                Linear(tensors=up_t,   weight_format=lin.weight_format,
                       data_format=lin.data_format))

    def _packed_expert_ffn(self, expert_pfx: str, expert_inter: int):
        base_pfx = expert_pfx.rsplit('.', 1)[0]
        expert_id = int(expert_pfx.rsplit('.', 1)[1])
        gate_up_lin = self._read_packed_expert(
            f'{base_pfx}.gate_up_proj', expert_id)
        down_lin = self._read_packed_expert(
            f'{base_pfx}.down_proj', expert_id)
        if gate_up_lin is None or down_lin is None:
            return None

        w1, w3 = self._deinterleave(gate_up_lin)

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = expert_inter
        cfg.fuse_silu  = False
        cfg.fused_moe  = True

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, down_lin, w3)
        return m
```

After:
```python
    def _packed_moe_ffn(self, mlp_pfx, expert_idx, inter_size):
        w1, w2, w3 = read_packed_moe_expert(
            self.params,
            f'{mlp_pfx}.experts.gate_up_proj',
            f'{mlp_pfx}.experts.down_proj',
            expert_idx,
            data_type=self._cpp_dtype(),
            weight_format=self._weight_format,
            interleaved=True,
            trans=True,
        )
        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.fuse_silu  = False
        cfg.fused_moe  = True
        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m
```

The section-header comment block is removed — the helper's name is self-documenting and the inline `interleaved=True, trans=True` kwargs make the gpt-oss specifics visible at the call site.

- [ ] **Step 7: Sanity-check the file imports and has no stale references**

```bash
python -c "from lmdeploy.turbomind.deploy.source_model import gpt_oss_spec; print('OK')"
```
Expected output: `OK`.

```bash
rg -n '_read_packed_expert|_deinterleave|_packed_expert_ffn' lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
```
Expected: zero matches — all three old helpers are gone.

```bash
rg -n '^import torch|^from \.\.linear import Linear|^from \.\.kind_map import build_linear' lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
```
Expected: zero matches — all three dropped imports are gone.

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "$(cat <<'EOF'
deploy: collapse gpt-oss packed MoE experts via read_packed_moe_expert

Replaces the _read_packed_expert / _deinterleave / _packed_expert_ffn
chain with a single call to the shared read_packed_moe_expert utility,
using interleaved=True for the stride-2 gate/up split and trans=True
for the trivial-layout transpose fixup. Drops now-unused torch, Linear,
and build_linear imports.
EOF
)"
```

---

### Task 4: Smoke-test the two affected specs

**Files:** None (verification only).

There are no unit tests for individual specs in this codebase; verification is end-to-end model inference through `scripts/test_turbomind_model.py`. Both packed-expert specs must be exercised because the two bool kwargs (`interleaved`, `trans`) select disjoint code paths.

- [ ] **Step 1: Check GPU availability**

Use the `get_gpu_usage` MCP tool to find an empty GPU. If none are free, wait or skip to a quieter time.

- [ ] **Step 2: Pick one model per packed-expert spec**

Use the `list_models` MCP tool to find one cached model per spec:

| Spec file           | Matching `model_type` (in `config.json`)              | Path exercised                |
|---------------------|-------------------------------------------------------|-------------------------------|
| `qwen3_5_spec.py`   | `qwen3_next_text` / `qwen3_5_moe` (whichever is registered) | `interleaved=False, trans=False` (contiguous split, trivial layout) |
| `gpt_oss_spec.py`   | `gpt_oss` with mxfp4 weights (native gpt-oss release) | `interleaved=True` on mxfp4 normalizer (no `trans` fixup — only applies to trivial) |
| `gpt_oss_spec.py`   | `gpt_oss` with BF16/trivial weights (if a dequantized variant is cached) | `interleaved=True, trans=True` on the trivial path |

If only one gpt-oss variant is cached, use it. The mxfp4 variant still exercises `interleaved=True` and goes through the shared helper; the `trans` branch is no-op for mxfp4 because `lin.weight_format.name == 'mxfp4'`, not `'trivial'`.

Prefer the smallest variant per spec that fits on the available GPU.

- [ ] **Step 3: Run one model test per selected model**

For each selected model, use the `get_model_cache_path` MCP tool to find its cache dir, then run:

```bash
python scripts/test_turbomind_model.py <model_path> <cache_dir> <tp> <gpus>
```

For each run, verify:
- The conversion completes without errors.
- The `--- response begin ---` block contains meaningful human text relevant to the prompt (not gibberish, not empty, not truncated).
- Generated token count (the `generated:` line in `--- tokens ---`) is at least 128.

Any gibberish, truncation, or crash indicates a regression. Return to Task 1, 2, or 3 and debug — do not stop with active bugs (per `AGENTS.md`'s debugging section).

- [ ] **Step 4: Commit only if a fix was needed**

If no regressions: no commit for this task.

If a fix was required during testing, commit it now:
```bash
git add <files-that-were-fixed>
git commit -m "deploy: fix <brief description of issue found in Task 4>"
```
