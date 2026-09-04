# TurboMind AWQ wrapper and Linear API promotion

## Status

Draft. Do not implement without explicit approval.

## Goal

Make `lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py` a thin compatibility adapter over the Torch-facing TurboMind `Linear` API.

Promote that API from the test tree to `lmdeploy/turbomind/linear.py`. Move the current build-time linear-weight bundle from that path to `lmdeploy/turbomind/builders/linear.py`, where it belongs.

The final ownership is:

```text
lmdeploy/turbomind/builders/linear.py
    Build-time Linear tensor bundle and its padding/fusion transforms.

lmdeploy/turbomind/linear.py
    Torch-facing Linear, WeightPlan, Weight, and ExecPlan API.

lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py
    AWQ interface compatibility only.
```

## Non-goals

- No C++ or CUDA changes.
- No GEMM-family, kernel-selection, packing, or numerical changes.
- No new validation or defensive checks.
- No Python execution-plan cache.
- No AWQ-specific context, allocator, stream, event, or runtime cache.
- No compatibility alias for the old build-time module location.
- No new tests or persistent verification helpers.

## 1. Move the build-time linear bundle into `builders`

Move the complete current contents of:

```text
lmdeploy/turbomind/linear.py
```

to:

```text
lmdeploy/turbomind/builders/linear.py
```

Keep the existing build-time `Linear` class and all existing transform functions unchanged. This is a file ownership change, not a class rename.

The one function-local import in `_dequant_linear` changes from:

```python
from .builders._base import _torch_dtype_to_cpp
from .weight_format import TrivialFormat
```

to:

```python
from ._base import _torch_dtype_to_cpp
from ..weight_format import TrivialFormat
```

The type-only import changes from:

```python
if TYPE_CHECKING:
    from .weight_format import WeightFormat
```

to:

```python
if TYPE_CHECKING:
    from ..weight_format import WeightFormat
```

Do not re-export this internal `Linear` from `lmdeploy.turbomind.builders.__init__`. Callers import it from the module that owns it.

### 1.1 Builder imports

The following files change their build-time bundle imports from `..linear` to `.linear`:

- `lmdeploy/turbomind/builders/_base.py`
- `lmdeploy/turbomind/builders/attention.py`
- `lmdeploy/turbomind/builders/deltanet.py`
- `lmdeploy/turbomind/builders/ffn.py`
- `lmdeploy/turbomind/builders/mla.py`
- `lmdeploy/turbomind/builders/text_model.py`

For example:

```python
from .linear import Linear, dequant_mixed, transform_output_dim
```

Each file imports only the names it already uses.

### 1.2 Model imports

The following files change their build-time bundle imports from `..linear` to `..builders.linear`:

- `lmdeploy/turbomind/models/internlm2.py`
- `lmdeploy/turbomind/models/internvl.py`
- `lmdeploy/turbomind/models/qwen2_vl.py`
- `lmdeploy/turbomind/models/qwen3_5.py`
- `lmdeploy/turbomind/models/utils.py`
- `lmdeploy/turbomind/models/vision_utils.py`

For example:

```python
from ..builders.linear import Linear, transform_input_dim, transform_output_dim
```

Again, each file imports only the names it already uses.

The type-only imports in `lmdeploy/turbomind/text_model.py` and `lmdeploy/turbomind/vision_model.py` become:

```python
if TYPE_CHECKING:
    from .builders.linear import Linear
```

### 1.3 `weight_format.py` dependency boundary

`lmdeploy/turbomind/weight_format.py` must not import the `builders` package merely to define format descriptors such as `AWQFormat`. `WeightFormatResolver` selects a format and collects its checkpoint tensors; it does not construct the build-time `Linear` bundle.

Remove the runtime module-level import:

```python
from .linear import Linear
```

No replacement `Linear` import is added. The existing `WeightFormatResolver.resolve()` becomes:

```python
def resolve(self, pfx, *, index: int | None = None, optional: bool = False) -> tuple[WeightFormat, dict[str, Tensor]] | None:
    """Resolve the selected format and its raw checkpoint tensors."""
    read = pfx.get if index is not None else pfx.pop
    available = {s: read(s, sep="", index=index) for s in self._suffixes if pfx.has(s, sep="")}

    if not available:
        if optional:
            return None
        raise KeyError(f"no checkpoint tensors found at prefix {pfx.prefix!r} (candidate suffixes: {sorted(self._suffixes)})")

    for fmt in self._formats:
        if fmt.accepts(available):
            return fmt, available

    raise ValueError(f"no weight format accepts tensors at {pfx.prefix!r}: got {sorted(available)}, tried {[f.name for f in self._formats]}")
```

Delete `WeightFormatResolver._build_linear()`.

Update the module documentation to describe `resolve()` as returning `(WeightFormat, available)` rather than a build-time `Linear`.

### 1.4 Builder-side construction

Add the construction function to `lmdeploy/turbomind/builders/linear.py`, next to the build-time `Linear` bundle it creates:

```python
def _build_linear(fmt: WeightFormat, available: dict[str, Tensor]) -> Linear:
    """Normalize resolved checkpoint tensors into a build-time Linear."""

    tensors = {kind: fmt.normalize(available[s], kind) for s, kind in fmt.suffix_map.items() if s in available}
    if fmt.zeros_dtype != _tm.DataType.TYPE_INVALID and "zeros" not in tensors:
        tensors["zeros"] = fmt.synthesize_zeros(tensors["scales"])
    if "zeros" in tensors:
        tensors["zeros"] = tensors["zeros"].to(tensors["scales"].dtype)
    return Linear(tensors=tensors, weight_format=fmt)
```

The module adds the corresponding imports:

```python
from .. import _tm

if TYPE_CHECKING:
    from ..weight_format import WeightFormat
```

The model-loading call sites explicitly perform the construction after resolution. `TextModel._linear()` and `VisionModel._linear()` become:

```python
def _linear(self, pfx: Prefix, *, optional: bool = False) -> Linear | None:
    resolved = self._resolver.resolve(pfx, optional=optional)
    if resolved is None:
        return None
    return _build_linear(*resolved)
```

Both modules import `_build_linear` from their builder-side owner:

```python
from .builders.linear import _build_linear
```

`read_packed_moe_expert()` resolves and constructs its two required bundles directly:

```python
gate_up = _build_linear(*resolver.resolve(gate_up_pfx, index=expert_idx))
down = _build_linear(*resolver.resolve(down_pfx, index=expert_idx))
```

`lmdeploy/turbomind/models/utils.py` imports `Linear`, `_build_linear`, and `_dequant_linear` from `..builders.linear`.

This keeps format selection independent of builders, while every build-time `Linear` is created by builder-owned code. The AWQ wrapper can import `AWQFormat` without loading the builders package.

## 2. Promote the Torch-facing Linear API

Move:

```text
tests/turbomind/linear/linear.py
```

to:

```text
lmdeploy/turbomind/linear.py
```

The implementation remains the single source of truth. Do not retain a forwarding module or copy under `tests/`.

Add the repository copyright header, then use package-relative production imports at the top of the promoted module:

```python
# Copyright (c) OpenMMLab. All rights reserved.

from . import _tm

if TYPE_CHECKING:
    from .weight_format import WeightFormat
```

Keep the four plan/weight/execution classes and add the device-scoped functional accessor to the public module surface:

```python
__all__ = ['ExecPlan', 'Linear', 'Weight', 'WeightPlan', 'get_linear']
```

No package-root re-export is added. External callers use:

```python
from lmdeploy.turbomind.linear import ExecPlan, Linear, Weight, WeightPlan, get_linear
```

### 2.1 One functional module per device

`Linear` owns reusable workspace and GEMM dispatch state. Individual model layers must not own separate instances. Add one process-local registry with exactly one shared `Linear` per CUDA device.

Add `weakref` to the promoted module imports. `Linear` binds to the current CUDA device at construction; it does not accept or normalize a device argument. Add weak-reference support to its slots so the device registry does not own it:

```python
import weakref


__slots__ = ('device', '_impl', '_context', '__weakref__')

def __init__(self):
    self.device = torch.device('cuda', torch.cuda.current_device())
    self._impl = None
    self._context = None
    with self._activate():
        try:
            self._impl = _tm.LlamaLinear()
        except Exception:
            self._impl = None
            raise

def __del__(self):
    try:
        self.close()
    except Exception:
        pass
```

After the `Linear` class definition, add a weak device registry:

```python
_linears: weakref.WeakValueDictionary[int, Linear] = weakref.WeakValueDictionary()


def get_linear() -> Linear:
    """Return the functional Linear module for the current CUDA device."""
    device = torch.cuda.current_device()
    linear = _linears.get(device)
    if linear is None:
        linear = Linear()
        _linears[device] = linear
    return linear
```

The registry does not own a `Linear`. Concrete Linear modules retain the strong references returned by `get_linear()`. Calls on the same current device reuse the live instance, and the functional `Linear` is destroyed as soon as its last concrete-module reference is gone. `Linear.__del__()` runs the existing device/context-aware `close()` path before the native executor is released; there is no process-lifetime `atexit` owner.

The caller selects the device with the ordinary Torch CUDA device context before calling `get_linear()`. The registry does not retain a context per stream: each shared `Linear` continues to cache only its most recently used context.

The shared functional `Linear` is non-reentrant because its native executor owns mutable GEMM workspace. There may be only one outstanding execution on a device. The supported engine path issues all layer calls in order on one model execution stream, including CUDA graph capture and replay. Concurrent independent model agents on the same device are outside this API contract and are undefined behavior. Do not add locks, event chaining, forward synchronization, or stream-scoped executors.

Layers cache the reference returned by `get_linear()` during weight preparation. They do not call `get_linear()` during forward and do not explicitly close the shared module. Dropping the last cached reference destroys it.

The promoted API otherwise keeps its current semantics:

- `Linear` owns the native executor and caches only the context for the most recently used Torch CUDA stream.
- `get_weight_plan()` selects the family once and publishes its shape constraints.
- `prepare_weight()` owns normalization, format packing, native allocation, byte copies, and native weight preparation.
- `get_exec_plan()` selects an immutable execution plan for one input problem.
- `Linear.__call__()` allocates omitted Torch outputs from the execution plan and runs dense GEMM.
- `forward_moe()` runs grouped GEMM.
- `tune()` measures feasible launches and returns the selected immutable plan.
- `Weight.close()` and `Linear.close()` retain their existing explicit lifetime contract; callers never close an object returned by `get_linear()`.
- Every operation uses the caller's current Torch CUDA stream.

No weight planning, preparation, execution, or output-allocation logic changes are part of this move.

## 3. Use the production API in the existing fixture

`tests/turbomind/linear/fixture.py` no longer imports a test-local implementation.

Replace:

```python
from .linear import Linear, Weight, _tm, _to_tm_dtype
```

with:

```python
from lmdeploy.turbomind import _tm
from lmdeploy.turbomind.linear import Weight, _to_tm_dtype, get_linear
```

Fixture construction selects its device and then uses the shared functional module:

```python
with torch.cuda.device(self.device):
    self.linear = get_linear()
    self._build_weights()
```

This replaces the existing separate `self.linear = Linear(self.device)` and `_build_weights()` statements in the constructor's `try` block.

Fixture cleanup closes its prepared weight and drops its cached reference, but does not close the shared module:

```python
if self.w_quant is not None:
    self.w_quant.close()
    self.w_quant = None
self.linear = None
```

No case list, benchmark logic, or assertion changes. The existing tests now exercise the same production API used by AWQ.

The existing dtype-contract test imports the relocated builder function:

```python
from lmdeploy.turbomind.builders.linear import _build_linear
```

Its existing direct construction becomes:

```python
linear = _build_linear(weight_format, available)
```

Delete `tests/turbomind/linear/linear.py`; do not replace it with a re-export.

## 4. Reduce the AWQ wrapper to a compatibility adapter

Replace `lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py` with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Optional W4A16 backend using the TurboMind Linear API."""

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.turbomind.linear import Linear, Weight, get_linear
from lmdeploy.turbomind.weight_format import AWQFormat

from ..awq_modules import LinearW4A16Impl


class TurbomindAwqLinearW4A16Impl(LinearW4A16Impl):
    """Adapt canonical AWQ parameters to the TurboMind Linear API."""

    def __init__(self, in_features: int, out_features: int, w_bit: int, group_size: int):
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self._linear: Linear | None = None
        self._weight: Weight | None = None

    def _release(self):
        weight, self._weight = self._weight, None
        if weight is None:
            self._linear = None
            return
        # Forward is asynchronous and may use any engine stream. Release the prepared weight only after all device work completes.
        torch.cuda.synchronize()
        weight.close()
        self._linear = None

    def __del__(self):
        try:
            linear = self._linear
            if linear is None:
                return
            with torch.cuda.device(linear.device):
                self._release()
        except Exception:
            pass

    def update_weights(self, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, bias: torch.Tensor | None = None):
        self._release()

        linear = get_linear()
        plan = linear.get_weight_plan(weight_format=AWQFormat(block_in=self.group_size), dtype=scales.dtype)
        weight = linear.prepare_weight(qweight, plan=plan, scales=scales, zeros=qzeros)
        # Weight preparation is asynchronous on the loading stream, while the engine executes on another stream. Publish only after packing completes.
        torch.cuda.current_stream().synchronize()

        self._linear = linear
        self._weight = weight
        return qweight, scales, qzeros, bias

    def forward(self, x, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, bias: torch.Tensor | None = None, all_reduce: bool = False, group: torch.distributed.ProcessGroup | None = None):
        linear = self._linear
        weight = self._weight

        exec_plan = linear.get_exec_plan(x, weight)
        out, _ = linear(x, weight, exec_plan=exec_plan)

        if bias is not None:
            out = out + bias

        if out.ndim == 2:
            out = out.unsqueeze(0)
        if all_reduce:
            dist.all_reduce(out, group=group)
        return out
```

The constructor keeps `w_bit` only because `LinearW4A16Impl` fixes that interface. The builder already admits only the supported W4A16 contract; the wrapper does not store or revalidate it.

The canonical AWQ `scales` parameter is allocated in the layer's resolved computation dtype before checkpoint loading. `update_weights()` therefore passes `scales.dtype` to `get_weight_plan()`. The adapter does not hard-code or separately store the computation dtype.

`update_weights()` returns the original checkpoint tensors unchanged. The native `Weight` is a private prepared representation owned separately by the wrapper.

Every AWQ layer caches the shared `Linear` reference obtained during `update_weights()`. Forward reads `self._linear` directly and performs no registry lookup. `_release()` closes only the layer's prepared `Weight`, then drops both references; it never closes the functional module.

The following definitions and imports disappear completely:

- `_LinearRuntime`
- `_runtime_pool`
- `_get_runtime`
- `_PreparedLinear`
- `_prepare_turbomind_linear`
- `weakref`
- direct `_tm` access
- `_is_turbomind_gemm_capability_supported` in this module
- manual `Context`, DLPack, `WeightQuery`, `LinearConfig`, `LinearWeight`, allocation, copy, and pack logic
- wrapper-level weight/input shape, dtype, device, and lifecycle checks

The following compatibility behavior remains because it belongs to the PyTorch AWQ interface rather than TurboMind execution:

- arbitrary-rank inputs whose leading dimensions are flattenable without copying, and their corresponding output shape, are handled directly by the `Linear` API;
- bias is applied separately;
- two-dimensional input retains the existing three-dimensional output compatibility behavior;
- optional all-reduce remains after the local result.

## 5. Stream and lifetime contract

`prepare_weight()` enqueues copies and packing on the current Torch CUDA stream. The engine may execute the first forward on another stream, so `update_weights()` synchronizes its current stream before publishing `_weight`. This preserves the existing AWQ adapter's ready-on-return contract without retaining an event or per-stream state:

```python
weight = linear.prepare_weight(qweight, plan=plan, scales=scales, zeros=qzeros)
# Weight preparation is asynchronous on the loading stream, while the engine executes on another stream. Publish only after packing completes.
torch.cuda.current_stream().synchronize()
self._linear = linear
self._weight = weight
```

Forward execution is also asynchronous and may occur on a stream other than the stream that prepared the weight. `_release()` therefore waits for all work on the owning device before closing the previously prepared weight:

```python
# Forward is asynchronous and may use any engine stream. Release the prepared weight only after all device work completes.
torch.cuda.synchronize()
weight.close()
```

A current-stream synchronization is sufficient for publishing a newly prepared weight because all preparation work was enqueued on that stream. Releasing an existing weight requires device-wide synchronization because its preceding executions may have used another stream.

Forward execution uses the caller's current Torch CUDA stream through `Linear._activate()`. The wrapper does not call `record_stream`, `wait_stream`, create an event, or retain per-stream objects.

Construction, explicit `update_weights()`, forward execution, and explicit release must occur with the owning CUDA device current. Implicit Python finalization cannot rely on the current device, so `__del__()` temporarily selects `self._linear.device` before calling `_release()`. The forward input dtype must match the input dtype selected by the prepared weight plan, and the caller consumes the native output dtype without wrapper conversion. The input must contain at least one GEMM row, and its leading dimensions must be flattenable into rows without copying. Empty inputs and incompatible non-contiguous layouts are undefined behavior. `update_weights()` must complete before `forward()`. Destruction must occur after outstanding uses are ordered. Calls using the same shared device module must not execute concurrently on different CUDA streams. These are unchecked preconditions inherited from the `Linear` API except for the device selection performed by `__del__()`.

## 6. Unchanged selection boundary

`lmdeploy/pytorch/backends/cuda/awq_modules.py` remains the only place that decides whether the TurboMind implementation is eligible. Its existing checks for W4A16, group size 128, K/N divisibility, FP16, CUDA availability, SM90, and native-module availability stay unchanged.

The implementation module assumes that the builder has already established those conditions. Direct construction outside that builder has the same unchecked preconditions.

## 7. Verification

All verification is transient. Add no tests, test cases, helper scripts, or CI steps.

### 7.1 Static and syntax checks

Require that the test-local API is gone and no production build-time consumer still imports the new execution API accidentally:

```bash
test ! -e tests/turbomind/linear/linear.py
rg -n 'from \.\.linear import' lmdeploy/turbomind/builders lmdeploy/turbomind/models
rg -n 'from \.linear import' lmdeploy/turbomind/text_model.py lmdeploy/turbomind/vision_model.py lmdeploy/turbomind/weight_format.py
rg -n 'from .*builders|import .*builders' lmdeploy/turbomind/weight_format.py
rg -n 'resolver\._build_linear' lmdeploy tests
```

The `rg` commands must have no matches. Build-time imports must point to `builders.linear` or `.linear` within the builders package, and `weight_format.py` must have no builder dependency.

Require that the AWQ module contains none of the removed machinery:

```bash
rg -n '_LinearRuntime|_runtime_pool|_get_runtime|_PreparedLinear|_prepare_turbomind_linear|create_device_context|from_dlpack|WeightQuery|LinearConfig|LinearWeight|copy_bytes_on_stream|weakref|_tm' lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py
```

The command must have no matches.

Compile every modified Python surface:

```bash
PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python -m compileall -q lmdeploy/turbomind lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py tests/turbomind/linear
```

### 7.2 Existing Linear API coverage

Before either CUDA command, use `get_gpu_usage` to select an empty SM90 GPU. Run both commands outside the sandbox. First, transiently confirm device sharing without adding a test:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python - <<'PY'
from lmdeploy.turbomind.linear import get_linear

assert get_linear() is get_linear()
PY
```

Then run the existing Linear suite on the same selected GPU:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm pytest -q tests/turbomind/linear/test_linear.py
```

This existing suite must pass without skips caused by the file move. It now exercises `lmdeploy.turbomind.linear` directly.

### 7.3 Transient AWQ comparison

Before the CUDA command, check `get_gpu_usage` again. Run outside the sandbox:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python - <<'PY'
import torch

from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl
from lmdeploy.pytorch.backends.cuda.turbomind_awq_modules import TurbomindAwqLinearW4A16Impl


def pack_awq(values):
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=values.device).view(2, 4).t().flatten()
    return torch.sum(values.unflatten(-1, (-1, 8)).to(torch.int64) << shifts, dim=-1).to(torch.int32)


torch.manual_seed(0)
device = torch.device('cuda')
k = 4096
n = 4096
group_size = 128

qweight = pack_awq(torch.randint(0, 16, (k, n), device=device))
qzeros = pack_awq(torch.randint(0, 16, (k // group_size, n), device=device))
scales = (torch.rand((k // group_size, n), device=device) * 0.02 + 0.001).to(torch.float16)
x = (torch.randn((1, 17, k), device=device) * 0.1).to(torch.float16)

triton_impl = AwqLinearW4A16Impl(k, n, 4, group_size)
tm_impl = TurbomindAwqLinearW4A16Impl(k, n, 4, group_size)
tm_impl.update_weights(qweight, scales, qzeros)

triton_output = triton_impl.forward(x, qweight, scales, qzeros)
tm_default_output = tm_impl.forward(x, qweight, scales, qzeros)

default_stream = torch.cuda.current_stream(device)
caller_stream = torch.cuda.Stream(device=device)
caller_stream.wait_stream(default_stream)
with torch.cuda.stream(caller_stream):
    pending_output = tm_impl.forward(x, qweight, scales, qzeros)

# Do not synchronize caller_stream. Replacing the prepared weight must wait for this forward before releasing the previous weight.
tm_impl.update_weights(qweight, scales, qzeros)
replacement_output = tm_impl.forward(x, qweight, scales, qzeros)

assert torch.equal(tm_default_output, pending_output)
assert torch.equal(tm_default_output, replacement_output)
max_abs_diff = (tm_default_output - triton_output).abs().max().item()
torch.testing.assert_close(tm_default_output, triton_output, rtol=1e-3, atol=2e-3)
print(f'max absolute difference: {max_abs_diff}')
PY
```

The pending alternate-stream result and the result from the replacement weight must both be bitwise identical to the original TurboMind result. The Triton comparison must satisfy the stated tolerance.

### 7.4 PyTorch AWQ model smoke with CUDA graphs

The adapter must be exercised through the PyTorch engine rather than the native TurboMind engine. Before testing, select an empty SM90 GPU with `get_gpu_usage`. Run outside the sandbox with the TurboMind W4A16 provider forced:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" LMDEPLOY_W4A16_GEMM_BACKEND=turbomind PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python - <<'PY'
import huggingface_hub.constants as hf_constants

hf_constants.HF_HUB_OFFLINE = 1
hf_constants.HF_HUB_CACHE = "/mnt_cfs/huggingface_hub/hub/"

from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline

prompt = "Explain how matrix multiplication is used in transformer language models, with concrete examples and enough detail to continue for the full response."
engine_config = PytorchEngineConfig(tp=1, max_batch_size=1, cache_max_entry_count=0.1, eager_mode=False)
generation_config = GenerationConfig(max_new_tokens=128, min_new_tokens=128)
pipe = pipeline("Qwen/Qwen3-8B-AWQ", backend_config=engine_config)
response = pipe(prompt, gen_config=generation_config)
print(response.text)
print(f"generated tokens: {response.generate_token_len}")
assert response.generate_token_len >= 128
pipe.close()
PY
```

`LMDEPLOY_W4A16_GEMM_BACKEND=turbomind` makes failure to instantiate the adapter fatal rather than falling back to another AWQ implementation. `eager_mode=False` exercises engine warmup and CUDA graph capture. Read the generated response and require meaningful human-language content relevant to the prompt.

### 7.5 Native TurboMind model-loading smoke

The build-time `Linear` relocation and the changed `WeightFormatResolver.resolve()` contract must also be exercised through the native TurboMind model-loading pipeline. Check `get_gpu_usage` again for an empty SM90 GPU, then run the required repository script as-is outside the sandbox:

```bash
PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python scripts/test_turbomind_model.py --model-id Qwen/Qwen3-8B --cache-dir /mnt_cfs/huggingface_hub/hub/ --gpus "$EMPTY_SM90_GPU" --tp 1 --max-new-tokens 128 --prompt "Explain how matrix multiplication is used in transformer language models, with concrete examples and enough detail to continue for the full response."
```

Read the generated response and require 128 meaningful human-language tokens relevant to the prompt.

## Completion criteria

- `lmdeploy/turbomind/linear.py` is the sole Torch-facing Linear API implementation.
- `lmdeploy/turbomind/builders/linear.py` owns the build-time tensor bundle and transforms.
- `WeightFormatResolver.resolve()` selects a format and returns its checkpoint tensors; builder-owned `_build_linear()` performs normalization and bundle construction.
- No test-local API implementation or compatibility re-export remains.
- One live functional `Linear` is shared by all concrete layers on each CUDA device without being retained for the lifetime of the process.
- The shared functional `Linear` is explicitly non-reentrant and is used by one ordered model execution stream per device.
- Each AWQ wrapper caches a reference to that shared `Linear`, owns its prepared `Weight`, and retains only PyTorch AWQ compatibility behavior.
- The AWQ wrapper contains no native binding calls, context management, stream tracking, events, manual packing, or duplicated feasibility checks; synchronization is limited to publishing a prepared weight and releasing a previously prepared weight.
- Existing Linear tests, the transient AWQ comparison, the PyTorch Qwen3-8B-AWQ CUDA-graph smoke, and the native TurboMind Qwen3-8B model-loading smoke pass.
