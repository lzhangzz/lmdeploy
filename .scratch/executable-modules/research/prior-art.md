# Prior art: module-level unification of weight storage and forward execution

Ticket: [01-prior-art-module-unification-research](../issues/01-prior-art-module-unification-research.md)
Researched 2026-09-17 against primary sources (framework source code fetched and read; links in [Sources](#sources)). Each pattern is tagged adopt / adapt / avoid as input to the Merged Linear module interface.

## The TurboMind shape under study

- Data side: `LinearWeight : core::Module` (`src/turbomind/models/linear_weight.h:42`) — params `weight/bias/scales/global_scale/zeros` declared via `TM_MODULE_DECLARE`, GEMM metadata (`gemm::Family`, three `DataFormat`s, `Epilogue`, `k_desc`/`q_desc`), lifecycle `prepare()` and `set_plan(gemm::WeightPlan)`. No forward.
- Exec side: `LlamaLinear` (`src/turbomind/models/llama/LlamaLinear.h:14`) — stateless executor with five `Forward` overloads that each take `const LinearWeight&` as an argument, plus `GetExecPlan`/`Tune` and a pimpl `Impl` owning `gemm::Gemm`; one instance per kind, owned by `UnifiedDecoder`.
- The merge under design: make the weight module executable (or let it own its executor/method) without moving checkpoint formats, the BatchOp/cache contract, or the Python binding surface.

## Patterns

### 1. torch `nn.Module`: one object owns storage and forward; behavior attaches at the call boundary — **adopt**

Where: [`torch/nn/modules/module.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py) — `_parameters`/`_buffers`/`_modules` registries, `__call__ → _wrapped_call_impl` (line 1785) running `_forward_pre_hooks` (471) / `_forward_hooks` (464), `register_forward_hook` (1698), `_apply` (932) for device/dtype moves, `state_dict` (2205).

What: the module is simultaneously the weight container and the executable; per-scheme behavior (observers, fake-quant, parametrizations) is attached as child modules and hooks fired by `__call__`, so `nn.Linear` itself never grows quant variants. Storage lifecycle (`_apply`, `state_dict`) and execution (`forward`) live behind one object with stable names.

Trade-off: attachment without subclass explosion, but the hook seam makes behavior invisible at the call site and `state_dict` is a parallel name-keyed contract that must be kept in sync by hand.

For TurboMind: the merged `core::Module` Linear should own both the declared params and `Forward`, with a small fixed set of lifecycle seams (`prepare`, plan/tune, export) instead of per-kind executor classes.

### 2. torch.ao quantized `Linear`: convert-time module swap with prepacked opaque weights — **adapt**

Where: [`torch/ao/nn/quantized/modules/linear.py`](https://github.com/pytorch/pytorch/blob/main/torch/ao/nn/quantized/modules/linear.py) — `Linear(WeightedQuantizedModule)` owning a `LinearPackedParams` submodule; `set_weight_bias` calls `torch.ops.quantized.linear_prepack`; `forward` is `torch.ops.quantized.linear(x, self._packed_params._packed_params, self.scale, self.zero_point)`; `from_float(mod)` performs the swap; `_load_from_state_dict` migrates version 1→2→3 payload shapes.

What: a second module class with an opaque packed weight and a one-kernel forward replaces the float module at `convert()` time; checkpointing unpacks back to the canonical form and re-packs on load.

Trade-off: clean runtime (single prepacked blob, single op call), but a two-representation lifecycle — a separate class to swap in, and a serialization format that had to be version-migrated three times.

For TurboMind: TurboMind already has the good half of this (`prepare()`/`WeightPlan` ≈ prepack). Keep one class across the lifecycle and derive the packed plan rather than ever serializing it — checkpoints stay as-is, and `LlamaLinear::Export/Import` remains a tuning cache only.

### 3. vLLM: two-axis composition — parallelism subclass × quant-method object — **adopt**

Where: [`vllm/model_executor/layers/linear.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py) — `LinearMethodBase(QuantizeMethodBase)` with abstract `create_weights(layer, ...)` ("The weights will be set as attributes of the layer") and `apply(layer, x, bias)` ("Expects create_weights to have been called before"); `UnquantizedLinearMethod` implements `create_weights` / `process_weights_after_loading` / `apply` (kernel impl chosen in `__init__` from `config.kernel_config.linear_backend`). `LinearBase` declares `self.quant_method`, resolved via `resolve_quant_method(quant_config, self, prefix)`; subclasses `ReplicatedLinear`, `ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear` each call `quant_method.create_weights(...)` in `__init__` and `quant_method.apply(self, ...)` in `forward`, and vary the sharding-aware `weight_loader`/`weight_loader_v2`. Extension happens by subclassing the module kind (`KimiK3MergedQKVGateLinear(MergedColumnParallelLinear)`), not by adding executors. Quant kinds resolve through a string-keyed lazy registry with an OOT hook ([`layers/quantization/__init__.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py): `get_quantization_config`, `register_quantization_config`).

What: the module class owns shape/sharding/loading; a separate method object owns quantized storage + kernel. N sharding strategies × M quant formats compose without N×M classes.

Trade-off: orthogonal growth at the cost of a loosely typed seam between the two axes (see pattern 4) and string-keyed quant dispatch.

For TurboMind: this is the target shape for merged Linear — `core::Module` subclass per module kind (dense / fused gate_up / QKV / row-parallel) holding a per-`DataFormat`/`gemm::Family` method object with `create/prepare/forward`.

### 4. vLLM: the method takes the whole layer and duck-types its attributes — **avoid**

Where: same file — `apply(self, layer, x, bias)` and `create_weights(self, layer, ...)`; methods read `layer.weight`, `layer.bias`, `layer.layer_scratch`, `layer._cpu_skip_gemm_dispatch` and register new attributes on the fly.

What: because Python has no interface for "the weights this method created", the contract is "whatever `create_weights` happened to attach" — discovered by attribute access at runtime.

Trade-off: maximal flexibility for quant methods, but breakage surfaces as runtime `AttributeError`s far from the cause, and each method implicitly constrains which module kinds it can attach to.

For TurboMind: in C++ pass (or own) a typed weights/handles struct between the module and its method — the `LinearWeight` param list (`weight/bias/scales/global_scale/zeros` + descriptors) is exactly the struct to formalize, so mismatch is a compile error.

### 5. vLLM: `process_weights_after_loading` as an explicit post-load transform stage — **adapt**

Where: `vllm/model_executor/layers/linear.py`, `UnquantizedLinearMethod.process_weights_after_loading(layer)` — platform-specific repack (CPU dispatch packing, optional XPU N-contiguous transpose; Marlin/FP8 variants repack or requantize) after all shards are loaded.

What: checkpoint loading stays format-only; kernel-preferred layouts are produced in a named lifecycle stage owned by the method, before the first `apply`.

Trade-off: loaders stay simple and transforms are discoverable, but transforms may reallocate (memory spike on large weights) and ordering relative to TP gather must be respected.

For TurboMind: this is `LinearWeight::prepare()`/`WeightPlan` — keep it as a named stage of the merged module, and tie its invocation to the existing warm-up/tuning path (`is_warm_up_`, `Gemm::Tune`) rather than to the loader.

### 6. vLLM `CustomOp`/`PluggableLayer`: per-platform forward methods + registry with OOT class replacement — **adapt**

Where: [`vllm/model_executor/custom_op.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/custom_op.py) — `CustomOp.dispatch_forward()` selects `forward_cuda/forward_hip/forward_cpu/forward_xpu/...` once in `__init__`; `op_registry` + `@CustomOp.register(name)`; `PluggableLayer.register_oot` swaps the instantiated class in `__new__` (e.g. an OOT `HPUUnquantizedFusedMoEMethod` replaces the in-tree one). `LinearBase` derives from `PluggableLayer`, and linear kinds are registered by name (`@PluggableLayer.register("column_parallel_linear")` in linear.py).

What: one class hosts all platform variants selected once at construction; implementations are registered per kind name and can be replaced out-of-tree without touching call sites.

Trade-off: no platform forks per layer and third-party swaps are cheap, but dispatch is fixed at construction and the effective implementation is only discoverable through the registry.

For TurboMind: mirrors what a `LinearConfig`-keyed factory would give C++ (per-kind, per-family executor selection); prefer an explicit factory function over runtime class-swap tricks, and keep the set of overridable kinds small and named.

### 7. vLLM + SGLang: staged interface migration via a per-method allowlist — **adopt**

Where: `WEIGHT_LOADER_V2_SUPPORTED` class-name list + `register_weight_loader_v2_supported_method` decorator in both [`vllm/.../linear.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py) and SGLang's fork; weight loaders branch on `self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED`.

What: a new weight-loading contract was rolled out method-by-method behind an allowlist; old and new paths coexist until every method opts in.

Trade-off: incremental, revertable rollout with no big-bang cutover; cost is a stale allowlist that must be burned down and per-method behavioral divergence while it lives.

For TurboMind: the Linear → FFN/MoE → Attention migration wants exactly this — a per-module-kind gate (merged path vs legacy `Llama*Layer` path) so each stage is independently revertable, with an explicit burn-down list in the spec's verification gates.

### 8. SGLang: inherited design + ambient runtime context instead of threaded state — **adapt**

Where: [`python/sglang/srt/layers/linear.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py) (header: "Adapted from vllm v0.6.4.post1") — same `LinearBase(torch.nn.Module)`/quant-method shape; forwards consult `from sglang.srt.runtime_context import get_exec, get_forward, get_parallel` — e.g. `get_forward().sp_active` in `ColumnParallelLinear.forward` and `get_exec().comm.enable_quant_communications` in `RowParallelLinear.forward`.

What: execution-mode state (sequence parallelism, quantized collectives) is read from an ambient typed context instead of being threaded through every signature — the sibling design closest to TurboMind's planned Context shared state. The file header is also fork-drift evidence: SGLang adapted vLLM v0.6.4 and manually re-merges interface changes since.

Trade-off: signatures stay clean while mode switches change behavior, but implicit reads make a layer untestable in isolation and hide dataflow; and forking the design means hand-merging every upstream contract change.

For TurboMind: a Context like SGLang's fits, but confine context reads to the module's forward entry point, make them typed and pimpl-cheap, and keep the core (input, output, plan) explicit — so tests can construct a module plus a fake Context.

### 9. TensorRT-LLM C++ decoding layers: stateful executors with params-struct I/O — weights stay outside — **avoid** (borrow the async contract)

Where: [`cpp/tensorrt_llm/layers/baseLayer.h`](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.15.0/cpp/tensorrt_llm/layers/baseLayer.h) (pinned to v0.15.0; the directory has since been restructured on current main) — `BaseLayer` holds a `BufferManager` (owns the stream) and `DecoderDomain`; pure virtual `setup(batchSize, beamWidth, batchSlots, setupParams, workspace)` ("must be called only once for new requests"), `forwardAsync(outputs, inputs, workspace)` ("no stream synchronization inside this function"), optional `forwardSync`; sibling [`layersFactory.h`](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.15.0/cpp/tensorrt_llm/layers/layersFactory.h) `createLayers<T>(mode, domain, bufferManager)` assembles the pipeline from an enum. Weight-carrying layers are *not* here — they live in the built network / plugins ([`cpp/tensorrt_llm/plugins/common/plugin.h`](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.15.0/cpp/tensorrt_llm/plugins/common/plugin.h), `BasePlugin : nvinfer1::IPluginV2DynamicExt`, TensorRT plugin registry).

What: the C++ side never unified weight storage with execution — executors are stream-bound state machines over params structs, and the weight path stays in a separate registry world.

Trade-off: the setup/forwardAsync/workspace shape is battle-tested for async decode (inputs/outputs as shared-ptr structs, explicit workspace sizing, stream discipline), but as a module model it is exactly the split TurboMind is trying to close — executors here, weight lives elsewhere.

For TurboMind: keep the contract discipline (setup-once for per-request state, no sync inside forward, workspace explicit — compatible with `checklist`/BatchOp meanings in `src/turbomind/engine/README.md`), do not keep the split ownership.

### 10. TVM relax `nn` / MLC-LLM: module owns symbolic params; side state is a typed forward argument — **adapt**

Where: [`python/tvm/relax/frontend/nn/core.py`](https://github.com/apache/tvm/blob/main/python/tvm/relax/frontend/nn/core.py) — `Module` (container of `Parameter`s and submodules, `state_dict`/`load_state_dict`, `__call__ → forward`, compile seams `export_tvm`/`jit`), `Parameter` (bound/unbound, per-param `attrs`), and `Effect` ("non-user-facing class that encloses potential side effects, for example, IO, impure external function callings, inplace mutation") / `Object` ("non-tensor frontend components such as KV caches"). Used by [`python/mlc_llm/model/llama/llama_model.py`](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/llama/llama_model.py): `LlamaFFN(nn.Module)` composes `nn.Linear` children (`gate_up_proj`, `down_proj`) with a plain `forward(x)`; `LlamaAttention.forward(hidden_states, paged_kv_cache: PagedKVCache, layer_id)`.

What: weights and forward are unified in one module tree; caches and other side state are explicit typed objects in the forward signature (and first-class to the compiler), not globals.

Trade-off: every forward threads the cache/effect objects — verbose, but it makes side effects visible, mockable, and compiler-analyzable, which is what lets MLC compile the whole model.

For TurboMind: the strongest argument for the Context shared-state contract being a *typed argument* of merged-module forwards (Attention/GatedDeltaNet especially) rather than ambient global state; Linear can take a trivial/empty context now and the interface stays uniform.

## Synthesis: what this implies for the merged Linear interface

Converging the adopted/adapted pieces, the prior art points at:

- One `core::Module` subclass per module kind, owning declared params and `Forward` (patterns 1, 3), with the kind registered in a small named factory (pattern 6).
- A per-`DataFormat`/`gemm::Family` method object with `create_weights` / `prepare` (post-load transform) / `forward` (patterns 3, 5), attached through a typed seam — the `LinearWeight` param set formalized as a struct, not duck-typed attributes (pattern 4, avoided).
- Lifecycle = load → `prepare()`/plan (≈ `process_weights_after_loading`) → tune/warm-up → `Forward`; one class throughout, derived (never serialized) packed state (pattern 2).
- Migration behind a per-kind gate with a burn-down list (pattern 7); Context as a typed forward argument, SGLang-style reads confined and debuggable (patterns 8, 10); async/workspace discipline retained, split ownership dropped (pattern 9).

Sketch (illustrative, not a spec):

```cpp
// per module kind; owns storage + execution (patterns 1, 3)
class Linear: public core::Module {
public:
    Linear(const LinearConfig& cfg);            // kind + dims + format from config
    void create_weights();                      // method_->create_weights(handles_)
    void prepare();                             // post-load transform + WeightPlan (patterns 2, 5)
    void Forward(const Tensor& x, Ref<Tensor> y, Context& ctx);  // ctx typed arg (pattern 10)
private:
    LinearHandles handles_;                     // typed seam: weight/bias/scales/zeros + k_desc/q_desc (pattern 4, avoided)
    std::unique_ptr<LinearMethod> method_;      // per gemm::Family / DataFormat (pattern 3)
    std::optional<gemm::ExecPlan> plan_;        // derived, never serialized (pattern 2)
};
```

Open questions for the spec effort (not answered here): whether the method object is per-module or shared per-family (tuning cache sharing argues shared; typed seam argues per-module), and how MoE expert linears compose with the kind registry.

## Sources

Fetched and read 2026-09-17 (main branches unless pinned):

- PyTorch: [torch/nn/modules/module.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py), [torch/ao/nn/quantized/modules/linear.py](https://github.com/pytorch/pytorch/blob/main/torch/ao/nn/quantized/modules/linear.py)
- vLLM: [vllm/model_executor/layers/linear.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py), [vllm/model_executor/layers/quantization/__init__.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py), [vllm/model_executor/custom_op.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/custom_op.py)
- SGLang: [python/sglang/srt/layers/linear.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py)
- TensorRT-LLM (pinned v0.15.0; `cpp/tensorrt_llm/layers/` no longer resolves on current main): [baseLayer.h](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.15.0/cpp/tensorrt_llm/layers/baseLayer.h), [layersFactory.h](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.15.0/cpp/tensorrt_llm/layers/layersFactory.h), [plugins/common/plugin.h](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.15.0/cpp/tensorrt_llm/plugins/common/plugin.h)
- TVM / MLC-LLM: [python/tvm/relax/frontend/nn/core.py](https://github.com/apache/tvm/blob/main/python/tvm/relax/frontend/nn/core.py), [python/mlc_llm/model/llama/llama_model.py](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/llama/llama_model.py)
- Local: `src/turbomind/models/linear_weight.h`, `src/turbomind/models/llama/LlamaLinear.h`

Not verified (searched, not confirmable from primary sources): the exact vLLM PR that renamed `Scheme` → `LinearMethodBase`/`QuantizeMethodBase`; the migration lesson is instead evidenced by the verified `WEIGHT_LOADER_V2_SUPPORTED` rollout gate (pattern 7).
