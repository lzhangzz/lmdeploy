# Prior-art research: module unification in sibling engines

Type: research
Status: resolved

## Question

What patterns do sibling inference frameworks use to unify weight storage and forward execution at the module level, and which are worth adopting for TurboMind's executable Module? Cover: torch `nn.Module` (weights + forward unity, how quantization hooks attach to it), vLLM's quantization-method objects owning both weights and `apply`, SGLang / TensorRT-LLM layer and plugin registries, and any engine that migrated from a split weight/executor shape to unified modules (what broke, what they'd do differently). Deliver 5-10 concrete patterns with class/file citations and a trade-off note each, tagged adopt / adapt / avoid as input to Merged Linear module interface. Write findings to `.scratch/executable-modules/research/prior-art.md`.

## Answer

Findings (10 patterns, each with primary-source citations and a trade-off, tagged adopt/adapt/avoid): [`.scratch/executable-modules/research/prior-art.md`](../research/prior-art.md). Gist:

- The dominant sibling shape is two-axis composition: a module class per parallelism kind owning weights+forward, times a per-quant method object owning storage format and kernel (vLLM `LinearBase`/`ColumnParallelLinear` × `LinearMethodBase.create_weights/apply`, `process_weights_after_loading`; SGLang forked this design and reads mode state from an ambient `runtime_context`).
- torch `nn.Module` is the reference for storage+forward unity with hook seams; torch.ao's quantized `Linear` adds the prepack/module-swap lifecycle whose versioned state-dict churn is the caution — derive packed state, never serialize it.
- TRT-LLM's C++ `BaseLayer` (setup/forwardAsync/workspace, v0.15.0) is the executor-only pole — async discipline worth borrowing, split weight ownership worth avoiding.
- TVM relax `nn`/MLC-LLM pass the KV cache as a typed forward argument (`LlamaAttention.forward(..., paged_kv_cache)`), the strongest precedent for TurboMind's Context-as-typed-argument contract.
- For the merged Linear interface: adopt module-owns-both + the method split + vLLM/SGLang's `WEIGHT_LOADER_V2_SUPPORTED`-style staged migration gate; adapt the post-load `prepare` stage and per-platform dispatch factory; avoid duck-typed layer-attribute seams (`apply(layer, ...)`) — formalize the `LinearWeight` param set as a typed handles struct so mismatch fails at compile time.

