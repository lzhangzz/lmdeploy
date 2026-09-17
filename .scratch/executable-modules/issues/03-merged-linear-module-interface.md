# Merged Linear module interface

Type: prototype
Status: resolved

## Question

Draft the merged Linear module interface as a concrete header to react to: `class Linear : core::Module` owning today's `LinearWeight` params (`weight/bias/scales/zeros/global_scale`) plus the `WeightPlan`, with `Forward(...)` overloads covering dense and grouped/MoE call sites, Context injection, the `GetOutputSpec`/`GetExecPlan`/`Tune` surface, construction-by-name from Python (binding surface unchanged), and the mapping of the standalone Torch Linear API (`lmdeploy/turbomind/linear.py`) onto it. Show the `LlamaFfnLayer.cc` and `moe_ffn_layer.cc` call sites rewritten against it. This header becomes the canonical module shape that Merged FFN and MoE module design and Merged Attention module design reuse.

## Answer

Prototype (user-validated, four reaction points confirmed) at [`prototype/linear-module-interface.md`](../prototype/linear-module-interface.md). Decisions:

- **Canonical shape**: `class Linear : core::Module` — one class owns the X-macro params, GEMM metadata, `prepare`/`set_plan` lifecycle, and the execution surface (five `Forward` overloads with the `const LinearWeight&` argument dropped, `GetOutputSpec`, `GetExecPlan`, `Tune`). `gemm::Family` remains the per-quant method axis (vLLM two-axis already in-tree); no new method-object layer.
- **Execution state**: `LlamaLinear` and its pimpl retire. GEMM engine + workspace + measure policy live in `ExecContext` (per the Context shared-state contract); `Linear` holds `ExecContext* xc_` from the `bind()` pass. Warm-up/tuning ops (`set_measure`, `Export`/`Import`, `GetTuningSeq`) are module methods delegating to `xc_` — no facade.
- **Naming**: C++ class `Linear`; `type()` keeps returning `"LinearWeight"` so Python `create_module` by name is untouched.
- **Call sites**: weight-is-executor — `mlp.w2->Forward(gating, inter_scales, param.output, unused_scales)`; grouped MoE forwards against `moe->block()`; attention projections land mechanically in the Attention ticket. `linear_` members and `Context::linear` disappear.
- **Python surface**: `linear.py` public names unchanged (`get_weight_plan`, `prepare_weight`, `get_exec_plan`, `tune`, `forward_dense`, `forward_moe`), bound on the module; the standalone path constructs its own `ExecContext`.

This header is the canonical module shape for Merged FFN and MoE module design, Merged Attention module design, BatchOp and decoder mapping, and Python binding-surface rules. The validated decision folds into the spec at assembly; the prototype stays a linked asset (no in-tree code yet — implementation is the follow-up effort).
