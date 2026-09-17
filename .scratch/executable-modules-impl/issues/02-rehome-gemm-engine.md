# 02: Re-home the GEMM engine into the engine root (prefactor)

**What to build:** Make the change easy, then make the easy change. The per-rank GEMM engine — engine object, dispatch cache, workspace, measure policy — moves out of the linear executor's private implementation into the engine-root Context, where the spec's engine-services section puts it. The old linear executor stays alive as a thin delegate over the re-homed engine so every existing call site keeps working unchanged. No module behavior changes in this ticket.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] GEMM engine, dispatch cache, workspace, and measure policy owned by the engine-root Context (one per rank)
- [x] The linear executor becomes a pure delegate: same public surface, zero owned engine state
- [x] Warm-up/tuning flow (measure mode, export/import) works through the re-homed engine
- [x] Build green; one core-model smoke run shows unchanged behavior

## Answer

New `GEMMEngine` (`src/turbomind/models/llama/GEMMEngine.{h,cu}`) holds the re-homed state: the `gemm::Gemm` object (dispatch cache lives inside it), the `DispatchPolicy` measure policy, and the CUDA `Workspace`. The engine-root `turbomind::Context` owns one per rank (`gemm_engine` member, spec-named `linear_engine()` accessor) and constructs `LlamaLinear` as a delegate over it. `LlamaLinear` keeps its exact public surface; its Impl holds only a captured `GEMMEngine*` plus the stateless operand/argument helpers. The standalone default ctor (the `_tm.LlamaLinear()` Python call site, no Context until ticket 04) transitionally owns a private engine — every engine-path instance holds none.

Warm-up/tuning in `turbomind.cc` (`WarmUp`: import, set_measure, GetTuningSeq, export) now runs directly through `ctx.linear_engine()`, and `TurboMind::Impl::gemm()` reads the re-homed engine. Zero behavior change: all layer call sites (`LlamaFfnLayer`, `unified_attention_layer`, `GatedDeltaNetLayer`, `moe_ffn_layer`, `language_model`, `model_executor`, internvit/qwenvit) still work through the unchanged `ctx.linear` surface.

Verified: `ninja` green; `scripts/test_turbomind_model.py --model-id Qwen/Qwen3-8B` end-to-end, 256-token meaningful response, pipeline load 6.63 s (baseline 6.99 s); standalone `_tm.LlamaLinear()` construct → `get_weight_plan` delegation → close verified. Review (standards + spec sub-agents) findings applied: file case matches class (`GEMMEngine.*`), message chains flattened.

Flagged pre-existing, moved verbatim and deliberately NOT fixed here (behavior-preserving prefactor): the workspace ctor allocates `tensormaps` with `partials_size` (32 MB) while the recorded `tensormaps_size` (8192*128 = 1 MB, "maximum 4096 tensor maps") is never used for the allocation — over-allocation in the safe direction; worth a dedicated cleanup ticket with kernel-side verification.
