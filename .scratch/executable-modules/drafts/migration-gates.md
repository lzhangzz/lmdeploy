# Migration order and verification gates

Draft spec section from [Migration order and verification gates](../issues/08-migration-order-and-verification-gates.md). Stage contents derive from the resolved design drafts; the matrix, perf method, and landing strategy are user-decided.

## Landing strategy

Stages land sequentially on `modules`, tree green and gates passed at every boundary. Temporary coexistence is allowed *inside* a stage (e.g. `LlamaLinear` surviving internally during stage 1); every retired class is deleted by the end of its owning stage's successor at the latest. No flagged dual paths.

## Stage 1 — Linear

In-tree moves:
- `LinearWeight` becomes `Linear` (`type()` string frozen); `LlamaLinear`'s execution logic absorbs into the module; its `Impl` contents (`gemm_`, `workspace_`, `dispatch_policy_`) move into the GEMM engine object owned by `turbomind::Context`.
- `core::Module` grows the default no-op `Run(BatchOp, phase, TensorMap&)` and `bind(Context&)`.
- All `linear_.Forward(input, *weight, ...)` call sites become `weight->Forward(...)`: attention qkv/wo projections (inside the still-existing `UnifiedAttentionLayer`), FFN and MoE GEMM calls (inside the still-existing layers), `LanguageModel`'s embedding/output GEMMs (own `Linear` modules).
- Python (the effort's only Python delta, lands here): `linear.py` construction block → `_tm.Context()` + `mod.bind(ctx)`; `_tm.Context` binding added; `TurboMind.register_cache()` and `TurboMind.bind()` entry points appear (register_cache is an empty walk until stage 3); `turbomind.py` driver sequence gains the two calls.

Gate: build (`ninja` in `build/`); frozen-list grep + builder import smoke; four core models below; perf ≥98% + load time.

## Stage 2 — FFN/MoE

In-tree moves: `FfnWeight`/`MoeWeight` become `Ffn`/`Moe` with full forwards; `LlamaFfnLayer`, `MoeFfnLayer`, `MoeFfnDefaultImpl`, `MoeFfnA2AImpl` retire; the thrice-copy-pasted GEMM orchestration collapses into `Ffn::Forward`; `MoeState` (TokenDispatcher, routing buffers, scales-copy stream) is created once per rank and bound; decoder loop swaps the FFN/MoE calls.

Gate: stage-1 checks, plus the general memory-footprint and startup-cost check (MoE-state sizing; the original per-module rationale was reversed by [Rework the ExecContext design](../issues/10-rework-execcontext-design.md)).

## Stage 3 — Attention + GatedDeltaNet

In-tree moves: `AttentionWeight`/`DeltaNetWeight` become `Attention`/`DeltaNet`; `UnifiedAttentionLayer` and `GatedDeltaNetLayer` retire; the `register_cache(CacheRegistry&)` pass becomes real (per-layer prefix KV bytes; conv/recurrent checkpoint bytes) with the batch-version-stamped `AttentionBatch` fill in `AttentionState`; decoder loop swaps attention/GDN calls; all README contract-sync amendments land same-change (class renames in `contracts.cache-content`/`unified-attention`/`gated-deltanet`, registration-timing rewording in `ownership.module-cache`/`contracts.category-registration`, lifecycle paragraph per `drafts/batchop-decoder-mapping.md`).

Gate: stage-2 checks, plus cache-contract behavior runs of the test script with `--enable-prefix-caching` and checkpointing variants.

## Verification recipe (every gate)

1. `ninja` in `build/` (clean build at stage boundaries).
2. Frozen-list grep (no new/renamed Python public names) + builder import smoke.
3. Four core models via `python scripts/test_turbomind_model.py --model-id <id> --cache-dir /mnt_cfs/huggingface_hub/hub/ --gpus <free>` with meaningful responses of at least 128 tokens, verified by the runner (never batched in a loop):
   - `Qwen/Qwen3-8B` — dense bf16
   - `Qwen/Qwen3-8B-AWQ` — dense W4A16
   - `Qwen/Qwen3-30B-A3B-Thinking-2507` — MoE bf16 (grouped GEMM)
   - `Qwen/Qwen3.5-35B-A3B-FP8` — MoE FP8 (fused-SiLU path)
4. Final gate adds `meta-Llama/Llama-3.1-70B-Instruct` at TP≥2 (comm/allreduce paths).
5. Perf parity: `benchmark_decode.py` before and after the stage on that stage's core models (same batch shapes, free GPUs); decode throughput ≥98% of the recorded baseline; pipeline-load time within noise. Baselines recorded in `.scratch/executable-modules/perf/baseline.md` before stage 1 begins.
6. GPU discipline: check GPU usage before any run; gates use free devices (currently 2-7 of the 8× SM90 hosts); checkpoint compat is implicit — every gate loads the same unmodified checkpoints.
