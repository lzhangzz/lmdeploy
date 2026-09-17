# Migration order and verification gates

Type: grilling
Status: resolved

## Question

Fix the staged migration order and per-stage gates: Linear, then FFN/MoE, then Attention (+GatedDeltaNet). For each stage: the in-tree moves, the checkpoint and Python-API compat checks, and the verification recipe (build with `ninja` in `build/`; `scripts/test_turbomind_model.py` with meaningful responses of at least 128 tokens, never batched; which quant × architecture × dense/MoE matrix is mandatory per gate; how perf parity is measured). The FFN/MoE stage must also gate on memory footprint and startup cost as a general check (the original per-module transport rationale was reversed to a shared `MoeState` by Rework the ExecContext design). Output: the migration section for the spec.

## Answer

Migration section drafted at [`drafts/migration-gates.md`](../drafts/migration-gates.md). Decisions:

- **Landing**: sequential stages on `modules`, tree green at every boundary; bounded coexistence inside a stage, every retired class deleted by its successor's end at the latest; no flagged dual paths.
- **Gate matrix (every stage)**: Qwen3-8B (dense bf16), Qwen3-8B-AWQ (dense W4A16), Qwen3-30B-A3B-Thinking-2507 (MoE bf16 grouped), Qwen3.5-35B-A3B-FP8 (MoE FP8 fused path); final gate adds Llama-3.1-70B-Instruct at TP≥2. All via `scripts/test_turbomind_model.py` (≥128 meaningful tokens, never batched), cache dir `/mnt_cfs/huggingface_hub/hub/`, free GPUs (2-7 of the 8× SM90 hosts).
- **Perf parity**: `benchmark_decode.py` before/after each stage on the stage's core models, same batch shapes; gate at ≥98% of baseline decode throughput plus pipeline-load time within noise; baselines recorded in `.scratch/executable-modules/perf/baseline.md` before stage 1.
- **Stage contents**: stage 1 = Linear merge + `bind(Context&)`/default no-op `Run` + the effort's only Python delta (`linear.py` construction, `_tm.Context`, the two new driver entry points); stage 2 = Ffn/Moe merge, impl classes retire, `MoeState`, general footprint/startup gate; stage 3 = Attention/DeltaNet merge, `register_cache` goes real, `AttentionState`/`AttentionState` batch fill, README contract-sync amendments same-change, cache-behavior gate variants (prefix caching, checkpointing).
- Checkpoint compat is implicit: every gate loads the same unmodified checkpoints.
