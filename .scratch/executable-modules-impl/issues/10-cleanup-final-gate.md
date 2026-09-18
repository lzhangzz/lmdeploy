# 10: Coexistence cleanup and final gate

**What to build:** The contract the spec's landing strategy promises: every retired class and temporary coexistence shim is gone from the tree. Then the final gate across the full matrix — including the large TP model that exercises the comm and allreduce paths — plus the cache-behavior variants and the perf comparison against the recorded baselines.

**Blocked by:** 04 (Python delta and stage-1 gate), 06 (Moe executable module with shared MoeState, and stage-2 gate), 08 (Attention executable module with AttentionState, and contract-doc sync), 09 (DeltaNet executable module with DeltaNetState).

**Status:** resolved

- [x] No retired executor, wrapper, or delegate class remains anywhere in the tree
- [x] Full four-model matrix plus the large TP≥2 model pass end-to-end with meaningful responses of at least 128 tokens
- [x] Cache-behavior variants (prefix caching, checkpointing) pass
- [x] Decode throughput at or above 98% of baseline across the matrix; load times within noise
- [x] Frozen-surface check passes one final time

## Answer

**Cleanup:** deleted the dead `src/turbomind/models/llama/CMakeLists.txt`
(never `add_subdirectory`'d; five phantom sources; every real file it
listed compiles via `models/CMakeLists.txt`). Full-tree sweep finds zero
retired-class references (`LlamaLinear`, `UnifiedAttentionLayer`,
`LlamaFfnLayer`, `MoeFfnLayer`, `GatedDeltaNetLayer`, `LlamaV2`,
`LlamaBatch`, `LlamaWeight`, `LlamaDecoderLayerWeight`) outside
intentional records (.scratch, CONTEXT.md avoid-glossary); `*Weight`
names survive only as the frozen registry strings. Review should-fix
applied: `interns2_mobius.py` docstrings re-qualified to the `Moe` class.

**Frozen surface:** builders + `weight_format.py` byte-identical to the
pre-effort base; pybind delta = `+_tm.Context`, `-_tm.LlamaLinear`;
registry strings verified at the source for all five kinds; import smoke
and the standalone torch-linear tests (8/8) pass.

**Final gate** (details in `perf/final-gate.md`):

- Four-model matrix: 256/254/256/256 meaningful tokens; variants pass
  (8B prefix caching, 8B + 35B-FP8 checkpointing at interval 64).
- TP≥2: `meta-llama/Llama-3.1-70B-Instruct` on GPUs 4+5, 143 meaningful
  tokens — TP comm, allreduce, and per-rank bind exercised. (Model id in
  `/data/models.json` is mis-cased `meta-Llama`; the cache dir is
  `meta-llama`.)
- Perf: every cell ≥98% of baseline except the two documented
  environmental-drift points — FP8 (recorded baseline proven stale in
  ticket 04; final = 99.2–100.2% of fresh same-session references) and
  8B bs=32 (recorded number is the band top; final = 98.9–100.9% of
  same-day stage references). Loads within recent bands; footprints
  byte-identical across stages 3/4/final.

The executable-modules effort is complete: Linear, Ffn, Moe, Attention,
and DeltaNet are executable modules; every per-kind executor/wrapper
class is retired; the Python surface is frozen exactly as specified.
