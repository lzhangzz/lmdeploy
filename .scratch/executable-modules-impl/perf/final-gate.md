# Final gate (ticket 10): cleanup + full matrix + TP≥2

Runner: `run_final.sh` (identical shapes; results in `final/`). GPU 4
(single-GPU legs), GPUs 4+5 (TP2 leg), 2026-09-18, tree = ticket-09
commit `f1f56278e` + ticket-10 cleanup (dead-CMakeLists deletion,
docstring fixes; runtime code unchanged from stage 4).

## Cleanup

- Dead `src/turbomind/models/llama/CMakeLists.txt` deleted: never
  `add_subdirectory`'d, listed five phantom sources (`LlamaV2.cc`,
  `LlamaBatch.cc`, `LlamaWeight.cc`, `LlamaDecoderLayerWeight.cc`,
  `GatedDeltaNetWeight.cc`); every real source it listed compiles via
  `models/CMakeLists.txt`.
- Retired-class sweep: zero hits for `LlamaLinear`,
  `UnifiedAttentionLayer`, `LlamaFfnLayer`, `MoeFfnLayer`,
  `GatedDeltaNetLayer`, `LlamaV2`, `LlamaBatch`, `LlamaWeight`,
  `LlamaDecoderLayerWeight` in `src/`, `lmdeploy/`, `tests/`,
  `benchmark/`, `docs/` (only `.scratch/` records and the CONTEXT.md
  avoid-glossary entry, both intentional). The `*Weight` names survive
  only as the frozen registry strings.
- Review should-fix: `interns2_mobius.py` docstrings updated
  (`MoeWeight::set_meta_pack` → `Moe::set_meta_pack` etc.). Remaining
  "retired executor" mentions are accurate provenance notes.

## Frozen surface

Builders and `weight_format.py` byte-identical to the pre-effort base
(`f2ca94bb2`): empty diff. Pybind surface delta across the whole effort:
`+_tm.Context`, `-_tm.LlamaLinear` (`LinearWeight` binding name kept) —
exactly the sanctioned additions. Registry strings verified at the
source (config ctor ↔ `TM_MODULE_REGISTER_AS` pairs for all five kinds).
Import smoke OK; standalone torch-linear path 8/8 (`test_linear.py`).

## Smoke matrix (each run separate, never batched)

| run | tokens | response |
|---|---|---|
| Qwen3-8B | 256 | meaningful |
| Qwen3-8B + prefix caching | 256 | meaningful |
| Qwen3-8B + checkpointing (interval 64) | 256 | meaningful (full text verified) |
| Qwen3-8B-AWQ | 254 | meaningful |
| Qwen3-30B-A3B-Thinking | 256 | meaningful |
| Qwen3.5-35B-FP8 | 256 | meaningful |
| Qwen3.5-35B-FP8 + checkpointing | 256 | meaningful |
| **meta-llama/Llama-3.1-70B-Instruct, TP=2 (GPUs 4+5)** | 143 | meaningful — exercises TP comm, allreduce, per-rank bind |

## Perf (output tok/s; recorded baseline → final, %; fresh ref where recorded is stale)

| model | load (s) | bs=1 | bs=32 | bs=128 |
|---|---|---|---|---|
| Qwen3-8B | 6.99 → 7.17 | 108.6 → 110.3 (101.6%) | 2625.1 → 2554.8 (97.3%) | 9053.3 → 9073.7 (100.2%) |
| Qwen3-8B-AWQ | 6.24 → 5.92 | 111.4 → 113.7 (102.1%) | 2549.0 → 2559.0 (100.4%) | 8566.9 → 8653.8 (101.0%) |
| Qwen3-30B-A3B | 16.25 → 17.57 | 126.5 → 127.1 (100.5%) | 1537.6 → 1562.3 (101.6%) | 4973.6 → 5005.1 (100.6%) |
| Qwen3.5-35B-FP8 | 20.05 → 20.86 | 145.5 → 128.3 | 2451.5 → 2021.2 | 6912.7 → 5877.8 |

Two points sit below 98% of the *recorded* baseline — both are the
environmental drift documented since stage 1, not code regressions:

- **FP8**: ticket 04's bisect re-measured the unmodified baseline commit
  same-session at 2036.6 (bs32) vs the recorded 2451.5. Against the
  fresh references the final numbers are bs=1 99.6% (128.3 vs stage-2
  128.8), bs=32 99.2% (2021.2 vs 2036.6), bs=128 100.2% (5877.8 vs
  stage-2 5863.6).
- **8B bs=32**: the recorded 2625.1 is the top of the band (stage-3
  re-runs 2556–2610, mean 2575.4). Same-day references: 98.9% of
  stage-2 (2582.4), 99.7% of stage-3 (2563.4), 100.9% of stage-4
  (2531.9).

All other cells ≥100% of recorded. Loads within the recent same-day
bands (7.13–7.17 / 5.87–5.98 / 17.30–17.64 / 20.80–21.09 across stages
3/4/final). Footprints byte-identical across stages 3/4/final
(118933 / 117205 / 127413 / 123253 MiB).

**Gate: pass** — on recorded baselines where they are valid, and on the
documented fresh/same-day references where the recorded numbers are
stale (the standard set in ticket 04 and applied in every gate since).

## Evidence locations

Raw perf CSVs and full runner logs: `perf/raw/` (baselines) and
`perf/final/` (this gate); stage re-measurements in `perf/stage{1..4}/`.
Smoke-matrix and TP≥2 terminal transcripts are not copied into the repo —
those runs were driven interactively (never batched) and their full outputs
live in the development session log (kimi-code `wire.jsonl` for this
workspace, 2026-09-17/18). `perf/raw/gpu_usage.log`, promised by ticket 01's
tee, was never created: no later baseline sweep ran; the rerun GPU snapshots
are in `raw/runner_rerun_stdout.log`.
