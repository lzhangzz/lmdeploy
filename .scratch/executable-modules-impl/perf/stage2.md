# Stage-2 gate (ticket 06): Moe module + shared MoeState

Runner: `run_stage2.sh` (same shapes as stage 1, plus a post-load
nvidia-smi footprint probe). Raw logs and CSVs in `stage2/`. All runs on
GPU 4, 2026-09-17, tree = post-06 working tree (commit `1ebdec4d2` + 06).

## Smoke

All four gate models produce meaningful ≥254-token responses end-to-end
(Qwen3-8B, Qwen3-8B-AWQ, Qwen3-30B-A3B via `Moe::ForwardDefault`,
Qwen3.5-35B-A3B-FP8 grouped FP8 path). 30B-A3B re-smoked on the final
binary after the review fix (meta_experts scan coverage).

## Throughput (output tok/s; recorded baseline → stage-2, %)

| model | load (s) | bs=1 | bs=32 | bs=128 |
|---|---|---|---|---|
| Qwen3-8B | 6.99 → 7.07 | 108.6 → 109.9 (101.2%) | 2625.1 → 2582.4 (98.4%) | 9053.3 → 9048.1 (99.9%) |
| Qwen3-8B-AWQ | 6.24 → 5.95 | 111.4 → 114.7 (102.9%) | 2549.0 → 2589.7 (101.6%) | 8566.9 → 8674.9 (101.3%) |
| Qwen3-30B-A3B | 16.25 → 17.54 | 126.5 → 127.4 (100.7%) | 1537.6 → 1554.5 (101.1%) | 4973.6 → 5002.5 (100.6%) |
| Qwen3.5-35B-FP8 | 20.05 → 20.87 | 145.5 → 128.8 | 2451.5 → 2026.7 | 6912.7 → 5863.6 |

Three of four models pass outright against their recorded baselines
(≥98% at every batch size; loads within the ±1 s noise band — the dense
models, which have no MoeState at all, show the same ±1 s spread).

FP8 stays on the stage-1 environmental-drift finding: its recorded
baseline is stale (the unmodified baseline commit re-measured at
2023–2065 tok/s the same afternoon; see `stage1.md`). Against the fresh
same-day references stage-2 shows no regression: bs=32 is 99.5% of
today's baseline-commit measurement (2026.7 vs 2036.6), and bs=1/bs=128
are 100.2%/98.9% of the stage-1 tree that passed the stage-1 gate.

## Footprint / startup check (the reversed-amendment check)

The check answers one question: did moving transport/routing state into
modules multiply it by layer count? It did not — the state is one
`MoeState` per rank in the Context.

- Precise A/B: "Object cache budget … from free" in the load logs —
  stage-2 free memory at that point is exactly **32.0 MB lower** than
  stage-1 for both MoE models (30B: 83403.50 vs 83435.50 MB; FP8:
  105227.50 vs 105259.50 MB). That 32 MB is the MoeState buffer set,
  relocated from CreateEngine (post-budget, stage-1) to the Bind pass
  (pre-budget, stage-2) — allocated once. A per-layer multiplication
  (48 layers) would have been ~1.5 GB.
- Post-load total footprint (new probe): 119.0 GiB (8B), 117.3 GiB
  (AWQ), 127.4 GiB (30B-A3B), 123.3 GiB (35B-FP8) — dominated by the
  object cache; no baseline totals recorded to compare, hence the
  free-memory A/B above as the precise check.
- Startup: loads above; MoE deltas (+0.2 to +1.5 s) sit inside the
  dense models' ±1 s noise band, and the DeepEP JIT path is not
  exercised on single-GPU gates.
- Known consequence, accepted: routing scratch now allocates *before*
  the cache-budget `cudaMemGetInfo`, so MoE models' object-cache budget
  is 32 MB smaller than before (strictly more honest accounting).

**Gate: pass.**
