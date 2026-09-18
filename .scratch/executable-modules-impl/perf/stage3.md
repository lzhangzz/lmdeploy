# Stage-3 gate (ticket 08): Attention module + AttentionState

Runner: `run_stage3.sh` (same shapes as stage 2; results in `stage3/`).
GPU 4, 2026-09-18, tree = post-08 working tree (commit `099fd0644` + 08,
final binary after review fixes).

## Smoke

All four gate models produce meaningful ≥254-token responses. Cache-behavior
variants all pass with 256-token meaningful responses:

- prefix caching: Qwen3-8B, Qwen3-30B-A3B
- checkpointing (`--cache-prompt all --cache-checkpoint-interval 64`):
  Qwen3.5-35B-A3B-FP8 (GDN checkpoint category under coexistence), Qwen3-8B
  (combined with prefix caching)

Re-smoked on the final post-review binary (atomic wave, shared RNG,
region-relative iterator offsets): 8B prefix-caching variant and the FP8
checkpoint variant.

## Throughput (output tok/s; recorded baseline → stage-3, %)

| model | load (s) | bs=1 | bs=32 | bs=128 |
|---|---|---|---|---|
| Qwen3-8B | 6.99 → 7.13 | 108.6 → 110.7 (101.9%) | 2625.1 → 2563.4 (97.6%) | 9053.3 → 9012.3 (99.5%) |
| Qwen3-8B-AWQ | 6.24 → 5.98 | 111.4 → 113.6 (101.9%) | 2549.0 → 2580.8 (101.2%) | 8566.9 → 8660.6 (101.1%) |
| Qwen3-30B-A3B | 16.25 → 17.30 | 126.5 → 129.6 (102.5%) | 1537.6 → 1571.4 (102.2%) | 4973.6 → 4978.3 (100.1%) |
| Qwen3.5-35B-FP8 | 20.05 → 20.87 | 145.5 → 126.6 | 2451.5 → 2058.3 | 6912.7 → 5932.2 |

8B bs=32 is 97.6% of the two-day-old recorded baseline on the single matrix
run. Follow-up: three re-runs measured 2571.4 / 2556.2 / 2610.4 (mean 2575.4,
best 99.4% of baseline) — the recorded number sits at the top of today's
band, the same inter-day drift signature stage-1 documented for fp8. The
clean same-day A/B is stage-2 (old attention path) vs stage-3 (new path):
2582.4 → 2575.4 mean = −0.3%, inside the ±1.1% same-binary run-to-run band,
while attention-dominated bs=1 is +1.9% and bs=128 is 99.5%. No
attention-path regression.

FP8 keeps the stage-1/2 environmental note: bs=32 is 101.1% of the fresh
same-day baseline-commit measurement (2058.3 vs 2036.6) and +1.6% over
stage-2. Loads within the dense models' noise band; footprints unchanged
(118.9/117.2/127.4/123.3 GiB, matching stage-2 within 0.1%).

**Gate: pass.**
