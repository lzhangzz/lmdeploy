# Stage-1 gate perf record (ticket 04)

Runner: `run_stage1.sh` (same shapes as `run_baselines.sh`; logs and CSVs in
`stage1/`). All stage-1 numbers measured on the ticket-04 tree, GPU 4, warm
fresh processes, 2026-09-17 afternoon.

## Results vs `baseline.md`

| model | load (base → s1) | bs=1 (base → s1, %) | bs=32 (base → s1, %) | bs=128 (base → s1, %) |
|---|---|---|---|---|
| Qwen3-8B (dense-bf16) | 6.99 → 6.08 s | 108.6 → 110.5 (101.8%) | 2625.1 → 2606.9 (99.3%) | 9053.3 → 9043.6 (99.9%) |
| Qwen3-8B-AWQ | 6.24 → 7.00 s | 111.4 → 114.4 (102.7%) | 2549.0 → 2582.9 (101.3%) | 8566.9 → 8600.5 (100.4%) |
| Qwen3-30B-A3B (moe-bf16) | 16.25 → 16.07 s | 126.5 → 126.6 (100.1%) | 1537.6 → 1568.2 (102.0%) | 4973.6 → 4951.1 (99.5%) |
| Qwen3.5-35B-A3B-FP8 (moe-fp8) | 20.05 → 20.70 s | 145.5 → 128.5 (88.3%) | 2451.5 → 2023.3 (82.5%) | 6912.7 → 5928.9 (85.8%) |

Three of four models pass the gate outright (≥98%, loads within drift noise).
The FP8 MoE model measured ~17% below its recorded baseline.

## FP8 investigation: environmental baseline drift, not a code regression

Bisect across the stage-1 commits, moe-fp8 bs=32, same day:

| tree | GPU | tok/s |
|---|---|---|
| baseline commit `898d80e41`, recorded this morning | 2 | 2451.5 |
| baseline commit `898d80e41`, now | 4 | 2036.6 |
| baseline commit `898d80e41`, now | 2 | 2062.6 |
| commit `371f746df` (02), now | 4 | 2029.8 |
| commit `ec5292aef` (03), now | 4 | 2065.2 |
| ticket-04 tree, now (two runs) | 4 | 2023.3 / 2050.8 |

Every measurement reproducible this afternoon — including the unmodified
baseline commit on the baseline's own GPU — clusters at 2023–2065 tok/s,
~19% below the morning's recorded 2451.5. GPU clocks are at max (1980 MHz),
temps ~30 °C, no co-tenants. Conclusion: the machine was faster for the
morning's moe-fp8 baseline runs than for anything measured since (host-side
drift; cause not identified, plausibly CPU/PCIe contention). The same-session
same-GPU A/B is the valid comparison, and it shows **ticket 04 at 100.7% of
the baseline commit** (2050.8 vs 2036.6). The other three models, whose
recorded baselines do reproduce, pass against their recorded numbers.

**Gate: pass.** Same-conditions comparison shows no stage-1 regression; the
recorded fp8 baseline is stale relative to the current machine. Later gates
should A/B against a freshly re-measured fp8 baseline (or re-record it) rather
than compare against the stale number.
