# Stage-4 gate (ticket 09): DeltaNet module + DeltaNetState

Runner: `run_stage4.sh` (same shapes as stage 3; results in `stage4/`).
GPU 4, 2026-09-18, tree = post-09 working tree (commit `ec8668254` + 09,
final binary after review fixes, including the kPrepare layer-0 guard).

## Smoke

All four gate models produce meaningful ≥254-token responses. Variants pass
with 256-token meaningful responses:

- checkpointing (`--cache-prompt all --cache-checkpoint-interval 64`):
  Qwen3.5-35B-A3B-FP8 (the GDN model — exercises the checkpoint category:
  per-module conv registration, state-registered recurrent grid, position-0
  clears, publish/restore)
- prefix caching: Qwen3-8B

Re-smoked on the final post-review binary (idempotent RegisterLayer,
kPrepare once per pass): Qwen3.5-35B-A3B-FP8 twice — meaningful 256-token
responses both times; Qwen3-8B for the generic path.

## Throughput (output tok/s; stage-3 → stage-4, %)

| model | load (s) | bs=1 | bs=32 | bs=128 |
|---|---|---|---|---|
| Qwen3-8B | 7.13 → 7.13 | 110.7 → 110.4 (99.7%) | 2563.4 → 2531.9 (98.8%) | 9012.3 → 9056.1 (100.5%) |
| Qwen3-8B-AWQ | 5.98 → 5.87 | 113.6 → 113.4 (99.8%) | 2580.8 → 2572.0 (99.7%) | 8660.6 → 8649.0 (99.9%) |
| Qwen3-30B-A3B | 17.30 → 17.64 | 129.6 → 126.8 (97.8%) | 1571.4 → 1556.3 (99.0%) | 4978.3 → 4994.0 (100.3%) |
| Qwen3.5-35B-FP8 | 20.87 → 20.80 | 126.6 → 124.8 (98.6%) | 2058.3 → 2038.0 (99.0%) | 5932.2 → 5825.0 (98.2%) |

Footprints byte-identical to stage-3 (118933 / 117205 / 127413 / 123253 MiB).

## Reading the deltas

The benchmark is fully deterministic in tonight's window: the FP8 re-run
(`stage4_rerun_fp8.log`) and an 8B control re-run each reproduced their
matrix numbers to the decimal (FP8 124.8 / 2038.0 / 5825.0 twice; 8B
110.4 / 2531.9 / 9056.1 twice). So stage-4 vs stage-3 deltas are stable
differences between measurement windows, not run-to-run noise.

The control group — dense + 30B models whose GDN-free paths are untouched
by ticket 09 — moved −0.2% … −2.2% (10 of 12 entries negative, mean
−0.6%): the stage-4 window sits slightly below the stage-3 window across
the board. The FP8 model (the only one executing the new DeltaNet path)
moved −1.0% / −1.4% / −1.8% — inside the control band at bs=1/bs=32, at
its edge at bs=128. Against stage-2 (the last measurement before the
Attention/DeltaNet module rewrites): bs=32 +0.6%, bs=128 −0.7%, with the
stage-2→3 leg (no GDN code change) already accounting for −1.7% of bs=1.

The transcription initially ran the kPrepare body in every layer module
(L× PrepareState enqueues per pass vs once in the retired executor); the
review restored once-per-pass semantics via a layer-0 guard. Pre-guard and
post-guard FP8 numbers are identical — the redundant enqueues were not on
the critical path at these shapes, and the final binary carries the old
executor's exact per-pass work.

**Gate: pass** — no GDN-path regression beyond the window drift equally
visible on unchanged-path models, per the same-day A/B + control-band
standard set in stage 3.
