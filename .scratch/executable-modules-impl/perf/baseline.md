# Decode-performance baseline — executable-modules implementation

Recorded by ticket 01 before stage 1 begins, on the unmodified tree at commit
`898d80e41` ("Publish executable-modules implementation tickets"). Every stage
gate compares against this table: decode throughput must stay ≥98% of baseline
and pipeline-load time within noise.

The gates draft referenced `.scratch/executable-modules/perf/baseline.md`; this
file (in the impl effort folder) is the canonical baseline.

## Method

**Benchmark:** `benchmark/profile_throughput.py --backend turbomind`, the repo's
fixed-concurrency decode benchmark driving `TurboMind.from_pretrained` directly.
The gates draft named `benchmark_decode.py`, but that script drives the pytorch
backend (`lmdeploy.pytorch.decode.Engine`) and is blind to the TurboMind engine
this effort refactors — substitution recorded here, same substitution applies at
every later gate.

**Workload:** random prompts sampled from
`/mnt_cfs/ShareGPT_V3_unfiltered_cleaned_split.json`, `--random-input-len 128`,
`--random-output-len 256`, `--seed 0`, `ignore_eos` (set by the harness),
streaming detokenize on. Batch shapes identical across all models:
concurrency (max_batch_size) **1, 32, 128**; `num_prompts` = 16 for bs=1
(sequential), 4×bs otherwise.

**Engine:** tp=1, single GPU (`CUDA_VISIBLE_DEVICES=2`), default
`cache_max_entry_count`, prefix caching off, session length default.
Model weights from `/mnt_cfs/huggingface_hub/hub/` (HF_HUB_OFFLINE=1).
Quantized checkpoints need `--model-format` matching their quant config
(`awq` for Qwen3-8B-AWQ, `fp8` for Qwen3.5-35B-A3B-FP8); the script default
`hf` makes the converter assert otherwise.

**Load time:** fresh process timing
`TurboMind.from_pretrained(model, TurbomindEngineConfig(tp=1, max_batch_size=128))`
from call to return, latest of two warm-cache runs per model (both in the
`<tag>_load.log` files). Page cache was warm for every run; later gates must
compare warm-cache against warm-cache — the sweep matters: moe-bf16 measured
35.48 s on its first-ever touch of the 61 GB checkpoint and 16.25 s warm, and
moe-fp8 moved 27.20 → 20.05 s.

**GPU discipline:** free devices only. Usage checked before each sweep via the
GPU monitor and per-model `nvidia-smi` in the runner. The first sweep's
per-model snapshots went to runner stdout that was not kept; the rerun's
snapshots are preserved in `raw/runner_rerun_stdout.log`, and the runner now
tees them into `raw/gpu_usage.log` for future runs.

**Environment:** 8× SM90 (143771 MiB each); repo at `/data/lmdeploy-gemm`,
in-tree python package with the `build/` ninja artifacts.

**Exact invocations:** see `run_baselines.sh` in this folder; per-run logs and
profiler CSVs in `raw/` (`<tag>_load.log`, `<tag>_bs<N>.{log,csv}`).

**Known noise source:** a pre-sweep smoke run (bs=1, two prompts) aborted at
process teardown (`std::system_error: Resource deadlock avoided`) *after*
writing its results. None of the recorded sweep runs aborted — every raw log
ends cleanly after its summary table.

## Baseline table

Decode throughput = `output_throughput` (token/s) from the profiler CSV; load
time = latest `LOAD_SECONDS` of the two warm runs in the load log.

| Model | Tag | Load (s) | bs=1 tok/s | bs=32 tok/s | bs=128 tok/s |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-8B (dense bf16) | dense-bf16 | 6.99 | 108.6 | 2625.1 | 9053.3 |
| Qwen/Qwen3-8B-AWQ (dense W4A16) | dense-awq-w4a16 | 6.24 | 111.4 | 2549.0 | 8566.9 |
| Qwen/Qwen3-30B-A3B-Thinking-2507 (MoE bf16) | moe-bf16 | 16.25 | 126.5 | 1537.6 | 4973.6 |
| Qwen/Qwen3.5-35B-A3B-FP8 (MoE FP8) | moe-fp8 | 20.05 | 145.5 | 2451.5 | 6912.7 |

Secondary metrics (TTFT/TPOT/ITL percentiles, request throughput) are in the
per-run CSVs under `raw/`; the gate compares the columns above.
