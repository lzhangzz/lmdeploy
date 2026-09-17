# 01: Record perf baselines

**What to build:** The decode-performance reference that every later stage gate compares against. Run the repo's decode benchmark on the four core gate models (dense bf16, dense W4A16 AWQ, MoE bf16, MoE FP8) at fixed, documented batch shapes on free GPUs, and record the results as the effort's baseline table.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Four core gate models benchmarked (dense bf16, dense AWQ, MoE bf16 grouped, MoE FP8) with identical batch shapes across runs
- [x] Batch shapes, GPUs, and benchmark invocation recorded so the runs are reproducible
- [x] Baseline table (decode throughput + pipeline-load time per model) written to the effort's `perf/baseline.md`
- [x] GPU discipline followed: free devices only, usage checked before each run

## Answer

Baselines recorded 2026-09-17 on the unmodified tree at `898d80e41`, GPU 2 of the 8× SM90 host (all devices verified free before each sweep; the rerun's per-model nvidia-smi snapshots are in `perf/raw/runner_rerun_stdout.log`, and the runner now tees future snapshots into `perf/raw/gpu_usage.log`). Table at `perf/baseline.md`; raw profiler CSVs and full logs in `perf/raw/`; exact invocations in `perf/run_baselines.sh` (supports tag-filtered reruns).

Decode throughput (token/s, `output_throughput`), identical shapes for every model — random prompts, input 128 / output 256, seed 0, tp=1, concurrency 1/32/128. Load time = latest of two warm runs:

| Model | Load (s) | bs=1 | bs=32 | bs=128 |
| --- | --- | --- | --- | --- |
| Qwen3-8B (dense bf16) | 6.99 | 108.6 | 2625.1 | 9053.3 |
| Qwen3-8B-AWQ (dense W4A16) | 6.24 | 111.4 | 2549.0 | 8566.9 |
| Qwen3-30B-A3B (MoE bf16) | 16.25 | 126.5 | 1537.6 | 4973.6 |
| Qwen3.5-35B-A3B-FP8 (MoE FP8) | 20.05 | 145.5 | 2451.5 | 6912.7 |

Method deviations from the gates draft, recorded in `baseline.md` and binding for later gates:

- `benchmark_decode.py` drives the pytorch backend and is blind to the TurboMind engine; the baseline uses `benchmark/profile_throughput.py --backend turbomind` instead (fixed-concurrency round-robin decode through `TurboMind.from_pretrained`).
- Quantized checkpoints require `--model-format awq|fp8` (converter asserts on the `hf` default).
- Load time is a fresh-process `TurboMind.from_pretrained` with the bs=128 engine config, latest of two warm-cache runs per model; later gates compare warm against warm (first-touch swing is large: moe-bf16 35.5 s cold vs 16.3 s warm).
- A pre-sweep smoke run aborted at process teardown (`std::system_error: Resource deadlock avoided`) after writing its results; none of the recorded sweep runs aborted.
