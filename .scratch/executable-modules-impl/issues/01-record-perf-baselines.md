# 01: Record perf baselines

**What to build:** The decode-performance reference that every later stage gate compares against. Run the repo's decode benchmark on the four core gate models (dense bf16, dense W4A16 AWQ, MoE bf16, MoE FP8) at fixed, documented batch shapes on free GPUs, and record the results as the effort's baseline table.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Four core gate models benchmarked (dense bf16, dense AWQ, MoE bf16 grouped, MoE FP8) with identical batch shapes across runs
- [ ] Batch shapes, GPUs, and benchmark invocation recorded so the runs are reproducible
- [ ] Baseline table (decode throughput + pipeline-load time per model) written to the effort's `perf/baseline.md`
- [ ] GPU discipline followed: free devices only, usage checked before each run
