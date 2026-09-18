# 04: Final gate — full seam run and frozen-surface sweep

**What to build:** The effort's verification story closes: every seam from the
spec run green in one place, the frozen surfaces proven unchanged by diff, and
the records (glossary, spec amendment, ADR) checked against the landed code.

**Blocked by:** 03 (MoE implementations as module subclasses).

**Status:** ready-for-agent

- [ ] `ninja` clean in the build folder
- [ ] `Qwen/Qwen3.5-35B-A3B` via `scripts/test_turbomind_model.py` (AS IS, never
      batched) answers with a meaningful ≥128-token response; optionally the
      FP8 variant, per prior gate matrices
- [ ] Standalone linear path smoke passes (bind shim → degenerate env)
- [ ] Frozen-surface sweep: no Python-visible deltas — registry strings, config
      classes and fields, `create_module`, and the standalone path all diff
      clean against pre-effort
- [ ] Checkpoints load unmodified
- [ ] The glossary entries, the executable-modules spec amendments, and ADR 0001
      still describe the landed code (no drift)
