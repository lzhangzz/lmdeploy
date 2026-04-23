# TurboMind model test harness (subagent-oriented) — design

Date: 2026-04-04

## Problem

`build/test_model.py` is a minimal CLI for smoke-testing Turbomind: fixed prompt,
rough readability heuristic, and a single elapsed time. **Agent workers** need a
stable contract: **process exit code = technical success only**, **labeled
metrics** (token counts, split timings), and **full decoded text** for manual or
downstream checks—without coupling exit status to output quality heuristics.

## Goals

1. **Exit code semantics** — `0` if the run completes with no uncaught exception
   (pipeline context entered, `pipe(...)` returns, script exits normally). Non-zero
   only on failure (including failures during pipeline **creation** or **inference**).
2. **No quality gate on exit** — Do not fail the process for “gibberish” or weak
   heuristics. Optional stderr or a non-fatal warning line for degenerate output is
   allowed if useful, but must not change exit code.
3. **Printed metrics** — Full response text; `input_token_len` and
   `generate_token_len` from the `lmdeploy` `Response` object (`lmdeploy.messages.Response`).
4. **Two timings** — Wall time for **pipeline creation** (from call to
   `pipeline(...)` until context manager **entered**) and wall time for the
   **`pipe(...)`** call only.
5. **Parse-friendly stdout** — Stable labels and clear separation between metadata
   lines and free-form model text (e.g. delimiters) so subprocesses and subagents
   can split output reliably.
6. **New location** — Implement as **`scripts/test_turbomind_model.py`** (creates
   repo-root `scripts/` if absent). **`build/test_model.py` remains** until callers
   are migrated; plans may cite either during transition.

Non-goals: JSON output mode, batch model lists, MCP integration inside the script,
changing default `TurbomindEngineConfig` / `GenerationConfig` beyond what
`build/test_model.py` already uses (unless a follow-up explicitly requests flags).

## CLI and environment

Positional arguments (same order and meaning as `build/test_model.py`):

1. `model_path`
2. `cache_dir` — HF hub cache; set via `huggingface_hub.constants` **before**
   importing lmdeploy (see `CLAUDE.md`).
3. `tp` — tensor parallel degree (int)
4. `gpus` — string for `CUDA_VISIBLE_DEVICES` (e.g. `"0"` or `"0,1"`)

Invocation expectation matches existing docs:

`PYTHONPATH=<repo>/lmdeploy:<repo>/build/lib` and appropriate `CUDA_VISIBLE_DEVICES`
(or set inside the script from argument 4 as today).

## Output format

Requirements:

- Lines (or a small header block) for: `tp`, GPU selection, **`create_s`**,
  **`infer_s`**, **`input_token_len`**, **`generate_token_len`** — each with a
  stable, documented label.
- **Full response text** after metadata, inside explicit **start/end delimiters**
  so parsers are not confused by model output containing similar words.
- Human-readable is preferred; consistency beats clever formatting.

Exact delimiter strings and label spelling are fixed at implementation time and
recorded in the script’s module docstring.

## Code layout (single file)

`scripts/test_turbomind_model.py` stays one module: **`parse_args`** (argv → tuple or exit 2),
**`run_smoke_infer`** (HF cache, `CUDA_VISIBLE_DEVICES`, lazy lmdeploy import, timed pipeline
+ `pipe`, returns a small **`SmokeResult`** `NamedTuple`), **`print_report`** (unchanged stdout
sections), and **`main`** wiring parse → run → print.

## Implementation notes

- Use `time.perf_counter()` for both intervals.
- If pipeline creation fails, do not report a meaningful `infer_s`; the exception
  path should exit non-zero with a clear message.
- Base engine and generation defaults on `build/test_model.py` unless product
  requirements change.

## Testing / verification

- Run once against a known local model with `PYTHONPATH` set; confirm exit `0`,
  non-empty text, token fields present, and `create_s` / `infer_s` both positive.
- Subagent/plan docs that required `PASS | tp=…` from `build/test_model.py` should
  be updated when they switch to this script: success = **exit code 0** (and
  optional checks on printed metrics).

## Self-review (2026-04-04)

- No placeholder TBDs.
- Location `scripts/test_turbomind_model.py` matches stakeholder decision; legacy
  harness left in `build/`.
- Scope is a single new script; no batch/MCP in this spec.
