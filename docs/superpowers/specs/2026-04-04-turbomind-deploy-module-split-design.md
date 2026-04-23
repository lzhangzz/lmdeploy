# TurboMind deploy `module.py` split — design

Date: 2026-04-04

## Problem

`lmdeploy/turbomind/deploy/module.py` (~1000 lines) mixes unrelated concerns: the
`TextModelSpec` contract, low-level tensor layout helpers, QKV merge / GDN fusion /
FFN fusion, and `commit_linear` / `commit_tensor` bridge to C++. That makes it
hard for contributors adding architectures to find the right code and understand
boundaries.

## Goals

1. **Clear structure** — smaller, responsibility-focused modules so new `TextModelSpec`
   subclasses are easier to author and review.
2. **Mechanical import updates are acceptable** — call sites in this repo may switch
   from `..module` / `.module` to explicit submodule imports (no requirement to
   preserve `from .module import …` compatibility).
3. **No intentional behavior change** — refactor only; same public operations for
   specs, loader, and `LoadContext`.

Non-goals: faster load, changing C++ lazy-child / `ensure_child` work, or redesigning
`TextModelLoader`’s imperative loop (those are separate efforts).

## Chosen approach

Use a **small package** under `lmdeploy/turbomind/deploy/weights/` (final name may
be `weights` or `weight_pipeline`; this document uses **`weights`**).

**Rejected alternatives:** (a) two giant files (`spec` + `ops`) — still a junk drawer;
(b) keeping `module.py` as a permanent facade — extra indirection when imports may
change anyway.

## Architecture

### Modules and responsibilities

| Module | Responsibility |
|--------|----------------|
| `weights/layout.py` | `SplitSide`, RoPE/permutation helpers (`permute_v2`, `permute_v2_partial`), `merge_qkv_v2`, `merge_qkvg_v2`, and any dim map constants shared with commit (e.g. `_SPLIT_SIDE_TO_DIM`). |
| `weights/qkv.py` | `merge_qkv_linear`, `_block_ops_need_dequant`, `_dequant_linear`, `_ensure_compatible_formats`; depends on `layout`. |
| `weights/gdn.py` | GDN input fusion (`fuse_gdn_in_proj`, `_tp_interleave_tensor`, `_GDN_IN_PROJ_KEYS`); depends on `layout` / `Linear` as today. |
| `weights/ffn.py` | `_should_fuse_silu`, `_can_fuse_w1w3`, `_shard_linear_for_tp`, `_fuse_and_commit_ffn`, `_torch_dtype_to_cpp` (and related small helpers co-located with FFN). |
| `weights/commit.py` | `commit_linear`, `commit_tensor`, `_infer_cpp_linear_dtype`, `_cast_shard_for_tm`, `_infer_compute_dtype`; depends on `layout` (for `SplitSide` / dim maps). |
| `weights/spec.py` | `TextModelSpec` only; imports `qkv` / `layout` (and others) for default method implementations such as `attn_linears`. |
| `weights/__init__.py` | Optional thin `__all__` re-exports for ergonomic `from lmdeploy.turbomind.deploy.weights import TextModelSpec, SplitSide` — **not** required if all call sites import from submodules. |

### Import rules (no cycles)

- `layout` → stdlib / `torch` / project types only (no imports from other `weights.*`).
- `qkv` → `layout` (+ `Linear`, formats).
- `gdn`, `ffn` → `layout` and existing deploy types as needed.
- `commit` → `layout` (+ `Linear`, `_turbomind` where applicable).
- `spec` → `qkv`, `layout`, and other helpers referenced by base-class methods — **never** the reverse (ops modules must not import `TextModelSpec`).

### Call sites to update (current)

- `lmdeploy/turbomind/deploy/text_model_loader.py`
- `lmdeploy/turbomind/deploy/load_context.py`
- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`

Re-scan with `rg 'from.*module import|deploy\.module'` before merging.

## Migration plan

1. Create `weights/` package and move code **verbatim** (same names, same semantics)
   from `module.py` into the modules above; run `ruff` / project linter on new files.
2. Fix internal cross-references inside the moved code (e.g. `merge_qkv_linear`
   calling helpers now in `layout`).
3. Update the six call sites listed above to import from `weights.*`.
4. Remove `module.py` (or replace with a one-line comment pointing to `weights/`
   if policy requires a tombstone — prefer full removal).
5. Smoke-check: Python import of deploy stack; run an existing convert/load or
   `build/toy_example.py` path per project convention; confirm no new warnings beyond
   baseline.

## Testing and verification

- **Regression bar:** identical tensor layouts and C++ commits for at least one
  representative model after the split (optional but recommended: quick compare
  before/after on a small checkpoint or golden test if available).
- **Lint:** clean diagnostics on all touched files.
- **No new tests required** solely for file moves if behavior is unchanged; add tests
  only if a moved helper is extracted in a way that benefits from direct unit tests.

## Error handling and observability

- Preserve existing `warnings.warn` paths (e.g. QKV merge dequant fallback) and
  exception messages from `commit_*` paths so downstream debugging does not regress.
- Do not introduce new exception types unless a moved function’s module boundary
  requires it (avoid).

## Self-check

- No TBD sections; scope is a single refactor deliverable.
- Single implementation plan should follow this spec (`writing-plans`).
