# 04: Final gate — full seam run and frozen-surface sweep

**What to build:** The effort's verification story closes: every seam from the
spec run green in one place, the frozen surfaces proven unchanged by diff, and
the records (glossary, spec amendment, ADR) checked against the landed code.

**Blocked by:** 03 (MoE implementations as module subclasses).

**Status:** resolved

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

## Answer

All seams green on the landed code (commit d0d5f6213, 2026-09-18):

- `ninja` clean in the build folder (DeepEP path compiled under USE_NCCL).
- `Qwen/Qwen3.5-35B-A3B` via `scripts/test_turbomind_model.py` AS IS: 256
  meaningful tokens. Also ran the optional FP8 variant
  (`Qwen/Qwen3.5-35B-A3B-FP8`): 256 meaningful tokens — one run covers the
  default MoE implementation plus the Attention and DeltaNet state
  migrations (all Qwen3.5 models are GDN).
- Standalone linear path smoke: dense `prepare_weight` (create → prepare →
  `bind(context)` shim → degenerate env) and grouped `LinkLinearExperts` →
  `bind` both pass on the final build.
- Frozen-surface sweep vs pre-effort (`ef0fe14c5`):
  - Python: the only delta is the approved `moe_a2a_backend='auto'`
    normalization in `update_parallel_config` (same site, now
    unconditional; no API/config surface change). No config class or field
    changed; `create_module` and the standalone path untouched.
  - pybind: zero `.def`/class registrations added, removed, or retyped —
    only shim internals and the guard plumbing changed.
  - Registry strings: exactly one swap, `TM_MODULE_REGISTER_AS("MoeWeight",
    Moe, ...)` → `register_type("MoeWeight", CreateMoe)` — the same frozen
    string, now a custom factory per ADR 0001. The other registrations are
    untouched.
  - Checkpoints load unmodified (both models load from the local cache;
    no converter or format change).
- Records checked against the landed code: `CONTEXT.md` glossary
  (Implementation, Context, BindEnv, Kind state) describes the landed
  shapes; the `bind(Context&)` reference in the engine README's lifecycle
  contract was updated to `bind(BindEnv&)` with the walk-time state
  creation; ADR 0001, the spec, and the executable-modules spec amendment
  no longer describe `update_params` reloads (unsupported per user,
  2026-09-18).

### Review follow-up (same day)

Two-axis review (standards + spec) vs `ef0fe14c5` landed four small fixes on
top of d0d5f6213: two glossary naming breaches in comments (attention/delta
net "executor" phrasing, a "backend" noun in the Python normalization
comment), a type check on the DecoderLayerWeight cast in DeltaNet's sibling
climb, and a loud `TM_CHECK(engine.ep_size > 1)` in `MoeDeepEPState` — the
string-only factory selects DeepEP for `moe_a2a_backend="deepep"` +
`ep_size == 1` (a misconfiguration the old code silently degraded to the
default transport), and the DeepEP side now rejects it precisely instead of
building a dispatcher on the never-split expert-parallel comm. The spec's
ambient-parking wording was amended to the landed guard mechanism. Post-fix
gate: `ninja` clean; `Qwen/Qwen3.5-35B-A3B` answers 256 meaningful tokens.
Review findings intentionally not acted on (settled design): the inline
find-or-create rendezvous (no helper indirection), the type-erased map with
string keys, and the per-implementation state structs without a shared base.
