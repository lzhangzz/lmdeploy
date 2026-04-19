# Weight classes and spec patterns — deduplication design

**Date:** 2026-04-19  
**Scope:** Document repeated patterns in C++ `*Weight` modules (`src/turbomind/models/`) and Python `TextModelSpec` subclasses (`lmdeploy/turbomind/deploy/source_model/*_spec.py`), and define a consolidation approach that **extends** the existing Python plan in [`2026-04-15-spec-dedup-design.md`](2026-04-15-spec-dedup-design.md).

---

## Relationship to prior work

The 2026-04-15 spec already targets high-value Python wins: shared `token_embeds` / `lm_head` / `norm` on `TextModelSpec`, single `reorder_rotary_emb` in `utils.py`, `InputModel` constructor hoisting, and removal of dead `model_info` / duplicate `_linear`. **This document assumes that work proceeds (or has proceeded) as written there.**

What remains after that pass is still substantial copy-paste:

- **`AttentionConfig` RoPE tail:** Every spec repeats `rope.type/base/dim/factor/max_position_embeddings` plus the `if yarn / elif llama3 / elif mrope` extended-field block (`qwen3_spec`, `qwen3_5_spec`, `gpt_oss_spec`, `glm4_moe_lite_spec`).
- **`ffn()` skeleton:** Clone `_ffn_cfg`, set `inter_size` / `fuse_silu` / `fused_moe`, construct `FfnBuilder`, `add_ffn` — same shape everywhere with small flag differences.
- **`attn()` skeleton:** `_linear` q/k/v/o, `reorder_rotary_emb_linear`, `AttentionBuilder`, `add_qkv_proj` / `add_o_proj`; variants add QK norm, sinks, per-layer `window_size`, or MLA-specific builders.
- **`moe()` template:** Large blocks assigning `MoeConfig` fields with family-specific defaults (`shared_gate`, `norm_topk_prob`, `topk_method`, `scoring_func`, etc.).
- **Packed experts:** Qwen3.5 and gpt-oss both implement “slice packed tensors → per-expert `Linear`” patterns with different naming; structure is parallel.

C++ side is largely **compositional** (most logic lives in `LinearWeight`), but several **mechanical** duplications remain.

---

## Problem (C++)

1. **Module registration:** Each `*.cc` file repeats an anonymous-namespace `struct FooRegistrar`, `ModuleRegistry::register_type("Foo", lambda)`, and a static registrar instance (`linear_weight.cc`, `attention_weight.cc`, `ffn_weight.cc`, `moe_weight.cc`, `delta_net_weight.cc`, `norm_weight.cc`, `decoder_layer_weight.cc`).
2. **Expert pointer linking:** `LinkLinearExperts` in `moe_weight.cc` is a near-copy of `LinkExperts` in `src/turbomind/kernels/gemm/test/testbed_v3.h`. Drift between production and test doubles risks subtle bugs.
3. **Thin `prepare()` methods:** `AttentionWeight::prepare` only forwards to `Module::prepare`; `DeltaNetWeight` / `NormWeight` share the “ensure float dtype on tensors” motif — low duplication but could use a tiny shared helper if more modules adopt it.

Non-problems (intentional boundaries):

- **`LinearWeight::prepare` / quantization:** Should stay centralized; other weights should not reimplement it.

---

## Approaches

### A. Incremental helpers + one shared C++ primitive (recommended)

**Python:** After the 2026-04-15 base-class methods land, add **small free functions** (or a private module `spec_config_helpers.py`) used by all specs:

- `apply_rope_fields(attn_cfg, rope_obj, max_position_embeddings)` — sets base rope fields and the yarn/llama3/mrope extensions from a shared `RopeView` or the existing rope object on the spec.
- Optional: `default_moe_config(engine_cfg, hf_cfg, *, overrides: dict)` returning a pre-filled `MoeConfig` with call-site overrides for GLM/gpt-oss/Qwen differences.

Keep `attn()` / `ffn()` / `moe()` as methods on each spec, but shrink them to “gather tensors → call helper → attach builder extras”. Preserves readability and per-family differences without deep inheritance trees.

**C++:** Introduce a macro or template in a single header, e.g. `TM_REGISTER_MODULE_TYPE(LinearWeight, "LinearWeight", core::LinearConfig)`, expanding to the current registrar boilerplate. Move **`LinkExperts` / `LinkLinearExperts` to one implementation** in a non-test header under `src/turbomind/models/detail/` or `kernels/gemm/` (depending on include layering), with `moe_weight.cc` and `testbed_v3.h` both calling it.

**Trade-offs:** Low risk, easy to review in small PRs. Requires discipline to avoid a “god helper” that knows every model.

### B. Richer `TextModelSpec` hierarchy (mixins / intermediate bases)

Add `StandardRopeAttentionSpec`, `MoeSpecMixin`, etc., and inherit in each concrete spec.

**Trade-offs:** Fewer lines in leaf specs, but harder navigation (multiple inheritance / MRO), and stronger coupling when one family needs a one-off field.

### C. Table- or schema-driven specs

Describe each model family as data (YAML/dict) that drives builder calls.

**Trade-offs:** Minimum duplication long-term, but high upfront cost, weaker IDE support, and harder debugging for contributors.

**Recommendation:** **A** for both Python and C++. Revisit **B** only if helper modules grow unwieldy; avoid **C** unless product requirements push toward many near-identical checkpoints.

---

## Design (Python — Phase 2, post 2026-04-15)

1. **RoPE on `AttentionConfig`:** Extract `apply_rope_fields(attn_cfg, rope, max_position_embeddings)` (name flexible) into `deploy/source_model/` next to `utils.py` or `spec_helpers.py`. Each spec’s `__init__` calls it once after constructing `_attn_cfg` and the spec’s `_rope` / `parse_rope_param` result. Family-specific geometry (MLA, partial rotary factor) stays in the spec **before** the call.
2. **`ffn()`:** Extract a helper `build_ffn_from_linears(spec, pfx, layer, *, inter_size=None, fused_moe=False)` that performs the repeated clone + `FfnBuilder` + `add_ffn` sequence, taking callables or the spec’s `_ffn_cfg` / contexts / ranks from `self`. Alternatively, keep `ffn()` inline but extract only the **four-line** `cfg = self._ffn_cfg.clone(); cfg.inter_size = ...` block if the team prefers minimal abstraction.
3. **`moe()` template:** Prefer a function `make_moe_config(hf_cfg, engine_cfg, dtype, defaults: Mapping)` that returns `_tm.MoeConfig()` with defaults merged from a per-family dict (e.g. Qwen3 vs Qwen3.5 vs gpt-oss vs GLM). Each spec passes overrides explicitly so differences stay visible at the call site.
4. **Packed experts:** Factor “split `gate_up` into gate/up tensors + `FfnBuilder`” into one helper shared by `qwen3_5_spec` and `gpt_oss_spec`, parameterized by key paths and tensor layout assumptions.

**Success criteria:** No intended behavior change; same checkpoint keys and builder outputs. Existing deploy / conversion tests and manual smoke loads remain green.

**Out of scope for this phase:** Changing C++ module boundaries beyond registration/linking; redesigning `LinearWeight`.

---

## Design (C++)

1. **Registration macro:** Add a header (e.g. `src/turbomind/core/register_module.h`) defining `TM_REGISTER_MODULE(ClassName, TypeString, ConfigType)` that emits the anonymous namespace registrar. Replace the seven copy-pasted registrars incrementally (one PR per module or one mechanical PR — team preference).
2. **Unified `LinkExperts`:** Move the canonical implementation to a shared location; `MoeWeight` and gemm tests include it. Delete the duplicate body from `testbed_v3.h` or reduce that file to a thin wrapper. Preserve existing FP8 blocked-pointer vs strided-pointer behavior exactly.
3. **Optional:** If `EnsureFloatDtype` patterns multiply, add `void PrepareFloatParams(Module& m, std::initializer_list<Tensor*> fields)` — only if at least three call sites benefit.

**Success criteria:** No change to weight tensors or runtime prepare order; unit tests and MoE paths unchanged modulo moved symbols.

---

## Testing

- **Python:** Run existing deploy/unit tests that touch `source_model` specs; add a focused test that two specs produce identical `AttentionConfig` rope fields for the same synthetic `hf_cfg` if such harness exists; otherwise rely on regression tests.
- **C++:** Build all targets that include `moe_weight.cc` and gemm tests; run tests that exercise `LinkExperts` / fused MoE.

---

## Rollout

1. Land or verify 2026-04-15 spec-dedup items first (reduces merge conflict surface).
2. Python Phase 2: `apply_rope_fields`, then `ffn`/`moe` helpers, then packed-expert helper — one PR per bullet where possible.
3. C++: macro first (noise-only PR), then `LinkExperts` dedup (behavior-sensitive — review carefully).

---

## Self-review checklist

- **Placeholders:** None; scope is explicit.
- **Consistency:** Aligns with 2026-04-15 for Python Phase 1; this doc is Phase 2 + C++.
- **Scope:** Single implementation plan is feasible; schema-driven specs explicitly deferred.
- **Ambiguity:** “Helper vs mixin” resolved in favor of helpers unless maintainers choose otherwise after Phase 2.
