# Python binding-surface rules

Type: grilling
Status: resolved

## Question

State the Python binding-surface rules precisely: which names and signatures are frozen (module creation by type name, `add_child_raw` child names, param names `weight/bias/scales/zeros`, the `linear.py` public API `get_weight_plan`/`prepare_weight`/`get_exec_plan`/`tune`/`forward_moe`), what internal reshape is allowed in `builders/_base.py` and `weight_format.py`, and the mechanical-adaptation recipe per builder (`linear`, `attention`, `ffn`, `moe`, `deltanet`, `mla`, `text_model`). Output: the compatibility section for the spec.

## Answer

Compatibility section drafted at [`drafts/python-binding-rules.md`](../drafts/python-binding-rules.md). Decisions (progress-note items folded in, deferred sub-question settled by [Rework the ExecContext design](10-rework-execcontext-design.md)):

- **Frozen surface**: registry type strings, config classes and field names, `add_child_raw` child names, param names and `PackedTensor` alloc semantics, module methods (`set_plan`/`param`/`prepare`/`set_meta_pack`/`LinkLinearExperts`), `_tm.create_module`/`WeightQuery`, `WeightPlan` fields, `linear.py` public names (`ExecPlan`/`Linear`/`Weight`/`WeightPlan`/`get_linear` + methods), driver names (`create_root`/`export`/`process_weight`/`gemm`).
- **Free to move**: internal `_tm` names only — `_tm.LlamaLinear` retires; `linear.py` constructs `_tm.Context()` and calls `mod.bind(ctx)` after `prepare`; `TurboMind.gemm` keeps its name but returns the GEMM engine object (whose `get_weight_plan` surface is already bound).
- **New Python-visible surface (deliberate)**: `_tm.Context` for the standalone path, plus `TurboMind.register_cache()` and `TurboMind.bind()` as separate entry points — driver sequence `create_root` → `export` → `process_weight` → `register_cache()` → finalize → `bind()` → engine start.
- **Per-builder recipe: zero changes** — merged classes keep identical registry strings, children, param kinds, and config fields; `weight_format.py` untouched (quant-dispatch unification out of scope). The effort's entire Python delta is `linear.py`'s construction block and `turbomind.py`'s two new driver calls.
- **Stage invariant**: builders byte-identical across all three stages; `linear.py`/`turbomind.py` internals shift once (Linear stage); every gate re-runs the frozen-list check, builder import smoke, and `scripts/test_turbomind_model.py` end-to-end.
