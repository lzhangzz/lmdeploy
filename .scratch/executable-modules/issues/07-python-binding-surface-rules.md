# Python binding-surface rules

Type: grilling
Status: open
Blocked by: 10

## Question

State the Python binding-surface rules precisely: which names and signatures are frozen (module creation by type name, `add_child_raw` child names, param names `weight/bias/scales/zeros`, the `linear.py` public API `get_weight_plan`/`prepare_weight`/`get_exec_plan`/`tune`/`forward_moe`), what internal reshape is allowed in `builders/_base.py` and `weight_format.py`, and the mechanical-adaptation recipe per builder (`linear`, `attention`, `ffn`, `moe`, `deltanet`, `mla`, `text_model`). Output: the compatibility section for the spec.

## Progress note (2026-09-17 session, unclaimed without resolution)

Confirmed sub-decisions, to fold into the eventual Answer:

- **Freeze list ratified as listed**: registry type strings, config field names, child/param names, module methods (`set_plan`/`param`/`add_child_raw`/`prepare`/`set_meta_pack`), `_tm.create_module`/`WeightQuery`/config classes, WeightPlan fields (`family`/`bridge`/`shape_constraints`), `linear.py` public names (`ExecPlan`/`Linear`/`Weight`/`WeightPlan`/`get_linear` and their methods), TurboMind driver names (`create_root`/`export`/`process_weight`/`gemm`). Internal `_tm` names not in the list (e.g. `_tm.LlamaLinear`) may move.
- **New engine passes reach Python as separate bound entry points** (`register_cache()`, `bind()`) that `turbomind.py` calls in sequence — not folded into `process_weight`.
- **Builders stay byte-identical across all three implementation stages**; `linear.py`/`turbomind.py` internals may shift exactly once (when `LlamaLinear` retires); every stage gate re-runs the same Python-side checks.

Deferred: the standalone Torch Linear construction path (module + own context vs facade) — blocked on [Rework the ExecContext design](10-rework-execcontext-design.md).
