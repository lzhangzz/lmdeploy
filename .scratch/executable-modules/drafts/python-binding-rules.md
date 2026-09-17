# Python binding-surface rules

Draft spec section from [Python binding-surface rules](../issues/07-python-binding-surface-rules.md). Folds in the progress-note decisions and the standalone-path settlement from [Rework the ExecContext design](../issues/10-rework-execcontext-design.md).

## Frozen surface (compat contract)

- **Module creation**: `_tm.create_module(cfg)` with the config classes and field names — `LinearConfig`, `FfnConfig`, `MoeConfig`, `AttentionConfig`, `DeltaNetConfig`, `ModelWeightConfig`, `RopeConfig` (bound in `bind.cpp:617+`). Registry type strings (`"LinearWeight"`, `"FfnWeight"`, `"MoeWeight"`, `"AttentionWeight"`, `"DeltaNetWeight"`, …) never change.
- **Tree construction**: `add_child_raw` child names (`w_qkv`, `wo`, `w1`, `w3`, `w2`, `w1w3`, `gate`, `shared_gate`, `shared`, `experts`, `attention`, `linear_attn`, `feed_forward`, `moe_ffn`, `attention_norm`, `ffn_norm`, …) and param names (`weight`, `bias`, `scales`, `zeros`, `global_scale`, `sinks`, `conv1d`, `A_log`, `dt_bias`, `score_correction_bias`).
- **Module methods**: `set_plan`, `param(kind)` alloc/copy semantics (including `PackedTensor` `alloc_shape`/`alloc_dtype` byte-equality asserts in `_copy_shard_to_param`), `prepare`, `set_meta_pack`, `_tm.LinkLinearExperts`.
- **Plan query**: `_tm.WeightQuery` fields, `get_weight_plan(query)` on the object `TurboMind.gemm(index)` returns, `WeightPlan` fields (`family`, `bridge`, `shape_constraints`).
- **Torch API (`linear.py`)**: public names `ExecPlan`, `Linear`, `Weight`, `WeightPlan`, `get_linear`, and their methods `get_weight_plan`/`prepare_weight`/`fuse_weight`/`get_exec_plan`/`tune`/`forward_dense`/`forward_moe`.
- **Driver (`turbomind.py`)**: `create_root`, `export`, `process_weight`, `gemm`.

## Free to move (internal `_tm` names not in the frozen list)

- `_tm.LlamaLinear` retires: `linear.py:139` constructs `_tm.Context()` instead and calls `mod.bind(ctx)` after `prepare` (the Rework the ExecContext design settlement); `forward_dense`/`forward_moe`/`get_exec_plan`/`tune` rebind onto the module.
- `TurboMind.gemm`'s return type changes from `LlamaLinear&` to the GEMM engine object (the `gemm::Gemm` surface with `get_weight_plan` is already bound at `linear_bind.cpp:152`); the method name stays frozen.

## New Python-visible surface (deliberate additions)

- `_tm.Context` — `turbomind::Context` exposed for the standalone path (constructor + engine accessor).
- `TurboMind.register_cache()` and `TurboMind.bind()` — separate bound entry points (not folded into `process_weight`); the driver calls them in sequence: `create_root` → `export` → `process_weight` (prepare walk) → `register_cache()` → category finalize (internal) → `bind()` → engine start.

## Per-builder recipe

`builders/_base.py` and every per-model builder (`linear`, `attention`, `ffn`, `moe`, `deltanet`, `mla`, `text_model`, `vision_model`) need **zero changes**: the merged classes keep identical registry strings, X-macro children, param kinds, and config fields. `weight_format.py` is untouched (quant dispatch stays twin-tabled; unification is out of scope). The only Python-side edits in the whole effort: `linear.py`'s construction block and `turbomind.py`'s driver sequence gaining the two new calls.

## Stage invariant (gate-checked)

Builders stay byte-identical across all three implementation stages; `linear.py`/`turbomind.py` internals shift exactly once (when `LlamaLinear` retires and the new entry points appear, staged with the Linear stage); every stage gate re-runs the same Python-side checks: the frozen-list grep (no new/renamed public names), a builder import smoke, and `scripts/test_turbomind_model.py` end-to-end.
