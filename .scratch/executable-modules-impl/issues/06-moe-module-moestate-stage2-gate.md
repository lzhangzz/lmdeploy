# 06: Moe executable module with shared MoeState, and stage-2 gate

**What to build:** The MoE weight module becomes the executable Moe: gate, routing (including the warm-up random path), transport gather/scatter with default-vs-DeepEP as an internal branch, grouped expert execution through the linked block, combine, and shared-expert fold. The per-rank transport state — dispatcher, routing buffers, scales-copy stream — becomes one shared MoeState bound to every MoE module per the spec's engine-services section; both old implementation classes retire. Then run the stage-2 gate including the memory-footprint and startup-cost check.

**Blocked by:** 05 (Ffn executable module).

**Status:** resolved

- [x] Moe forward owns routing through shared fold; default-vs-DeepEP is an internal branch, not separate classes
- [x] One shared per-rank MoeState (dispatcher, routing buffers, scales stream) bound to all MoE modules
- [x] Both old MoE implementation classes retired and deleted
- [x] Stage-2 gate passes: four-model matrix with meaningful responses, decode ≥98% of baseline, load time within noise, footprint/startup check recorded

## Answer

**Shape landed:**

- `MoeWeight` → `Moe` (`models/moe_weight.{h,cc}` renamed to `moe.{h,cc}`); registry string stays
  `"MoeWeight"` via `TM_MODULE_REGISTER_AS`; macros `MOE_CHILDREN`/`MOE_PARAMS` mirror ticket 05's
  `FFN_CHILDREN`/`FFN_PARAMS`. `set_meta_pack` binding cast updated to `ft::Moe`.
- `Moe::Forward(input, Ref<Tensor> output, token_mask, local_token_nums)` dispatches internally on
  `state_->use_a2a` between private `ForwardDefault` and `ForwardA2A` — verbatim transcriptions of the
  retired `MoeFfnDefaultImpl::Forward` / `MoeFfnA2AImpl::Forward` (reviewers verified every kernel
  call, buffer size, memset, branch predicate, and the warm-up SampleUniform path
  argument-by-argument). The A2A path keeps its legacy in-place contract (writes through the input
  tensor; decoder passes the same tensor for input and output) — documented on the Forward decl.
- `MoeState` (`moe.h`): the retired executors' state once per rank — routing buffers, DeepEP
  `TokenDispatcher`, scales-copy stream + events, engine geometry, `is_warm_up` ref. Branch
  selection (BUILD_MULTI_GPU && ep>1 && deepep; bf16 check on A2A only) lives in its ctor.
- Ownership: `Context` (`models/llama/context.h`) holds `unique_ptr<MoeState>` (fwd-declared; ctor,
  dtor, and setter out-of-line in new `llama/context.cc` — the kind-state type is incomplete in the
  header). Created once per rank in `TurboMind::Impl::Bind` **before** the bind walk (Python order is
  register_cache → bind → create_engine, and WarmUp inside create_engine runs the decoder, so the
  state must exist by bind time), guarded `!ctx.moe_state()` so `update_params` reloads reuse it
  (no dispatcher re-JIT). Buffers allocate under `{ctx.core_stream, ctx.allocator}` — the engine
  stream/allocator, where the retired executor's buffers lived, not the weight-loading pool.
  `Moe::bind` captures the state and still descends `block_` explicitly.
- `MoeFfnLayer`/`MoeFfnLayerImpl`/`MoeFfnDefaultImpl`/`MoeFfnA2AImpl` deleted
  (`llama/moe_ffn_layer.{h,cc}` gone); decoder calls
  `weights.at(layer)->moe_ffn->Forward(global_hidden_states, global_hidden_states, token_mask,
  local_token_nums)`; `ffn_group_` logic keyed on `has_moe`. Dead `llama/CMakeLists.txt` no longer
  lists the deleted file.

**Conscious accepts (from review):** routing scratch now allocates before the cache-budget
`cudaMemGetInfo` (MoE models' object-cache budget 32 MB smaller — more honest accounting); DeepEP
JIT warm-up moved from CreateEngine to Bind (rank-uniform; comm groups fully built in CreateContext,
which precedes Bind).

**Gate (perf/stage2.md):** four models end-to-end, meaningful ≥254-token responses. Throughput vs
recorded baselines: 8B 98.4–101.2%, AWQ 101.3–102.9%, 30B-A3B 100.6–101.1%; FP8 carries the stage-1
environmental finding (recorded 2451.5 unreproducible on the unmodified baseline commit) and shows
no regression against fresh same-day references (99.5% of today's baseline-commit bs=32; 100.2%/
98.9% of the stage-1 tree at bs=1/128). Loads within the dense models' ±1 s noise band. Footprint
check: free-memory-at-budget A/B vs stage-1 shows exactly 32.0 MB once-per-rank MoeState
relocation on both MoE models — no per-layer multiplication (×48 would be ~1.5 GB).
