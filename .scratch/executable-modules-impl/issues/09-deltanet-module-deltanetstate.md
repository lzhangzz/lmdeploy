# 09: DeltaNet executable module with DeltaNetState

**What to build:** The gated-delta-net weight module becomes the executable DeltaNet on the same pattern as attention: per-layer modules registering their conv and recurrent state byte ranges with the checkpoint category during the registration pass, preserving the engine contracts — rank-local shards published and restored in lockstep at the same global position, position-zero state clears, the block-grid recurrent layout with its per-layer in-block offsets. The GDN layer class retires; per-kind workspaces move into DeltaNetState.

**Blocked by:** 07 (Make the cache-registration pass real).

**Status:** resolved

- [x] DeltaNet module runs its forward with plan/prepare-state lifecycle intact and contracts preserved (lockstep publish/restore, position-0 clears, block-grid offsets)
- [x] Conv/recurrent ranges registered during the registration pass; offsets module-held
- [x] GDN layer class retired and deleted; per-kind workspaces in DeltaNetState
- [x] Build green; if a locally cached model exercises GDN, its end-to-end run passes with meaningful responses — otherwise verification is the build plus the engine contract checks, stated in the ticket close-out

## Answer

`DeltaNetWeight` is renamed `DeltaNet` (`models/delta_net.{h,cc}`, registry
string unchanged via `TM_MODULE_REGISTER_AS("DeltaNetWeight", …)` so the
Python builder name is stable) and now executes the full GDN forward:
fused in-proj, beta/gate, fused conv1d, L2-normalized QK, the delta-rule
recurrent/chunked kernels, gated RMS norm, out-proj — a line-faithful
transcription of the retired executor with the module holding per-layer
state (conv element offset, layer index, in-block recurrent offset).

Registration split (agreed during planning; deviation from the ticket's
"recurrent … during the registration pass" wording): each module registers
its conv bytes per-layer in the `register_cache` pass (accumulation
channel, alignment 1 — byte-identical layout to the old single bulk
registration, offsets element-converted), while the recurrent block grid
is registered ONCE per rank by `DeltaNetState` at creation during `Bind`,
before finalization — per-module registration would double-register shared
block-rows when `TM_GDN_BLOCK_CONFIG` sets `L_b > 1`. `rec_base_ == 1`
either way (sole checkpoint composite registrant; part ids are
channel-independent). Documented in `contracts.gated-deltanet` and the
code.

`DeltaNetState` (created in `TurboMind::Impl::Bind`, held by the Context,
mirroring AttentionState) owns the per-rank workspaces: per-phase batch
slots, delta-rule planner, staging pointer buffers, aux stream/events,
work counter, block-grid geometry, wave counter. Bind assigns each module
its layer index (`RegisterLayer`: checks `module.tp_size ==
attn_tp*attn_cp`, checks bind order == layer order, idempotent across
`update_params` re-binds — the reload path re-runs `_bind`, which would
otherwise abort on the second walk). Wave guard identical to Attention;
`kPrepare` is batch-level and runs once per pass (layer-0 guard) exactly
as the retired executor did.

`GatedDeltaNetLayer.{h,cc}` deleted; `UnifiedDecoder` lost its
`linear_attn_layer_` member and constructs nothing (GDN forward dispatched
like attention's); `CacheRegistry&` removed from the UnifiedDecoder /
LanguageModel ctor chain; `decoder_layer_weight` child renamed; stale
comments and README sections updated (`module-lifecycle` now states all
kinds register pre-finalization).

Verification: build green; all four gate models smoke-tested with
meaningful ≥254-token responses (35B-FP8 is the GDN model — also passed
with `--cache-prompt all --cache-checkpoint-interval 64`); 8B passed with
prefix caching; re-smoked after review fixes. Perf stage 4
(`perf/stage4.md`): footprints byte-identical to stage 3; throughput
within the same-window control band (dense/30B models with untouched
paths drifted −0.2…−2.2%; FP8 −1.0…−1.8%) — deterministic re-runs
confirmed stable measurements; gate pass. Two review subagents (standards
+ spec/fidelity vs the retired executor) found one blocker (the
`update_params` re-bind abort — fixed via idempotent `RegisterLayer`) and
minor items, all addressed.
