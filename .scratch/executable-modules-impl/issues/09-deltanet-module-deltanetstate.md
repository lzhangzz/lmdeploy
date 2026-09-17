# 09: DeltaNet executable module with DeltaNetState

**What to build:** The gated-delta-net weight module becomes the executable DeltaNet on the same pattern as attention: per-layer modules registering their conv and recurrent state byte ranges with the checkpoint category during the registration pass, preserving the engine contracts — rank-local shards published and restored in lockstep at the same global position, position-zero state clears, the block-grid recurrent layout with its per-layer in-block offsets. The GDN layer class retires; per-kind workspaces move into DeltaNetState.

**Blocked by:** 07 (Make the cache-registration pass real).

**Status:** ready-for-agent

- [ ] DeltaNet module runs its forward with plan/prepare-state lifecycle intact and contracts preserved (lockstep publish/restore, position-0 clears, block-grid offsets)
- [ ] Conv/recurrent ranges registered during the registration pass; offsets module-held
- [ ] GDN layer class retired and deleted; per-kind workspaces in DeltaNetState
- [ ] Build green; if a locally cached model exercises GDN, its end-to-end run passes with meaningful responses — otherwise verification is the build plus the engine contract checks, stated in the ticket close-out
