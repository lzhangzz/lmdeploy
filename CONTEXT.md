# LMDeploy / TurboMind

The serving engine this repo builds: an async C++ engine loop, a scheduler transaction over shared cache resources, and model modules participating in `BatchOp`. This glossary is the shared language for engine work.

## Language

**External Memory**:
Sequence cache state held outside the engine's local pool, backed by LMCache; the second place sequence cache state can live.
_Avoid_: LMCache (as an engine/scheduler concept), remote cache, KV offload, tier

**External Coverage**:
The per-sequence extent of cache state, across all cache categories, known to exist in external memory.
_Avoid_: matched length, lmcache_matched_end, external prefix hit

**External Move**:
A scheduler-planned relocation of sequence cache state between the local pool and external memory; planned as an intent inside the scheduler transaction, executed asynchronously by external memory's I/O service. Retrieve is an inbound move; store is an outbound move.
_Avoid_: transfer (as a scheduling concept), retrieval (for the scheduling-level intent), tier move
