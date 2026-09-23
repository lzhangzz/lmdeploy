# 07: Pin ownership and hard pin protection

**What to build:** Hard pin protection for in-flight external moves: move sources and destinations are transfer-pinned by the seam at submission and released at fold (`CacheBlock::transfer_refs`), with the fatal checks in the deallocation paths remaining the safety net; no coordinator-era acquire sites remain on the native path. Pins are unbounded by design (same as today) — no budget, no enforcement point: healthy waits are carried by liveness suppression, and the unhealthy window's head-of-line failures are the declared same-as-today posture. Per spec §10, §4.

**Blocked by:** 03 (retrieve), 04 (store).

**Status:** ready-for-agent

- [ ] Pins are acquired and released only by the seam along intent lifecycle; no coordinator-era acquire sites remain on the native path
- [ ] Uncertain transfers keep their pins until terminal (no abort API) and release at fold with no leaks
- [ ] The external-move-pins contract leaf is landed in the engine contract README
