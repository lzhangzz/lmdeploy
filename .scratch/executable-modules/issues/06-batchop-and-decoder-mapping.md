# BatchOp and decoder mapping

Type: grilling
Status: resolved

## Question

Map executable modules onto the engine loop: which module implements or forwards which BatchOp op (`kAdd`/`kSetup`/`kPrepare`/`kForward`/`kUnprep`/`kFetch`/`kUpdate`/`kDel`) with thread placement per `contracts.batchop*`; how `UnifiedDecoder` shrinks mechanically to a loop over per-layer modules; what `LanguageModel` touches; the kSetup fill flow for the pooled per-phase `AttentionBatch` (Merged Attention module design); and the engine pass order prepare → register_cache → finalize → bind → first Run. Contract meanings must hold; if the README needs rewording, flag it per `checklist.contract-sync` (same-change doc update).

## Answer

Mapping drafted at [`drafts/batchop-decoder-mapping.md`](../drafts/batchop-decoder-mapping.md). Decisions:

- **Glue stays in the decoder**: the fused `AllreduceResidualRMSnorm`, per-layer `ffn_group` mapping, global/local hidden-state plumbing, and next-layer norm staging remain decoder-owned; only executor calls swap to module forwards. `DecoderLayerWeight` stays the Python-built data composite (no executable layer composite — that would be the orchestration redesign ruled out of scope). Zero behavior change.
- **Default no-op `Run(BatchOp, int, TensorMap&)` on `core::Module`**: the decoder's generic walk calls every layer child for every op; kinds implement their subset (Attention: kSetup fill + kForward + kUnprep as today; DeltaNet adds kPrepare state clears and frontier ops; Linear/Ffn/Moe: kForward only) and ignore the rest, per `contracts.batchop`. `kForward` keeps the decoder's orchestrated loop — two paths, as today.
- **Versioned fill guard**: the pooled `AttentionBatch` slot carries a batch-version stamp; `Attention::kSetup` fills batch-level parts only when stale — order-independent, no coordinator.
- **`LanguageModel::Run` dispatch unchanged** (single BatchOp entry per `concepts.module`); its `linear_` uses (embedding/output GEMMs) migrate to owned `Linear` modules per the migration order.
- **README contract-sync list compiled**: `contracts.cache-content`, `contracts.unified-attention`, `contracts.gated-deltanet` (class renames + new shape), `ownership.module-cache` + `contracts.category-registration` ("during construction" → "during the register_cache pass"), plus a new lifecycle paragraph for the pass order and dispatch rule. Same-change updates during implementation; `concepts.module` needs no change.
