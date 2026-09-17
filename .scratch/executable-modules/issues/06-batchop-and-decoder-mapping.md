# BatchOp and decoder mapping

Type: grilling
Status: open
Blocked by: 03

## Question

Map executable modules onto the engine loop: which module implements or forwards which BatchOp op (`kAdd`/`kSetup`/`kPrepare`/`kForward`/`kUnprep`/`kFetch`/`kUpdate`/`kDel`) with thread placement per `contracts.batchop*`; how `UnifiedDecoder` shrinks mechanically to a loop over per-layer modules; what `LanguageModel` touches. Contract meanings must hold; if the README needs rewording, flag it per `checklist.contract-sync` (same-change doc update).
