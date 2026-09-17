# Assemble and review the spec

Type: prototype
Status: resolved

## Question

Assemble `.scratch/executable-modules/spec.md` from the closed tickets' answers: module interfaces with the concrete snippets from Merged Linear module interface, the Context shared-state contract, the BatchOp and decoder mapping, the Python binding-surface rules, and the migration order with verification gates, plus a risk register. Then walk the user through it for review. The map closes when the user accepts the spec.

## Answer

Spec assembled at [`spec.md`](../spec.md): problem and concepts, architecture with the normative lifecycle, the seven design sections (Linear, Ffn/Moe, Attention/DeltaNet, engine services, BatchOp/decoder mapping, Python binding surface, migration and verification) with all amendments folded to their final form (no `LinearState`, shared `MoeState`, `bind(Context&)`), a seven-item risk register with mitigations, and a decision log pointing back through the tickets. Walkthrough presented section by section; **user accepted — the map closes with all tickets resolved**. Implementation is the follow-up effort, sequenced by the stage table (Linear → FFN/MoE → Attention/GDN) with the per-gate matrix and perf baselines.
