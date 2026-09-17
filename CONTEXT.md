# LMDeploy TurboMind

Glossary for the TurboMind inference engine context: the C++ module system under `src/turbomind/` and the Python loader under `lmdeploy/turbomind/`.

## Language

**Module**:
The unit that owns both a module's weights and its Forward; the single thing Python constructs by name. In the target design every Module is executable.
_Avoid_: Layer, executor, weight class

**Layer**:
Legacy name for the pre-merge per-kind executor classes (`LlamaLinear`, `UnifiedAttentionLayer`, `LlamaFfnLayer`, `MoeFfnLayer`, `GatedDeltaNetLayer`); retires as Module absorbs forward execution. Do not use for new design.

**Context**:
The per-rank provider of shared execution resources (workspace, GEMM plans, tuning state) that Modules request at forward time.
_Avoid_: device resources, global state

**Binding surface**:
The Python-visible construction surface of the module tree: type names, child names, param names. Frozen by contract while internals may change.
_Avoid_: API (too broad), loader internals
