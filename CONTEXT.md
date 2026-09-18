# LMDeploy TurboMind

Glossary for the TurboMind inference engine context: the C++ module system under `src/turbomind/` and the Python loader under `lmdeploy/turbomind/`.

## Language

**Module**:
The unit that owns both a module's weights and its Forward; the single thing Python constructs by name. In the target design every Module is executable.
_Avoid_: Layer, executor, weight class

**Implementation**:
A concrete executable Module subclass of a kind — it holds the kind's weights and its whole Forward, and is selected at construction from the engine configuration. A kind may have several sibling Implementations (Moe: default and DeepEP).
_Avoid_: executor, weight class, backend, internal branch

**Layer**:
Legacy name for the pre-merge per-kind executor classes (`LlamaLinear`, `UnifiedAttentionLayer`, `LlamaFfnLayer`, `MoeFfnLayer`, `GatedDeltaNetLayer`); retires as Module absorbs forward execution. Do not use for new design.

**Context**:
The per-rank infrastructure holder: GEMM engine, build stream and allocator, comm, the warm-up flag. Holds no module state of any kind.
_Avoid_: ExecContext (dead name), services bundle, god-object framings, kind-state slots

**BindEnv**:
The transient per-rank bundle alive only during the bind pass: engine parameters, the Context, the cache registry, the warm-up flag, phases, and the shared-state map.
_Avoid_: composition root, services bundle

**Kind state**:
The per-rank shared state of a module kind or of one Implementation of a kind; created by the first Module of the kind during the bind pass, shared with the rest through the BindEnv map, co-owned by the Modules afterwards.
_Avoid_: Context slot, executor, per-layer copies

**Binding surface**:
The Python-visible construction surface of the module tree: type names, child names, param names. Frozen by contract while internals may change.
_Avoid_: API (too broad), loader internals
