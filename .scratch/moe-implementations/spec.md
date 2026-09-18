# Spec: MoE implementations as module subclasses + bind-pass state map

Status: ready-for-agent

## Problem Statement

The MoE code merged two genuinely different implementations — the AG/RS default
transport and the DeepEP transport — into a single class behind an internal
branch, and the module-state wiring parks every kind's shared state in slots on
the engine Context. Contributors cannot add or evolve an implementation without
editing the merged class; the Context accretes domain state it should not know
about; and module kinds that will need multiple implementations in the future
(attention, transports, FFN/Linear variants, DeltaNet) have no repeatable
pattern to follow. The architecture does not match what the module system is
supposed to be.

## Solution

Each MoE implementation becomes an executable Module subclass — `MoeDefaultImpl`
and `MoeDeepEPImpl`, each holding its layer's weights and its whole forward —
selected at construction by the registry factory from the ambient per-rank
engine configuration. Per-rank state moves off the Context entirely: each
implementation's state is created once by the first module during the bind pass
and shared with its siblings through a transient, type-erased map carried on a
`BindEnv`. The Context slims to pure infrastructure. Attention and DeltaNet
migrate to the same rendezvous. Behavior is verbatim, the Python surface is
untouched, and the pattern is recorded (ADR 0001, glossary) for future kinds.

## User Stories

1. As a TurboMind engine developer, I want the AG/RS default and DeepEP MoE paths
   to live in separate implementation classes, so that each can evolve without
   contaminating the other.
2. As a TurboMind engine developer, I want each implementation to be an
   executable Module subclass holding its own weights, so that the code never
   regresses to an executor-beside-a-weight-class split.
3. As a TurboMind engine developer, I want implementation selection to happen at
   module construction from the engine configuration, so that what a module
   *is* is fixed at birth and never reconsidered later.
4. As a TurboMind engine developer, I want selection to trust the backend string
   only (`default` or `deepep`, anything else fatal), so that misconfiguration
   fails loudly instead of silently degrading.
5. As a TurboMind engine developer, I want the bfloat16 invariant enforced on
   the DeepEP side, so that unsupported dtypes fail fast at creation.
6. As a module-system maintainer, I want the MoE config class to remain a
   constant model declaration, so that configs never accumulate engine policy or
   sizing data.
7. As a module-system maintainer, I want per-rank state created by the first
   module of a kind and shared with the rest, so that state sharing needs no
   engine glue and no scanning.
8. As a module-system maintainer, I want the shared-state map alive only during
   the bind walk with modules co-owning the states afterwards, so that no
   long-lived rendezvous object exists anywhere.
9. As a module-system maintainer, I want the Context to hold no module state of
   any kind, so that it stays a pure per-rank infrastructure holder.
10. As a module-system maintainer, I want the bind pass to receive a `BindEnv`
    bundle (engine params, context, cache registry, warm-up flag, phases, shared
    map), so that everything a binding module needs arrives under honest,
    existing names.
11. As a module-system maintainer, I want the engine's bind step to only build
    the env and walk the tree, so that no kind-module scan and no pre-walk state
    creation exists in engine glue.
12. As a module-system maintainer, I want Attention and DeltaNet states on the
    same rendezvous, so that all kinds follow one mechanism.
13. As a module-system maintainer, I want DeltaNet's per-layer registration and
    bind-order enforcement preserved, so that recurrent state semantics are
    unchanged.
14. As a module-system maintainer, I want the warm-up flag to remain an
    engine-owned stable reference exposed through the bind env, so that all its
    existing consumers keep working.
15. As a single-node model runner, I want the default path's behavior preserved
    verbatim, so that model outputs are identical before and after the change.
16. As a multi-node DeepEP operator, I want the DeepEP path's code preserved
    verbatim (compiled, structurally intact), so that the eventual multi-GPU
    execution behaves as before.
17. As a multi-node DeepEP operator, I accept that parameter reloads re-create
    state and re-run the dispatcher JIT, so that the sharing mechanism stays
    simple.
18. As a standalone GEMM user, I want the Python `Module.bind(context)` surface
    unchanged, so that the standalone linear path keeps working unmodified.
19. As a contributor adding a third MoE transport, I want the pattern (subclass,
    per-rank state, factory mapping) documented and isolated per file pair, so
    that I can add an implementation without touching the existing two.
20. As a future attention-implementation author, I want the same named
    convention available without any core API or registry change, so that kinds
    adopt it only when they actually split.
21. As a reviewer, I want the implementation bodies moved verbatim, so that
    review reduces to verifying moves rather than re-deriving correctness.
22. As a release engineer, I want zero Python-visible surface change and zero
    config class change, so that loading pipelines and checkpoints are
    unaffected.
23. As a TurboMind engine developer, I want the decoder's MoE call surface
    unchanged, so that the orchestration layer is untouched.
24. As a TurboMind engine developer, I want meta-pack aliasing and grouped-block
    building unchanged, so that MoE models with shared expert packs keep
    loading.
25. As a TurboMind engine developer, I want each implementation in its own file
    pair together with its state, so that navigation is by implementation.

## Implementation Decisions

- **Module subclasses.** The MoE base keeps the frozen `"MoeWeight"` type
  string, the X-macro children/params, the config-field ctor, `prepare` /
  `link_block`, the shared `Gate` helper, and declares a pure-virtual `Forward`
  with today's four arguments. `MoeDefaultImpl` and `MoeDeepEPImpl` subclass it;
  each is the executable module holding its layer's weights through the
  inherited children. Bodies are verbatim moves of the current internal
  branches; the DeepEP in-place output contract stays documented on the
  declaration.
- **Registry factory.** One custom factory callable registered under the frozen
  `"MoeWeight"` string (the registry already supports arbitrary factories)
  constructs the selected subclass.
- **Selection input.** The ambient per-rank EngineParam, parked on the
  module-build thread by the per-rank weight-context guard (beside the
  stream/allocator it pushes) and read at module construction. String-only
  semantics: `"default"` → default subclass, `"deepep"` → DeepEP subclass,
  anything else fatal; the bfloat16 check lives on
  the DeepEP side; no expert-parallel-size conditioning.
- **Config purity.** The MoE config class is untouched — a constant model
  declaration.
- **State topology.** `MoeDefaultState` (routing scratch) and `MoeDeepEPState`
  (dispatcher, top-k buffers, scales-copy stream/events): created once per rank
  by the first module of that implementation during the bind pass; rendezvous on
  `std::unordered_map<std::string, std::shared_ptr<void>>` keyed `"moe.default"`
  / `"moe.deepep"`; the other modules fetch and cast; all co-own via
  `shared_ptr`. Buffers allocate under the engine build stream/allocator, not
  the weight-loading pool.
- **Bind pass.** `bind(BindEnv&)` replaces `bind(Context&)`. BindEnv carries
  exactly: `const EngineParam& engine`, `turbomind::Context& context`,
  `CacheRegistry& registry`, `int& is_warm_up`, the shared-state map, and
  `int& phases`. It is defined beside the Context and forward-declared from
  core, as the Context is today.
- **Engine glue.** The engine's bind step builds the env and walks the tree; the
  kind-module scan and pre-walk kind-state creation are deleted; the Context's
  three kind-state slots and accessors are deleted. Linear's GEMM-engine capture
  is unchanged, via the env's context.
- **Attention/DeltaNet.** Their states rendezvous on the same map (single
  implementation each; MLA remains an internal branch of Attention's forward).
  DeltaNet's per-layer registration and layer-order enforcement are unchanged;
  the recurrent grid still registers while the cache category is open.
- **Reload semantics.** A re-bind runs against a fresh map: states are
  re-created and the DeepEP dispatcher re-runs its JIT. Accepted. (`update_params`,
  the reload path at design time, has since been dropped as unsupported.)
- **Standalone path.** The pybind shim keeps the Python-visible
  `Module.bind(context)` signature and internally assembles a degenerate BindEnv
  (default engine params, empty registry, local flags, fresh map).
- **Layout.** The base keeps its own file pair (with the factory); each
  implementation gets its own file pair containing its subclass and its state;
  BindEnv lives beside the Context.
- **Recording.** Terminology is the Implementation axis; the glossary
  (`CONTEXT.md`), the executable-modules spec, and ADR 0001 are already updated.

## Testing Decisions

- A good test here asserts external behavior only: a clean build, meaningful
  model responses, and unchanged Python surfaces — never internal structure
  (class shape, map contents, bind mechanics).
- Seams — all existing, none new:
  1. **Build** (`ninja` in the build folder): the compile gate and the DeepEP
     code's only coverage.
  2. **End-to-end model verification** via the normative test script, used AS IS
     (never modified, never batched), responses ≥128 tokens and verified
     meaningful: `Qwen/Qwen3.5-35B-A3B` (a MoE + GDN model — one run covers the
     default MoE implementation, the Attention state migration, and the DeltaNet
     state migration); the FP8 variant optionally, per prior gate matrices.
  3. **Standalone linear path**: any existing use exercises the bind shim →
     degenerate BindEnv → bind walk without a model.
- Prior art: the executable-modules stage gates (four-model matrix, perf ≥98%).
  This effort deliberately narrows to compile + default-path per the accepted
  decision; the DeepEP path remains unexecuted by gates (recorded in ADR 0001
  and the stage-2 record).
- No new test seams are introduced.

## Out of Scope

- Executing the DeepEP path (no multi-GPU expert-parallel run; the caveat is
  recorded, not hidden).
- Performance gates (no ≥98% throughput requirement for this effort).
- New implementation kinds (no attention split; MLA stays an internal branch).
- Any Python-visible surface change; any config-class change.
- A core-level implementation registry or API (named convention only).
- Context cleanup beyond the kind-state slots (write-only device props, unused
  stream handle, unconsumed comm split).
- Behavioral normalization (the DeepEP in-place contract, warm-up random
  routing, and local-token-count semantics are preserved as-is).

## Further Notes

- The decision record with rejected alternatives is `docs/adr/0001`; the
  glossary terms Implementation and BindEnv live in `CONTEXT.md`; the
  executable-modules spec carries the amendment notes.
- The landed stage-2 gate ran all four models at tp=1/ep=1 — the DeepEP path has
  never been part of the standing gate matrix in this repo (the normative script
  cannot express a `deepep` config on a single node). After this effort landed,
  one-off runs at ep=2/4/8 (tp matching ep) verified the DeepEP implementation
  end-to-end at every size (dispatcher init, bind-time a2a JIT, forward
  dispatch/combine), each with a meaningful 256-token response.
- All Qwen3.5 models are GDN models, so Qwen3.5-35B-A3B provides MoE + GDN
  coverage in a single default-path run.
