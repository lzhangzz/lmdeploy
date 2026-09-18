# 03: MoE implementations as module subclasses

**What to build:** The two MoE implementations become executable module
subclasses under the single frozen `"MoeWeight"` registry string — selected at
construction by a custom factory reading the ambient EngineParam — each holding
its layer's weights, each owning its per-rank state on the bind-pass map. The
merged state struct dies. Behavior is verbatim; the DeepEP path remains compiled
but unexecuted.

**Blocked by:** 01 (bind pass redesign), 02 (ambient EngineParam).

**Status:** ready-for-agent

- [ ] The MoE base keeps the frozen type string, children/params surface,
      config-field ctor, `prepare`/`link_block`, shared `Gate`, and declares a
      pure-virtual `Forward` with today's four arguments; the decoder's call
      surface is unchanged
- [ ] `MoeDefaultImpl` and `MoeDeepEPImpl` subclass the base, each in its own
      file pair together with its state; bodies are verbatim moves of the
      current internal branches; the DeepEP in-place output contract is
      documented on the declaration
- [ ] One custom factory under the frozen `"MoeWeight"` string selects from the
      ambient EngineParam: `"default"` → default subclass, `"deepep"` → DeepEP
      subclass, anything else fatal; the bfloat16 invariant lives on the DeepEP
      side; no expert-parallel-size conditioning
- [ ] `MoeDefaultState` (routing scratch) and `MoeDeepEPState` (dispatcher,
      top-k buffers, scales-copy stream/events) rendezvous on the map under
      `"moe.default"` / `"moe.deepep"`; buffers allocate on the engine build
      stream/allocator; the merged state struct and its slot are gone
- [ ] The MoE config class and the entire Python surface are untouched
- [ ] `ninja` clean (the DeepEP path compiles); `Qwen/Qwen3.5-35B-A3B`
      default-path run via `scripts/test_turbomind_model.py` answers with a
      meaningful ≥128-token response; the DeepEP-unexecuted caveat holds
