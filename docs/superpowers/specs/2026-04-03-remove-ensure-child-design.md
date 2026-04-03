# Remove `ensure_child` from C++ Modules

**Date:** 2026-04-03
**Status:** Draft

## Goal

Remove all `ensure_child` overrides from C++ composite modules. The Python loading pipeline (`TextModelLoader`) now creates all children explicitly via `create_child`, making the lazy-creation path dead code. Removing it eliminates dual-path confusion and the risk of crashes from uninitialized constructor parameters.

## Current State

`Module::get(segment)` does two things:
1. Look up an existing child via `child(segment)`
2. If not found, call virtual `ensure_child(segment)` to lazily create it

Each composite module overrides `ensure_child` to create children using constructor parameters (head dims, TP config, etc.). The execution side calls typed accessors like `weights.w_qkv()` which call `get<T>("w_qkv")`, triggering lazy creation on first access.

## New State

`Module::get(segment)` just returns `child(segment)` — no lazy creation. All children are created explicitly by the Python pipeline before inference starts. The typed `get<T>()` template wraps the result with `TM_CHECK` to produce clear errors if a child is missing, rather than returning nullptr.

## Changes

### 1. `src/turbomind/core/module.h`

- Remove `virtual Module* ensure_child(const std::string& segment)` from Module
- Change `Module::get(segment)` (non-const) to return `child(segment)` only
- Add `TM_CHECK` to `get<T>()` template so missing children produce clear errors:
  ```cpp
  template<typename T>
  T* get(const std::string& name) const {
      auto* c = child(name);
      TM_CHECK(c != nullptr) << "child '" << name << "' not found in " << type();
      return static_cast<T*>(c);
  }
  ```
- Remove `ModuleList::Factory` typedef and constructor parameter
- Remove `ModuleList::ensure_child` override
- Add `ModuleList()` default constructor
- Keep `ModuleList::add_child` override and `indexed_` (still needed for O(1) indexed access)
- Update doc comments to remove lazy-creation language

### 2. `src/turbomind/core/module.cc`

- Remove `Module::ensure_child` base implementation
- Simplify `Module::get(segment)` to `return child(segment)`
- Remove `ModuleList::ensure_child` implementation
- Change `ModuleList` constructor to default (no factory)
- Simplify `ModuleListRegistrar` to create a plain `ModuleList` (no crashing factory)

### 3. Per-composite module (delete `ensure_child` override)

Remove the `ensure_child` override declaration from `.h` and definition from `.cc` for:

| File | Children previously created by ensure_child |
|------|---------------------------------------------|
| `attention_weight.h/.cc` | w_qkv, q_proj, q_a_proj, q_b_proj, kv_a_proj, wo, q_norm, k_norm, q_a_layernorm, kv_a_layernorm, sinks |
| `decoder_layer_weight.h/.cc` | attention_norm, ffn_norm, attention, feed_forward, moe_ffn, linear_attn |
| `ffn_weight.h/.cc` | w1, w3, w2, w1w3 |
| `moe_weight.h/.cc` | gate, shared_gate, experts (ModuleList factory) |
| `delta_net_weight.h/.cc` | in_proj_all, out_proj, conv1d, A_log, dt_bias, norm |
| `model_weight.h/.cc` | tok_embeddings, output, norm, layers (ModuleList factory) |

### 4. No changes to execution-side code

Typed accessors like `AttentionWeight::w_qkv()` call `get<T>()` which now includes the `TM_CHECK` guard. No other execution-side changes needed.

## What stays unchanged

- `Module::child()` — const lookup, unchanged
- `Module::create_child()` — registry-driven explicit creation
- `ModuleList::add_child` override — tracks `indexed_` for O(1) access
- `ModuleList::size()` — works from `indexed_`
- All typed accessors — call `get<T>()` which calls `child()`
- All execution-side code — uses typed accessors, no changes

## Verification

All 13 models must pass with TP=1 and TP=2 after the change. The `TM_CHECK` in `get<T>()` will catch any missing children with a clear error message during testing.
