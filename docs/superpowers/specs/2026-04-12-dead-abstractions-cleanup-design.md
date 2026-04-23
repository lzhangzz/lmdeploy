# Remove Dead Abstractions from Model Loading Pipeline

**Date:** 2026-04-12
**Scope:** Python only, model loading pipeline

## Context

The turbomind model loading pipeline has accumulated dead code from the transition to the `TextModelLoader` + `Distributor` architecture. Two abstractions are completely unreferenced:

1. **`LoadContext` class** — superseded by `Distributor`, zero external callers
2. **`TextModelSpec.load_layer()` / `load_global()`** — `NotImplementedError` stubs from a previous pipeline, zero callers

## Changes

### 1. Delete `LoadContext` class from `load_context.py`

Remove lines 362–507 (the section header and entire class body). The module-level functions (`commit_linear`, `commit_tensor`, TP rules, dtype helpers) remain — they are actively used by `Distributor`.

### 2. Delete `load_layer()` / `load_global()` from `TextModelSpec` in `spec.py`

Remove lines 303–323 (section comment and two methods). `TextModelLoader` has its own `_load_layer()` / `_load_global()` that use `Distributor`.

## Impact

- ~160 lines deleted, 0 lines added
- No behavioral change — both deletions have zero callers
- `load_context.py` retains its role as the commit infrastructure module (functions + constants used by `Distributor`)
