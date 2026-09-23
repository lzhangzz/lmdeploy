# Cache layouts are owned by the modules that fill them

`object_cache_plan.cc` grew from a page-size tuner into the place where attention and GatedDeltaNet cache geometry (object sizes, per-layer offsets, state block shapes) is computed from weights, handed down to the modules that consume it, and finally written back into weight structs as runtime fields. We decided each cache-producing module owns its cache layout: the module computes it via static functions on the module class, enumerates its own admissible geometries, and applies page-fit padding itself, while planning (`object_cache_plan`) keeps only composition and page tuning, selecting among module-offered candidates and never writing module fields. The split keeps the cross-module page fit (LCM across all part sizes) where it belongs, with the planner, and the shape knowledge where it belongs, with the module. Full detail: `.scratch/cache-layout-ownership/spec.md`.

## Considered Options

- **Planner computes all geometry from weights** (status quo): one place to read, but planner and modules mutate the same facts in both directions and weights end up carrying scheduler-cache state.
- **Modules register geometry into `CacheRegistry` and planning reads it back**: rejected because the page size must be known before the allocator and registry wiring exist, so registry state cannot drive tuning.
- **Module-internal optimization** (`TuneForPage` style, module optimizes under page constraints): rejected because the waste/LCM tradeoff spans attention and GDN parts jointly; no single module can see it.
