# MoE implementations are module subclasses; kind states live on the bind-pass map

Status: accepted (design settled 2026-09-18; implementation pending).

Ticket 06 of the executable-modules effort merged the AG/RS default and DeepEP MoE
paths into one class behind an internal `use_a2a` branch, parking both transports'
state in a Context-owned `MoeState`. We reversed both halves: each implementation
is an executable Module subclass (`MoeDefaultImpl`, `MoeDeepEPImpl : Moe`)
constructed by the frozen `"MoeWeight"` registry factory — selected at construction
from the ambient per-rank `EngineParam` (the `moe_a2a_backend` string only;
unknown → fatal; bf16 invariant on the DeepEP side) — and per-rank state is created
by the first module of a kind during the bind pass and shared through a transient
type-erased map carried on `BindEnv`, never stored on `Context`. Drivers: modules
must stay executable; a config class must stay a constant model declaration; engine
data reaches domain code as its own typed argument; per-rank state must be shared by
all layers without `Context` becoming a god object.

## Considered options

- Internal branch in one `Moe::Forward` (ticket 06's landing) — the two flows differ
  end-to-end (routing format, token partitioning, shared fold, output contract), and
  the merged class re-created the executor-inside-a-weight-class problem.
- Implementation choice or sizing fields in `MoeConfig` — rejected: a config is a
  constant model declaration; engine policy and geometry are not model structure.
- Kind states as Context slots, created by `Impl::Bind` — rejected: Context is an
  infrastructure holder, and creation rules do not belong in engine glue.
- Creation in `bind`, or a per-rank executor object beside the modules — rejected:
  what a module *is* is chosen at construction; an executor beside a weight class is
  the retired Layer shape.
- A second registry string per implementation — rejected: the binding surface's
  registry strings never change.

## Consequences

- `update_params` reloads re-create kind states (fresh map per binding); the DeepEP
  dispatcher re-runs its JIT — accepted.
- `MoeState`, the Context kind-state slots, and `Impl::Bind`'s kind-module scan are
  deleted; Attention/DeltaNet states move to the same map rendezvous.
- Gate is compile + default-path run; the DeepEP path remains unexecuted by gates
  (as in ticket 06) — recorded, not hidden.
- Python surface unchanged (`create_module(cfg)`; the standalone `Module.bind(context)`
  pybind shim assembles a degenerate `BindEnv` internally).
