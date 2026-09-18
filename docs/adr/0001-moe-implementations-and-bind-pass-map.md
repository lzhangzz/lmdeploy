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

- Each bind pass creates the kind states against a fresh map (`update_params`
  reloads — the original concern here — are no longer supported).
- `MoeState`, the Context kind-state slots, and `Impl::Bind`'s kind-module scan are
  deleted; Attention/DeltaNet states move to the same map rendezvous.
- Gate is compile + default-path run; the normative gate matrix still does
  not include DeepEP (`scripts/test_turbomind_model.py` cannot express
  `moe_a2a_backend="deepep"` on a single node). A one-off verification run
  after landing (ep=2/tp=2, bf16, 2 GPUs, one node) did execute the DeepEP
  implementation end-to-end — dispatcher init with NCCL symmetric memory,
  bind-time a2a JIT, dispatch/combine in forward — with a meaningful
  256-token response.
- Python surface unchanged (`create_module(cfg)`; the standalone `Module.bind(context)`
  pybind shim assembles a degenerate `BindEnv` internally).
