# 03: Linear executable module

**What to build:** The linear weight module becomes the executable Linear of the spec: it owns its forwards — dense, grouped (expert routing tables), act-scale companions, and plan replay — and the call sites flip to weight-is-executor (`weight->Forward(...)`, no executor argument, no executor member). The module base gains the default no-op BatchOp Run and the engine-capture bind from the spec's lifecycle section. Every existing linear call site — attention projections, FFN and MoE GEMMs inside the still-existing layers, and the model driver's embedding/output GEMMs — swaps to the module form. The shape, param names, and registry string are frozen per the spec's Linear section; the per-quant axis stays the GEMM family.

**Blocked by:** 02 (Re-home the GEMM engine into the engine root).

**Status:** ready-for-agent

- [ ] Linear owns all five forward forms plus output-spec/exec-plan/tune and the run-mode ops delegating to the captured engine
- [ ] Module base has the default no-op Run and bind; bind captures the engine reference; forward-before-bind asserts
- [ ] All linear call sites use module forwards; no executor-form linear calls remain
- [ ] Build green; dense and grouped (MoE) smoke runs produce meaningful output
