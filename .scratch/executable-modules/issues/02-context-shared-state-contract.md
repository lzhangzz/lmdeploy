# Context shared-state contract

Type: grilling
Status: open

## Question

Define the Context shared-state contract for executable modules: the exact inventory of per-rank shared execution resources (GEMM engine plus dispatch/exec-plan cache, tuning state including `Gemm::Tune` and `Export`/`Import` plan serialization, workspace pools, `OutputSpec` allocations, streams/events), what each executable Module may request from Context, and the ownership and thread-placement rules consistent with `src/turbomind/engine/README.md` (cite `contracts.executor-only` and friends). Also decide what a module instance may never hold. Output: the contract section destined for the spec, with concrete C++ snippets of the Context accessor surface.
