# 06: Moe executable module with shared MoeState, and stage-2 gate

**What to build:** The MoE weight module becomes the executable Moe: gate, routing (including the warm-up random path), transport gather/scatter with default-vs-DeepEP as an internal branch, grouped expert execution through the linked block, combine, and shared-expert fold. The per-rank transport state — dispatcher, routing buffers, scales-copy stream — becomes one shared MoeState bound to every MoE module per the spec's engine-services section; both old implementation classes retire. Then run the stage-2 gate including the memory-footprint and startup-cost check.

**Blocked by:** 05 (Ffn executable module).

**Status:** ready-for-agent

- [ ] Moe forward owns routing through shared fold; default-vs-DeepEP is an internal branch, not separate classes
- [ ] One shared per-rank MoeState (dispatcher, routing buffers, scales stream) bound to all MoE modules
- [ ] Both old MoE implementation classes retired and deleted
- [ ] Stage-2 gate passes: four-model matrix with meaningful responses, decode ≥98% of baseline, load time within noise, footprint/startup check recorded
