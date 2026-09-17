# Merged FFN and MoE module design

Type: grilling
Status: open
Blocked by: 03

## Question

Design the merged FFN and MoE modules: the `Ffn` module (w1w3 fused versus w1/w3 separate, the FP8 fused-SiLU `inter_scales` path) and the `Moe` module (gate, shared expert, `LinkLinearExperts` block ownership, grouped GEMM indices/offsets). Eliminate the GEMM orchestration block copy-pasted three times (`LlamaFfnLayer.cc:51-90`, `MoeFfnDefaultImpl::Forward`, `MoeFfnA2AImpl::Forward`). Decide where routing (`invokeMoeGate_V2`/`NoAuxTC`) lives. Respect the hard constraints: checkpoints load as-is, no perf regression, engine contract meanings hold.
