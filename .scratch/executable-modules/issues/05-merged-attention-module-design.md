# Merged Attention module design

Type: grilling
Status: open
Blocked by: 03

## Question

Design the merged Attention module: per-layer module instances versus today's all-layers `UnifiedAttentionLayer` plus a single `CacheRegistry`; where KV cache category registration and offsets live (cite `contracts.category-registration`, `ownership.module-cache`); the MLA branch; where rope parameters live (`init_rope_kernel_param` currently sits in the weight file but serves the executor). Decide GatedDeltaNet's place: its own module following the Attention pattern, with its recurrent/conv state cache contracts (`contracts.gated-deltanet`) preserved.
