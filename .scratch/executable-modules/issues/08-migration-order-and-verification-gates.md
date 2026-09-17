# Migration order and verification gates

Type: grilling
Status: open
Blocked by: 04, 05, 06, 07

## Question

Fix the staged migration order and per-stage gates: Linear, then FFN/MoE, then Attention (+GatedDeltaNet). For each stage: the in-tree moves, the checkpoint and Python-API compat checks, and the verification recipe (build with `ninja` in `build/`; `scripts/test_turbomind_model.py` with meaningful responses of at least 128 tokens, never batched; which quant × architecture × dense/MoE matrix is mandatory per gate; how perf parity is measured). The FFN/MoE stage must also gate on memory footprint and startup cost: per-module transport state (Merged FFN and MoE module design amendment) multiplies dispatcher buffers and JIT compilation by layer count. Output: the migration section for the spec.
