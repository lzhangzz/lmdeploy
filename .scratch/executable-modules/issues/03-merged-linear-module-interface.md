# Merged Linear module interface

Type: prototype
Status: open
Blocked by: 01, 02

## Question

Draft the merged Linear module interface as a concrete header to react to: `class Linear : core::Module` owning today's `LinearWeight` params (`weight/bias/scales/zeros/global_scale`) plus the `WeightPlan`, with `Forward(...)` overloads covering dense and grouped/MoE call sites, Context injection, the `GetOutputSpec`/`GetExecPlan`/`Tune` surface, construction-by-name from Python (binding surface unchanged), and the mapping of the standalone Torch Linear API (`lmdeploy/turbomind/linear.py`) onto it. Show the `LlamaFfnLayer.cc` and `moe_ffn_layer.cc` call sites rewritten against it. This header becomes the canonical module shape that Merged FFN and MoE module design and Merged Attention module design reuse.
