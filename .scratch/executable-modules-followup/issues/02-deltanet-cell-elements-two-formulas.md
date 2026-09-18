# 02: DeltaNet cell size computed two ways (key×value vs head_dim²)

**What:** `src/turbomind/models/delta_net.cc` computes `cell_elements` as
`first.key_head_dim * first.value_head_dim` when validating state sizes, but
`DeltaNetState::layer_state_offset` uses `head_dim_ * head_dim_`, where
`head_dim_` was assigned from `first.key_head_dim` only. The two agree only
while `key_head_dim == value_head_dim` — true for every current GDN
checkpoint (e.g. Qwen3-Next 128/128). A model with asymmetric dims would
compute wrong in-block recurrent offsets with no check failing.

**Direction:** assert `key_head_dim == value_head_dim` in the DeltaNet ctor
(cheapest), or store both dims and use key×value in the offset math.

**Status:** needs-triage
