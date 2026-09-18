# 01: Attention::bind hard-requires AttentionState — latent vision+GDN trap

**What:** `Attention::bind` (src/turbomind/models/attention.cc) does
`state_ = TM_CHECK_NOTNULL(ctx.attention_state())`. The attention kind state
is created only when the *text* model has attention layers
(`TurboMind::Impl::Bind`, src/turbomind/turbomind.cc), but the bind walk
reaches the vision tree too — `VisionModelWeight` stops only
`register_cache`, not `bind` — and both internvit and qwenvit instantiate
`Attention` modules.

A vision-language model whose text tower is pure GDN (no attention layers)
would abort at bind when its first vision Attention binds against a null
state. No current model has that shape (interns2_mobius is 3×GDN + 1×Attn
per layer cycle), so this is latent, not live.

**Direction:** either create the AttentionState whenever any Attention
module exists in the tree (text or vision), or give vision-side Attention a
state-free bind path.

**Status:** needs-triage
