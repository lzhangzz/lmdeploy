# 04: Production attention header includes kernels/gemm/test/test_utils.h

**What:** `src/turbomind/models/attention.h` includes
`src/turbomind/kernels/gemm/test/test_utils.h` for the RNG used to fill
warm-up output noise (`AttentionState::rng_`, used in the attention forward
paths when `is_warm_up` is set). Carried over verbatim from the retired
`unified_attention_layer.h`; a production module depending on a test header
is layering debt (the test utils are compiled into test targets and a bench,
not the runtime library's own surface).

**Direction:** move the small RNG utility (or an equivalent runtime helper)
into `src/turbomind/utils/` and include that from attention.h.

**Status:** needs-triage
