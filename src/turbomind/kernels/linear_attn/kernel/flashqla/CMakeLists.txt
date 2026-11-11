# FlashQLA GDN prefill adapter. Kernels are exported at model load by
# lmdeploy/turbomind/kernels/tilelang/flashqla/ and found at runtime; none are needed to build.
add_library(linear_attn_kernels_flashqla STATIC entry.cu host.cc)
configure_linear_attn_kernel_target(linear_attn_kernels_flashqla)
target_link_libraries(linear_attn_kernels_flashqla PRIVATE tilelang_utils)
list(APPEND LINEAR_ATTN_GDR_KERNEL_TARGETS linear_attn_kernels_flashqla)
set(LINEAR_ATTN_GDR_KERNEL_TARGETS "${LINEAR_ATTN_GDR_KERNEL_TARGETS}" PARENT_SCOPE)
