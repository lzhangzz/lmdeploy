# Copyright (c) OpenMMLab. All rights reserved.

add_library(attention STATIC attention.cu)
target_compile_options(attention PRIVATE
  --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr --keep)
set_property(TARGET attention PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(attention PRIVATE nvidia::cutlass::cutlass)

add_executable(test_attention test_utils.cu test_attention.cu)
# target_compile_options(test_decoder_multihead_attention PRIVATE
#   --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr)
target_link_libraries(test_attention PRIVATE 
    attention
    decoder_multihead_attention 
    decoder_masked_multihead_attention
    cublas)
