# Copyright (c) OpenMMLab. All rights reserved.

add_library(gemm_sm80 STATIC gemm_sm80.cu)
target_link_libraries(gemm_sm80 PUBLIC nvidia::cutlass::cutlass)
set_property(TARGET gemm_sm80 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gemm_sm80 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_executable(test_gemm_sm80 test_gemm_sm80.cu)
target_link_libraries(test_gemm_sm80 PRIVATE gemm_sm80)
