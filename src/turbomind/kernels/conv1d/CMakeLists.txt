# Copyright (c) OpenMMLab. All rights reserved.

add_library(conv1d STATIC conv1d_silu.cu)
set_property(TARGET conv1d PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET conv1d PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(conv1d PUBLIC core)

if(BUILD_TEST)
    add_executable(bench_conv1d_silu
            bench_conv1d_silu.cc)
    target_link_libraries(bench_conv1d_silu PRIVATE
            conv1d
            gemm2_core
            core
            CUDA::cudart)
endif()
