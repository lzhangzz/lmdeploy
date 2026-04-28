# Mixed-precision GEMM kernels (split A loading POC)
set(MIXED_GEMM_TARGETS
    bf16_gemm_sm90_split_a_pack
    bf16_gemm_sm90_split_a_wgmma
)

foreach(target ${MIXED_GEMM_TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PRIVATE nvidia::cutlass::cutlass)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "90a-real")
endforeach()
