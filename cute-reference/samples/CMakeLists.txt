# CuTe reference GEMM sample kernels
cmake_minimum_required(VERSION 3.11)

set(CUTE_GEMM_SAMPLES
    bf16_gemm_sm80
    bf16_gemm_sm80_opt
    bf16_gemm_sm80_pipe
    bf16_gemm_sm80_pipe_256x128
    bf16_gemm_sm80_pipe_epilogue
    bf16_gemm_sm80_pipe_tma
    bf16_gemm_sm80_pipe_tma_ws
)

foreach(sample ${CUTE_GEMM_SAMPLES})
    add_executable(${sample} ${sample}.cu)
    target_link_libraries(${sample} PRIVATE nvidia::cutlass::cutlass)
    target_compile_options(${sample} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    set_target_properties(${sample} PROPERTIES CUDA_ARCHITECTURES "90a-real")
endforeach()
