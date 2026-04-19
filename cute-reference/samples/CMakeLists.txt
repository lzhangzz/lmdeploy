# CuTe reference GEMM sample kernels
# Built via: cmake -B build -DBUILD_CUTE_SAMPLES=ON && ninja -C build

set(CUTE_GEMM_SAMPLES
    01_bf16_gemm_sm80
    02_bf16_gemm_sm80_opt
    03_bf16_gemm_sm80_pipe
    04_bf16_gemm_sm80_pipe_256x128
    05_bf16_gemm_sm80_pipe_epilogue
    06_bf16_gemm_sm80_pipe_tma
    07_bf16_gemm_sm80_pipe_tma_ws
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
