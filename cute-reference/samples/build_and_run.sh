#!/bin/bash
set -e

CUTLASS_INC=/data/lmdeploy-cute/build/_deps/repo-cutlass-src/include

echo "Compiling bf16_gemm_sm80.cu (plain) ..."
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     bf16_gemm_sm80.cu \
     -o bf16_gemm_sm80

echo "Compiling bf16_gemm_sm80_opt.cu (swizzle+LDSM) ..."
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     bf16_gemm_sm80_opt.cu \
     -o bf16_gemm_sm80_opt

echo "Compiling bf16_gemm_sm80_pipe.cu (cp.async pipeline) ..."
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     bf16_gemm_sm80_pipe.cu \
     -o bf16_gemm_sm80_pipe

echo ""
echo "=== Running plain sample ==="
echo "--- 1024x1024x1024 ---"
./bf16_gemm_sm80 1024 1024 1024

echo ""
echo "=== Running optimized sample (swizzle+LDSM) ==="
echo "--- 1024x1024x1024 ---"
./bf16_gemm_sm80_opt 1024 1024 1024

echo ""
echo "=== Running pipelined sample (cp.async 3-stage) ==="
echo "--- 1024x1024x1024 ---"
./bf16_gemm_sm80_pipe 1024 1024 1024

echo ""
echo "Compiling bf16_gemm_sm80_pipe_256x128.cu (256x128 tile, 256 threads) ..."
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     bf16_gemm_sm80_pipe_256x128.cu \
     -o bf16_gemm_sm80_pipe_256x128

echo ""
echo "=== Running 256x128 tile sample (256 threads) ==="
echo "--- 1024x1024x1024 ---"
./bf16_gemm_sm80_pipe_256x128 1024 1024 1024

echo ""
echo "Compiling bf16_gemm_sm80_pipe_epilogue.cu (STSM BF16 epilogue) ..."
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     bf16_gemm_sm80_pipe_epilogue.cu \
     -o bf16_gemm_sm80_pipe_epilogue

echo ""
echo "=== Running STSM epilogue sample (BF16 output) ==="
echo "--- 1024x1024x1024 ---"
./bf16_gemm_sm80_pipe_epilogue 1024 1024 1024

echo ""
echo "Compiling bf16_gemm_sm80_pipe_tma.cu (TMA load/store) ..."
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     bf16_gemm_sm80_pipe_tma.cu \
     -o bf16_gemm_sm80_pipe_tma

echo ""
echo "=== Running TMA load/store sample ==="
echo "--- 1024x1024x1024 ---"
./bf16_gemm_sm80_pipe_tma 1024 1024 1024
