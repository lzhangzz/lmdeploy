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
