#!/bin/bash
set -e

CUTLASS_INC=/data/lmdeploy-cute/build/_deps/repo-cutlass-src/include

echo "Compiling bf16_gemm_sm80.cu ..."
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     bf16_gemm_sm80.cu \
     -o bf16_gemm_sm80

echo "Running ..."
./bf16_gemm_sm80 "$@"
