#!/bin/bash
set -e

BIN=../../build/bin

echo ""
echo "=== Running plain sample ==="
echo "--- 1024x1024x1024 ---"
$BIN/01_bf16_gemm_sm80 1024 1024 1024

echo ""
echo "=== Running optimized sample (swizzle+LDSM) ==="
echo "--- 1024x1024x1024 ---"
$BIN/02_bf16_gemm_sm80_opt 1024 1024 1024

echo ""
echo "=== Running pipelined sample (cp.async 3-stage) ==="
echo "--- 1024x1024x1024 ---"
$BIN/03_bf16_gemm_sm80_pipe 1024 1024 1024

echo ""
echo "=== Running 256x128 tile sample (256 threads) ==="
echo "--- 1024x1024x1024 ---"
$BIN/04_bf16_gemm_sm80_pipe_256x128 1024 1024 1024

echo ""
echo "=== Running STSM epilogue sample (BF16 output) ==="
echo "--- 1024x1024x1024 ---"
$BIN/05_bf16_gemm_sm80_pipe_epilogue 1024 1024 1024

echo ""
echo "=== Running TMA load + STSM/TMA store sample ==="
echo "--- 1024x1024x1024 ---"
$BIN/06_bf16_gemm_sm80_pipe_tma 1024 1024 1024

echo ""
echo "=== Running warp-specialized TMA load/store sample ==="
echo "--- 1024x1024x1024 ---"
$BIN/07_bf16_gemm_sm80_pipe_tma_ws 1024 1024 1024
