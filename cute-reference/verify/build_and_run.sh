#!/bin/bash
set -e

CUTLASS_INC=/data/lmdeploy-cute/build/_deps/repo-cutlass-src/include

echo "=== Compiling all verification files ==="
for f in verify_*.cu; do
  bin="${f%.cu}"
  echo "  Compiling $f -> $bin"
  nvcc -std=c++17 -arch=sm_89 -I${CUTLASS_INC} "$f" -o "$bin"
done

echo ""
echo "=== Running all verification binaries ==="
PASS=0
FAIL=0
TOTAL=0
for bin in verify_0[0-9]_*; do
  [ -x "$bin" ] || continue
  echo ""
  echo "--- $bin ---"
  if ./"$bin"; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1))
  fi
  TOTAL=$((TOTAL+1))
done

echo ""
echo "=== Summary: ${PASS}/${TOTAL} passed, ${FAIL} failed ==="
