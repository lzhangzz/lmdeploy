# CuTe Layout Algebra Verification — Design Spec

## Goal

Verify every claim in `cute-reference/02-layout-algebra.md` by compiling and running
actual CuTe code. Fix any errors found in the documentation.

## Approach

One standalone `.cu` file per doc section. Each file:
- Includes `cute/tensor.hpp` (header-only, no linking needed)
- Runs the examples from the doc on CPU (layout algebra is host-side)
- Prints actual results
- Compares against expected values, reports PASS/FAIL per claim

No GPU kernels needed — `coalesce`, `composition`, `complement`, `logical_divide`,
`logical_product` all operate on `Layout` objects at compile-time/host.

## File Structure

```
cute-reference/verify/
  verify_00_inttuple.cu       — Section 0: IntTuple & Layout foundations
  verify_01_coalesce.cu       — Section 1: Coalesce
  verify_02_composition.cu    — Section 2: Composition
  verify_03_complement.cu     — Section 3: Complement
  verify_04_divide.cu         — Section 4: Division (Tiling)
  verify_05_product.cu        — Section 5: Product (Replication)
  verify_06_partitioning.cu   — Section 6: Higher-Level Partitioning
  verify_07_mode_select.cu    — Section 7: Mode Selection & Manipulation
  verify_08_utilities.cu      — Section 8: Construction & Validation Utilities
  build_and_run.sh            — Compile and run all files, report summary
```

## Compilation

```bash
CUTLASS_INC=/data/lmdeploy-cute/build/_deps/repo-cutlass-src/include
nvcc -std=c++17 -arch=sm_89 -I${CUTLASS_INC} verify_XX.cu -o verify_XX && ./verify_XX
```

## Verification Claims Per File

### verify_00_inttuple.cu
- `Int<4>{}` compiles
- `make_tuple(4, make_tuple(2, 3))` — rank, size, depth
- `make_layout(make_shape(4, 8), make_stride(1, 4))` — layout(2,3)==14
- `size(layout)==32`, `cosize(layout)==?` (doc says 29, likely bug — max index = 3*1+7*4 = 31, cosize = 32)
- `shape(layout)`, `stride(layout)` accessors

### verify_01_coalesce.cu
- Rule 1: size-1 mode ignored (left): `_1:d0 ++ s1:d1 => s1:d1`
- Rule 2: size-1 mode ignored (right): `s0:d0 ++ _1:d1 => s0:d0`
- Rule 3: contiguous merge: `s0:d0 ++ s1:s0*d0 => s0*s1:d0`
- Rule 4: non-contiguous stays separate
- Doc example: `(2,(1,6)):(1,(6,2))` coalesces to `(2,6):(_1,_2)`
- By-mode coalesce with Step

### verify_02_composition.cu
- Example 1: `20:2 ∘ (5,4):(4,1)` = `(5,4):(8,2)`
  - Verify: R(0,0)==0, R(1,0)==8, R(0,1)==2
- Example 2: `(10,2):(16,4) ∘ (5,4):(1,5)` = `(_5,(_2,_2)):(_16,(_80,_4))`
- By-mode composition with make_tile tiler

### verify_03_complement.cu
- `complement(4:1, 24)` = `6:4`
- `complement(4:2, 24)` = `(2,3):(1,8)`
- `complement((2,2):(1,6), 24)` = `(3,2):(2,12)`
- `complement((4,6):(1,4), 24)` = `1:0`
- Overload: `complement(layout)` (default cotarget)
- Overload: `complement(layout, cotarget)`
- Overload: `complement(layout, shape)`

### verify_04_divide.cu
- `logical_divide((4,2,3):(2,1,8), 4:2)` produces claimed result
- Intermediate: `complement(4:2, 24)` = `(2,3):(1,8)`
- zipped_divide, tiled_divide, flat_divide produce correct mode rearrangements

### verify_05_product.cu
- `logical_product((2,2):(4,1), 6:1)` = `((2,2),(2,3)):((4,1),(2,8))`
- blocked_product exists and produces result
- raked_product exists and produces result
- tile_to_shape works

### verify_06_partitioning.cu
- `ceil_div` produces correct shapes
- `congruent` validates shapes correctly
- `make_coord` constructs coordinates
- `make_identity_tensor` creates identity tensor
- `local_tile`, `local_partition` — verify they compile (tensor-dependent, hard to fully verify standalone)

### verify_07_mode_select.cu
- `select<0,2>(shape)` extracts correct modes
- `take<0,2>(tuple)` extracts range
- `append<4>(tuple, val)` adds mode
- `replace<2>(tuple, val)` replaces mode
- `flatten(layout)` flattens modes
- `make_tile(layouts...)` constructs tiler

### verify_08_utilities.cu
- `make_ordered_layout(shape, LayoutRight{})` = `(8,4):(4,1)`
- `recast<float4>(tensor)` compiles and produces correct layout
- `make_shape`, `make_stride` basic construction

## build_and_run.sh

```bash
#!/bin/bash
set -e
CUTLASS_INC=/data/lmdeploy-cute/build/_deps/repo-cutlass-src/include
PASS=0; FAIL=0; TOTAL=0

for f in verify_*.cu; do
  echo "=== Compiling $f ==="
  nvcc -std=c++17 -arch=sm_89 -I${CUTLASS_INC} "$f" -o "${f%.cu}" 2>&1
  if [ $? -eq 0 ]; then
    echo "=== Running $f ==="
    ./"${f%.cu}"
  else
    echo "COMPILE FAILED: $f"
    FAIL=$((FAIL+1))
  fi
  TOTAL=$((TOTAL+1))
done

echo "=== Summary: ${PASS}/${TOTAL} passed, ${FAIL} failed ==="
```

## Error Handling

- If a claim fails: print expected vs actual, continue to next claim
- If compilation fails: print error, continue to next file
- Final summary of all PASS/FAIL across all files

## Out of Scope

- Verifying architecture-specific files (06-sm70.md through 09-sm90.md) — these need GPU execution
- Verifying tensor algorithms (copy, gemm) — these need device memory
- Modifying the doc during verification — fix errors in a separate pass after all results are known
