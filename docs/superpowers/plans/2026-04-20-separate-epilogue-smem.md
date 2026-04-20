# Separate Epilogue Smem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated shared memory buffer for the epilogue C output tile, eliminating the smem overlap race between producer loads and consumer epilogue in the persistent GEMM kernel.

**Architecture:** Add a `C` member to `SharedStorage`, change `sC` to point at it instead of overlaying `smem.A`. The A/B pipeline smem and C epilogue smem become disjoint, enabling the producer to safely load tile N+1 while the consumer does tile N's epilogue. No new sync mechanisms needed.

**Tech Stack:** CUDA, CuTe, CUTLASS PipelineTmaAsync, SM80 HMMA + SM90 TMA

---

### Task 1: Update SharedStorage and all references

**Files:**
- Modify: `cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`

This task makes all 4 code changes plus the file header update. The changes are all in one file and logically one unit of work.

- [ ] **Step 1: Update the SharedStorage struct template and body**

File: `cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`, lines 45-51

Change the SharedStorage struct from:

```cpp
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB, int Stages>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};
```

To:

```cpp
template <class ElementA, class ElementB, class ElementC,
          class SmemLayoutA, class SmemLayoutB, class SmemLayoutC, int Stages>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  alignas(128) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> C;
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};
```

- [ ] **Step 2: Update the SharedStorage type alias in the kernel**

File: `cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`, line 102

Change:

```cpp
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB, cute::size<2>(SmemLayoutA{})>;
```

To:

```cpp
  using SharedStorage = SharedStorage<TA, TB, TC, SmemLayoutA, SmemLayoutB, SmemLayoutC, cute::size<2>(SmemLayoutA{})>;
```

`TC` and `SmemLayoutC` are already template parameters of the kernel (line 60).

- [ ] **Step 3: Update sC tensor construction to use the dedicated C buffer**

File: `cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`, lines 228-230

Change:

```cpp
    Tensor sC = make_tensor(
        make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin())),
        SmemLayoutC{});
```

To:

```cpp
    Tensor sC = make_tensor(make_smem_ptr(smem.C.begin()), SmemLayoutC{});
```

No `reinterpret_cast` needed — `smem.C` is already typed as `ElementC` (bf16_t).

- [ ] **Step 4: Update host function SharedStorage instantiation**

File: `cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`, line 407

Change:

```cpp
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, decltype(sA), decltype(sB), cute::size<2>(decltype(sA){})>));
```

To:

```cpp
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, bf16_t, decltype(sA), decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sA){})>));
```

The new arguments are: `bf16_t` (for `ElementC`), `decltype(sC_layout)` (for `SmemLayoutC`).

- [ ] **Step 5: Update the file header comment**

File: `cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`, lines 1-21

Add to the "Key changes from 07" list (after the existing 5 bullet points, before the blank line):

```
 *   - Separate epilogue smem buffer (C output tile uses dedicated smem, not overlay on smem.A)
```

This documents the difference from 07 for future readers.

- [ ] **Step 6: Build the binary**

Run:

```bash
cd /data/lmdeploy-cute/build && ninja 08_bf16_gemm_sm80_pipe_tma_ws_persistent
```

Expected: Clean build with no errors. PTXAS should report ~213 KB smem usage (up from ~147 KB previously).

- [ ] **Step 7: Run correctness test**

Run:

```bash
cd /data/lmdeploy-cute/build && ./bin/08_bf16_gemm_sm80_pipe_tma_ws_persistent 1024 1024 1024
```

Expected output includes:

```
Correctness (1024^3): max error <some_value> — PASS
```

The max error must be < 0.5f. If it fails, the smem layout or pointer is wrong — double-check that `smem.C.begin()` returns a valid pointer to the dedicated C buffer.

- [ ] **Step 8: Run full benchmark sweep**

Run:

```bash
cd /data/lmdeploy-cute/build && ./bin/08_bf16_gemm_sm80_pipe_tma_ws_persistent
```

Expected: All 6 sizes (256³ through 8192³) run without errors. Performance should be comparable to the previous version (the extra smem may cause a slight occupancy reduction, but the tile-level pipelining should compensate). Compare the TFLOP/s numbers with the previous run.

- [ ] **Step 9: Commit**

```bash
cd /data/lmdeploy-cute && git add cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu && git commit -m "Add separate epilogue smem buffer for tile-level pipelining

Eliminates the race where producer TMA loads could overlap with
consumer STSM epilogue writes to the same smem.A region. Adds a
dedicated C buffer to SharedStorage (~64 KB), bringing total smem
to ~208 KB (fits in 228 KB on L20Y/H800)."
```
