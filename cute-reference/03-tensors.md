# Tensors & Memory Types

`Tensor<Engine, Layout>` pairs a data store (Engine) with a coordinate mapping (Layout).
The Engine provides the iterator; the Layout maps coordinates to offsets.

## Tensor Engines

The Engine concept wraps an iterator:
```cpp
using iterator     =  // The iterator type
using value_type   =  // The iterator value-type
using reference    =  // The iterator reference-type
iterator begin()      // The iterator
```

Three built-in engines:
- `ArrayEngine<T,N>` — owning array (register memory)
- `ViewEngine<Iter>` — non-owning view
- `ConstViewEngine<Iter>` — const non-owning view

## Tagged Pointers (Memory Spaces)

Tagging a pointer with its memory space allows CuTe to dispatch to optimized operations.

```cpp
// Global memory
make_gmem_ptr(ptr)        // tag raw pointer as global
make_gmem_ptr<T>(ptr)     // tag and reinterpret as type T

// Shared memory
make_smem_ptr(ptr)        // tag raw pointer as shared
make_smem_ptr<T>(ptr)     // tag and reinterpret as type T

// Register memory
make_rmem_ptr(ptr)        // tag raw pointer as register
```

## Tensor Creation

### Nonowning (views of existing memory)

```cpp
float* A = ...;

// Untagged
Tensor t1 = make_tensor(A, make_layout(Int<8>{}));    // shape only
Tensor t2 = make_tensor(A, Int<8>{});                 // shorthand
Tensor t3 = make_tensor(A, 8, 2);                     // shape + stride

// Global memory
Tensor gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);   // (M,K)
Tensor gB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);   // (N,K)

// Shared memory
__shared__ float smem[256];
Tensor sA = make_tensor(make_smem_ptr(smem), SmemLayout{});
```

### Owning (register memory, static layout only)

```cpp
Tensor rA = make_tensor<float>(Shape<_4, _8>{});                     // col-major 4×8
Tensor rB = make_tensor<float>(Shape<_4, _8>{}, LayoutRight{});      // row-major 4×8
Tensor rC = make_tensor<float>(Shape<_4, _8>{}, Stride<_32, _1>{});  // custom stride
Tensor rD = make_tensor_like(gA(_, 0));                               // same type+shape as gA(_,0)
```

## Tensor Access

```cpp
// Element access
tensor(coord)           // single coordinate
tensor(m, n)            // variadic
tensor[linear_idx]      // 1D linear access

// Properties
tensor.data()           // the iterator
tensor.size()           // total logical size
rank(tensor)            // number of modes
shape(tensor)           // shape IntTuple
stride(tensor)          // stride IntTuple
size<I...>(tensor)      // hierarchical size access
shape<I...>(tensor)     // hierarchical shape access
```

## Tensor Slicing

Slicing with `_` (Underscore) returns a subtensor. The `_` modes are retained.

```cpp
// ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
Tensor A = make_tensor(ptr, make_shape(make_shape(Int<3>{},2),
                                       make_shape(2,Int<5>{},Int<2>{})),
                              make_stride(make_stride(4,1),
                                          make_stride(Int<2>{},13,100)));

Tensor B = A(2, _);                   // ((2,_5,_2)):((_2,13,100)) — slice mode 0 to index 2
Tensor C = A(_, 5);                   // ((_3,_2)):((4,1)) — slice mode 1 to index 5
Tensor D = A(make_coord(_,_), 5);     // (_3,2):(4,1) — same elements as C, different rank
Tensor E = A(make_coord(_,1), make_coord(0,_,1));  // (_3,_5):(4,13) — partial slice
```

## Tensor Partitioning

Partitioning = tiling + slicing. Three common patterns:

### Inner partition (CTA-level) — `local_tile`

Keep the tile mode, index into the rest mode:
```cpp
Tensor tiled = zipped_divide(A, tiler);      // ((Tile), (Rest))
Tensor my_tile = tiled(_, cta_coord);        // (Tile) — one tile per threadblock
// Equivalent to: local_tile(A, tiler, cta_coord, ...)
```

### Outer partition (thread-level) — `local_partition`

Index into the tile mode, keep the rest mode:
```cpp
Tensor tiled = zipped_divide(A, tiler);      // ((Tile), (Rest))
Tensor thr_view = tiled(threadIdx.x, _);     // (Rest) — one thread's view
// Equivalent to: local_partition(A, tiler, threadIdx.x, ...)
```

### Thread-Value partitioning

Compose with a TV-layout that maps `(thread, value) → coordinate`:
```cpp
auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                        Stride<Stride<_8,_1>,Stride<_4,_16>>>{};  // (8,4)

Tensor A = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});     // (4,8)
Tensor tv = composition(A, tv_layout);                             // (8,4)
Tensor v  = tv(threadIdx.x, _);                                    // (4) — this thread's values
```

## Algorithms on Tensors

```cpp
copy(src, dst)                              // element-wise or CopyAtom-based copy
copy_if(pred, src, dst)                     // conditional copy
gemm(D, A, B, C)                            // D = A*B + C (dispatches by rank)
fill(tensor, value)                         // fill with value
clear(tensor)                               // zero a tensor
axpby(alpha, x, beta, y)                   // y = alpha*x + beta*y
prefetch(tensor)                            // prefetch data

// Cooperative (multi-threaded)
cooperative_copy(thread_group, src, dst)
cooperative_gemm(thread_group, D, A, B, C)
```

The `gemm` algorithm dispatches by tensor rank:
```
(V) x (V) => (V)           element-wise, dispatches to FMA/MMA
(M) x (N) => (M,N)         outer product
(M,K) x (N,K) => (M,N)     matrix product, dispatches to outer product per K
(V,M) x (V,N) => (V,M,N)   batched outer product
(V,M,K) x (V,N,K) => (V,M,N)  batched matrix product
```

## Kernel Composition Patterns

### Pattern 1: Global → CTA tile partitioning

```cpp
Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);  // (M,K)
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);           // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
```

### Pattern 2: Shared memory staging

```cpp
extern __shared__ char shared_memory[];
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{});  // (BLK_M,BLK_K,PIPE)
```

### Pattern 3: Thread partitioning

```cpp
ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);             // (MMA, MMA_M, MMA_K, PIPE)
Tensor tCrA = thr_mma.make_fragment_A(tCsA);       // register fragment
Tensor tCrC = thr_mma.make_fragment_C(tCgC);       // register C fragment
clear(tCrC);                                        // zero accumulators
```

### Pattern 4: Epilogue (register → global)

```cpp
axpby(alpha, tCrC, beta, tCgC);                    // simple scaled write-back
```
