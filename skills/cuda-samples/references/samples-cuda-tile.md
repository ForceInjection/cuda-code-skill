# CUDA Tile Samples

CUDA Tile C++ is a new programming model introduced in CUDA 13.0 (SM 10.0+, Blackwell). It provides:
- `__tile_global__` — tile kernel launch syntax (`kernel<<<grid>>>(args)`, threads-per-block always 1)
- `cuda::tiles` namespace — `tensor_span`, `partition_view`, `tile<T>`, `mma`, reductions
- Compile-time shapes and static optimization hints (`assume_aligned`, `assume_divisible`, `irange`, latency hints)
- Persistent grid support for grid-stride loops

**Arch**: SM 10.0+ (Blackwell)
**Lines**: ~340

---

## 10.1 helloTile

- **Path**: `cpp/9_CUDA_Tile/helloTile/helloTile.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/9_CUDA_Tile/helloTile/helloTile.cu>
- **Pattern**: Minimal CUDA Tile program: launches a SIMT kernel and a tile kernel side by side, demonstrating `__tile_global__`, triple-chevron launch without threads-per-block, and data sharing through global memory.
- **Arch**: SM 10.0+
- **Lines**: ~76

```cuda
__global__ void simtKernel(int* x) {
  printf("Hello, SIMT!\n");
  *x = 100;
}

__tile_global__ void tileKernel(int* x) {
  printf("Hello, Tile!\n");
  *x = 200;
}

int main() {
  // ...
  simtKernel<<<1, 1>>>(d_x);          // SIMT launch: grid + block dims
  tileKernel<<<1>>>(d_x);             // Tile launch: grid only, block always 1
}
```

## 10.2 tileMatmul

- **Path**: `cpp/9_CUDA_Tile/tileMatmul/tileMatmul.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/9_CUDA_Tile/tileMatmul/tileMatmul.cu>
- **Pattern**: FP16 → FP32 mixed-precision GEMM with `cuda::tiles::mma`. Compares naive vs. optimized kernels. Optimized version adds `__restrict__`, `assume_aligned`, `assume_divisible`, `irange` loops, and latency hints for the compiler.
- **Arch**: SM 10.0+
- **Lines**: ~282

```cuda
__tile_global__ void matmul(float* __restrict__ _C,
                            const __half* __restrict__ _A,
                            const __half* __restrict__ _B,
                            int _M, int _N, int _K) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    float* C = ct::assume_aligned(_C, 16_ic);
    const __half* A = ct::assume_aligned(_A, 16_ic);
    const __half* B = ct::assume_aligned(_B, 16_ic);
    int M = ct::assume_divisible(_M, TILE_BLOCK_M);
    // ...same for N, K...

    auto a_span = ct::tensor_span{A, ct::extents{M, K}};
    auto b_span = ct::tensor_span{B, ct::extents{K, N}};
    auto c_span = ct::tensor_span{C, ct::extents{M, N}};

    auto a_view = ct::partition_view{a_span, ct::shape<TILE_BLOCK_M, TILE_BLOCK_K>{}};
    auto b_view = ct::partition_view{b_span, ct::shape<TILE_BLOCK_K, TILE_BLOCK_N>{}};
    auto c_view = ct::partition_view{c_span, ct::shape<TILE_BLOCK_M, TILE_BLOCK_N>{}};

    auto [pid_m, pid_n, dummy] = ct::bid();

    auto acc = ct::zeros<ct::tile<float, ct::shape<TILE_BLOCK_M, TILE_BLOCK_N>>>();

    ct::irange k_range(0, K, TILE_BLOCK_K, LOAD_LATENCY);
    for (int k_block : k_range) {
        ct::tile<__half, ct::shape<TILE_BLOCK_M, TILE_BLOCK_K>> a_tile;
        ct::tile<__half, ct::shape<TILE_BLOCK_K, TILE_BLOCK_N>> b_tile;
        a_tile = a_view.load_masked(pid_m, k_block);
        b_tile = b_view.load_masked(k_block, pid_n);
        acc = ct::mma(a_tile, b_tile, acc);
    }

    c_view.store_masked(acc, pid_m, pid_n, STORE_LATENCY);
}
```

## 10.3 tileBmm

- **Path**: `cpp/9_CUDA_Tile/tileBmm/tileBmm.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/9_CUDA_Tile/tileBmm/tileBmm.cu>
- **Pattern**: Persistent-grid batched GEMM (BMM) with rank-3 `cuda::tiles::mma`. A fixed number of persistent blocks (sized from SM count) walk the (M, N, Q-chunk) tile space via grid-stride loop. Each K-step issues a single batched rank-3 mma over tiles of shape (BLOCK_SIZE_Q, BLOCK_SIZE_M, BLOCK_SIZE_K) × (BLOCK_SIZE_Q, BLOCK_SIZE_K, BLOCK_SIZE_N). Demonstrates `grouped_2d_grid` for L2 reuse.
- **Arch**: SM 10.0+
- **Lines**: ~268

```cuda
// Persistent blocks sized by device SM count
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
dim3 grid(num_sms);
tileBmm<<<grid>>>(d_C, d_A, d_B, Q, M, N, K, num_sms);

// Inside the kernel: grid-stride loop over (pid_m, pid_n, batch chunks)
auto [pid_m, pid_n, dummy] = ct::bid();  // grouped_2d grids ignore dim z
for (int bi = pid_m; bi < num_batches; bi += grid.x) {
    for (int bj = pid_n; bj < num_batches_n; bj += grid.y) {
        // rank-3 mma: (BLOCK_Q, BLOCK_M, BLOCK_K) x (BLOCK_Q, BLOCK_K, BLOCK_N)
        acc = ct::mma(a_tile, b_tile, acc);  // batched over BLOCK_Q
    }
}
```

## 10.4 tileLayerNorm

- **Path**: `cpp/9_CUDA_Tile/tileLayerNorm/tileLayerNorm.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/9_CUDA_Tile/tileLayerNorm/tileLayerNorm.cu>
- **Pattern**: Persistent-grid LayerNorm forward pass: `y = (x - mean) * rsqrt(var + eps) * weight + bias`. Uses `cuda::tiles` row reductions (mean, variance) across the column dimension. Compile-time template parameters for N, D, NUM_SMS, and EPS allow the tile compiler to fold loop steps and reciprocals at compile time.
- **Arch**: SM 10.0+
- **Lines**: ~270

```cuda
template <int N, int D, int NUM_SMS, float EPS>
__tile_global__ void layernorm_forward_persistent(
    float* __restrict__ Y, const float* __restrict__ X,
    const float* __restrict__ weight, const float* __restrict__ bias) {
    namespace ct = cuda::tiles;
    // ...
    // Per-row mean reduction across column dimension
    auto sum = ct::reduce(ct::tile_ops::plus{}, x_tile, 1);  // reduce over dim 1
    auto mean = sum * rcp_D;
    // Broadcast mean, subtract, square, reduce again for variance
    auto var = ct::reduce(ct::tile_ops::plus{}, diff_sq, 1);
    auto inv_std = ct::rsqrt(var * rcp_D + EPS);
    // Apply weight/bias and store
}
```

## 10.5 tileRope

- **Path**: `cpp/9_CUDA_Tile/tileRope/tileRope.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/9_CUDA_Tile/tileRope/tileRope.cu>
- **Pattern**: Rotary Position Embedding (RoPE) with GPT-NeoX split-half convention. Each token position `s` rotates pairs `(q[i], q[i + D/2])` by `theta = s * 10000^(-2i/D)`. Uses `cuda::tiles::partition_view` over (heads, half_rope_dim), processing all heads for one (batch, position) token in parallel. Applies precomputed cos/sin tables in-place.
- **Arch**: SM 10.0+
- **Lines**: ~274

```cuda
__tile_global__ void rope(__half* Q, const float* cos_table, const float* sin_table,
                           int B, int H, int S, int D) {
    namespace ct = cuda::tiles;
    auto q_view = ct::partition_view{
        ct::tensor_span{Q, ct::extents{B, H, S, D}},
        ct::shape<1_ic, BLOCK_H, 1_ic, BLOCK_D>{}};
    // For each (batch, seq) position, load Q tile, rotate by cos/sin
    auto q_tile = q_view.load(b, h, s, 0);
    auto q_rotated = ct::rope_split_half(q_tile, cos_tile, sin_tile);
    q_view.store(q_rotated, b, h, s, 0);
}
```

## 10.6 tileSpMV

- **Path**: `cpp/9_CUDA_Tile/tileSpMV/tileSpMV.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/9_CUDA_Tile/tileSpMV/tileSpMV.cu>
- **Pattern**: Sparse matrix-vector multiplication `y = A * x` using CUDA Tile C++. Demonstrates irregular memory access patterns in the Tile programming model — each row of the sparse matrix has a different number of non-zeros, yet tile blocks handle this variability through masked operations and sparse index lookups.
- **Arch**: SM 10.0+
- **Lines**: ~494

```cuda
__tile_global__ void spmv(float* y, const float* values, const int* col_idx,
                           const int* row_ptr, const float* x, int M) {
    namespace ct = cuda::tiles;
    auto [pid, dummy1, dummy2] = ct::bid();
    // Each block processes BLOCK_M rows
    // For each row, gather non-zero values and accumulate dot product
    auto acc = ct::zeros<ct::tile<float, ct::shape<BLOCK_M, 1>>>();
    // ...sparse gather from values/col_idx by row_ptr offset...
}
```

## 10.7 tileMatmulAutotuner

- **Path**: `cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp>
- **Pattern**: NVRTC/NVCC-based autotuner that sweeps tile sizes (BLOCK_M, BLOCK_N, BLOCK_K) and optimization hints (alignment, divisibility, latency) to find the best configuration for a given GEMM shape. Demonstrates how to programmatically generate and compile CUDA Tile kernels with different parameters at runtime.
- **Arch**: SM 10.0+
- **Lines**: ~340

## 10.8 tileVectorAdd, tileTranspose

- **tileVectorAdd** (`cpp/9_CUDA_Tile/tileVectorAdd/tileVectorAdd.cu`, ~136 lines): Simple elementwise vector addition using `partition_view` with BLOCK_SIZE=1024. Demonstrates masked loads/stores for boundary handling.
- **tileTranspose** (`cpp/9_CUDA_Tile/tileTranspose/tileTranspose.cu`, ~126 lines): 2D matrix transpose — each block loads an n×m chunk, transposes locally, and stores to the transposed position. Uses `cuda::tiles::partition_view` for chunking.
