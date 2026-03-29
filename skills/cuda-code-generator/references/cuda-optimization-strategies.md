# CUDA Optimization Strategies Reference

When there is no NCU analysis report, choose the default optimization strategy based on the algorithm type.

## Default Strategies by Algorithm Type

### MatMul (Matrix Multiplication)

**Default Optimization Combination**: P0 + P1 + P2

| Parameter     | Recommended Value                                  |
| ------------- | -------------------------------------------------- |
| TILE_SIZE     | 16 (conservative) or 32 (aggressive, needs sm_80+) |
| Block         | (TILE_SIZE, TILE_SIZE, 1)                          |
| Grid          | (ceil(N/TILE), ceil(M/TILE), 1)                    |
| Shared Memory | 2 × TILE × (TILE+1) × sizeof(T)                    |

**Bottleneck Characteristic**: L1_PRESSURE_BOUND → Preferred: Shared Memory Tiling

---

### Reduction

**Default Optimization Combination**: Warp Shuffle + Multi-pass Reduction

| Parameter | Recommended Value         |
| --------- | ------------------------- |
| Block     | (256, 1, 1)               |
| Grid      | (ceil(N/Block.x/2), 1, 1) |

**Core Pattern**:

```cuda
// Warp-level reduction (no __syncthreads needed)
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);

// Block-level reduction (via shared memory)
__shared__ float smem[32];  // One value per warp
if (lane == 0) smem[wid] = val;
__syncthreads();
```

**Bottleneck Characteristic**: MEMORY_BOUND → Reduce GMEM access passes

---

### 1D Convolution

**Default Optimization Combination**: P0 (Shared Memory Halo) + P2 (Vectorized Load)

| Parameter     | Recommended Value                   |
| ------------- | ----------------------------------- |
| Block         | (256, 1, 1)                         |
| Shared Memory | (BLOCK + 2\*RADIUS) × sizeof(float) |

**Core Pattern**:

```cuda
#define RADIUS 3  // Convolution radius (= (FILTER_SIZE-1)/2)

__shared__ float smem[BLOCK_SIZE + 2 * RADIUS];

// Load halo region
int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x - RADIUS;
smem[threadIdx.x] = (gx >= 0 && gx < N) ? input[gx] : 0.0f;
// Load right halo (thread tail handling)
if (threadIdx.x < 2 * RADIUS) {
    int gx2 = blockIdx.x * BLOCK_SIZE + BLOCK_SIZE + threadIdx.x - RADIUS;
    smem[BLOCK_SIZE + threadIdx.x] = (gx2 < N) ? input[gx2] : 0.0f;
}
__syncthreads();

float result = 0.0f;
for (int r = -RADIUS; r <= RADIUS; r++)
    result += smem[threadIdx.x + RADIUS + r] * filter[r + RADIUS];
```

---

### General Element-wise Kernel

**Default Optimization Combination**: Grid-Stride Loop + Vectorized Load

```cuda
__global__ void elementwise(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop: One thread processes multiple elements
    for (int i = idx; i < N / 4; i += stride) {
        float4 a4 = reinterpret_cast<const float4*>(A)[i];
        float4 b4 = reinterpret_cast<const float4*>(B)[i];
        float4 c4 = {a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w};
        reinterpret_cast<float4*>(C)[i] = c4;
    }
    // Handle tail (when N is not divisible by 4)
    for (int i = (N / 4) * 4 + idx; i < N; i += stride)
        C[i] = A[i] + B[i];
}
```

---

## Detailed Explanation of Optimization Measures

### Thread Block Configuration Reference

| Scenario       | blockDim                       | Description                          |
| -------------- | ------------------------------ | ------------------------------------ |
| 2D Matrix Algo | `(16, 16, 1)` or `(32, 32, 1)` | TILE aligned, 32 available on sm_80+ |
| Reduction / 1D | `(256, 1, 1)`                  | General, good for Warp Shuffle       |
| Custom         | `(128, 1, 1)`                  | Conservative choice                  |

> blockDim.x must be a multiple of 32 (warp aligned).

---

### Shared Memory Tiling

**Purpose**: Reduce redundant L1/GMEM loads, each data block is loaded from global memory only once.

**Applicable Conditions**:

- `L1/TEX Cache Throughput > 85%`
- `DRAM Throughput < 30%` (Data is repeatedly accessed at the L1 level)
- The algorithm has data reuse (matrix multiplication, convolution, etc.)

**TILE_SIZE Selection**:

| sm_XX       | Recommended TILE_SIZE | Shared Memory Consumption |
| ----------- | --------------------- | ------------------------- |
| sm_70~sm_75 | 16                    | 2KB per tile              |
| sm_80~sm_89 | 16 or 32              | 2KB or 8KB                |
| sm_90       | 32                    | 8KB per tile              |

**Complete MatMul Tiling Template**:

```cuda
#define TILE_SIZE 16  // Adjustable: 16 or 32

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 to eliminate Bank Conflict
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Boundary safe load (tail Tile out-of-bounds padded with 0)
        As[threadIdx.y][threadIdx.x] = (row < M && t * TILE_SIZE + threadIdx.x < K)
            ? A[row * K + t * TILE_SIZE + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (t * TILE_SIZE + threadIdx.y < K && col < N)
            ? B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Host launch
dim3 blockDim(TILE_SIZE, TILE_SIZE);
dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
matmul_tiled<<<gridDim, blockDim>>>(A, B, C, M, N, K);
```

---

### Bank Conflict Elimination (Padding)

**Principle**: 32 banks, 4 bytes stride. When TILE_SIZE is 32, elements of each column fall into the same bank → 32-way conflict.

**Solution**: Add 1 element padding in the column dimension:

```cuda
__shared__ float As[TILE][TILE + 1];  // TILE+1 breaks alignment
```

**When padding is not needed**: When TILE_SIZE = 16, bank conflicts are mild (at most 2-way), and can be omitted at discretion.

---

### Vectorized Load (float4)

**Requirements**:

1. Global memory address is **16-byte aligned** (`cudaMalloc` satisfies this by default)
2. Data length is a multiple of 4 (tail handled separately)

**Benefit**: Merges 4 Load instructions into 1, reducing instruction issue pressure and improving L1/L2 bandwidth utilization.

**Half precision version**:

```cuda
half2 h2 = reinterpret_cast<const half2*>(A)[idx / 2];
```

---

### Double Buffering / cp.async (sm_80+)

**Purpose**: Asynchronously prefetch the next Tile while computing the current Tile, hiding GMEM latency.

**Applicable Conditions**:

- GPU architecture >= sm_80 (Ampere)
- `Warp Cycles Per Issued > 20` (Latency is not hidden)
- LATENCY_BOUND still exists after Tiling optimization

**Template** (Requires `#include <cuda/pipeline>`):

```cuda
__shared__ float As[2][TILE][TILE + 1];
__shared__ float Bs[2][TILE][TILE + 1];

cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

// Preload the 0th Tile
cuda::memcpy_async(As[0][threadIdx.y] + threadIdx.x,
                   A + row * K + 0 * TILE + threadIdx.x,
                   sizeof(float), pipe);
pipe.producer_commit();

for (int t = 0; t < numTiles; t++) {
    int next = (t + 1) % 2;
    // Asynchronously load the next Tile
    if (t + 1 < numTiles) {
        cuda::memcpy_async(As[next][threadIdx.y] + threadIdx.x,
                           A + row * K + (t+1) * TILE + threadIdx.x,
                           sizeof(float), pipe);
        pipe.producer_commit();
    }
    // Wait for the current Tile to be ready
    pipe.consumer_wait();
    __syncthreads();

    // Compute using the current Tile
    #pragma unroll
    for (int k = 0; k < TILE; k++)
        sum += As[t % 2][threadIdx.y][k] * Bs[t % 2][k][threadIdx.x];

    pipe.consumer_release();
    __syncthreads();
}
```

---

### `__launch_bounds__` Tuning

**Purpose**: Prompt the compiler to control register allocation, avoiding spilling or occupancy drops.

```cuda
// maxThreadsPerBlock: Must match launch blockDim
// minBlocksPerMultiprocessor: Minimum blocks corresponding to target occupancy
__global__ __launch_bounds__(256, 4)
void kernel(...) { ... }
```

**Reference Table** (sm_89, max 1536 threads per SM):

| maxThreads | minBlocks | Max Registers/Thread |
| ---------- | --------- | -------------------- |
| 256        | 6         | 42                   |
| 256        | 4         | 64                   |
| 256        | 2         | 128                  |

---

### Prefetching (sm_80+)

**Purpose**: In DRAM_MEMORY_BOUND scenarios, use `__builtin_assume_aligned` and `prefetch` to issue load requests early, hiding DRAM latency.

**Applicable Conditions**:

- `DRAM Throughput > 70%`
- GPU architecture >= sm_80 (Ampere) supports `cp.async`

**Template**:

```cuda
// Method 1: Use __ldg read-only cache (sm_35+)
float val = __ldg(&A[idx]);  // Goes through texture/read-only cache, reducing L1 pressure

// Method 2: cp.async asynchronous prefetch (sm_80+, used with Double Buffering)
// See Double Buffering section

// Method 3: prefetch instruction (suitable for stride access)
asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr + prefetch_offset));
```

---

### Data Transpose (Access Pattern Adjustment)

**Purpose**: In L1_PRESSURE_BOUND + column access scenarios, convert uncoalesced column accesses into coalesced row accesses, eliminating L1 bandwidth waste.

**Applicable Conditions**:

- Kernel has column-major accesses like `A[col * M + row]`
- `L1/TEX Throughput > 80%`, and L2 Hit Rate is low

**Template** (Transpose implemented via Shared Memory intermediary):

```cuda
// Coalesced read a Tile of A into shared memory, then write out transposed
__shared__ float tile[TILE][TILE + 1];  // +1 to eliminate Bank Conflict

// Coalesced read (row access)
tile[threadIdx.y][threadIdx.x] = A[(blockIdx.y * TILE + threadIdx.y) * N
                                   + blockIdx.x * TILE + threadIdx.x];
__syncthreads();

// Transposed write (converted to column write, merging row accesses for matrix B)
B[(blockIdx.x * TILE + threadIdx.y) * M
  + blockIdx.y * TILE + threadIdx.x] = tile[threadIdx.x][threadIdx.y];
```

---

### Fragment Caching (Register-level Caching)

**Purpose**: When L1_PRESSURE_BOUND + small data scale, place frequently accessed small arrays (like convolution weights, LUT tables) into register arrays, bypassing L1 entirely.

**Applicable Conditions**:

- Data size <= available registers per thread (usually < 256 bytes)
- The same thread reads data from a fixed index multiple times in a loop

**Template**:

```cuda
// Load weights / small LUT into register array (bypasses L1)
float reg_filter[FILTER_SIZE];
#pragma unroll
for (int i = 0; i < FILTER_SIZE; i++)
    reg_filter[i] = filter[i];  // Load only once

// Use registers directly in the loop, no longer accessing global/shared memory
for (int i = 0; i < N; i++)
    output[i] += input[i] * reg_filter[i % FILTER_SIZE];
```

---

### ILP (Instruction-Level Parallelism)

**Purpose**: In LATENCY_BOUND scenarios, each thread processes multiple independent elements simultaneously, allowing the scheduler to issue computation instructions for one set while waiting for memory operations of another.

**Applicable Conditions**:

- `Warp Cycles Per Issued > 15`, but Occupancy cannot be further increased
- Sufficient registers (will not cause spilling)

**Template** (Each thread processes 4 elements, ILP=4):

```cuda
// Each thread is responsible for consecutive ILP elements
int base = (blockIdx.x * blockDim.x + threadIdx.x) * ILP;

float a0 = A[base + 0], a1 = A[base + 1];
float a2 = A[base + 2], a3 = A[base + 3];
float b0 = B[base + 0], b1 = B[base + 1];
float b2 = B[base + 2], b3 = B[base + 3];

// Independent computation, scheduler can pipeline execution
C[base + 0] = a0 + b0;
C[base + 1] = a1 + b1;
C[base + 2] = a2 + b2;
C[base + 3] = a3 + b3;
```

---

### Loop Unrolling

**Purpose**: In LATENCY_BOUND scenarios, reduce loop control overhead, providing more instruction-level parallelism opportunities for the compiler and scheduler.

**Template**:

```cuda
// Static unroll (loop count known at compile time)
#pragma unroll
for (int k = 0; k < TILE_SIZE; k++)
    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

// Partial unroll (loop count unknown, unroll N times)
#pragma unroll 4
for (int i = 0; i < N; i++)
    sum += A[i] * B[i];
```

**Note**: Excessive unrolling (unroll factor too large) increases register pressure, needing control via `__launch_bounds__`.

---

### FP16 / TF32 / Tensor Core (COMPUTE_BOUND)

**Purpose**: In COMPUTE_BOUND scenarios, massively increase arithmetic throughput by reducing precision or using dedicated matrix multiplication hardware units (Tensor Cores).

**Applicable Conditions**:

- `SM Busy > 80%`, compute intensive
- sm_70+ (Volta) supports FP16 Tensor Cores; sm_80+ (Ampere) supports TF32

**FP16 Tensor Core Template (Using WMMA API)**:

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// Fragment size: 16x16x16
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

fill_fragment(c_frag, 0.0f);
load_matrix_sync(a_frag, A + warp_row * 16 * K, K);
load_matrix_sync(b_frag, B + warp_col * 16,     N);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(C + warp_row * 16 * N + warp_col * 16, c_frag, N, mem_row_major);
```

**TF32 (Ampere, no data type modification needed, just enable compilation option)**:

```bash
nvcc -O3 -arch=sm_80 --use_fast_math -o kernel kernel.cu
# Or use cublasMath_t CUBLAS_TF32_TENSOR_OP_MATH in code
```

---

### Block Size Tuning (OCCUPANCY_BOUND)

**Purpose**: In OCCUPANCY_BOUND scenarios, balance registers, shared memory, and occupancy by adjusting `blockDim`.

**Diagnostic Steps**:

1. Check the limiting factors `Block Limit Registers` / `Block Limit Shared Mem` / `Block Limit Warps` in the NCU report.
2. If `Block Limit Registers` is the smallest → Increase Block size or use `__launch_bounds__` to lower registers/thread.
3. If `Block Limit Shared Mem` is the smallest → Reduce shared memory allocation or use dynamic shared memory.
4. If `Block Limit Warps` is the smallest → Increase Block size (but not exceeding 1024).

**Common Configurations Comparison** (sm_89, 1536 threads per SM):

| blockDim | Blocks per SM | Occupancy              | Applicable Scenario                   |
| -------- | ------------- | ---------------------- | ------------------------------------- |
| (64, 1)  | 24            | 100%                   | Simple element-wise                   |
| (128, 1) | 12            | 100%                   | General 1D kernel                     |
| (256, 1) | 6             | 100%                   | Reduction                             |
| (16, 16) | 6             | 100%                   | 2D Matrix Tiling                      |
| (32, 32) | 1~2           | Low (Reg/Smem limited) | Large Tile, needs `__launch_bounds__` |

**Grid-Stride Loop (Alternative when Block is too small)**:

```cuda
// Without changing gridDim, let each thread process multiple rows via a loop
for (int row = blockIdx.y * blockDim.y + threadIdx.y;
         row < M;
         row += gridDim.y * blockDim.y) {
    // ... processing logic
}
```

---

## Common Bottleneck → Optimization Mapping

| NCU Characteristic                      | Bottleneck Type   | Priority Optimization                                             |
| --------------------------------------- | ----------------- | ----------------------------------------------------------------- |
| L1 Throughput > 90%, DRAM < 30%         | L1_PRESSURE_BOUND | P0: Shared Memory Tiling → P1: Padding → P2: Data Transpose       |
| DRAM Throughput > 70%, SM Busy < 30%    | DRAM_MEMORY_BOUND | P0: Block Tiling → P1: Vectorized Load → P2: Prefetching          |
| Warp Cycles > 20, Eligible < 40%        | LATENCY_BOUND     | P0: Double Buffering → P1: ILP → P2: Loop Unrolling               |
| SM Busy > 80%, Compute Throughput > 80% | COMPUTE_BOUND     | P0: FMA → P1: FP16/TF32 → P2: Tensor Core                         |
| Occupancy < 30%, SM Busy > 70%          | OCCUPANCY_BOUND   | P0: Adjust Block Size → P1: `__launch_bounds__` → P2: Reduce smem |
