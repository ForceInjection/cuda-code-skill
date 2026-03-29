# cuBLAS Reference

**Related guides:** cuda-runtime.md (memory management), cuda-math.md (numeric types), ptx-isa.md (instruction-level)

## Table of Contents

- [Local Documentation](#local-documentation) — 319 markdown files, 2.9MB
- [When to Use cuBLAS Documentation](#when-to-use-cublas-documentation) — GEMM, batched ops, cuBLASLt, FP8
- [Quick Search Examples](#quick-search-examples) — GEMM, epilogues, compute types, FP8
- [cuBLAS vs cuBLASLt vs cuBLASXt](#cublas-vs-cublaslt-vs-cublasxt) — Which API to use
- [Documentation Structure](#documentation-structure) — Chapter layout
- [Notable Large Files](#notable-large-files) — Level 3, extensions, cublasLt
- [Search Tips](#search-tips) — Function names, types, enums
- [Common Workflows](#common-workflows) — GemmEx, cublasLtMatmul, batched GEMM
- [Troubleshooting](#troubleshooting) — Workspace size, compute type mismatch, epilogue errors

## Local Documentation

**Complete cuBLAS 13.2 documentation is available locally at `cublas-docs/`**

The documentation has been converted to markdown with:

- ✅ All function signatures, parameters, and return values preserved
- ✅ 319 files organized by chapter (2.9 MB)
- ✅ Full searchability with grep/ripgrep
- ✅ Type, enum, and function names preserved (redundant URLs removed)
- ✅ Detailed descriptions, notes, and caveats

**Note:** Documentation is local and searchable with grep. Links to online resources provided for reference only.

## When to Use cuBLAS Documentation

Consult cuBLAS reference when:

1. **GEMM operations** — Looking up `cublas<t>gemm` (sgemm/dgemm/hgemm/cgemm), batched variants, strided batched
2. **Mixed-precision GEMM** — `cublasGemmEx` / `cublasGemmBatchedEx` with `CUDA_R_16F`, `CUDA_R_8F_E4M3`, etc.
3. **FP8/BF16 narrow-precision** — FP8 GEMM with scale factors, `CUBLAS_COMPUTE_32F_FAST_TF32`
4. **Fused GEMM with epilogues** — cuBLASLt `cublasLtMatmul` with bias, ReLU, GELU, aux output
5. **Tensor Core control** — `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`, `CUBLAS_TF32_TENSOR_OP_MATH`
6. **Batched/grouped GEMM** — vLLM's paged attention, grouped query attention, MoE routing
7. **BLAS Level 1/2** — DOT, AXPY, GEMV for attention score scaling
8. **cuBLASXt** — Multi-GPU GEMM for extremely large matrices

## Quick Search Examples

### Find GEMM variants

```bash
# Standard typed GEMM (sgemm/dgemm/hgemm)
grep -r "cublasHgemm\|cublasSgemm\|cublasDgemm" cublas-docs/

# Mixed-precision GemmEx
grep -A 20 "cublasGemmEx" cublas-docs/2-using-the-cublas-api/2.8.12-cublasgemmex.md

# Strided batched GEMM (vLLM paged attention)
grep -A 30 "cublasGemmStridedBatchedEx" cublas-docs/2-using-the-cublas-api/2.8.14-cublasgemmstridedbatchedex.md
```

### Find cuBLASLt epilogue options

```bash
# All epilogue types (bias, ReLU, GELU, dropout...)
grep -r "CUBLASLT_EPILOGUE_" cublas-docs/3-using-the-cublaslt-api/

# cublasLtMatmul full API
grep -A 40 "cublasLtMatmul" cublas-docs/3-using-the-cublaslt-api/3.4-cublaslt-api-reference.md | head -60

# Layout and matrix descriptor attributes
grep "CUBLASLT_MATRIX_LAYOUT_\|CUBLASLT_MATMUL_DESC_" cublas-docs/3-using-the-cublaslt-api/3.3-cublaslt-datatypes-reference.md
```

### Find compute types and data types

```bash
# Compute types (TF32, FP16, FP8)
grep "CUBLAS_COMPUTE_" cublas-docs/2-using-the-cublas-api/2.2.11-cublascomputetype_t.md

# cudaDataType_t values (R_16F, R_8F_E4M3, etc.)
grep "CUDA_R_\|CUDA_C_" cublas-docs/2-using-the-cublas-api/2.3.1-cudadatatype_t.md

# FP8 / narrow precision support
grep -r "E4M3\|E5M2\|FP8\|fp8" cublas-docs/
```

### Find workspace requirements

```bash
grep -r "workspace\|setWorkspace\|cublasSetWorkspace" cublas-docs/
grep -A 10 "cublasSetWorkspace" cublas-docs/2-using-the-cublas-api/2.4.8-cublassetworkspace.md
```

## cuBLAS vs cuBLASLt vs cuBLASXt

| API          | Use for                                                      | Key function                    |
| ------------ | ------------------------------------------------------------ | ------------------------------- |
| **cuBLAS**   | Standard typed GEMM, BLAS L1/L2/L3                           | `cublas<t>gemm`, `cublasGemmEx` |
| **cuBLASLt** | Fused GEMM + epilogue (bias/activation), FP8, custom layouts | `cublasLtMatmul`                |
| **cuBLASXt** | Multi-GPU very large GEMM                                    | `cublasXtSgemm`                 |

**vLLM typically uses:**

- `cublasGemmEx` / `cublasGemmStridedBatchedEx` for attention and MLP projection
- `cublasLtMatmul` for fused linear + bias + activation layers on Hopper+
- FP8 paths via cuBLASLt with E4M3/E5M2 data types and FP32 scale factors

## Documentation Structure

```text
cublas-docs/
├── 1-introduction/
│   ├── 1.1-data-layout.md                     # Row/column major, leading dimension
│   ├── 1.5-floating-point-emulation.md        # BF16x9, fixed-point emulation
│   └── 1.5.2-fixed-point.md                   # Dynamic mantissa control
├── 2-using-the-cublas-api/
│   ├── 2.1-general-description.md             # Thread safety, streams, Tensor Core usage
│   ├── 2.1.11-tensor-core-usage.md            # When and how Tensor Cores activate
│   ├── 2.2-cublas-datatypes-reference.md      # All enums overview
│   ├── 2.2.11-cublascomputetype_t.md          # CUBLAS_COMPUTE_16F/32F/32F_FAST_TF32/64F
│   ├── 2.3.1-cudadatatype_t.md                # CUDA_R_16F/32F/8F_E4M3 etc.
│   ├── 2.7-cublas-level-3-function-reference.md  # GEMM, SYMM, TRSM
│   ├── 2.7.1-cublastgemm.md                   # cublas<t>gemm
│   ├── 2.7.4-cublastgemmstridedbatched.md     # strided batched GEMM
│   ├── 2.8-blas-like-extension.md             # GemmEx, GemmBatchedEx
│   ├── 2.8.12-cublasgemmex.md                 # cublasGemmEx (mixed precision)
│   └── 2.8.14-cublasgemmstridedbatchedex.md   # batched mixed-precision GEMM
├── 3-using-the-cublaslt-api/
│   ├── 3.3-cublaslt-datatypes-reference.md    # All cublasLt types and attributes
│   └── 3.4-cublaslt-api-reference.md          # cublasLtMatmul and all Lt functions
├── 4-using-the-cublasxt-api/                  # Multi-GPU GEMM
└── 8-interaction-with-other-libraries-and-tools/  # CUDA graphs, stream integration
```

## Notable Large Files

1. **3.3-cublaslt-datatypes-reference.md** — All `cublasLtEpilogue_t`, layout attributes, matmul desc fields
2. **2.7-cublas-level-3-function-reference.md** — GEMM, TRSM, SYMM, SYRK overview
3. **2.8-blas-like-extension.md** — GemmEx, GemmBatchedEx, GemmStridedBatchedEx signatures
4. **2.2-cublas-datatypes-reference.md** — All enum types (operation, fill mode, side mode, pointer mode)

## Search Tips

1. **Standard GEMM**: `cublas<t>gemm` uses lowercase type prefix (`S`, `D`, `H`, `C`, `Z`)

   ```bash
   grep "cublasHgemm\b" cublas-docs/2-using-the-cublas-api/2.7.1-cublastgemm.md
   ```

2. **Epilogue types**: all start with `CUBLASLT_EPILOGUE_`

   ```bash
   grep "CUBLASLT_EPILOGUE_" cublas-docs/3-using-the-cublaslt-api/3.3-cublaslt-datatypes-reference.md
   ```

3. **Compute types** for mixed precision: `CUBLAS_COMPUTE_`

   ```bash
   grep "CUBLAS_COMPUTE_32F\|CUBLAS_COMPUTE_16F" cublas-docs/
   ```

4. **Data types**: `CUDA_R_` prefix in `cudaDataType_t`

   ```bash
   grep "CUDA_R_8F_E4M3\|CUDA_R_8F_E5M2" cublas-docs/
   ```

5. **cuBLASLt attributes**: `CUBLASLT_MATMUL_DESC_` and `CUBLASLT_MATRIX_LAYOUT_`

   ```bash
   grep "CUBLASLT_MATMUL_DESC_" cublas-docs/3-using-the-cublaslt-api/3.3-cublaslt-datatypes-reference.md
   ```

## Common Workflows

### Mixed-precision GEMM with GemmEx

```bash
# Full cublasGemmEx signature and parameter docs
grep -A 50 "^cublasGemmEx" cublas-docs/2-using-the-cublas-api/2.8.12-cublasgemmex.md

# Supported type combinations (Atype/Btype/Ctype/computeType)
grep "CUDA_R_16F\|CUDA_R_32F" cublas-docs/2-using-the-cublas-api/2.8.12-cublasgemmex.md
```

### cuBLASLt fused GEMM + epilogue

```bash
# cublasLtMatmul full API
cat cublas-docs/3-using-the-cublaslt-api/3.4-cublaslt-api-reference.md

# Set epilogue (bias/ReLU/GELU) on matmul descriptor
grep -B 2 -A 10 "CUBLASLT_EPILOGUE_RELU\|CUBLASLT_EPILOGUE_GELU\|CUBLASLT_EPILOGUE_BIAS" \
  cublas-docs/3-using-the-cublaslt-api/3.3-cublaslt-datatypes-reference.md

# Workspace size query
grep "cublasLtMatmulAlgoGetHeuristic\|workspaceSize" \
  cublas-docs/3-using-the-cublaslt-api/3.4-cublaslt-api-reference.md
```

### Batched GEMM (paged/grouped attention)

```bash
# Strided batched GEMM parameters
cat cublas-docs/2-using-the-cublas-api/2.7.4-cublastgemmstridedbatched.md

# Mixed-precision strided batched
cat cublas-docs/2-using-the-cublas-api/2.8.14-cublasgemmstridedbatchedex.md
```

### FP8 GEMM setup

```bash
# FP8 type definitions (E4M3/E5M2)
grep "CUDA_R_8F_E4M3\|CUDA_R_8F_E5M2" cublas-docs/2-using-the-cublas-api/2.3.1-cudadatatype_t.md

# Scale factor attributes for FP8
grep "CUBLASLT_MATMUL_DESC_.*SCALE\|aScale\|bScale\|dScale" \
  cublas-docs/3-using-the-cublaslt-api/3.3-cublaslt-datatypes-reference.md

# FP8 emulation strategies
cat cublas-docs/2-using-the-cublas-api/2.2.12-cublasemulationstrategy_t.md
```

## Troubleshooting

### `CUBLAS_STATUS_INVALID_VALUE` on GemmEx

```bash
grep "CUBLAS_STATUS_INVALID_VALUE\|supported combination" cublas-docs/2-using-the-cublas-api/2.8.12-cublasgemmex.md
```

- **Cause**: Unsupported type combination (Atype/Btype/Ctype/computeType)
- **Fix**: Check the supported type matrix table in `2.8.12-cublasgemmex.md`

### `CUBLAS_STATUS_NOT_SUPPORTED` on cublasLtMatmul

```bash
grep "CUBLAS_STATUS_NOT_SUPPORTED\|AlgoGetHeuristic" cublas-docs/3-using-the-cublaslt-api/3.4-cublaslt-api-reference.md
```

- **Cause**: No suitable algorithm found for the requested epilogue + layout combo
- **Fix**: Use `cublasLtMatmulAlgoGetHeuristic` to query available algos before calling matmul

### Workspace size errors

```bash
grep -A 5 "workspace" cublas-docs/2-using-the-cublas-api/2.4.8-cublassetworkspace.md
```

- **Cause**: Workspace not set, or set too small for cuBLASLt ops
- **Fix**: Allocate at least 4MB (32MB recommended for cublasLt); call `cublasSetWorkspace`

## Version Information

- **cuBLAS Version**: 13.2
- **Total Size**: 2.9 MB
- **Files**: 319 files across 8 chapters
- **Source**: https://docs.nvidia.com/cuda/cublas/index.html
