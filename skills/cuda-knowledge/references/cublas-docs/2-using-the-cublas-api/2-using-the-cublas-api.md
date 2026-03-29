# 2. Using the cuBLAS API


##  2.1. General Description   
  
This section describes how to use the cuBLAS library API.

###  2.1.1. Error Status 

All cuBLAS library function calls return the error status [cublasStatus_t](#cublasstatus-t).

###  2.1.2. cuBLAS Context 

The application must initialize a handle to the cuBLAS library context by calling the [cublasCreate()](#cublascreate) function. Then, the handle is explicitly passed to every subsequent library function call. Once the application finishes using the library, it must call the function [cublasDestroy()](#cublasdestroy) to release the resources associated with the cuBLAS library context.

This approach allows the user to explicitly control the library setup when using multiple host threads and multiple GPUs. For example, the application can use `cudaSetDevice()` to associate different devices with different host threads and in each of those host threads it can initialize a unique handle to the cuBLAS library context, which will use the particular device associated with that host thread. Then, the cuBLAS library function calls made with different handles will automatically dispatch the computation to different devices.

The device associated with a particular cuBLAS context is assumed to remain unchanged between the corresponding [cublasCreate()](#cublascreate) and [cublasDestroy()](#cublasdestroy) calls. In order for the cuBLAS library to use a different device in the same host thread, the application must set the new device to be used by calling `cudaSetDevice()` and then create another cuBLAS context, which will be associated with the new device, by calling [cublasCreate()](#cublascreate). When multiple devices are available, applications must ensure that the device associated with a given cuBLAS context is current (e.g. by calling `cudaSetDevice()`) before invoking cuBLAS functions with this context.

A cuBLAS library context is tightly coupled with the CUDA context that is current at the time of the [cublasCreate()](#cublascreate) call. An application that uses multiple CUDA contexts is required to create a cuBLAS context per CUDA context and make sure the former never outlives the latter. Starting from version 12.8, cuBLAS detects if the underlying CUDA context is tied to a graphics context and follows the shared memory size limits that are set in such case.

###  2.1.3. Thread Safety 

The library is thread safe and its functions can be called from multiple host threads, even with the same handle. When multiple threads share the same handle, extreme care needs to be taken when the handle configuration is changed because that change will affect potentially subsequent cuBLAS calls in all threads. It is even more true for the destruction of the handle. So it is not recommended that multiple thread share the same cuBLAS handle.

Additional considerations apply when the same handle is used from multiple threads with a user provided workspace. See [cublasSetWorkspace()](#cublassetworkspace) for details.

###  2.1.4. Results Reproducibility 

By design, all cuBLAS API routines from a given toolkit version, generate the same bit-wise results at every run when executed on GPUs with the same architecture and the same number of SMs. However, bit-wise reproducibility is not guaranteed across toolkit versions because the implementation might differ due to some implementation changes.

This guarantee no longer holds when multiple CUDA streams are active or [fixed-point](#fixed-point) emulation is used. If multiple concurrent streams are active, the library may optimize total performance by picking different internal implementations.

Note

The non-deterministic behavior of multi-stream execution is due to library optimizations in selecting internal workspace for the routines running in parallel streams. To avoid this effect user can either:

  * provide a separate workspace for each used stream using the [cublasSetWorkspace()](#cublassetworkspace) function, or

  * have one cuBLAS handle per stream, or

  * use [cublasLtMatmul()](#cublasltmatmul) instead of GEMM-family of functions and provide user owned workspace, or

  * set a debug environment variable `CUBLAS_WORKSPACE_CONFIG` to `:16:8` (may limit overall performance) or `:4096:8` (will increase library footprint in GPU memory by approximately 24MiB).


The non-deterministic behavior of [fixed-point](#fixed-point) emulation is due to the large workspace memory requirements (see [Fixed-Point Workspace Requirements](#id2) for details). This requires dynamically allocating memory with cudaMallocAsync() and allocation failures result in fallbacks to non-emulated routines. To avoid this effect, users can provide workspace via [cublasSetWorkspace()](#cublassetworkspace) to meet fixed-point emulation workspace requirements.

Any of those settings will allow for deterministic behavior even with multiple concurrent streams sharing a single cuBLAS handle.

This behavior is expected to change in a future release.

For some routines such as [cublas<t>symv()](#cublas-t-symv) and [cublas<t>hemv()](#cublas-t-hemv), an alternate significantly faster routine can be chosen using the routine [cublasSetAtomicsMode()](#cublassetatomicsmode). In that case, the results are not guaranteed to be bit-wise reproducible because atomics are used for the computation.

###  2.1.5. Scalar Parameters 

There are two categories of the functions that use scalar parameters :

  * Functions that take `alpha` and/or `beta` parameters by reference on the host or the device as scaling factors, such as `gemm`.

  * Functions that return a scalar result on the host or the device such as `amax()`, `amin`, `asum()`, `rotg()`, `rotmg()`, `dot()` and `nrm2()`.


For the functions of the first category, when the pointer mode is set to `CUBLAS_POINTER_MODE_HOST`, the scalar parameters `alpha` and/or `beta` can be on the stack or allocated on the heap, shouldn’t be placed in managed memory. Underneath, the CUDA kernels related to those functions will be launched with the value of `alpha` and/or `beta`. Therefore if they were allocated on the heap, they can be freed just after the return of the call even though the kernel launch is asynchronous. When the pointer mode is set to `CUBLAS_POINTER_MODE_DEVICE`, `alpha` and/or `beta` must be accessible on the device and their values should not be modified until the kernel is done. Note that since `cudaFree()` does an implicit `cudaDeviceSynchronize()`, `cudaFree()` can still be called on `alpha` and/or `beta` just after the call but it would defeat the purpose of using this pointer mode in that case.

For the functions of the second category, when the pointer mode is set to `CUBLAS_POINTER_MODE_HOST`, these functions block the CPU, until the GPU has completed its computation and the results have been copied back to the Host. When the pointer mode is set to `CUBLAS_POINTER_MODE_DEVICE`, these functions return immediately. In this case, similar to matrix and vector results, the scalar result is ready only when execution of the routine on the GPU has completed. This requires proper synchronization in order to read the result from the host.

In either case, the pointer mode `CUBLAS_POINTER_MODE_DEVICE` allows the library functions to execute completely asynchronously from the Host even when `alpha` and/or `beta` are generated by a previous kernel. For example, this situation can arise when iterative methods for solution of linear systems and eigenvalue problems are implemented using the cuBLAS library.

###  2.1.6. Parallelism with Streams 

If the application uses the results computed by multiple independent tasks, CUDA™ streams can be used to overlap the computation performed in these tasks.

The application can conceptually associate each stream with each task. In order to achieve the overlap of computation between the tasks, the user should create CUDA™ streams using the function `cudaStreamCreate()` and set the stream to be used by each individual cuBLAS library routine by calling [cublasSetStream()](#cublassetstream) just before calling the actual cuBLAS routine. Note that [cublasSetStream()](#cublassetstream) resets the user-provided workspace to the default workspace pool; see [cublasSetWorkspace()](#cublassetworkspace). Then, the computation performed in separate streams would be overlapped automatically when possible on the GPU. This approach is especially useful when the computation performed by a single task is relatively small and is not enough to fill the GPU with work.

We recommend using the new cuBLAS API with scalar parameters and results passed by reference in the device memory to achieve maximum overlap of the computation when using streams.

A particular application of streams, batching of multiple small kernels, is described in the following section.

###  2.1.7. Batching Kernels 

In this section, we explain how to use streams to batch the execution of small kernels. For instance, suppose that we have an application where we need to make many small independent matrix-matrix multiplications with dense matrices.

It is clear that even with millions of small independent matrices we will not be able to achieve the same _GFLOPS_ rate as with a one large matrix. For example, a single \\(n \times n\\) large matrix-matrix multiplication performs \\(n^{3}\\) operations for \\(n^{2}\\) input size, while 1024 \\(\frac{n}{32} \times \frac{n}{32}\\) small matrix-matrix multiplications perform \\(1024\left( \frac{n}{32} \right)^{3} = \frac{n^{3}}{32}\\) operations for the same input size. However, it is also clear that we can achieve a significantly better performance with many small independent matrices compared with a single small matrix.

The architecture family of GPUs allows us to execute multiple kernels simultaneously. Hence, in order to batch the execution of independent kernels, we can run each of them in a separate stream. In particular, in the above example we could create 1024 CUDA™ streams using the function `cudaStreamCreate()`, then preface each call to [cublas<t>gemm()](#id8) with a call to [cublasSetStream()](#cublassetstream) with a different stream for each of the matrix-matrix multiplications (note that [cublasSetStream()](#cublassetstream) resets user-provided workspace to the default workspace pool, see [cublasSetWorkspace()](#cublassetworkspace)). This will ensure that when possible the different computations will be executed concurrently. Although the user can create many streams, in practice it is not possible to have more than 32 concurrent kernels executing at the same time.

###  2.1.8. Cache Configuration 

On some devices, L1 cache and shared memory use the same hardware resources. The cache configuration can be set directly with the CUDA Runtime function cudaDeviceSetCacheConfig. The cache configuration can also be set specifically for some functions using the routine cudaFuncSetCacheConfig. Please refer to the CUDA Runtime API documentation for details about the cache configuration settings.

Because switching from one configuration to another can affect kernels concurrency, the cuBLAS Library does not set any cache configuration preference and relies on the current setting. However, some cuBLAS routines, especially Level-3 routines, rely heavily on shared memory. Thus the cache preference setting might affect adversely their performance.

###  2.1.9. Static Library Support 

The cuBLAS Library is also delivered in a static form as `libcublas_static.a` on Linux. The static cuBLAS library and all other static math libraries depend on a common thread abstraction layer library called `libculibos.a`.

For example, on Linux, to compile a small application using cuBLAS, against the dynamic library, the following command can be used:
    
    
    nvcc myCublasApp.c  -lcublas  -o myCublasApp
    

Whereas to compile against the static cuBLAS library, the following command must be used:
    
    
    nvcc myCublasApp.c  -lcublas_static   -lculibos -o myCublasApp
    

It is also possible to use the native Host C++ compiler. Depending on the Host operating system, some additional libraries like `pthread` or `dl` might be needed on the linking line. The following command on Linux is suggested :
    
    
    g++ myCublasApp.c  -lcublas_static   -lculibos -lcudart_static -lpthread -ldl -I <cuda-toolkit-path>/include -L <cuda-toolkit-path>/lib64 -o myCublasApp
    

Note that in the latter case, the library `cuda` is not needed. The CUDA Runtime will try to open explicitly the `cuda` library if needed. In the case of a system which does not have the CUDA driver installed, this allows the application to gracefully manage this issue and potentially run if a CPU-only path is available.

Starting with release 11.2, using the typed functions instead of the extension functions (cublas**Ex()) helps in reducing the binary size when linking to static cuBLAS Library.

###  2.1.10. GEMM Algorithms Numerical Behavior 

Some GEMM algorithms split the computation along the dimension K to increase the GPU occupancy, especially when the dimension K is large compared to dimensions M and N. When this type of algorithm is chosen by the cuBLAS heuristics or explicitly by the user, the results of each split is summed deterministically into the resulting matrix to get the final result.

For the routines [cublas<t>gemmEx()](#id14) and [cublasGemmEx()](#cublasgemmex), when the compute type is greater than the output type, the sum of the split chunks can potentially lead to some intermediate overflows thus producing a final resulting matrix with some overflows. Those overflows might not have occurred if all the dot products had been accumulated in the compute type before being converted at the end in the output type. This computation side-effect can be easily exposed when the computeType is `CUDA_R_32F` and Atype, Btype and Ctype are `CUDA_R_16F`. This behavior can be controlled using the compute precision mode `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` with [cublasSetMathMode()](#cublassetmathmode)

###  2.1.11. Tensor Core Usage 

Tensor cores were first introduced with Volta GPUs (compute capability 7.0 and above) and significantly accelerate matrix multiplications. Starting with cuBLAS version 11.0.0, the library may automatically make use of Tensor Core capabilities wherever possible, unless they are explicitly disabled by selecting pedantic compute modes in cuBLAS (see [cublasSetMathMode()](#cublassetmathmode), [cublasMath_t](#cublasmath-t)).

It should be noted that the library will pick a Tensor Core enabled implementation wherever it determines that it would provide the best performance.

The best performance when using Tensor Cores can be achieved when the matrix dimensions and pointers meet certain memory alignment requirements. Specifically, all of the following conditions must be satisfied to get the most performance out of Tensor Cores:

  * `((op_A == CUBLAS_OP_N ? m : k) * AtypeSize) % 16 == 0`

  * `((op_B == CUBLAS_OP_N ? k : n) * BtypeSize) % 16 == 0`

  * `(m * CtypeSize) % 16 == 0`

  * `(lda * AtypeSize) % 16 == 0`

  * `(ldb * BtypeSize) % 16 == 0`

  * `(ldc * CtypeSize) % 16 == 0`

  * `intptr_t(A) % 16 == 0`

  * `intptr_t(B) % 16 == 0`

  * `intptr_t(C) % 16 == 0`


To conduct matrix multiplication with FP8 types (see [8-bit Floating Point Data Types (FP8) Usage](#bit-floating-point-data-types-fp8-usage)), you must ensure that your matrix dimensions and pointers meet the optimal requirements listed above. Aside from FP8, there are no longer any restrictions on matrix dimensions and memory alignments to use Tensor Cores (starting with cuBLAS version 11.0.0).

###  2.1.12. CUDA Graphs Support 

cuBLAS routines can be captured in CUDA Graph stream capture without restrictions in most situations.

The exception are routines that output results into host buffers (e.g. [cublas<t>dot()](#id6) while pointer mode `CUBLAS_POINTER_MODE_HOST` is configured), as it enforces synchronization.

For input coefficients (such as `alpha`, `beta`) behavior depends on the pointer mode setting:

  * In the case of `CUBLAS(LT)_POINTER_MODE_HOST`, coefficient values are captured in the graph.

  * In the case of pointer modes with device pointers, coefficient value is accessed using the device pointer at the time of graph execution.


Note

When captured in CUDA Graph stream capture, cuBLAS routines can create [memory nodes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graph-memory-nodes) through the use of stream-ordered allocation APIs, `cudaMallocAsync` and `cudaFreeAsync`. However, as there is currently no support for memory nodes in [child graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#node-types) or graphs launched [from the device](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-graph-launch), attempts to capture cuBLAS routines in such scenarios may fail. To avoid this issue, use the [cublasSetWorkspace()](#cublassetworkspace) function to provide user-owned workspace memory.

###  2.1.13. 64-bit Integer Interface 

cuBLAS version 12 introduced 64-bit integer capable functions. Each 64-bit integer function is equivalent to a 32-bit integer function with the following changes:

  * The function name has `_64` suffix.

  * The dimension (problem size) data type changed from `int` to `int64_t`. Examples of dimension: `m`, `n`, and `k`.

  * The leading dimension data type changed from `int` to `int64_t`. Examples of leading dimension: `lda`, `ldb`, and `ldc`.

  * The vector increment data type changed from `int` to `int64_t`. Examples of vector increment: `incx` and `incy`.


For example, consider the following 32-bit integer functions:
    
    
    cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
    cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float *x, int incx, int *result);
    cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *A, int lda);
    

The equivalent 64-bit integer functions are:
    
    
    cublasStatus_t cublasSetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda, void *B, int64_t ldb);
    cublasStatus_t cublasIsamax_64(cublasHandle_t handle, int64_t n, const float *x, int64_t incx, int64_t *result);
    cublasStatus_t cublasSsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float *alpha, const float *x, int64_t incx, float *A, int64_t lda);
    

Not every function has a 64-bit integer equivalent. For instance, [cublasSetMathMode()](#cublassetmathmode) doesn’t have any arguments that could meaningfully be `int64_t`. For documentation brevity, the 64-bit integer APIs are not explicitly listed, but only mentioned that they exist for the relevant functions.


##  2.2. cuBLAS Datatypes Reference 

###  2.2.1. cublasHandle_t 

The [cublasHandle_t](#cublashandle-t) type is a pointer type to an opaque structure holding the cuBLAS library context. The cuBLAS library context must be initialized using [cublasCreate()](#cublascreate) and the returned handle must be passed to all subsequent library function calls. The context should be destroyed at the end using [cublasDestroy()](#cublasdestroy).

###  2.2.2. cublasStatus_t 

The type is used for function status returns. All cuBLAS library functions return their status, which can have the following values.

Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` |  The cuBLAS library was not initialized. This is usually caused by the lack of a prior [cublasCreate()](#cublascreate) call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup. To correct: call [cublasCreate()](#cublascreate) before the function call; and check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.  
`CUBLAS_STATUS_ALLOC_FAILED` |  Resource allocation failed inside the cuBLAS library. This is usually caused by a `cudaMalloc()` failure. To correct: prior to the function call, deallocate previously allocated memory as much as possible.  
`CUBLAS_STATUS_INVALID_VALUE` |  An unsupported value or parameter was passed to the function (a negative vector size, for example). To correct: ensure that all the parameters being passed have valid values.  
`CUBLAS_STATUS_ARCH_MISMATCH` |  The function requires a feature absent from the device architecture; usually caused by compute capability lower than 5.0. To correct: compile and run the application on a device with appropriate compute capability.  
`CUBLAS_STATUS_MAPPING_ERROR` |  An access to GPU memory space failed, which is usually caused by a failure to bind a texture. To correct: before the function call, unbind any previously bound textures.  
`CUBLAS_STATUS_EXECUTION_FAILED` |  The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons. To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.  
`CUBLAS_STATUS_INTERNAL_ERROR` |  An internal cuBLAS operation failed. This error is usually caused by a `cudaMemcpyAsync()` failure. To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.  
`CUBLAS_STATUS_NOT_SUPPORTED` | The functionality requested is not supported.  
`CUBLAS_STATUS_LICENSE_ERROR` | The functionality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.  
  
###  2.2.3. cublasOperation_t 

The [cublasOperation_t](#cublasoperation-t) type indicates which operation needs to be performed with the dense matrix. Its values correspond to Fortran characters `‘N’` or `‘n’` (non-transpose), `‘T’` or `‘t’` (transpose) and `‘C’` or `‘c’` (conjugate transpose) that are often used as parameters to legacy BLAS implementations.

Value | Meaning  
---|---  
`CUBLAS_OP_N` | The non-transpose operation is selected.  
`CUBLAS_OP_T` | The transpose operation is selected.  
`CUBLAS_OP_C` | The conjugate transpose operation is selected.  
  
###  2.2.4. cublasFillMode_t 

The type indicates which part (lower or upper) of the dense matrix was filled and consequently should be used by the function. Its values correspond to Fortran characters `L` or `l` (lower) and `U` or `u` (upper) that are often used as parameters to legacy BLAS implementations.

Value | Meaning  
---|---  
`CUBLAS_FILL_MODE_LOWER` | The lower part of the matrix is filled.  
`CUBLAS_FILL_MODE_UPPER` | The upper part of the matrix is filled.  
`CUBLAS_FILL_MODE_FULL` | The full matrix is filled.  
  
###  2.2.5. cublasDiagType_t 

The type indicates whether the main diagonal of the dense matrix is unity and consequently should not be touched or modified by the function. Its values correspond to Fortran characters `‘N’` or `‘n’` (non-unit) and `‘U’` or `‘u’` (unit) that are often used as parameters to legacy BLAS implementations.

Value | Meaning  
---|---  
`CUBLAS_DIAG_NON_UNIT` | The matrix diagonal has non-unit elements.  
`CUBLAS_DIAG_UNIT` | The matrix diagonal has unit elements.  
  
###  2.2.6. cublasSideMode_t 

The type indicates whether the dense matrix is on the left or right side in the matrix equation solved by a particular function. Its values correspond to Fortran characters `‘L’` or `‘l’` (left) and `‘R’` or `‘r’` (right) that are often used as parameters to legacy BLAS implementations.

Value | Meaning  
---|---  
`CUBLAS_SIDE_LEFT` | The matrix is on the left side in the equation.  
`CUBLAS_SIDE_RIGHT` | The matrix is on the right side in the equation.  
  
###  2.2.7. cublasPointerMode_t 

The [cublasPointerMode_t](#cublaspointermode-t) type indicates whether the scalar values are passed by reference on the host or device. It is important to point out that if several scalar values are present in the function call, all of them must conform to the same single pointer mode. The pointer mode can be set and retrieved using [cublasSetPointerMode()](#cublassetpointermode) and [cublasGetPointerMode()](#cublasgetpointermode) routines, respectively.

Value | Meaning  
---|---  
`CUBLAS_POINTER_MODE_HOST` | The scalars are passed by reference on the host.  
`CUBLAS_POINTER_MODE_DEVICE` | The scalars are passed by reference on the device.  
  
###  2.2.8. cublasAtomicsMode_t 

The type indicates whether cuBLAS routines which has an alternate implementation using atomics can be used. The atomics mode can be set and queried using [cublasSetAtomicsMode()](#cublassetatomicsmode) and [cublasGetAtomicsMode()](#cublasgetatomicsmode) and routines, respectively.

Value | Meaning  
---|---  
`CUBLAS_ATOMICS_NOT_ALLOWED` | The usage of atomics is not allowed.  
`CUBLAS_ATOMICS_ALLOWED` | The usage of atomics is allowed.  
  
###  2.2.9. cublasGemmAlgo_t 

cublasGemmAlgo_t type is an enumerant to specify the algorithm for matrix-matrix multiplication on GPU architectures up to `sm_75`. On `sm_80` and newer GPU architectures, this enumerant has no effect. cuBLAS has the following algorithm options:

Value | Meaning  
---|---  
`CUBLAS_GEMM_DEFAULT` | Apply Heuristics to select the GEMM algorithm  
`CUBLAS_GEMM_ALGO0` to `CUBLAS_GEMM_ALGO23` | Explicitly choose an Algorithm `0..23`. Note: Doesn’t have effect on NVIDIA Ampere architecture GPUs and newer.  
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`[DEPRECATED] | This mode is deprecated and will be removed in a future release. Apply Heuristics to select the GEMM algorithm, while allowing use of reduced precision CUBLAS_COMPUTE_32F_FAST_16F kernels (for backward compatibility).  
`CUBLAS_GEMM_ALGO0_TENSOR_OP` to `CUBLAS_GEMM_ALGO15_TENSOR_OP`[DEPRECATED] | Those values are deprecated and will be removed in a future release. Explicitly choose a Tensor core GEMM Algorithm `0..15`. Allows use of reduced precision CUBLAS_COMPUTE_32F_FAST_16F kernels (for backward compatibility). Note: Doesn’t have effect on NVIDIA Ampere architecture GPUs and newer.  
`CUBLAS_GEMM_AUTOTUNE` | [EXPERIMENTAL] The library will benchmark a number of available algorithms and choose the optimal one for the given problem configuration. Solution is cached in cublas handle so that next calls with the problem size will use the cached configuration. Note: To avoid overwriting the user’s data, the library will allocate the amount of memory corresponding to the size of the output. Note: The benchmarking is not supported during stream capture; CUBLAS_STATUS_NOT_SUPPORTED will be returned under stream capture if no configuration was found in the cache for the given problem size.  
  
###  2.2.10. cublasMath_t 

[cublasMath_t](#cublasmath-t) enumerate type is used in [cublasSetMathMode()](#cublassetmathmode) to choose compute precision modes as defined in the following table. Since this setting does not directly control the use of Tensor Cores, the mode `CUBLAS_TENSOR_OP_MATH` is being deprecated, and will be removed in a future release.

Value | Meaning  
---|---  
`CUBLAS_DEFAULT_MATH` | This is the default and highest-performance mode that uses compute and intermediate storage precisions with at least the same number of mantissa and exponent bits as requested. Tensor Cores will be used whenever possible.  
`CUBLAS_PEDANTIC_MATH` | This mode uses the prescribed precision and standardized arithmetic for all phases of calculations and is primarily intended for numerical robustness studies, testing, and debugging. This mode might not be as performant as the other modes.  
`CUBLAS_TF32_TENSOR_OP_MATH` | Enable acceleration of single-precision routines using TF32 tensor cores. Note that input conversions round to nearest even.  
`CUBLAS_FP32_EMULATED_BF16X9_MATH` | Enable acceleration of single-precision routines using the BF16x9 algorithm. See [Floating Point Emulation](#floating-point-emulation) for more details. For single precision GEMM routines cuBLAS will use the `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` compute type.  
`CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` | Enable acceleration of double-precision routines using fixed-point emulation algorithms. See [Floating Point Emulation](#floating-point-emulation) for more details.  
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` | Forces any reductions during matrix multiplications to use the accumulator type (that is, compute type) and not the output type in case of mixed precision routines where output type precision is less than the compute type precision. This is a flag that can be set (using a bitwise or operation) alongside any of the other values.  
`CUBLAS_TENSOR_OP_MATH` [DEPRECATED] | This mode is deprecated and will be removed in a future release. Allows the library to use Tensor Core operations whenever possible. For single precision GEMM routines cuBLAS will use the `CUBLAS_COMPUTE_32F_FAST_16F` compute type.  
  
###  2.2.11. cublasComputeType_t 

[cublasComputeType_t](#cublascomputetype-t) enumerate type is used in [cublasGemmEx()](#cublasgemmex) and [cublasLtMatmul()](#cublasltmatmul) (including all batched and strided batched variants) to choose compute precision modes as defined below.

Value | Meaning  
---  
`CUBLAS_COMPUTE_16F` | This is the default and highest-performance mode for 16-bit half precision floating point and all compute and intermediate storage precisions with at least 16-bit half precision. Tensor Cores will be used whenever possible.  
`CUBLAS_COMPUTE_16F_PEDANTIC` | This mode uses 16-bit half precision floating point standardized arithmetic for all phases of calculations and is primarily intended for numerical robustness studies, testing, and debugging. This mode might not be as performant as the other modes since it disables use of tensor cores.  
`CUBLAS_COMPUTE_32F` | This is the default 32-bit single precision floating point and uses compute and intermediate storage precisions of at least 32-bits.  
`CUBLAS_COMPUTE_32F_PEDANTIC` | Uses 32-bit single precision floating point arithmetic for all phases of calculations and also disables algorithmic optimizations such as Gaussian complexity reduction (3M).  
`CUBLAS_COMPUTE_32F_FAST_16F` | Allows the library to use Tensor Cores with automatic down-conversion and 16-bit half-precision compute for 32-bit input and output matrices.  
`CUBLAS_COMPUTE_32F_FAST_16BF` | Allows the library to use Tensor Cores with automatic down-convesion and bfloat16 compute for 32-bit input and output matrices. See [Alternate Floating Point](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-altfp) section for more details on bfloat16.  
`CUBLAS_COMPUTE_32F_FAST_TF32` | Allows the library to use Tensor Cores with TF32 compute for 32-bit floating point input and output matrices. Note that input conversions round to nearest even. See [Alternate Floating Point](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-altfp) section for more details on TF32 compute.  
`CUBLAS_COMPUTE_32F_EMULATED_16BFX9` | Allows the library to use the BF16x9 floating point emulation algorithm for 32-bit floating point arithmetic. See [Floating Point Emulation](#floating-point-emulation) for more details.  
`CUBLAS_COMPUTE_64F` | This is the default 64-bit double precision floating point and uses compute and intermediate storage precisions of at least 64-bits.  
`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` | Allows the library to use fixed-point emulation algorithms for 64-bit double precision floating point arithmetic. See [Floating Point Emulation](#floating-point-emulation) for more details.  
`CUBLAS_COMPUTE_64F_PEDANTIC` | Uses 64-bit double precision floating point arithmetic for all phases of calculations and also disables algorithmic optimizations such as Gaussian complexity reduction (3M).  
`CUBLAS_COMPUTE_32I` | This is the default 32-bit integer mode and uses compute and intermediate storage precisions of at least 32-bits.  
`CUBLAS_COMPUTE_32I_PEDANTIC` | Uses 32-bit integer arithmetic for all phases of calculations.  
  
Note

Setting the environment variable `NVIDIA_TF32_OVERRIDE = 0` will override any defaults or programmatic configuration of NVIDIA libraries, and consequently, cuBLAS will not accelerate single-precision computations with TF32 tensor cores.

###  2.2.12. cublasEmulationStrategy_t 

[cublasEmulationStrategy_t](#cublasemulationstrategy-t) enumerate type is used in [cublasSetEmulationStrategy()](#cublassetemulationstrategy) to choose how to leverage floating point emulation algorithms.

Value | Meaning  
---|---  
`CUBLAS_EMULATION_STRATEGY_DEFAULT` | This is the default emulation strategy and is equivalent to `CUBLAS_EMULATION_STRATEGY_PERFORMANT` unless the `CUBLAS_EMULATION_STRATEGY` environment variable is set.  
`CUBLAS_EMULATION_STRATEGY_PERFORMANT` | A strategy which utilizes emulation whenever it provides a performance benefit.  
`CUBLAS_EMULATION_STRATEGY_EAGER` | A strategy which utilizes emulation whenever possible.  
  
Note

In general, the [cublasSetEmulationStrategy()](#cublassetemulationstrategy) function takes precedence over the environment variable setting. However, setting the environment variable `CUBLAS_EMULATION_STRATEGY` to `performant` or `eager` will override the default emulation strategy with the corresponding emulation strategy, even if the default strategy was set by the function call.


##  2.3. CUDA Datatypes Reference 

The chapter describes types shared by multiple CUDA Libraries and defined in the header file `library_types.h`.

###  2.3.1. cudaDataType_t 

The `cudaDataType_t` type is an enumerant to specify the data precision. It is used when the data reference does not carry the type itself (e.g void *)

For example, it is used in the routine [cublasSgemmEx()](#cublas-t-gemmex).

Value | Meaning  
---|---  
`CUDA_R_16F` | The data type is a 16-bit real half precision floating-point  
`CUDA_C_16F` | The data type is a 32-bit structure comprised of two half precision floating-points representing a complex number.  
`CUDA_R_16BF` | The data type is a 16-bit real bfloat16 floating-point  
`CUDA_C_16BF` | The data type is a 32-bit structure comprised of two bfloat16 floating-points representing a complex number.  
`CUDA_R_32F` | The data type is a 32-bit real single precision floating-point  
`CUDA_C_32F` | The data type is a 64-bit structure comprised of two single precision floating-points representing a complex number.  
`CUDA_R_64F` | The data type is a 64-bit real double precision floating-point  
`CUDA_C_64F` | The data type is a 128-bit structure comprised of two double precision floating-points representing a complex number.  
`CUDA_R_8I` | The data type is a 8-bit real signed integer  
`CUDA_C_8I` | The data type is a 16-bit structure comprised of two 8-bit signed integers representing a complex number.  
`CUDA_R_8U` | The data type is a 8-bit real unsigned integer  
`CUDA_C_8U` | The data type is a 16-bit structure comprised of two 8-bit unsigned integers representing a complex number.  
`CUDA_R_32I` | The data type is a 32-bit real signed integer  
`CUDA_C_32I` | The data type is a 64-bit structure comprised of two 32-bit signed integers representing a complex number.  
`CUDA_R_8F_E4M3` | The data type is an 8-bit real floating point in E4M3 format  
`CUDA_R_8F_E5M2` | The data type is an 8-bit real floating point in E5M2 format  
`CUDA_R_4F_E2M1` | The data type is a 4-bit real floating point in E2M1 format  
  
###  2.3.2. cudaEmulationStrategy_t 

The `cudaEmulationStrategy_t` is a parameter to specify how to leverage floating point emulation algorithms. This is equivalent to [cublasEmulationStrategy_t](#cublasemulationstrategy-t).

###  2.3.3. cudaEmulationMantissaControl_t 

The `cudaEmulationMantissaControl_t` is an enumerated type to specify how to configure how the number of mantissa bits are calculated in floating point emulation algorithms. See See [cublasSetFixedPointEmulationMantissaControl()](#cublassetfixedpointemulationmantissacontrol) and [cublasGetFixedPointEmulationMaxMantissaBitCount()](#cublasgetfixedpointemulationmaxmantissabitcount).

Value | Meaning  
---|---  
`CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC` | The number of retained mantissa bits is computed at runtime to ensure the same or better accuracy than the native floating point representation.  
`CUDA_EMULATION_MANTISSA_CONTROL_FIXED` | The number of retained mantissa bits is fixed at runtime.  
  
###  2.3.4. cudaEmulationSpecialValuesSupport_t 

The `cudaEmulationSpecialValuesSupport_t` is an enumerated type to specify how to configure which floating point special values are required to be supported by floating point emulation algorithms. See [cublasSetEmulationSpecialValuesSupport()](#cublassetemulationspecialvaluessupport) and [cublasGetEmulationSpecialValuesSupport()](#cublasgetemulationspecialvaluessupport).

Value | Meaning  
---|---  
`CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT` | The default special value support mask which contains support for signed infinities and NaN values.  
`CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE` | There are no requirements for emulation algorithms to support special values.  
`CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY` | Require emulation algorithms to handle signed infinity inputs and outputs.  
`CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN` | Require emulation algorithms to handle NaN inputs and outputs.  
  
Note

In general, the [cublasSetEmulationSpecialValuesSupport()](#cublassetemulationspecialvaluessupport) function takes precedence over the environment variable setting. However, setting the environment variable `CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK` to a bitmask value will override the default special values support with the specified mask, even if the default was set by the function call.

###  2.3.5. libraryPropertyType_t 

The `libraryPropertyType_t` is used as a parameter to specify which property is requested when using the routine [cublasGetProperty()](#cublasgetproperty)

Value | Meaning  
---|---  
`MAJOR_VERSION` | enumerant to query the major version  
`MINOR_VERSION` | enumerant to query the minor version  
`PATCH_LEVEL` | number to identify the patch level


##  2.4. cuBLAS Helper Function Reference   
  
###  2.4.1. cublasCreate() 
    
    
    cublasStatus_t
    cublasCreate(cublasHandle_t *handle)
    

This function initializes the cuBLAS library and creates a handle to an opaque structure holding the cuBLAS library context. It allocates hardware resources on the host and device and must be called prior to making any other cuBLAS library calls.

The cuBLAS library context is tied to the current CUDA device. To use the library on multiple devices, one cuBLAS handle needs to be created for each device. See also [cuBLAS Context](#cublas-context).

For a given device, multiple cuBLAS handles with different configurations can be created. For multi-threaded applications that use the same device from different threads, the recommended programming model is to create one cuBLAS handle per thread and use that cuBLAS handle for the entire life of the thread.

Because [cublasCreate()](#cublascreate) allocates some internal resources and the release of those resources by calling [cublasDestroy()](#cublasdestroy) will implicitly call `cudaDeviceSynchronize()`, it is recommended to minimize the number of times these functions are called.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The initialization succeeded  
`CUBLAS_STATUS_NOT_INITIALIZED` | The CUDA™ Runtime initialization failed  
`CUBLAS_STATUS_ALLOC_FAILED` | The resources could not be allocated  
`CUBLAS_STATUS_INVALID_VALUE` | `handle` is NULL  
  
###  2.4.2. cublasDestroy() 
    
    
    cublasStatus_t
    cublasDestroy(cublasHandle_t handle)
    

This function releases hardware resources used by the cuBLAS library. This function is usually the last call with a particular handle to the cuBLAS library. Because [cublasCreate()](#cublascreate) allocates some internal resources and the release of those resources by calling [cublasDestroy()](#cublasdestroy) will implicitly call `cudaDeviceSynchronize()`, it is recommended to minimize the number of times these functions are called.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the shut down succeeded  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
  
###  2.4.3. cublasGetVersion() 
    
    
    cublasStatus_t
    cublasGetVersion(cublasHandle_t handle, int *version)
    

This function returns the version number of the cuBLAS library.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | `version` is NULL  
  
Note

This function can be safely called with `handle` set to NULL. This allows users to get the version of the library without a handle. Another way to do this is with [cublasGetProperty()](#cublasgetproperty).

###  2.4.4. cublasGetProperty() 
    
    
    cublasStatus_t
    cublasGetProperty(libraryPropertyType type, int *value)
    

This function returns the value of the requested property in memory pointed to by value. Refer to `libraryPropertyType` for supported types.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` |  Invalid type or value

  * If `type` has an invalid value, or
  * if `value` is NULL

  
  
###  2.4.5. cublasGetStatusName() 
    
    
    const char* cublasGetStatusName(cublasStatus_t status)
    

This function returns the string representation of a given status.

Return Value | Meaning  
---|---  
NULL-terminated string | The string representation of the `status`  
  
###  2.4.6. cublasGetStatusString() 
    
    
    const char* cublasGetStatusString(cublasStatus_t status)
    

This function returns the description string for a given status.

Return Value | Meaning  
---|---  
NULL-terminated string | The description of the `status`  
  
###  2.4.7. cublasSetStream() 
    
    
    cublasStatus_t
    cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
    

This function sets the cuBLAS library stream, which will be used to execute all subsequent calls to the cuBLAS library functions. If the cuBLAS library stream is not set, all kernels use the _default_ NULL stream. In particular, this routine can be used to change the stream between kernel launches and then to reset the cuBLAS library stream back to NULL. Additionally this function unconditionally resets the cuBLAS library workspace back to the default workspace pool (see [cublasSetWorkspace()](#cublassetworkspace)).

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the stream was set successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
  
###  2.4.8. cublasSetWorkspace() 
    
    
    cublasStatus_t
    cublasSetWorkspace(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes)
    

This function sets the cuBLAS library workspace to a user-owned device buffer, which will be used to execute all subsequent calls to the cuBLAS library functions (on the currently set stream). If the cuBLAS library workspace is not set, all kernels will use the default workspace pool allocated during the cuBLAS context creation. In particular, this routine can be used to change the workspace between kernel launches. The workspace pointer has to be aligned to at least 256 bytes, otherwise `CUBLAS_STATUS_INVALID_VALUE` error is returned. The [cublasSetStream()](#cublassetstream) function unconditionally resets the cuBLAS library workspace back to the default workspace pool. Calling this function, including with `workspaceSizeInBytes` equal to 0, will prevent the cuBLAS library from utilizing the default workspace. Too small value of `workspaceSizeInBytes` may cause some routines to fail with `CUBLAS_STATUS_ALLOC_FAILED` error returned or cause large regressions in performance. Workspace size equal to or larger than 16KiB is enough to prevent `CUBLAS_STATUS_ALLOC_FAILED` error, while a larger workspace can provide performance benefits for some routines.

Note

If the stream set by [cublasSetStream()](#cublassetstream) is `cudaStreamPerThread` and there are multiple threads using the same cuBLAS library handle, then users must manually manage synchronization to avoid possible race conditions in the user provided workspace. Alternatively, users may rely on the default workspace pool which safely guards against race conditions.

Warning

cuBLAS functions may invoke more than one CUDA kernel, and rely on workspace being intact between the invocations. Hence, if cuBLAS handle is configured with user-provided workspace and is being used from multiple threads, it is user’s responsibility to serialize cuBLAS calls between threads, as otherwise the kernels from different cuBLAS invocations might interleave and invalidate the assumptions each of them makes regarding workspace intactness. The default workspace pool managed by cuBLAS is thread safe.

The table below shows the recommended size of user-provided workspace. This is based on the cuBLAS default workspace pool size which is GPU architecture dependent.

GPU Architecture | Recommended workspace size  
---|---  
NVIDIA Hopper Architecture (sm90) | 32 MiB  
NVIDIA Blackwell Architecture (sm10x) | 32 MiB  
NVIDIA Blackwell Architecture (sm12x) | 32 MiB  
Other | 4 MiB  
  
Note

If the cuBLAS library is configured to utilize [fixed-point](#fixed-point) emulation, which can be done by setting the corresponding math mode in [cublasSetMathMode()](#cublassetmathmode) or calling APIs with CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, it can be beneficial to provide more workspace than recommended for the GPU architecture. See [Fixed-Point Workspace Requirements](#id2) for more details.

The possible error values returned by this function and their meanings are listed below.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The stream was set successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | The `workspace` pointer wasn’t aligned to at least 256 bytes  
  
###  2.4.9. cublasGetStream() 
    
    
    cublasStatus_t
    cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId)
    

This function gets the cuBLAS library stream, which is being used to execute all calls to the cuBLAS library functions. If the cuBLAS library stream is not set, all kernels use the _default_ NULL stream.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the stream was returned successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | `streamId` is NULL  
  
###  2.4.10. cublasGetPointerMode() 
    
    
    cublasStatus_t
    cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode)
    

This function obtains the pointer mode used by the cuBLAS library. Please see the section on the [cublasPointerMode_t](#cublaspointermode-t) type for more details.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The pointer mode was obtained successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | `mode` is NULL  
  
###  2.4.11. cublasSetPointerMode() 
    
    
    cublasStatus_t
    cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode)
    

This function sets the pointer mode used by the cuBLAS library. The _default_ is for the values to be passed by reference on the host. Please see the section on the [cublasPointerMode_t](#cublaspointermode-t) type for more details.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The pointer mode was set successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | `mode` is not `CUBLAS_POINTER_MODE_HOST` or `CUBLAS_POINTER_MODE_DEVICE`  
  
###  2.4.12. cublasSetVector() 
    
    
    cublasStatus_t
    cublasSetVector(int n, int elemSize,
                    const void *x, int incx, void *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function copies `n` elements from a vector `x` in host memory space to a vector `y` in GPU memory space. Elements in both vectors are assumed to have a size of `elemSize` bytes. The storage spacing between consecutive elements is given by `incx` for the source vector `x` and by `incy` for the destination vector `y`.

Since column-major format for two-dimensional matrices is assumed, if a vector is part of a matrix, a vector increment equal to `1` accesses a (partial) column of that matrix. Similarly, using an increment equal to the leading dimension of the matrix results in accesses to a (partial) row of that matrix.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `incx`, `incy`, or `elemSize` are not positive  
`CUBLAS_STATUS_MAPPING_ERROR` | There was an error accessing GPU memory  
  
###  2.4.13. cublasGetVector() 
    
    
    cublasStatus_t
    cublasGetVector(int n, int elemSize,
                    const void *x, int incx, void *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function copies `n` elements from a vector `x` in GPU memory space to a vector `y` in host memory space. Elements in both vectors are assumed to have a size of `elemSize` bytes. The storage spacing between consecutive elements is given by `incx` for the source vector and `incy` for the destination vector `y`.

Since column-major format for two-dimensional matrices is assumed, if a vector is part of a matrix, a vector increment equal to `1` accesses a (partial) column of that matrix. Similarly, using an increment equal to the leading dimension of the matrix results in accesses to a (partial) row of that matrix.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `incx`, `incy`, or `elemSize` are not positive  
`CUBLAS_STATUS_MAPPING_ERROR` | There was an error accessing GPU memory  
  
###  2.4.14. cublasSetMatrix() 
    
    
    cublasStatus_t
    cublasSetMatrix(int rows, int cols, int elemSize,
                    const void *A, int lda, void *B, int ldb)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function copies a tile of `rows x cols` elements from a matrix `A` in host memory space to a matrix `B` in GPU memory space. It is assumed that each element requires storage of `elemSize` bytes and that both matrices are stored in column-major format, with the leading dimension of the source matrix `A` and destination matrix `B` given in `lda` and `ldb`, respectively. The leading dimension indicates the number of rows of the allocated matrix, even if only a submatrix of it is being used.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `rows` or `cols` are negative, or `elemSize`, `lda` `ldb` are not positive.  
`CUBLAS_STATUS_MAPPING_ERROR` | There was an error accessing GPU memory  
  
###  2.4.15. cublasGetMatrix() 
    
    
    cublasStatus_t
    cublasGetMatrix(int rows, int cols, int elemSize,
                    const void *A, int lda, void *B, int ldb)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function copies a tile of `rows x cols` elements from a matrix `A` in GPU memory space to a matrix `B` in host memory space. It is assumed that each element requires storage of `elemSize` bytes and that both matrices are stored in column-major format, with the leading dimension of the source matrix `A` and destination matrix `B` given in `lda` and `ldb`, respectively. The leading dimension indicates the number of rows of the allocated matrix, even if only a submatrix of it is being used.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `rows` or `cols` are negative, or `elemSize`, `lda` `ldb` are not positive.  
`CUBLAS_STATUS_MAPPING_ERROR` | There was an error accessing GPU memory  
  
###  2.4.16. cublasSetVectorAsync() 
    
    
    cublasStatus_t
    cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx,
                         void *devicePtr, int incy, cudaStream_t stream)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function has the same functionality as [cublasSetVector()](#cublassetvector), with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `incx`, `incy`, or `elemSize` are not positive  
`CUBLAS_STATUS_MAPPING_ERROR` | There was an error accessing GPU memory  
  
###  2.4.17. cublasGetVectorAsync() 
    
    
    cublasStatus_t
    cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx,
                         void *hostPtr, int incy, cudaStream_t stream)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function has the same functionality as [cublasGetVector()](#cublasgetvector), with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `incx`, `incy`, or `elemSize` are not positive  
`CUBLAS_STATUS_MAPPING_ERROR` | There was an error accessing GPU memory  
  
###  2.4.18. cublasSetMatrixAsync() 
    
    
    cublasStatus_t
    cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A,
                         int lda, void *B, int ldb, cudaStream_t stream)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function has the same functionality as [cublasSetMatrix()](#cublassetmatrix), with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `rows` or `cols` are negative, or `elemSize`, `lda` `ldb` are not positive.  
`CUBLAS_STATUS_MAPPING_ERROR` | There was an error accessing GPU memory  
  
###  2.4.19. cublasGetMatrixAsync() 
    
    
    cublasStatus_t
    cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A,
                         int lda, void *B, int ldb, cudaStream_t stream)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function has the same functionality as [cublasGetMatrix()](#cublasgetmatrix), with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `rows` or `cols` are negative, or `elemSize`, `lda` `ldb` are not positive.  
`CUBLAS_STATUS_MAPPING_ERROR` | There was an error accessing GPU memory  
  
###  2.4.20. cublasSetAtomicsMode() 
    
    
    cublasStatus_t cublasSetAtomicsMode(cublasHandlet handle, cublasAtomicsMode_t mode)
    

Some routines like [cublas<t>symv()](#cublas-t-symv) and [cublas<t>hemv()](#cublas-t-hemv) have an alternate implementation that use atomics to cumulate results. This implementation is generally significantly faster but can generate results that are not strictly identical from one run to the others. Mathematically, those different results are not significant but when debugging those differences can be prejudicial.

This function allows or disallows the usage of atomics in the cuBLAS library for all routines which have an alternate implementation. When not explicitly specified in the documentation of any cuBLAS routine, it means that this routine does not have an alternate implementation that use atomics. When atomics mode is disabled, each cuBLAS routine should produce the same results from one run to the other when called with identical parameters on the same Hardware.

The default atomics mode of default initialized [cublasHandle_t](#cublashandle-t) object is `CUBLAS_ATOMICS_NOT_ALLOWED`. Please see the section on the type for more details.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the atomics mode was set successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
  
###  2.4.21. cublasGetAtomicsMode() 
    
    
    cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode)
    

This function queries the atomic mode of a specific cuBLAS context.

The default atomics mode of default initialized [cublasHandle_t](#cublashandle-t) object is `CUBLAS_ATOMICS_NOT_ALLOWED`. Please see the section on the type for more details.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The atomics mode was queried successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | The argument `mode` is a NULL pointer  
  
###  2.4.22. cublasSetMathMode() 
    
    
    cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
    

The [cublasSetMathMode()](#cublassetmathmode) function enables you to choose the compute precision modes as defined by [cublasMath_t](#cublasmath-t). Users are allowed to set the compute precision mode as a logical combination of them (except the deprecated `CUBLAS_TENSOR_OP_MATH`). For example, `cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)`. Please note that the default math mode is `CUBLAS_DEFAULT_MATH`.

For matrix and compute precisions allowed for [cublasGemmEx()](#cublasgemmex) and [cublasLtMatmul()](#cublasltmatmul) APIs and their strided variants please refer to: [cublasGemmEx()](#cublasgemmex) , [cublasGemmBatchedEx()](#cublasgemmbatchedex), [cublasGemmStridedBatchedEx()](#cublasgemmstridedbatchedex), and [cublasLtMatmul()](#cublasltmatmul).

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The math mode was set successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | An invalid value for mode was specified.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
  
###  2.4.23. cublasGetMathMode() 
    
    
    cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode)
    

This function returns the math mode used by the library routines.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The math type was returned successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | If `mode` is NULL.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
  
###  2.4.24. cublasSetSmCountTarget() 
    
    
    cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget)
    

The [cublasSetSmCountTarget()](#cublassetsmcounttarget) function allows overriding the number of multiprocessors available to the library during kernels execution.

This option can be used to improve the library performance when cuBLAS routines are known to run concurrently with other work on different CUDA streams. For example, on an NVIDIA A100 GPU, which has 108 multiprocessors, when there is a concurrent kenrel running with grid size of 8, one can use [cublasSetSmCountTarget()](#cublassetsmcounttarget) with `smCountTarget` set to `100` to override the library heuristics to optimize for running on the remaining 100 multiprocessors.

When set to `0`, the library returns to its default behavior. The input value should not exceed the device’s multiprocessor count, which can be obtained using `cudaDeviceGetAttribute`. Negative values are not accepted.

The user must ensure thread safety when modifying the library handle with this routine similar to when using [cublasSetStream()](#cublassetstream), etc.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | SM count target was set successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | The value of `smCountTarget` outside of the allowed range.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
  
###  2.4.25. cublasGetSmCountTarget() 
    
    
    cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int *smCountTarget)
    

This function obtains the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | SM count target was returned successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | smCountTarget is NULL.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.26. cublasSetEmulationStrategy() 
    
    
    cublasStatus_t cublasSetEmulationStrategy(cublasHandle_t handle, cublasEmulationStrategy_t emulationStrategy)
    

The [cublasSetEmulationStrategy()](#cublassetemulationstrategy) function enables you to select how the library should make use of [floating point emulation](#floating-point-emulation). For more details, please see [cublasEmulationStrategy_t](#cublasemulationstrategy-t).

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The emulation strategy was set successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | An invalid value for emulation strategy was specified.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
  
###  2.4.27. cublasGetEmulationStrategy() 
    
    
    cublasStatus_t cublasGetEmulationStrategy(cublasHandle_t handle, cublasEmulationStrategy_t *emulationStrategy)
    

This function obtains the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | emulation strategy was returned successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | emulationStrategy is NULL.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.28. cublasGetEmulationSpecialValuesSupport() 
    
    
    cublasStatus_t cublasGetEmulationSpecialValuesSupport(cublasHandle_t handle, cudaEmulationSpecialValuesSupport *mask)
    

This function obtains the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | emulation special values support was returned successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | mask is NULL.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.29. cublasSetEmulationSpecialValuesSupport() 
    
    
    cublasStatus_t cublasSetEmulationSpecialValuesSupport(cublasHandle_t handle, cudaEmulationSpecialValuesSupport mask)
    

This function sets the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | emulation special values support was set successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | mask is outside of the allowed range.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.30. cublasGetFixedPointEmulationMantissaControl() 
    
    
    cublasStatus_t cublasGetFixedPointEmulationMantissaControl(cublasHandle_t handle, cudaEmulationMantissaControl *mantissaControl)
    

This function obtains the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | fixed-point emulation mantissa control was returned successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | mantissaControl is NULL.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.31. cublasSetFixedPointEmulationMantissaControl() 
    
    
    cublasStatus_t cublasSetFixedPointEmulationMantissaControl(cublasHandle_t handle, cudaEmulationMantissaControl mantissaControl)
    

This function sets the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | fixed-point emulation mantissa control was set successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | mantissaControl is outside of the allowed range.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.32. cublasGetFixedPointEmulationMaxMantissaBitCount() 
    
    
    cublasStatus_t cublasGetFixedPointEmulationMaxMantissaBitCount(cublasHandle_t handle, int *maxMantissaBitCount)
    

This function obtains the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | maxMantissaBitCount was returned successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | maxMantissaBitCount is NULL.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.33. cublasSetFixedPointEmulationMaxMantissaBitCount() 
    
    
    cublasStatus_t cublasSetFixedPointEmulationMaxMantissaBitCount(cublasHandle_t handle, int maxMantissaBitCount)
    

This function sets the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | maxMantissaBitCount was set successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | maxMantissaBitCount is outside of the allowed range.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.34. cublasGetFixedPointEmulationMantissaBitOffset() 
    
    
    cublasStatus_t cublasGetFixedPointEmulationMantissaBitOffset(cublasHandle_t handle, int *mantissaBitOffset)
    

This function obtains the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | mantissaBitOffset was returned successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | mantissaBitOffset is NULL.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.35. cublasSetFixedPointEmulationMantissaBitOffset() 
    
    
    cublasStatus_t cublasSetFixedPointEmulationMantissaBitOffset(cublasHandle_t handle, int mantissaBitOffset)
    

This function sets the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | mantissaBitOffset was set successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | mantissaBitOffset is outside of the allowed range.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.36. cublasGetFixedPointEmulationMantissaBitCountPointer() 
    
    
    cublasStatus_t cublasGetFixedPointEmulationMantissaBitCountPointer(cublasHandle_t handle, int **mantissaBitCount)
    

This function obtains the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | mantissaBitCount was returned successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | mantissaBitCount is NULL.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.37. cublasSetFixedPointEmulationMantissaBitCountPointer() 
    
    
    cublasStatus_t cublasSetFixedPointEmulationMantissaBitCountPointer(cublasHandle_t handle, int *mantissaBitCount)
    

This function sets the value previously programmed to the library handle.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | mantissaBitCount was set successfully.  
`CUBLAS_STATUS_INVALID_VALUE` | mantissaBitCount is outside of the allowed range.  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized.  
  
###  2.4.38. cublasLoggerConfigure() 
    
    
    cublasStatus_t cublasLoggerConfigure(
        int             logIsOn,
        int             logToStdOut,
        int             logToStdErr,
        const char*     logFileName)
    

This function configures logging during runtime. Besides this type of configuration, it is possible to configure logging with special environment variables which will be checked by libcublas:

  * `CUBLAS_LOGINFO_DBG` \- setting this environment variable to `1` means turning logging on (by default logging is off).

  * `CUBLAS_LOGDEST_DBG` \- this environment variable encodes where to write the log to: `stdout`, `stderr` mean to write log messages to standard output or error streams, respectively. Other values are interpreted as file names.


**Parameters**

Param. | Memory | In/out | Meaning  
---|---|---|---  
logIsOn | host | input | Turn on/off logging completely. By default is off, but is turned on by calling [cublasSetLoggerCallback()](#cublassetloggercallback) to user defined callback function.  
logToStdOut | host | input | Turn on/off logging to standard output I/O stream. By default is off.  
logToStdErr | host | input | Turn on/off logging to standard error I/O stream. By default is off.  
logFileName | host | input | Turn on/off logging to file in filesystem specified by it’s name. [cublasLoggerConfigure()](#cublasloggerconfigure) copies the content of `logFileName`. You should provide null pointer if you are not interested in this type of logging.  
Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
  
###  2.4.39. cublasGetLoggerCallback() 
    
    
    cublasStatus_t cublasGetLoggerCallback(
        cublasLogCallback* userCallback)
    

This function retrieves function pointer to previously installed custom user defined callback function via [cublasSetLoggerCallback()](#cublassetloggercallback) or zero otherwise.

Param. | Memory | In/out | Meaning  
---|---|---|---  
userCallback | host | output | Pointer to user defined callback function.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_INVALID_VALUE` | `userCallback` is NULL  
  
###  2.4.40. cublasSetLoggerCallback() 
    
    
    cublasStatus_t cublasSetLoggerCallback(
        cublasLogCallback   userCallback)
    

This function installs a custom user defined callback function via cublas C public API.

Param. | Memory | In/out | Meaning  
---|---|---|---  
userCallback | host | input | Pointer to user defined callback function.  
Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully


##  2.5. cuBLAS Level-1 Function Reference   
  
In this chapter we describe the Level-1 Basic Linear Algebra Subprograms (BLAS1) functions that perform scalar and vector based operations. We will use abbreviations <_type_ > for type and <_t_ > for the corresponding short type to make a more concise and clear presentation of the implemented functions. Unless otherwise specified <_type_ > and <_t_ > have the following meanings:

<_type_ > | <_t_ > | Meaning  
---|---|---  
`float` | `s` or `S` | real single-precision  
`double` | `d` or `D` | real double-precision  
`cuComplex` | `c` or `C` | complex single-precision  
`cuDoubleComplex` | `z` or `Z` | complex double-precision  
  
When the parameters and returned values of the function differ, which sometimes happens for complex input, the <_t_ > can also be `Sc`, `Cs`, `Dz` and `Zd`.

The abbreviation \\(\mathbf{Re}(\cdot)\\) and \\(\mathbf{Im}(\cdot)\\) will stand for the real and imaginary part of a number, respectively. Since imaginary part of a real number does not exist, we will consider it to be zero and can usually simply discard it from the equation where it is being used. Also, the \\(\bar{\alpha}\\) will denote the complex conjugate of \\(\alpha\\) .

In general throughout the documentation, the lower case Greek symbols \\(\alpha\\) and \\(\beta\\) will denote scalars, lower case English letters in bold type \\(\mathbf{x}\\) and \\(\mathbf{y}\\) will denote vectors and capital English letters \\(A\\) , \\(B\\) and \\(C\\) will denote matrices.

###  2.5.1. cublasI<t>amax() 
    
    
    cublasStatus_t cublasIsamax(cublasHandle_t handle, int n,
                                const float *x, int incx, int *result)
    cublasStatus_t cublasIdamax(cublasHandle_t handle, int n,
                                const double *x, int incx, int *result)
    cublasStatus_t cublasIcamax(cublasHandle_t handle, int n,
                                const cuComplex *x, int incx, int *result)
    cublasStatus_t cublasIzamax(cublasHandle_t handle, int n,
                                const cuDoubleComplex *x, int incx, int *result)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function finds the (smallest) index of the element of the maximum magnitude. Hence, the result is the first \\(i\\) such that \\(\left| \mathbf{Im}\left( {x\lbrack j\rbrack} \right) \middle| + \middle| \mathbf{Re}\left( {x\lbrack j\rbrack} \right) \right|\\) is maximum for \\(i = 1,\ldots,n\\) and \\(j = 1 + \left( {i - 1} \right)*\text{ incx}\\) . Notice that the last equation reflects 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vector `x`.  
`x` | device | input | <_type_ > vector with elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`result` | host or device | output | The resulting index, which is set to `0` if `n <= 0` or `incx <= 0`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_ALLOC_FAILED` | The reduction buffer could not be allocated  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_INVALID_VALUE` | `result` is NULL  
  
For references please refer to NETLIB documentation:

[isamax()](http://www.netlib.org/blas/isamax.f), [idamax()](http://www.netlib.org/blas/idamax.f), [icamax()](http://www.netlib.org/blas/icamax.f), [izamax()](http://www.netlib.org/blas/izamax.f)

###  2.5.2. cublasI<t>amin() 
    
    
    cublasStatus_t cublasIsamin(cublasHandle_t handle, int n,
                                const float *x, int incx, int *result)
    cublasStatus_t cublasIdamin(cublasHandle_t handle, int n,
                                const double *x, int incx, int *result)
    cublasStatus_t cublasIcamin(cublasHandle_t handle, int n,
                                const cuComplex *x, int incx, int *result)
    cublasStatus_t cublasIzamin(cublasHandle_t handle, int n,
                                const cuDoubleComplex *x, int incx, int *result)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function finds the (smallest) index of the element of the minimum magnitude. Hence, the result is the first \\(i\\) such that \\(\left| \mathbf{Im}\left( {x\lbrack j\rbrack} \right) \middle| + \middle| \mathbf{Re}\left( {x\lbrack j\rbrack} \right) \right|\\) is minimum for \\(i = 1,\ldots,n\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incx}\\) Notice that the last equation reflects 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vector `x`.  
`x` | device | input | <_type_ > vector with elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`result` | host or device | output | The resulting index, which is set to `0` if `n <= 0` or `incx <= 0`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_ALLOC_FAILED` | The reduction buffer could not be allocated  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_INVALID_VALUE` | `result` is NULL  
  
For references please refer to NETLIB documentation:

[isamin()](http://www.netlib.org/scilib/blass.f)

###  2.5.3. cublas<t>asum() 
    
    
    cublasStatus_t  cublasSasum(cublasHandle_t handle, int n,
                                const float           *x, int incx, float  *result)
    cublasStatus_t  cublasDasum(cublasHandle_t handle, int n,
                                const double          *x, int incx, double *result)
    cublasStatus_t cublasScasum(cublasHandle_t handle, int n,
                                const cuComplex       *x, int incx, float  *result)
    cublasStatus_t cublasDzasum(cublasHandle_t handle, int n,
                                const cuDoubleComplex *x, int incx, double *result)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function computes the sum of the absolute values of the elements of vector `x`. Hence, the result is \\(\left. \sum_{i = 1}^{n} \middle| \mathbf{Im}\left( {x\lbrack j\rbrack} \right) \middle| + \middle| \mathbf{Re}\left( {x\lbrack j\rbrack} \right) \right|\\) where \\(j = 1 + \left( {i - 1} \right)*\text{incx}\\) . Notice that the last equation reflects 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vector `x`.  
`x` | device | input | <_type_ > vector with elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`result` | host or device | output | The resulting sum, which is set to `0` if `n <= 0` or `incx <= 0`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_ALLOC_FAILED` | The reduction buffer could not be allocated  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_INVALID_VALUE` | `result` is NULL  
  
For references please refer to NETLIB documentation:

[sasum()](http://www.netlib.org/blas/sasum.f), [dasum()](http://www.netlib.org/blas/dasum.f), [scasum()](http://www.netlib.org/blas/scasum.f), [dzasum()](http://www.netlib.org/blas/dzasum.f)

###  2.5.4. cublas<t>axpy() 
    
    
    cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
                               const float           *alpha,
                               const float           *x, int incx,
                               float                 *y, int incy)
    cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n,
                               const double          *alpha,
                               const double          *x, int incx,
                               double                *y, int incy)
    cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n,
                               const cuComplex       *alpha,
                               const cuComplex       *x, int incx,
                               cuComplex             *y, int incy)
    cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               cuDoubleComplex       *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function multiplies the vector `x` by the scalar \\(\alpha\\) and adds it to the vector `y` overwriting the latest vector with the result. Hence, the performed operation is \\(\mathbf{y}\lbrack j\rbrack = \alpha \times \mathbf{x}\lbrack k\rbrack + \mathbf{y}\lbrack j\rbrack\\) for \\(i = 1,\ldots,n\\) , \\(k = 1 + \left( {i - 1} \right)*\text{incx}\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incy}\\) . Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`n` |  | input | Number of elements in the vector `x` and `y`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[saxpy()](http://www.netlib.org/blas/saxpy.f), [daxpy()](http://www.netlib.org/blas/daxpy.f), [caxpy()](http://www.netlib.org/blas/caxpy.f), [zaxpy()](http://www.netlib.org/blas/zaxpy.f)

###  2.5.5. cublas<t>copy() 
    
    
    cublasStatus_t cublasScopy(cublasHandle_t handle, int n,
                               const float           *x, int incx,
                               float                 *y, int incy)
    cublasStatus_t cublasDcopy(cublasHandle_t handle, int n,
                               const double          *x, int incx,
                               double                *y, int incy)
    cublasStatus_t cublasCcopy(cublasHandle_t handle, int n,
                               const cuComplex       *x, int incx,
                               cuComplex             *y, int incy)
    cublasStatus_t cublasZcopy(cublasHandle_t handle, int n,
                               const cuDoubleComplex *x, int incx,
                               cuDoubleComplex       *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function copies the vector `x` into the vector `y`. Hence, the performed operation is \\(\mathbf{y}\lbrack j\rbrack = \mathbf{x}\lbrack k\rbrack\\) for \\(i = 1,\ldots,n\\) , \\(k = 1 + \left( {i - 1} \right)*\text{incx}\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incy}\\) . Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vector `x` and `y`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[scopy()](http://www.netlib.org/blas/scopy.f), [dcopy()](http://www.netlib.org/blas/dcopy.f), [ccopy()](http://www.netlib.org/blas/ccopy.f), [zcopy()](http://www.netlib.org/blas/zcopy.f)

###  2.5.6. cublas<t>dot() 
    
    
    cublasStatus_t cublasSdot (cublasHandle_t handle, int n,
                               const float           *x, int incx,
                               const float           *y, int incy,
                               float           *result)
    cublasStatus_t cublasDdot (cublasHandle_t handle, int n,
                               const double          *x, int incx,
                               const double          *y, int incy,
                               double          *result)
    cublasStatus_t cublasCdotu(cublasHandle_t handle, int n,
                               const cuComplex       *x, int incx,
                               const cuComplex       *y, int incy,
                               cuComplex       *result)
    cublasStatus_t cublasCdotc(cublasHandle_t handle, int n,
                               const cuComplex       *x, int incx,
                               const cuComplex       *y, int incy,
                               cuComplex       *result)
    cublasStatus_t cublasZdotu(cublasHandle_t handle, int n,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *result)
    cublasStatus_t cublasZdotc(cublasHandle_t handle, int n,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex       *result)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function computes the dot product of vectors `x` and `y`. Hence, the result is \\(\sum_{i = 1}^{n}\left( {\mathbf{x}\lbrack k\rbrack \times \mathbf{y}\lbrack j\rbrack} \right)\\) where \\(k = 1 + \left( {i - 1} \right)*\text{incx}\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incy}\\) . Notice that in the first equation the conjugate of the element of vector x should be used if the function name ends in character ‘c’ and that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vectors `x` and `y`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | input | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`result` | host or device | output | The resulting dot product, which is set to `0` if `n <= 0`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_ALLOC_FAILED` | The reduction buffer could not be allocated  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sdot()](http://www.netlib.org/blas/sdot.f), [ddot()](http://www.netlib.org/blas/ddot.f), [cdotu()](http://www.netlib.org/blas/cdotu.f), [cdotc()](http://www.netlib.org/blas/cdotc.f), [zdotu()](http://www.netlib.org/blas/zdotu.f), [zdotc()](http://www.netlib.org/blas/zdotc.f)

###  2.5.7. cublas<t>nrm2() 
    
    
    cublasStatus_t  cublasSnrm2(cublasHandle_t handle, int n,
                                const float           *x, int incx, float  *result)
    cublasStatus_t  cublasDnrm2(cublasHandle_t handle, int n,
                                const double          *x, int incx, double *result)
    cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n,
                                const cuComplex       *x, int incx, float  *result)
    cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n,
                                const cuDoubleComplex *x, int incx, double *result)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function computes the Euclidean norm of the vector `x`. The code uses a multiphase model of accumulation to avoid intermediate underflow and overflow, with the result being equivalent to \\(\sqrt{\sum_{i = 1}^{n}\left( {\mathbf{x}\lbrack j\rbrack \times \mathbf{x}\lbrack j\rbrack} \right)}\\) where \\(j = 1 + \left( {i - 1} \right)*\text{incx}\\) in exact arithmetic. Notice that the last equation reflects 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vector `x`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`result` | host or device | output | The resulting norm, which is set to `0` if `n <= 0` or `incx <= 0`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_ALLOC_FAILED` | The reduction buffer could not be allocated  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_INVALID_VALUE` | `result` is NULL  
  
For references please refer to NETLIB documentation:

[snrm2()](http://www.netlib.org/blas/snrm2.f90), [dnrm2()](http://www.netlib.org/blas/dnrm2.f90), [scnrm2()](http://www.netlib.org/blas/scnrm2.f90), [dznrm2()](http://www.netlib.org/blas/dznrm2.f90)

###  2.5.8. cublas<t>rot() 
    
    
    cublasStatus_t  cublasSrot(cublasHandle_t handle, int n,
                               float           *x, int incx,
                               float           *y, int incy,
                               const float  *c, const float           *s)
    cublasStatus_t  cublasDrot(cublasHandle_t handle, int n,
                               double          *x, int incx,
                               double          *y, int incy,
                               const double *c, const double          *s)
    cublasStatus_t  cublasCrot(cublasHandle_t handle, int n,
                               cuComplex       *x, int incx,
                               cuComplex       *y, int incy,
                               const float  *c, const cuComplex       *s)
    cublasStatus_t cublasCsrot(cublasHandle_t handle, int n,
                               cuComplex       *x, int incx,
                               cuComplex       *y, int incy,
                               const float  *c, const float           *s)
    cublasStatus_t  cublasZrot(cublasHandle_t handle, int n,
                               cuDoubleComplex *x, int incx,
                               cuDoubleComplex *y, int incy,
                               const double *c, const cuDoubleComplex *s)
    cublasStatus_t cublasZdrot(cublasHandle_t handle, int n,
                               cuDoubleComplex *x, int incx,
                               cuDoubleComplex *y, int incy,
                               const double *c, const double          *s)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by \\(cos(alpha) = c\\), \\(sin(alpha) = s\\)):

\\(G = \begin{pmatrix} c & s \\\ {- s} & c \\\ \end{pmatrix}\\)

to vectors `x` and `y`.

Hence, the result is \\(\mathbf{x}\lbrack k\rbrack = c \times \mathbf{x}\lbrack k\rbrack + s \times \mathbf{y}\lbrack j\rbrack\\) and \\(\mathbf{y}\lbrack j\rbrack = - s \times \mathbf{x}\lbrack k\rbrack + c \times \mathbf{y}\lbrack j\rbrack\\) where \\(k = 1 + \left( {i - 1} \right)*\text{incx}\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incy}\\) . Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vectors `x` and `y`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`c` | host or device | input | Cosine element of the rotation matrix.  
`s` | host or device | input | Sine element of the rotation matrix.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[srot()](http://www.netlib.org/blas/srot.f), [drot()](http://www.netlib.org/blas/drot.f), [crot()](http://www.netlib.org/lapack/lapack_routine/crot.f), [csrot()](http://www.netlib.org/blas/csrot.f), [zrot()](http://www.netlib.org/lapack/lapack_routine/zrot.f), [zdrot()](http://www.netlib.org/blas/zdrot.f)

###  2.5.9. cublas<t>rotg() 
    
    
    cublasStatus_t cublasSrotg(cublasHandle_t handle,
                               float           *a, float           *b,
                               float  *c, float           *s)
    cublasStatus_t cublasDrotg(cublasHandle_t handle,
                               double          *a, double          *b,
                               double *c, double          *s)
    cublasStatus_t cublasCrotg(cublasHandle_t handle,
                               cuComplex       *a, cuComplex       *b,
                               float  *c, cuComplex       *s)
    cublasStatus_t cublasZrotg(cublasHandle_t handle,
                               cuDoubleComplex *a, cuDoubleComplex *b,
                               double *c, cuDoubleComplex *s)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function constructs the Givens rotation matrix

\\(G = \begin{pmatrix} c & s \\\ {- s} & c \\\ \end{pmatrix}\\)

that zeros out the second entry of a \\(2 \times 1\\) vector \\(\left( {a,b} \right)^{T}\\) .

Then, for real numbers we can write

\\(\begin{pmatrix} c & s \\\ {- s} & c \\\ \end{pmatrix}\begin{pmatrix} a \\\ b \\\ \end{pmatrix} = \begin{pmatrix} r \\\ 0 \\\ \end{pmatrix}\\)

where \\(c^{2} + s^{2} = 1\\) and \\(r = \pm \sqrt{a^{2} + b^{2}}\\) . The parameters \\(a\\) and \\(b\\) are overwritten with \\(r\\) and \\(z\\) , respectively. The value of \\(z\\) is such that \\(c\\) and \\(s\\) may be recovered using the following rules:

\\(\left( {c,s} \right) = \begin{cases} \left( {\sqrt{1 - z^{2}},z} \right) & {\text{ if }\left| z \middle| < 1 \right.} \\\ \left( {0.0,1.0} \right) & {\text{ if }\left| z \middle| = 1 \right.} \\\ \left( 1/z,\sqrt{1 - z^{2}} \right) & {\text{ if }\left| z \middle| > 1 \right.} \\\ \end{cases}\\)

For complex numbers we can write

\\(\begin{pmatrix} c & s \\\ {- \bar{s}} & c \\\ \end{pmatrix}\begin{pmatrix} a \\\ b \\\ \end{pmatrix} = \begin{pmatrix} r \\\ 0 \\\ \end{pmatrix}\\)

where \\(c^{2} + \left( {\bar{s} \times s} \right) = 1\\) and \\(r = \frac{a}{|a|} \times \parallel \left( {a,b} \right)^{T} \parallel_{2}\\) with \\(\parallel \left( {a,b} \right)^{T} \parallel_{2} = \sqrt{\left| a|^{2} + \middle| B|^{2} \right.}\\) for \\(a \neq 0\\) and \\(r = b\\) for \\(a = 0\\) . Finally, the parameter \\(a\\) is overwritten with \\(r\\) on exit.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`a` | host or device | in/out | <_type_ > scalar that is overwritten with \\(r\\) .  
`b` | host or device | in/out | <_type_ > scalar that is overwritten with \\(z\\) .  
`c` | host or device | output | Cosine element of the rotation matrix.  
`s` | host or device | output | Sine element of the rotation matrix.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[srotg()](http://www.netlib.org/blas/srotg.f90), [drotg()](http://www.netlib.org/blas/drotg.f90), [crotg()](http://www.netlib.org/blas/crotg.f90), [zrotg()](http://www.netlib.org/blas/zrotg.f90)

###  2.5.10. cublas<t>rotm() 
    
    
    cublasStatus_t cublasSrotm(cublasHandle_t handle, int n, float  *x, int incx,
                               float  *y, int incy, const float*  param)
    cublasStatus_t cublasDrotm(cublasHandle_t handle, int n, double *x, int incx,
                               double *y, int incy, const double* param)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function applies the modified Givens transformation

\\(H = \begin{pmatrix} h_{11} & h_{12} \\\ h_{21} & h_{22} \\\ \end{pmatrix}\\)

to vectors `x` and `y`.

Hence, the result is \\(\mathbf{x}\lbrack k\rbrack = h_{11} \times \mathbf{x}\lbrack k\rbrack + h_{12} \times \mathbf{y}\lbrack j\rbrack\\) and \\(\mathbf{y}\lbrack j\rbrack = h_{21} \times \mathbf{x}\lbrack k\rbrack + h_{22} \times \mathbf{y}\lbrack j\rbrack\\) where \\(k = 1 + \left( {i - 1} \right)*\text{incx}\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incy}\\) . Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.

The elements , , and of matrix \\(H\\) are stored in `param[1]`, `param[2]`, `param[3]` and `param[4]`, respectively. The `flag = param[0]` defines the following predefined values for the matrix \\(H\\) entries

`flag == -1.0` | `flag == 0.0` | `flag == 1.0` | `flag == -2.0`  
---|---|---|---  
\\(\begin{pmatrix} h_{11} & h_{12} \\\ h_{21} & h_{22} \\\ \end{pmatrix}\\) | \\(\begin{pmatrix} {1.0} & h_{12} \\\ h_{21} & {1.0} \\\ \end{pmatrix}\\) | \\(\begin{pmatrix} h_{11} & {1.0} \\\ {- 1.0} & h_{22} \\\ \end{pmatrix}\\) | \\(\begin{pmatrix} {1.0} & {0.0} \\\ {0.0} & {1.0} \\\ \end{pmatrix}\\)  
  
Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vectors `x` and `y`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`param` | host or device | input | <_type_ > vector of 5 elements, where `param[0]` and `param[1..4]` contain the flag and matrix \\(H\\).  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[srotm()](http://www.netlib.org/blas/srotm.f), [drotm()](http://www.netlib.org/blas/drotm.f)

###  2.5.11. cublas<t>rotmg() 
    
    
    cublasStatus_t cublasSrotmg(cublasHandle_t handle, float  *d1, float  *d2,
                                float  *x1, const float  *y1, float  *param)
    cublasStatus_t cublasDrotmg(cublasHandle_t handle, double *d1, double *d2,
                                double *x1, const double *y1, double *param)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function constructs the modified Givens transformation

\\(H = \begin{pmatrix} h_{11} & h_{12} \\\ h_{21} & h_{22} \\\ \end{pmatrix}\\)

that zeros out the second entry of a \\(2 \times 1\\) vector \\(\left( {\sqrt{d1}*x1,\sqrt{d2}*y1} \right)^{T}\\) .

The `flag = param[0]` defines the following predefined values for the matrix \\(H\\) entries

`flag == -1.0` | `flag == 0.0` | `flag == 1.0` | `flag == -2.0`  
---|---|---|---  
\\(\begin{pmatrix} h_{11} & h_{12} \\\ h_{21} & h_{22} \\\ \end{pmatrix}\\) | \\(\begin{pmatrix} {1.0} & h_{12} \\\ h_{21} & {1.0} \\\ \end{pmatrix}\\) | \\(\begin{pmatrix} h_{11} & {1.0} \\\ {- 1.0} & h_{22} \\\ \end{pmatrix}\\) | \\(\begin{pmatrix} {1.0} & {0.0} \\\ {0.0} & {1.0} \\\ \end{pmatrix}\\)  
  
Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`d1` | host or device | in/out | <_type_ > scalar that is overwritten on exit.  
`d2` | host or device | in/out | <_type_ > scalar that is overwritten on exit.  
`x1` | host or device | in/out | <_type_ > scalar that is overwritten on exit.  
`y1` | host or device | input | <_type_ > scalar.  
`param` | host or device | output | <_type_ > vector of 5 elements, where `param[0]` and `param[1-4]` contain the flag and matrix \\(H\\).  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[srotmg()](http://www.netlib.org/blas/srotmg.f), [drotmg()](http://www.netlib.org/blas/drotmg.f)

###  2.5.12. cublas<t>scal() 
    
    
    cublasStatus_t  cublasSscal(cublasHandle_t handle, int n,
                                const float           *alpha,
                                float           *x, int incx)
    cublasStatus_t  cublasDscal(cublasHandle_t handle, int n,
                                const double          *alpha,
                                double          *x, int incx)
    cublasStatus_t  cublasCscal(cublasHandle_t handle, int n,
                                const cuComplex       *alpha,
                                cuComplex       *x, int incx)
    cublasStatus_t cublasCsscal(cublasHandle_t handle, int n,
                                const float           *alpha,
                                cuComplex       *x, int incx)
    cublasStatus_t  cublasZscal(cublasHandle_t handle, int n,
                                const cuDoubleComplex *alpha,
                                cuDoubleComplex *x, int incx)
    cublasStatus_t cublasZdscal(cublasHandle_t handle, int n,
                                const double          *alpha,
                                cuDoubleComplex *x, int incx)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function scales the vector `x` by the scalar \\(\alpha\\) and overwrites it with the result. Hence, the performed operation is \\(\mathbf{x}\lbrack j\rbrack = \alpha \times \mathbf{x}\lbrack j\rbrack\\) for \\(i = 1,\ldots,n\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incx}\\) . Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`n` |  | input | Number of elements in the vector `x`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
  
The possible error values returned by this function and their meanings are listed below.

:class: table-no-stripes Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sscal()](http://www.netlib.org/blas/sscal.f), [dscal()](http://www.netlib.org/blas/dscal.f), [csscal()](http://www.netlib.org/blas/csscal.f), [cscal()](http://www.netlib.org/blas/cscal.f), [zdscal()](http://www.netlib.org/blas/zdscal.f), [zscal()](http://www.netlib.org/blas/zscal.f)

###  2.5.13. cublas<t>swap() 
    
    
    cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float           *x,
                               int incx, float           *y, int incy)
    cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double          *x,
                               int incx, double          *y, int incy)
    cublasStatus_t cublasCswap(cublasHandle_t handle, int n, cuComplex       *x,
                               int incx, cuComplex       *y, int incy)
    cublasStatus_t cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex *x,
                               int incx, cuDoubleComplex *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function interchanges the elements of vector `x` and `y`. Hence, the performed operation is \\(\left. \mathbf{y}\lbrack j\rbrack\Leftrightarrow\mathbf{x}\lbrack k\rbrack \right.\\) for \\(i = 1,\ldots,n\\) , \\(k = 1 + \left( {i - 1} \right)*\text{incx}\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incy}\\) . Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vectors `x` and `y`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sswap()](http://www.netlib.org/blas/sswap.f), [dswap()](http://www.netlib.org/blas/dswap.f), [cswap()](http://www.netlib.org/blas/cswap.f), [zswap()](http://www.netlib.org/blas/zswap.f)


##  2.6. cuBLAS Level-2 Function Reference 

In this chapter we describe the Level-2 Basic Linear Algebra Subprograms (BLAS2) functions that perform matrix-vector operations.

###  2.6.1. cublas<t>gbmv() 
    
    
    cublasStatus_t cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n, int kl, int ku,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *x, int incx,
                               const float           *beta,
                               float           *y, int incy)
    cublasStatus_t cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n, int kl, int ku,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *x, int incx,
                               const double          *beta,
                               double          *y, int incy)
    cublasStatus_t cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n, int kl, int ku,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *x, int incx,
                               const cuComplex       *beta,
                               cuComplex       *y, int incy)
    cublasStatus_t cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n, int kl, int ku,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the banded matrix-vector multiplication

\\(\mathbf{y} = \alpha\text{ op}(A)\mathbf{x} + \beta\mathbf{y}\\)

where \\(A\\) is a banded matrix with \\(kl\\) subdiagonals and \\(ku\\) superdiagonals, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) and \\(\beta\\) are scalars. Also, for matrix \\(A\\)

\\(\text{ op}(A) = \begin{cases} A & \text{ if trans == $\mathrm{CUBLAS\\_OP\\_N}$} \\\ A^{T} & \text{ if trans == $\mathrm{CUBLAS\\_OP\\_T}$} \\\ A^{H} & \text{ if trans == $\mathrm{CUBLAS\\_OP\\_C}$} \\\ \end{cases}\\)

The banded matrix \\(A\\) is stored column by column, with the main diagonal stored in row \\(ku + 1\\) (starting in first position), the first superdiagonal stored in row \\(ku\\) (starting in second position), the first subdiagonal stored in row \\(ku + 2\\) (starting in first position), etc. So that in general, the element \\(A\left( {i,j} \right)\\) is stored in the memory location `A(ku+1+i-j,j)` for \\(j = 1,\ldots,n\\) and \\(i \in \left\lbrack {\max\left( {1,j - ku} \right),\min\left( {m,j + kl} \right)} \right\rbrack\\) . Also, the elements in the array \\(A\\) that do not conceptually correspond to the elements in the banded matrix (the top left \\(ku \times ku\\) and bottom right \\(kl \times kl\\) triangles) are not referenced.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix `A`.  
`n` |  | input | Number of columns of matrix `A`.  
`kl` |  | input | Number of subdiagonals of matrix `A`.  
`ku` |  | input | Number of superdiagonals of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x n` with `lda >= kl + ku + 1`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`x` | device | input | <_type_ > vector with `n` elements if `trans == CUBLAS_OP_N` and `m` elements otherwise.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `y` does not have to be a valid input.  
`y` | device | in/out | <_type_ > vector with `m` elements if `trans == CUBLAS_OP_N` and `n` elements otherwise.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0`, `n < 0`, `kl < 0` or `ku < 0`, or
  * if `lda < (kl + ku + 1)`, or
  * if `incx == 0` or `incy == 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T`, `CUBLAS_OP_C`, or
  * if `alpha` or `beta` are NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sgbmv()](http://www.netlib.org/blas/sgbmv.f), [dgbmv()](http://www.netlib.org/blas/dgbmv.f), [cgbmv()](http://www.netlib.org/blas/cgbmv.f), [zgbmv()](http://www.netlib.org/blas/zgbmv.f)

###  2.6.2. cublas<t>gemv() 
    
    
    cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *x, int incx,
                               const float           *beta,
                               float           *y, int incy)
    cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *x, int incx,
                               const double          *beta,
                               double          *y, int incy)
    cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *x, int incx,
                               const cuComplex       *beta,
                               cuComplex       *y, int incy)
    cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans,
                               int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-vector multiplication

\\(\textbf{y} = \alpha\text{ op}(A)\textbf{x} + \beta\textbf{y}\\)

where \\(A\\) is a \\(m \times n\\) matrix stored in column-major format, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) and \\(\beta\\) are scalars. Also, for matrix \\(A\\)

\\(\text{ op}(A) = \begin{cases} A & \text{ if trans == $\mathrm{CUBLAS\\_OP\\_N}$} \\\ A^{T} & \text{ if trans == $\mathrm{CUBLAS\\_OP\\_T}$} \\\ A^{H} & \text{ if trans == $\mathrm{CUBLAS\\_OP\\_C}$} \\\ \end{cases}\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix `A`.  
`n` |  | input | Number of columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x n` with `lda >= max(1, m)`. Before entry, the leading `m` by `n` part of the array `A` must contain the matrix of coefficients. Unchanged on exit.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`. `lda` must be at least `max(1, m)`.  
`x` | device | input | <_type_ > vector at least `(1 + (n - 1) * abs(incx))` elements if `trans == CUBLAS_OP_N` and at least `(1 + (m - 1) * abs(incx))` elements otherwise.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `y` does not have to be a valid input.  
`y` | device | in/out | <_type_ > vector at least `(1 + (m - 1) * abs(incy))` elements if `trans == CUBLAS_OP_N` and at least `(1 + (n - 1) * abs(incy))` elements otherwise.  
`incy` |  | input | Stride between consecutive elements of `y`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `m < 0` or `n < 0`, or `incx == 0` or `incy == 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sgemv()](http://www.netlib.org/blas/sgemv.f), [dgemv()](http://www.netlib.org/blas/dgemv.f), [cgemv()](http://www.netlib.org/blas/cgemv.f), [zgemv()](http://www.netlib.org/blas/zgemv.f)

###  2.6.3. cublas<t>ger() 
    
    
    cublasStatus_t  cublasSger(cublasHandle_t handle, int m, int n,
                               const float           *alpha,
                               const float           *x, int incx,
                               const float           *y, int incy,
                               float           *A, int lda)
    cublasStatus_t  cublasDger(cublasHandle_t handle, int m, int n,
                               const double          *alpha,
                               const double          *x, int incx,
                               const double          *y, int incy,
                               double          *A, int lda)
    cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n,
                               const cuComplex       *alpha,
                               const cuComplex       *x, int incx,
                               const cuComplex       *y, int incy,
                               cuComplex       *A, int lda)
    cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n,
                               const cuComplex       *alpha,
                               const cuComplex       *x, int incx,
                               const cuComplex       *y, int incy,
                               cuComplex       *A, int lda)
    cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *A, int lda)
    cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *A, int lda)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the rank-1 update

\\(A = \begin{cases} {\alpha\mathbf{xy}^{T} + A} & \text{if ger(),geru() is called} \\\ {\alpha\mathbf{xy}^{H} + A} & \text{if gerc() is called} \\\ \end{cases}\\)

where \\(A\\) is a \\(m \times n\\) matrix stored in column-major format, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) is a scalar.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`m` |  | input | Number of rows of matrix `A`.  
`n` |  | input | Number of columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`x` | device | input | <_type_ > vector with `m` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | input | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`A` | device | in/out | <_type_ > array of dimension `lda x n` with `lda >= max(1, m)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0`, or
  * if `incx == 0` or `incy == 0`, or
  * if `alpha` is NULL, or
  * if `lda < max(1, m)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sger()](http://www.netlib.org/blas/sger.f), [dger()](http://www.netlib.org/blas/dger.f), [cgeru()](http://www.netlib.org/blas/cgeru.f), [cgerc()](http://www.netlib.org/blas/cgerc.f), [zgeru()](http://www.netlib.org/blas/zgeru.f), [zgerc()](http://www.netlib.org/blas/zgerc.f)

###  2.6.4. cublas<t>sbmv() 
    
    
    cublasStatus_t cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, int k, const float  *alpha,
                               const float  *A, int lda,
                               const float  *x, int incx,
                               const float  *beta, float *y, int incy)
    cublasStatus_t cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, int k, const double *alpha,
                               const double *A, int lda,
                               const double *x, int incx,
                               const double *beta, double *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the symmetric banded matrix-vector multiplication

\\(\textbf{y} = \alpha A\textbf{x} + \beta\textbf{y}\\)

where \\(A\\) is a \\(n \times n\\) symmetric banded matrix with \\(k\\) subdiagonals and superdiagonals, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) and \\(\beta\\) are scalars.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the symmetric banded matrix \\(A\\) is stored column by column, with the main diagonal of the matrix stored in row 1, the first subdiagonal in row 2 (starting at first position), the second subdiagonal in row 3 (starting at first position), etc. So that in general, the element \\(A(i,j)\\) is stored in the memory location `A(1+i-j,j)` for \\(j = 1,\ldots,n\\) and \\(i \in \lbrack j,\min(m,j + k)\rbrack\\) . Also, the elements in the array `A` that do not conceptually correspond to the elements in the banded matrix (the bottom right \\(k \times k\\) triangle) are not referenced.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the symmetric banded matrix \\(A\\) is stored column by column, with the main diagonal of the matrix stored in row `k + 1`, the first superdiagonal in row `k` (starting at second position), the second superdiagonal in row `k-1` (starting at third position), etc. So that in general, the element \\(A(i,j)\\) is stored in the memory location `A(1+k+i-j,j)` for \\(j = 1,\ldots,n\\) and \\(i \in \lbrack\max(1,j - k),j\rbrack\\) . Also, the elements in the array `A` that do not conceptually correspond to the elements in the banded matrix (the top left \\(k \times k\\) triangle) are not referenced.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`k` |  | input | Number of sub- and super-diagonals of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x n` with `lda >= k + 1`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `y` does not have to be a valid input.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `incx == 0` or `incy == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `alpha` or `beta` are NULL, or
  * if `lda < (1 + k)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssbmv()](http://www.netlib.org/blas/ssbmv.f), [dsbmv()](http://www.netlib.org/blas/dsbmv.f)

###  2.6.5. cublas<t>spmv() 
    
    
    cublasStatus_t cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float  *alpha, const float  *AP,
                               const float  *x, int incx, const float  *beta,
                               float  *y, int incy)
    cublasStatus_t cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha, const double *AP,
                               const double *x, int incx, const double *beta,
                               double *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the symmetric packed matrix-vector multiplication

\\(\textbf{y} = \alpha A\textbf{x} + \beta\textbf{y}\\)

where \\(A\\) is a \\(n \times n\\) symmetric matrix stored in packed format, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) and \\(\beta\\) are scalars.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements in the lower triangular part of the symmetric matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+((2*n-j+1)*j)/2]` for \\(j = 1,\ldots,n\\) and \\(i \geq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements in the upper triangular part of the symmetric matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+(j*(j+1))/2]` for \\(j = 1,\ldots,n\\) and \\(i \leq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix \\(A\\) lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix \\(A\\) .  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`AP` | device | input | <_type_ > array with \\(A\\) stored in packed format.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `y` does not have to be a valid input.  
`y` | device | input | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0` or `incy == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `alpha` or `beta` are NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sspmv()](http://www.netlib.org/blas/sspmv.f), [dspmv()](http://www.netlib.org/blas/dspmv.f)

###  2.6.6. cublas<t>spr() 
    
    
    cublasStatus_t cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float  *alpha,
                              const float  *x, int incx, float  *AP)
    cublasStatus_t cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double *alpha,
                              const double *x, int incx, double *AP)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the packed symmetric rank-1 update

\\(A = \alpha\textbf{x}\textbf{x}^{T} + A\\)

where \\(A\\) is a \\(n \times n\\) symmetric matrix stored in packed format, \\(\mathbf{x}\\) is a vector, and \\(\alpha\\) is a scalar.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements in the lower triangular part of the symmetric matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+((2*n-j+1)*j)/2]` for \\(j = 1,\ldots,n\\) and \\(i \geq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements in the upper triangular part of the symmetric matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+(j*(j+1))/2]` for \\(j = 1,\ldots,n\\) and \\(i \leq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix \\(A\\) lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix \\(A\\) .  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`AP` | device | in/out | <_type_ > array with \\(A\\) stored in packed format.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `alpha` is NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sspr()](http://www.netlib.org/blas/sspr.f), [dspr()](http://www.netlib.org/blas/dspr.f)

###  2.6.7. cublas<t>spr2() 
    
    
    cublasStatus_t cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float  *alpha,
                               const float  *x, int incx,
                               const float  *y, int incy, float  *AP)
    cublasStatus_t cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double *alpha,
                               const double *x, int incx,
                               const double *y, int incy, double *AP)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the packed symmetric rank-2 update

\\(A = \alpha\left( {\textbf{x}\textbf{y}^{T} + \textbf{y}\textbf{x}^{T}} \right) + A\\)

where \\(A\\) is a \\(n \times n\\) symmetric matrix stored in packed format, \\(\mathbf{x}\\) is a vector, and \\(\alpha\\) is a scalar.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements in the lower triangular part of the symmetric matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+((2*n-j+1)*j)/2]` for \\(j = 1,\ldots,n\\) and \\(i \geq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements in the upper triangular part of the symmetric matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+(j*(j+1))/2]` for \\(j = 1,\ldots,n\\) and \\(i \leq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix \\(A\\) lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix \\(A\\) .  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | input | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`AP` | device | in/out | <_type_ > array with \\(A\\) stored in packed format.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0` or `incy == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `alpha` is NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sspr2()](http://www.netlib.org/blas/sspr2.f), [dspr2()](http://www.netlib.org/blas/dspr2.f)

###  2.6.8. cublas<t>symv() 
    
    
    cublasStatus_t cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const float           *alpha,
                               const float           *A, int lda,
                               const float           *x, int incx, const float           *beta,
                               float           *y, int incy)
    cublasStatus_t cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const double          *alpha,
                               const double          *A, int lda,
                               const double          *x, int incx, const double          *beta,
                               double          *y, int incy)
    cublasStatus_t cublasCsymv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex       *alpha, /* host or device pointer */
                               const cuComplex       *A, int lda,
                               const cuComplex       *x, int incx, const cuComplex       *beta,
                               cuComplex       *y, int incy)
    cublasStatus_t cublasZsymv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the symmetric matrix-vector multiplication.

\\(\textbf{y} = \alpha A\textbf{x} + \beta\textbf{y}\\) where \\(A\\) is a \\(n \times n\\) symmetric matrix stored in lower or upper mode, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) and \\(\beta\\) are scalars.

This function has an alternate faster implementation using atomics that can be enabled with [cublasSetAtomicsMode()](#cublassetatomicsmode).

Please see the section on the function [cublasSetAtomicsMode()](#cublassetatomicsmode) for more details about the usage of atomics.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x n` with `lda >= max(1, n)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `y` does not have to be a valid input.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0` or `incy == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < n`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssymv()](http://www.netlib.org/blas/ssymv.f), [dsymv()](http://www.netlib.org/blas/dsymv.f)

###  2.6.9. cublas<t>syr() 
    
    
    cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float           *alpha,
                              const float           *x, int incx, float           *A, int lda)
    cublasStatus_t cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double          *alpha,
                              const double          *x, int incx, double          *A, int lda)
    cublasStatus_t cublasCsyr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuComplex       *alpha,
                              const cuComplex       *x, int incx, cuComplex       *A, int lda)
    cublasStatus_t cublasZsyr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the symmetric rank-1 update

\\(A = \alpha\textbf{x}\textbf{x}^{T} + A\\)

where \\(A\\) is a \\(n \times n\\) symmetric matrix stored in column-major format, \\(\mathbf{x}\\) is a vector, and \\(\alpha\\) is a scalar.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`A` | device | in/out | <_type_ > array of dimensions `lda x n`, with `lda >= max(1, n)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)`, or
  * if `alpha` is NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssyr()](http://www.netlib.org/blas/ssyr.f), [dsyr()](http://www.netlib.org/blas/dsyr.f)

###  2.6.10. cublas<t>syr2() 
    
    
    cublasStatus_t cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                               const float           *alpha, const float           *x, int incx,
                               const float           *y, int incy, float           *A, int lda
    cublasStatus_t cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                               const double          *alpha, const double          *x, int incx,
                               const double          *y, int incy, double          *A, int lda
    cublasStatus_t cublasCsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                               const cuComplex       *alpha, const cuComplex       *x, int incx,
                               const cuComplex       *y, int incy, cuComplex       *A, int lda
    cublasStatus_t cublasZsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                               const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the symmetric rank-2 update

\\(A = \alpha\left( {\textbf{x}\textbf{y}^{T} + \textbf{y}\textbf{x}^{T}} \right) + A\\)

where \\(A\\) is a \\(n \times n\\) symmetric matrix stored in column-major format, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) is a scalar.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | input | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`A` | device | in/out | <_type_ > array of dimensions `lda x n`, with `lda >= max(1,n)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0` or `incy == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `alpha` is NULL, or
  * if `lda < max(1, n)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssyr2()](http://www.netlib.org/lapack/explore-html/db/d99/ssyr2_8f_source.html), [dsyr2()](http://www.netlib.org/lapack/explore-html/de/d41/dsyr2_8f_source.html)

###  2.6.11. cublas<t>tbmv() 
    
    
    cublasStatus_t cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, int k, const float           *A, int lda,
                               float           *x, int incx)
    cublasStatus_t cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, int k, const double          *A, int lda,
                               double          *x, int incx)
    cublasStatus_t cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, int k, const cuComplex       *A, int lda,
                               cuComplex       *x, int incx)
    cublasStatus_t cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, int k, const cuDoubleComplex *A, int lda,
                               cuDoubleComplex *x, int incx)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the triangular banded matrix-vector multiplication

\\(\textbf{x} = \text{op}(A)\textbf{x}\\)

where \\(A\\) is a triangular banded matrix, and \\(\mathbf{x}\\) is a vector. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

If `uplo == CUBLAS_FILL_MODE_LOWER` then the triangular banded matrix \\(A\\) is stored column by column, with the main diagonal of the matrix stored in row `1`, the first subdiagonal in row `2` (starting at first position), the second subdiagonal in row `3` (starting at first position), etc. So that in general, the element \\(A(i,j)\\) is stored in the memory location `A(1+i-j,j)` for \\(j = 1,\ldots,n\\) and \\(i \in \lbrack j,\min(m,j + k)\rbrack\\) . Also, the elements in the array `A` that do not conceptually correspond to the elements in the banded matrix (the bottom right \\(k \times k\\) triangle) are not referenced.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the triangular banded matrix \\(A\\) is stored column by column, with the main diagonal of the matrix stored in row `k + 1`, the first superdiagonal in row `k` (starting at second position), the second superdiagonal in row `k-1` (starting at third position), etc. So that in general, the element \\(A(i,j)\\) is stored in the memory location `A(1+k+i-j,j)` for \\(j = 1,\ldots,n\\) and \\(i \in \lbrack\max(1,j - k,j)\rbrack\\) . Also, the elements in the array `A` that do not conceptually correspond to the elements in the banded matrix (the top left \\(k \times k\\) triangle) are not referenced.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | Indicates if the elements on the main diagonal of matrix `A` are unity and should not be accessed.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`k` |  | input | Number of sub- and super-diagonals of matrix .  
`A` | device | input | <_type_ > array of dimension `lda x n`, with `lda >= k + 1`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `incx == 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `diag` is not one of `CUBLAS_DIAG_UNIT` and `CUBLAS_DIAG_NON_UNIT`, or
  * if `lda < (1 + k)`

  
`CUBLAS_STATUS_ALLOC_FAILED` | The allocation of internal scratch memory failed  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[stbmv()](http://www.netlib.org/blas/stbmv.f), [dtbmv()](http://www.netlib.org/blas/dtbmv.f), [ctbmv()](http://www.netlib.org/blas/ctbmv.f), [ztbmv()](http://www.netlib.org/blas/ztbmv.f)

###  2.6.12. cublas<t>tbsv() 
    
    
    cublasStatus_t cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, int k, const float           *A, int lda,
                               float           *x, int incx)
    cublasStatus_t cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, int k, const double          *A, int lda,
                               double          *x, int incx)
    cublasStatus_t cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, int k, const cuComplex       *A, int lda,
                               cuComplex       *x, int incx)
    cublasStatus_t cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, int k, const cuDoubleComplex *A, int lda,
                               cuDoubleComplex *x, int incx)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function solves the triangular banded linear system with a single right-hand-side

\\(\text{op}(A)\textbf{x} = \textbf{b}\\)

where \\(A\\) is a triangular banded matrix, and \\(\mathbf{x}\\) and \\(\mathbf{b}\\) are vectors. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

The solution \\(\mathbf{x}\\) overwrites the right-hand-sides \\(\mathbf{b}\\) on exit.

No test for singularity or near-singularity is included in this function.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the triangular banded matrix \\(A\\) is stored column by column, with the main diagonal of the matrix stored in row `1`, the first subdiagonal in row `2` (starting at first position), the second subdiagonal in row `3` (starting at first position), etc. So that in general, the element \\(A(i,j)\\) is stored in the memory location `A(1+i-j,j)` for \\(j = 1,\ldots,n\\) and \\(i \in \lbrack j,\min(m,j + k)\rbrack\\) . Also, the elements in the array `A` that do not conceptually correspond to the elements in the banded matrix (the bottom right \\(k \times k\\) triangle) are not referenced.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the triangular banded matrix \\(A\\) is stored column by column, with the main diagonal of the matrix stored in row `k + 1`, the first superdiagonal in row `k` (starting at second position), the second superdiagonal in row `k-1` (starting at third position), etc. So that in general, the element \\(A(i,j)\\) is stored in the memory location `A(1+k+i-j,j)` for \\(j = 1,\ldots,n\\) and \\(i \in \lbrack\max(1,j - k,j)\rbrack\\) . Also, the elements in the array `A` that do not conceptually correspond to the elements in the banded matrix (the top left \\(k \times k\\) triangle) are not referenced.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | Indicates if the elements on the main diagonal of matrix `A` are unity and should not be accessed.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`k` |  | input | Number of sub- and super-diagonals of matrix `A`.  
`A` | device | input | <_type_ > array of dimension `lda x n`, with `lda >= k+1`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `incx == 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `diag` is not one of `CUBLAS_DIAG_UNIT` and `CUBLAS_DIAG_NON_UNIT`, or
  * if `lda < (1 + k)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[stbsv()](http://www.netlib.org/blas/stbsv.f), [dtbsv()](http://www.netlib.org/blas/dtbsv.f), [ctbsv()](http://www.netlib.org/blas/ctbsv.f), [ztbsv()](http://www.netlib.org/blas/ztbsv.f)

###  2.6.13. cublas<t>tpmv() 
    
    
    cublasStatus_t cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const float           *AP,
                               float           *x, int incx)
    cublasStatus_t cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const double          *AP,
                               double          *x, int incx)
    cublasStatus_t cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuComplex       *AP,
                               cuComplex       *x, int incx)
    cublasStatus_t cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuDoubleComplex *AP,
                               cuDoubleComplex *x, int incx)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the triangular packed matrix-vector multiplication

\\(\textbf{x} = \text{op}(A)\textbf{x}\\)

where \\(A\\) is a triangular matrix stored in packed format, and \\(\mathbf{x}\\) is a vector. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements in the lower triangular part of the triangular matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+((2*n-j+1)*j)/2]` for \\(j = 1,\ldots,n\\) and \\(i \geq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements in the upper triangular part of the triangular matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+(j*(j+1))/2]` for \\(A(i,j)\\) and \\(i \leq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | Indicates if the elements on the main diagonal of matrix `A` are unity and should not be accessed.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`AP` | device | input | <_type_ > array with \\(A\\) stored in packed format.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `diag` is not one of `CUBLAS_DIAG_UNIT` and `CUBLAS_DIAG_NON_UNIT`

  
`CUBLAS_STATUS_ALLOC_FAILED` | The allocation of internal scratch memory failed  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[stpmv()](http://www.netlib.org/blas/stpmv.f), [dtpmv()](http://www.netlib.org/blas/dtpmv.f), [ctpmv()](http://www.netlib.org/blas/ctpmv.f), [ztpmv()](http://www.netlib.org/blas/ztpmv.f)

###  2.6.14. cublas<t>tpsv() 
    
    
    cublasStatus_t cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const float           *AP,
                               float           *x, int incx)
    cublasStatus_t cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const double          *AP,
                               double          *x, int incx)
    cublasStatus_t cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuComplex       *AP,
                               cuComplex       *x, int incx)
    cublasStatus_t cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuDoubleComplex *AP,
                               cuDoubleComplex *x, int incx)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function solves the packed triangular linear system with a single right-hand-side

\\(\text{op}(A)\textbf{x} = \textbf{b}\\)

where \\(A\\) is a triangular matrix stored in packed format, and \\(\mathbf{x}\\) and \\(\mathbf{b}\\) are vectors. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

The solution \\(\mathbf{x}\\) overwrites the right-hand-sides \\(\mathbf{b}\\) on exit.

No test for singularity or near-singularity is included in this function.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements in the lower triangular part of the triangular matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+((2*n-j+1)*j)/2]` for \\(j = 1,\ldots,n\\) and \\(i \geq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements in the upper triangular part of the triangular matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+(j*(j+1))/2]` for \\(j = 1,\ldots,n\\) and \\(i \leq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | Indicates if the elements on the main diagonal of matrix are unity and should not be accessed.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`AP` | device | input | <_type_ > array with \\(A\\) stored in packed format.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `diag` is not one of `CUBLAS_DIAG_UNIT` and `CUBLAS_DIAG_NON_UNIT`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[stpsv()](http://www.netlib.org/blas/stpsv.f), [dtpsv()](http://www.netlib.org/blas/dtpsv.f), [ctpsv()](http://www.netlib.org/blas/ctpsv.f), [ztpsv()](http://www.netlib.org/blas/ztpsv.f)

###  2.6.15. cublas<t>trmv() 
    
    
    cublasStatus_t cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const float           *A, int lda,
                               float           *x, int incx)
    cublasStatus_t cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const double          *A, int lda,
                               double          *x, int incx)
    cublasStatus_t cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuComplex       *A, int lda,
                               cuComplex       *x, int incx)
    cublasStatus_t cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuDoubleComplex *A, int lda,
                               cuDoubleComplex *x, int incx)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the triangular matrix-vector multiplication

\\(\textbf{x} = \text{op}(A)\textbf{x}\\)

where \\(A\\) is a triangular matrix stored in lower or upper mode with or without the main diagonal, and \\(\mathbf{x}\\) is a vector. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | Indicates if the elements on the main diagonal of matrix `A` are unity and should not be accessed.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`A` | device | input | <_type_ > array of dimensions `lda x n` , with `lda >= max(1, n)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `diag` is not one of `CUBLAS_DIAG_UNIT` and `CUBLAS_DIAG_NON_UNIT`, or
  * if `lda < max(1, n)`

  
`CUBLAS_STATUS_ALLOC_FAILED` | The allocation of internal scratch memory failed  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[strmv()](http://www.netlib.org/blas/strmv.f), [dtrmv()](http://www.netlib.org/blas/dtrmv.f), [ctrmv()](http://www.netlib.org/blas/ctrmv.f), [ztrmv()](http://www.netlib.org/blas/ztrmv.f)

###  2.6.16. cublas<t>trsv() 
    
    
    cublasStatus_t cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const float           *A, int lda,
                               float           *x, int incx)
    cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const double          *A, int lda,
                               double          *x, int incx)
    cublasStatus_t cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuComplex       *A, int lda,
                               cuComplex       *x, int incx)
    cublasStatus_t cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int n, const cuDoubleComplex *A, int lda,
                               cuDoubleComplex *x, int incx)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function solves the triangular linear system with a single right-hand-side

\\(\text{op}(A)\textbf{x} = \textbf{b}\\)

where \\(A\\) is a triangular matrix stored in lower or upper mode with or without the main diagonal, and \\(\mathbf{x}\\) and \\(\mathbf{b}\\) are vectors. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

The solution \\(\mathbf{x}\\) overwrites the right-hand-sides \\(\mathbf{b}\\) on exit.

No test for singularity or near-singularity is included in this function.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | Indicates if the elements on the main diagonal of matrix `A` are unity and should not be accessed.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`A` | device | input | <_type_ > array of dimension `lda x n`, with `lda >= max(1, n)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `diag` is not one of `CUBLAS_DIAG_UNIT` and `CUBLAS_DIAG_NON_UNIT`, or
  * if `lda < max(1, n)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[strsv()](http://www.netlib.org/blas/strsv.f), [dtrsv()](http://www.netlib.org/blas/dtrsv.f), [ctrsv()](http://www.netlib.org/blas/ctrsv.f), [ztrsv()](http://www.netlib.org/blas/ztrsv.f)

###  2.6.17. cublas<t>hemv() 
    
    
    cublasStatus_t cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *x, int incx,
                               const cuComplex       *beta,
                               cuComplex       *y, int incy)
    cublasStatus_t cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the Hermitian matrix-vector multiplication

\\(\textbf{y} = \alpha A\textbf{x} + \beta\textbf{y}\\)

where \\(A\\) is a \\(n \times n\\) Hermitian matrix stored in lower or upper mode, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) and \\(\beta\\) are scalars.

This function has an alternate faster implementation using atomics that can be enabled with

Please see the section on the for more details about the usage of atomics

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x n`, with `lda >= max(1, n)`. The imaginary parts of the diagonal elements are assumed to be zero.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `y` does not have to be a valid input.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0` or `incy == 0`, or
  * if `uplo` != `CUBLAS_FILL_MODE_LOWER` and `uplo != CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < n`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[chemv()](http://www.netlib.org/blas/chemv.f), [zhemv()](http://www.netlib.org/blas/zhemv.f)

###  2.6.18. cublas<t>hbmv() 
    
    
    cublasStatus_t cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, int k, const cuComplex       *alpha,
                              const cuComplex       *A, int lda,
                              const cuComplex       *x, int incx,
                              const cuComplex       *beta,
                              cuComplex       *y, int incy)
    cublasStatus_t cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, int k, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *beta,
                              cuDoubleComplex *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the Hermitian banded matrix-vector multiplication

\\(\textbf{y} = \alpha A\textbf{x} + \beta\textbf{y}\\)

where \\(A\\) is a \\(n \times n\\) Hermitian banded matrix with \\(k\\) subdiagonals and superdiagonals, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) and \\(\beta\\) are scalars.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the Hermitian banded matrix \\(A\\) is stored column by column, with the main diagonal of the matrix stored in row `1`, the first subdiagonal in row `2` (starting at first position), the second subdiagonal in row `3` (starting at first position), etc. So that in general, the element \\(A(i,j)\\) is stored in the memory location `A(1+i-j,j)` for \\(j = 1,\ldots,n\\) and \\(i \in \lbrack j,\min(m,j + k)\rbrack\\) . Also, the elements in the array `A` that do not conceptually correspond to the elements in the banded matrix (the bottom right \\(k \times k\\) triangle) are not referenced.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the Hermitian banded matrix \\(A\\) is stored column by column, with the main diagonal of the matrix stored in row `k + 1`, the first superdiagonal in row `k` (starting at second position), the second superdiagonal in row `k-1` (starting at third position), etc. So that in general, the element \\(A(i,j)\\) is stored in the memory location `A(1+k+i-j,j)` for \\(j = 1,\ldots,n\\) and \\(i \in \lbrack\max(1,j - k),j\rbrack\\) . Also, the elements in the array `A` that do not conceptually correspond to the elements in the banded matrix (the top left \\(k \times k\\) triangle) are not referenced.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`k` |  | input | Number of sub- and super-diagonals of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimensions `lda x n`, with `lda >= k + 1`. The imaginary parts of the diagonal elements are assumed to be zero.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then does not have to be a valid input.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `incx == 0` or `incy == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < (1 + k)`, or
  * if `alpha` or `beta` are NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[chbmv()](http://www.netlib.org/blas/chbmv.f), [zhbmv()](http://www.netlib.org/blas/zhbmv.f)

###  2.6.19. cublas<t>hpmv() 
    
    
    cublasStatus_t cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex       *alpha,
                               const cuComplex       *AP,
                               const cuComplex       *x, int incx,
                               const cuComplex       *beta,
                               cuComplex       *y, int incy)
    cublasStatus_t cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *AP,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *y, int incy)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the Hermitian packed matrix-vector multiplication

\\(\textbf{y} = \alpha A\textbf{x} + \beta\textbf{y}\\)

where \\(A\\) is a \\(n \times n\\) Hermitian matrix stored in packed format, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) and \\(\beta\\) are scalars.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements in the lower triangular part of the Hermitian matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+((2*n-j+1)*j)/2]` for \\(j = 1,\ldots,n\\) and \\(i \geq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements in the upper triangular part of the Hermitian matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+(j*(j+1))/2]` for \\(j = 1,\ldots,n\\) and \\(i \leq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`AP` | device | input | <_type_ > array with \\(A\\) stored in packed format. The imaginary parts of the diagonal elements are assumed to be zero.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `y` does not have to be a valid input.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0` or `incy == 0`, or
  * if `uplo` != `CUBLAS_FILL_MODE_LOWER` and `uplo != CUBLAS_FILL_MODE_UPPER`, or
  * if `alpha` or `beta` are NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[chpmv()](http://www.netlib.org/blas/chpmv.f), [zhpmv()](http://www.netlib.org/blas/zhpmv.f)

###  2.6.20. cublas<t>her() 
    
    
    cublasStatus_t cublasCher(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float  *alpha,
                              const cuComplex       *x, int incx,
                              cuComplex       *A, int lda)
    cublasStatus_t cublasZher(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double *alpha,
                              const cuDoubleComplex *x, int incx,
                              cuDoubleComplex *A, int lda)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the Hermitian rank-1 update

\\(A = \alpha\textbf{x}\textbf{x}^{H} + A\\)

where \\(A\\) is a \\(n \times n\\) Hermitian matrix stored in column-major format, \\(\mathbf{x}\\) is a vector, and \\(\alpha\\) is a scalar.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`A` | device | in/out | <_type_ > array of dimensions `lda x n`, with `lda >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)`, or
  * if `alpha` is NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[cher()](http://www.netlib.org/blas/cher.f), [zher()](http://www.netlib.org/blas/zher.f)

###  2.6.21. cublas<t>her2() 
    
    
    cublasStatus_t cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex       *alpha,
                               const cuComplex       *x, int incx,
                               const cuComplex       *y, int incy,
                               cuComplex       *A, int lda)
    cublasStatus_t cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *A, int lda)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the Hermitian rank-2 update

\\(A = \alpha\textbf{x}\textbf{y}^{H} + \overset{ˉ}{\alpha}\textbf{y}\textbf{x}^{H} + A\\)

where \\(A\\) is a \\(n \times n\\) Hermitian matrix stored in column-major format, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) is a scalar.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | input | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`A` | device | in/out | <_type_ > array of dimension `lda x n` with `lda >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)`, or
  * if `alpha` is NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[cher2()](http://www.netlib.org/blas/cher2.f), [zher2()](http://www.netlib.org/blas/zher2.f)

###  2.6.22. cublas<t>hpr() 
    
    
    cublasStatus_t cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float *alpha,
                              const cuComplex       *x, int incx,
                              cuComplex       *AP)
    cublasStatus_t cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double *alpha,
                              const cuDoubleComplex *x, int incx,
                              cuDoubleComplex *AP)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the packed Hermitian rank-1 update

\\(A = \alpha\textbf{x}\textbf{x}^{H} + A\\)

where \\(A\\) is a \\(n \times n\\) Hermitian matrix stored in packed format, \\(\mathbf{x}\\) is a vector, and \\(\alpha\\) is a scalar.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements in the lower triangular part of the Hermitian matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+((2*n-j+1)*j)/2]` for \\(j = 1,\ldots,n\\) and \\(i \geq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements in the upper triangular part of the Hermitian matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+(j*(j+1))/2]` for \\(j = 1,\ldots,n\\) and \\(i \leq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`AP` | device | in/out | <_type_ > array with \\(A\\) stored in packed format. The imaginary parts of the diagonal elements are assumed and set to zero.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `alpha` is NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[chpr()](http://www.netlib.org/blas/chpr.f), [zhpr()](http://www.netlib.org/blas/zhpr.f)

###  2.6.23. cublas<t>hpr2() 
    
    
    cublasStatus_t cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuComplex       *alpha,
                               const cuComplex       *x, int incx,
                               const cuComplex       *y, int incy,
                               cuComplex       *AP)
    cublasStatus_t cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo,
                               int n, const cuDoubleComplex *alpha,
                               const cuDoubleComplex *x, int incx,
                               const cuDoubleComplex *y, int incy,
                               cuDoubleComplex *AP)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the packed Hermitian rank-2 update

\\(A = \alpha\textbf{x}\textbf{y}^{H} + \overset{ˉ}{\alpha}\textbf{y}\textbf{x}^{H} + A\\)

where \\(A\\) is a \\(n \times n\\) Hermitian matrix stored in packed format, \\(\mathbf{x}\\) and \\(\mathbf{y}\\) are vectors, and \\(\alpha\\) is a scalar.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements in the lower triangular part of the Hermitian matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+((2*n-j+1)*j)/2]` for \\(j = 1,\ldots,n\\) and \\(i \geq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements in the upper triangular part of the Hermitian matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+(j*(j+1))/2]` for \\(j = 1,\ldots,n\\) and \\(i \leq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`x` | device | input | <_type_ > vector with `n` elements.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | input | <_type_ > vector with `n` elements.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`AP` | device | in/out | <_type_ > array with \\(A\\) stored in packed format. The imaginary parts of the diagonal elements are assumed and set to zero.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `incx == 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `alpha` is NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

chpr2, zhpr2

###  2.6.24. cublas<t>gemvBatched() 
    
    
    cublasStatus_t cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans,
                                      int m, int n,
                                      const float           *alpha,
                                      const float           *const Aarray[], int lda,
                                      const float           *const xarray[], int incx,
                                      const float           *beta,
                                      float           *const yarray[], int incy,
                                      int batchCount)
    cublasStatus_t cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans,
                                      int m, int n,
                                      const double          *alpha,
                                      const double          *const Aarray[], int lda,
                                      const double          *const xarray[], int incx,
                                      const double          *beta,
                                      double          *const yarray[], int incy,
                                      int batchCount)
    cublasStatus_t cublasCgemvBatched(cublasHandle_t handle, cublasOperation_t trans,
                                      int m, int n,
                                      const cuComplex       *alpha,
                                      const cuComplex       *const Aarray[], int lda,
                                      const cuComplex       *const xarray[], int incx,
                                      const cuComplex       *beta,
                                      cuComplex       *const yarray[], int incy,
                                      int batchCount)
    cublasStatus_t cublasZgemvBatched(cublasHandle_t handle, cublasOperation_t trans,
                                      int m, int n,
                                      const cuDoubleComplex *alpha,
                                      const cuDoubleComplex *const Aarray[], int lda,
                                      const cuDoubleComplex *const xarray[], int incx,
                                      const cuDoubleComplex *beta,
                                      cuDoubleComplex *const yarray[], int incy,
                                      int batchCount)
    
    #if defined(__cplusplus)
    cublasStatus_t cublasHSHgemvBatched(cublasHandle_t handle, cublasOperation_t trans,
                                        int m, int n,
                                        const float           *alpha,
                                        const __half          *const Aarray[], int lda,
                                        const __half          *const xarray[], int incx,
                                        const float           *beta,
                                        __half                *const yarray[], int incy,
                                        int batchCount)
    cublasStatus_t cublasHSSgemvBatched(cublasHandle_t handle, cublasOperation_t trans,
                                        int m, int n,
                                        const float           *alpha,
                                        const __half          *const Aarray[], int lda,
                                        const __half          *const xarray[], int incx,
                                        const float           *beta,
                                        float                 *const yarray[], int incy,
                                        int batchCount)
    cublasStatus_t cublasTSTgemvBatched(cublasHandle_t handle, cublasOperation_t trans,
                                        int m, int n,
                                        const float           *alpha,
                                        const __nv_bfloat16   *const Aarray[], int lda,
                                        const __nv_bfloat16   *const xarray[], int incx,
                                        const float           *beta,
                                        __nv_bfloat16         *const yarray[], int incy,
                                        int batchCount)
    cublasStatus_t cublasTSSgemvBatched(cublasHandle_t handle, cublasOperation_t trans,
                                        int m, int n,
                                        const float           *alpha,
                                        const __nv_bfloat16   *const Aarray[], int lda,
                                        const __nv_bfloat16   *const xarray[], int incx,
                                        const float           *beta,
                                        float                 *const yarray[], int incy,
                                        int batchCount)
    #endif
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-vector multiplication of a batch of matrices and vectors. The batch is considered to be “uniform”, i.e. all instances have the same dimensions (m, n), leading dimension (lda), increments (incx, incy) and transposition (trans) for their respective A matrix, x and y vectors. The address of the input matrix and vector, and the output vector of each instance of the batch are read from arrays of pointers passed to the function by the caller.

\\(\textbf{y}\lbrack i\rbrack = \alpha\text{op}(A\lbrack i\rbrack)\textbf{x}\lbrack i\rbrack + \beta\textbf{y}\lbrack i\rbrack,\text{ for i} \in \lbrack 0,batchCount - 1\rbrack\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) is an array of pointers to matrice \\(A\lbrack i\rbrack\\) stored in column-major format with dimension \\(m \times n\\) , and \\(\textbf{x}\\) and \\(\textbf{y}\\) are arrays of pointers to vectors. Also, for matrix \\(A\lbrack i\rbrack\\) ,

\\(\text{op}(A\lbrack i\rbrack) = \left\\{ \begin{matrix} {A\lbrack i\rbrack} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A\lbrack i\rbrack}^{T} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ {A\lbrack i\rbrack}^{H} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Note

\\(\textbf{y}\lbrack i\rbrack\\) vectors must not overlap, i.e. the individual gemv operations must be computable independently; otherwise, undefined behavior is expected.

On certain problem sizes, it might be advantageous to make multiple calls to [cublas<t>gemv()](#cublas-t-gemv) in different CUDA streams, rather than use this API.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`trans` |  | input | Operation op(`A[i]`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix `A[i]`.  
`n` |  | input | Number of columns of matrix `A[i]`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`Aarray` | device | input |  Array of pointers to <_type_ > array, with each array of dim. `lda x n` with `lda >= max(1, m)`. All pointers must meet certain alignment criteria. Please see below for details.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `A[i]`.  
`xarray` | device | input |  Array of pointers to <_type_ > array, with each dimension `n` if `trans == CUBLAS_OP_N` and `m` otherwise. All pointers must meet certain alignment criteria. Please see below for details.  
`incx` |  | input | Stride of each one-dimensional array x[i].  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0`, `y` does not have to be a valid input.  
`yarray` | device | in/out |  Array of pointers to <_type_ > array. It has dimensions `m` if `trans == CUBLAS_OP_N` and `n` otherwise. Vectors `y[i]` should not overlap; otherwise, undefined behavior is expected. All pointers must meet certain alignment criteria. Please see below for details.  
`incy` |  | input | Stride of each one-dimensional array y[i].  
`batchCount` |  | input | Number of pointers contained in Aarray, xarray and yarray.  
  
If math mode enables fast math modes when using [cublasSgemvBatched()](#cublas-t-gemvbatched), pointers (not the pointer arrays) placed in the GPU memory must be properly aligned to avoid misaligned memory access errors. Ideally all pointers are aligned to at least 16 Bytes. Otherwise it is recommended that they meet the following rule:

  * if `k % 4 == 0` then ensure `intptr_t(ptr) % 16 == 0`,


The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | `m < 0`, `n < 0`, or `batchCount < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
###  2.6.25. cublas<t>gemvStridedBatched() 
    
    
    cublasStatus_t cublasSgemvStridedBatched(cublasHandle_t handle,
                                             cublasOperation_t trans,
                                             int m, int n,
                                             const float           *alpha,
                                             const float           *A, int lda,
                                             long long int         strideA,
                                             const float           *x, int incx,
                                             long long int         stridex,
                                             const float           *beta,
                                             float                 *y, int incy,
                                             long long int         stridey,
                                             int batchCount)
    cublasStatus_t cublasDgemvStridedBatched(cublasHandle_t handle,
                                             cublasOperation_t trans,
                                             int m, int n,
                                             const double          *alpha,
                                             const double          *A, int lda,
                                             long long int         strideA,
                                             const double          *x, int incx,
                                             long long int         stridex,
                                             const double          *beta,
                                             double                *y, int incy,
                                             long long int         stridey,
                                             int batchCount)
    cublasStatus_t cublasCgemvStridedBatched(cublasHandle_t handle,
                                             cublasOperation_t trans,
                                             int m, int n,
                                             const cuComplex       *alpha,
                                             const cuComplex       *A, int lda,
                                             long long int         strideA,
                                             const cuComplex       *x, int incx,
                                             long long int         stridex,
                                             const cuComplex       *beta,
                                             cuComplex             *y, int incy,
                                             long long int         stridey,
                                             int batchCount)
    cublasStatus_t cublasZgemvStridedBatched(cublasHandle_t handle,
                                             cublasOperation_t trans,
                                             int m, int n,
                                             const cuDoubleComplex *alpha,
                                             const cuDoubleComplex *A, int lda,
                                             long long int         strideA,
                                             const cuDoubleComplex *x, int incx,
                                             long long int         stridex,
                                             const cuDoubleComplex *beta,
                                             cuDoubleComplex       *y, int incy,
                                             long long int         stridey,
                                             int batchCount)
    cublasStatus_t cublasHSHgemvStridedBatched(cublasHandle_t handle,
                                               cublasOperation_t trans,
                                               int m, int n,
                                               const float           *alpha,
                                               const __half          *A, int lda,
                                               long long int         strideA,
                                               const __half          *x, int incx,
                                               long long int         stridex,
                                               const float           *beta,
                                               __half                *y, int incy,
                                               long long int         stridey,
                                               int batchCount)
    cublasStatus_t cublasHSSgemvStridedBatched(cublasHandle_t handle,
                                               cublasOperation_t trans,
                                               int m, int n,
                                               const float           *alpha,
                                               const __half          *A, int lda,
                                               long long int         strideA,
                                               const __half          *x, int incx,
                                               long long int         stridex,
                                               const float           *beta,
                                               float                 *y, int incy,
                                               long long int         stridey,
                                               int batchCount)
    cublasStatus_t cublasTSTgemvStridedBatched(cublasHandle_t handle,
                                               cublasOperation_t trans,
                                               int m, int n,
                                               const float           *alpha,
                                               const __nv_bfloat16   *A, int lda,
                                               long long int         strideA,
                                               const __nv_bfloat16   *x, int incx,
                                               long long int         stridex,
                                               const float           *beta,
                                               __nv_bfloat16         *y, int incy,
                                               long long int         stridey,
                                               int batchCount)
    cublasStatus_t cublasTSSgemvStridedBatched(cublasHandle_t handle,
                                               cublasOperation_t trans,
                                               int m, int n,
                                               const float           *alpha,
                                               const __nv_bfloat16   *A, int lda,
                                               long long int         strideA,
                                               const __nv_bfloat16   *x, int incx,
                                               long long int         stridex,
                                               const float           *beta,
                                               float                 *y, int incy,
                                               long long int         stridey,
                                               int batchCount)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-vector multiplication of a batch of matrices and vectors. The batch is considered to be “uniform”, i.e. all instances have the same dimensions (m, n), leading dimension (lda), increments (incx, incy) and transposition (trans) for their respective A matrix, x and y vectors. Input matrix A and vector x, and output vector y for each instance of the batch are located at fixed offsets in number of elements from their locations in the previous instance. Pointers to A matrix, x and y vectors for the first instance are passed to the function by the user along with offsets in number of elements - strideA, stridex and stridey that determine the locations of input matrices and vectors, and output vectors in future instances.

\\(\textbf{y} + i*{stridey} = \alpha\text{op}(A + i*{strideA})(\textbf{x} + i*{stridex}) + \beta(\textbf{y} + i*{stridey}),\text{ for i } \in \lbrack 0,batchCount - 1\rbrack\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) is an array of pointers to matrix stored in column-major format with dimension \\(A\lbrack i\rbrack\\) \\(m \times n\\) , and \\(\textbf{x}\\) and \\(\textbf{y}\\) are arrays of pointers to vectors. Also, for matrix \\(A\lbrack i\rbrack\\)

\\(\text{op}(A\lbrack i\rbrack) = \left\\{ \begin{matrix} {A\lbrack i\rbrack} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A\lbrack i\rbrack}^{T} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ {A\lbrack i\rbrack}^{H} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Note

\\(\textbf{y}\lbrack i\rbrack\\) matrices must not overlap, i.e. the individual gemv operations must be computable independently; otherwise, undefined behavior is expected.

On certain problem sizes, it might be advantageous to make multiple calls to [cublas<t>gemv()](#cublas-t-gemv) in different CUDA streams, rather than use this API.

Note

In the table below, we use `A[i], x[i], y[i]` as notation for A matrix, and x and y vectors in the ith instance of the batch, implicitly assuming they are respectively offsets in number of elements `strideA, stridex, stridey` away from `A[i-1], x[i-1], y[i-1]`. The unit for the offset is number of elements and must not be zero .

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`trans` |  | input | Operation op(`A[i]`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix `A[i]`.  
`n` |  | input | Number of columns of matrix `A[i]`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ >* pointer to the A matrix corresponding to the first instance of the batch, with dimensions `lda x n` with `lda >= max(1, m)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `A[i]`.  
`strideA` |  | input | Value of type long long int that gives the offset in number of elements between `A[i]` and `A[i+1]`  
`x` | device | input | <_type_ >* pointer to the x vector corresponding to the first instance of the batch, with each dimension `n` if `trans == CUBLAS_OP_N` and `m` otherwise.  
`incx` |  | input | Stride of each one-dimensional array `x[i]`.  
`stridex` |  | input | Value of type long long int that gives the offset in number of elements between `x[i]` and `x[i+1]`  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0`, `y` does not have to be a valid input.  
`y` | device | in/out | <_type_ >* pointer to the y vector corresponding to the first instance of the batch, with each dimension `m` if `trans == CUBLAS_OP_N` and `n` otherwise. Vectors `y[i]` should not overlap; otherwise, undefined behavior is expected.  
`incy` |  | input | Stride of each one-dimensional array `y[i]`.  
`stridey` |  | input | Value of type long long int that gives the offset in number of elements between `y[i]` and `y[i+1]`  
`batchCount` |  | input | Number of GEMVs to perform in the batch.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | `m < 0`, `n < 0`, or `batchCount < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU


##  2.7. cuBLAS Level-3 Function Reference   
  
In this chapter we describe the Level-3 Basic Linear Algebra Subprograms (BLAS3) functions that perform matrix-matrix operations.

###  2.7.1. cublas<t>gemm() 
    
    
    cublasStatus_t cublasSgemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *B, int ldb,
                               const float           *beta,
                               float           *C, int ldc)
    cublasStatus_t cublasDgemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *B, int ldb,
                               const double          *beta,
                               double          *C, int ldc)
    cublasStatus_t cublasCgemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *B, int ldb,
                               const cuComplex       *beta,
                               cuComplex       *C, int ldc)
    cublasStatus_t cublasZgemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *C, int ldc)
    cublasStatus_t cublasHgemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const __half *alpha,
                               const __half *A, int lda,
                               const __half *B, int ldb,
                               const __half *beta,
                               __half *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-matrix multiplication

\\(C = \alpha\text{op}(A)\text{op}(B) + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are matrices stored in column-major format with dimensions \\(\text{op}(A)\\) \\(m \times k\\) , \\(\text{op}(B)\\) \\(k \times n\\) and \\(C\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B)\\) is defined similarly for matrix \\(B\\) .

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`transa` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`transb` |  | input | Operation op(`B`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix op(`A`) and `C`.  
`n` |  | input | Number of columns of matrix op(`B`) and `C`.  
`k` |  | input | Number of columns of op(`A`) and rows of op(`B`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimensions `lda x k` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store the matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, k)` if `transb == CUBLAS_OP_N` and `ldb x k` with `ldb >= max(1,n)` otherwise.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0`, `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | Leading dimension of a two-dimensional array used to store the matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `k < 0`, or
  * if `transa` and `transb` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda < max(1, m)` when `transa == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, k)` when `transb == CUBLAS_OP_N` and `ldb < max(1, n)` otherwise, or
  * if `ldc < max(1, m)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_ARCH_MISMATCH` | In the case of [cublasHgemm()](#cublas-t-gemm) the device does not support math in half precision.  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sgemm()](http://www.netlib.org/blas/sgemm.f), [dgemm()](http://www.netlib.org/blas/dgemm.f), [cgemm()](http://www.netlib.org/blas/cgemm.f), [zgemm()](http://www.netlib.org/blas/zgemm.f)

###  2.7.2. cublas<t>gemm3m() 
    
    
    cublasStatus_t cublasCgemm3m(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *B, int ldb,
                               const cuComplex       *beta,
                               cuComplex       *C, int ldc)
    cublasStatus_t cublasZgemm3m(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the complex matrix-matrix multiplication, using Gauss complexity reduction algorithm. This can lead to an increase in performance up to 25%

\\(C = \alpha\text{op}(A)\text{op}(B) + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are matrices stored in column-major format with dimensions \\(\text{op}(A)\\) \\(m \times k\\) , \\(\text{op}(B)\\) \\(k \times n\\) and \\(C\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B)\\) is defined similarly for matrix \\(B\\) .

Note

These 2 routines are only supported on GPUs with architecture capabilities equal to or greater than 5.0

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`transa` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`transb` |  | input | Operation op(`B`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix op(`A`) and `C`.  
`n` |  | input | Number of columns of matrix op(`B`) and `C`.  
`k` |  | input | Number of columns of op(`A`) and rows of op(`B`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimensions `lda x k` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store the matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, k)` if `transb == CUBLAS_OP_N` and `ldb x k` with `ldb >= max(1,n)` otherwise.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0`, `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | Leading dimension of a two-dimensional array used to store the matrix `C`.  
  
The possible error values returned by this function and their meanings are listed in the following table:

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `k < 0`, or
  * if `transa` and `transb` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda < max(1, m)` when `transa == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, k)` when `transb == CUBLAS_OP_N` and `ldb < max(1, n)` otherwise, or
  * if `ldc < max(1, m)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_ARCH_MISMATCH` | The device has a compute capabilites lower than 5.0.  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU.  
  
For references please refer to NETLIB documentation:

[cgemm()](http://www.netlib.org/blas/cgemm.f), [zgemm()](http://www.netlib.org/blas/zgemm.f)

###  2.7.3. cublas<t>gemmBatched() 
    
    
    cublasStatus_t cublasHgemmBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const __half          *alpha,
                                      const __half          *const Aarray[], int lda,
                                      const __half          *const Barray[], int ldb,
                                      const __half          *beta,
                                      __half          *const Carray[], int ldc,
                                      int batchCount)
    cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const float           *alpha,
                                      const float           *const Aarray[], int lda,
                                      const float           *const Barray[], int ldb,
                                      const float           *beta,
                                      float           *const Carray[], int ldc,
                                      int batchCount)
    cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const double          *alpha,
                                      const double          *const Aarray[], int lda,
                                      const double          *const Barray[], int ldb,
                                      const double          *beta,
                                      double          *const Carray[], int ldc,
                                      int batchCount)
    cublasStatus_t cublasCgemmBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const cuComplex       *alpha,
                                      const cuComplex       *const Aarray[], int lda,
                                      const cuComplex       *const Barray[], int ldb,
                                      const cuComplex       *beta,
                                      cuComplex       *const Carray[], int ldc,
                                      int batchCount)
    cublasStatus_t cublasZgemmBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const cuDoubleComplex *alpha,
                                      const cuDoubleComplex *const Aarray[], int lda,
                                      const cuDoubleComplex *const Barray[], int ldb,
                                      const cuDoubleComplex *beta,
                                      cuDoubleComplex *const Carray[], int ldc,
                                      int batchCount)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-matrix multiplication of a batch of matrices. The batch is considered to be “uniform”, i.e. all instances have the same dimensions (m, n, k), leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. The address of the input matrices and the output matrix of each instance of the batch are read from arrays of pointers passed to the function by the caller.

\\(C\lbrack i\rbrack = \alpha\text{op}(A\lbrack i\rbrack)\text{op}(B\lbrack i\rbrack) + \beta C\lbrack i\rbrack,\text{ for i } \in \lbrack 0,batchCount - 1\rbrack\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are arrays of pointers to matrices stored in column-major format with dimensions \\(\text{op}(A\lbrack i\rbrack)\\) \\(m \times k\\) , \\(\text{op}(B\lbrack i\rbrack)\\) \\(k \times n\\) and \\(C\lbrack i\rbrack\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B\lbrack i\rbrack)\\) is defined similarly for matrix \\(B\lbrack i\rbrack\\) .

Note

\\(C\lbrack i\rbrack\\) matrices must not overlap, that is, the individual gemm operations must be computable independently; otherwise, undefined behavior is expected.

On certain problem sizes, it might be advantageous to make multiple calls to [cublas<t>gemm()](#id8) in different CUDA streams, rather than use this API.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`transa` |  | input | Operation op(`A[i]`) that is non- or (conj.) transpose.  
`transb` |  | input | Operation op(`B[i]`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix op(`A[i]`) and `C[i]`.  
`n` |  | input | Number of columns of op(`B[i]`) and `C[i]`.  
`k` |  | input | Number of columns of op(`A[i]`) and rows of op(`B[i]`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`Aarray` | device | input |  Array of pointers to <_type_ > array, with each array of dim. `lda x k` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, k)` otherwise. All pointers must meet certain alignment criteria. Please see below for details.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `A[i]`.  
`Barray` | device | input |  Array of pointers to <_type_ > array, with each array of dim. `ldb x n` with `ldb >= max(1, k)` if `transb == CUBLAS_OP_N` and `ldb x k` with `ldb >= max(1,n)` otherwise. All pointers must meet certain alignment criteria. Please see below for details.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store each matrix `B[i]`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0`, `C` does not have to be a valid input.  
`Carray` | device | in/out |  Array of pointers to <_type_ > array. It has dimensions `ldc x n` with `ldc >= max(1, m)`. Matrices `C[i]` should not overlap; otherwise, undefined behavior is expected. All pointers must meet certain alignment criteria. Please see below for details.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store each matrix `C[i]`.  
`batchCount` |  | input | Number of pointers contained in Aarray, Barray and Carray.  
  
If math mode enables fast math modes when using [cublasSgemmBatched()](#cublas-t-gemmbatched), pointers (not the pointer arrays) placed in the GPU memory must be properly aligned to avoid misaligned memory access errors. Ideally all pointers are aligned to at least 16 Bytes. Otherwise it is recommended that they meet the following rule:

  * if `k % 4 == 0` then ensure `intptr_t(ptr) % 16 == 0`,


The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `k < 0`, or
  * if `transa` and `transb` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda < max(1, m)` when `transa == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, k)` when `transb == CUBLAS_OP_N` and `ldb < max(1, n)` otherwise, or
  * if `ldc < max(1, m)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_ARCH_MISMATCH` | [cublasHgemmBatched()](#cublas-t-gemmbatched) is only supported for GPU with architecture capabilities equal or greater than 5.3  
  
###  2.7.4. cublas<t>gemmStridedBatched() 
    
    
    cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const __half           *alpha,
                                      const __half           *A, int lda,
                                      long long int          strideA,
                                      const __half           *B, int ldb,
                                      long long int          strideB,
                                      const __half           *beta,
                                      __half                 *C, int ldc,
                                      long long int          strideC,
                                      int batchCount)
    cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const float           *alpha,
                                      const float           *A, int lda,
                                      long long int          strideA,
                                      const float           *B, int ldb,
                                      long long int          strideB,
                                      const float           *beta,
                                      float                 *C, int ldc,
                                      long long int          strideC,
                                      int batchCount)
    cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const double          *alpha,
                                      const double          *A, int lda,
                                      long long int          strideA,
                                      const double          *B, int ldb,
                                      long long int          strideB,
                                      const double          *beta,
                                      double                *C, int ldc,
                                      long long int          strideC,
                                      int batchCount)
    cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const cuComplex       *alpha,
                                      const cuComplex       *A, int lda,
                                      long long int          strideA,
                                      const cuComplex       *B, int ldb,
                                      long long int          strideB,
                                      const cuComplex       *beta,
                                      cuComplex             *C, int ldc,
                                      long long int          strideC,
                                      int batchCount)
    cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const cuComplex       *alpha,
                                      const cuComplex       *A, int lda,
                                      long long int          strideA,
                                      const cuComplex       *B, int ldb,
                                      long long int          strideB,
                                      const cuComplex       *beta,
                                      cuComplex             *C, int ldc,
                                      long long int          strideC,
                                      int batchCount)
    cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const cuDoubleComplex *alpha,
                                      const cuDoubleComplex *A, int lda,
                                      long long int          strideA,
                                      const cuDoubleComplex *B, int ldb,
                                      long long int          strideB,
                                      const cuDoubleComplex *beta,
                                      cuDoubleComplex       *C, int ldc,
                                      long long int          strideC,
                                      int batchCount)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-matrix multiplication of a batch of matrices. The batch is considered to be “uniform”, i.e. all instances have the same dimensions (m, n, k), leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. Input matrices A, B and output matrix C for each instance of the batch are located at fixed offsets in number of elements from their locations in the previous instance. Pointers to A, B and C matrices for the first instance are passed to the function by the user along with offsets in number of elements - strideA, strideB and strideC that determine the locations of input and output matrices in future instances.

\\(C + i*{strideC} = \alpha\text{op}(A + i*{strideA})\text{op}(B + i*{strideB}) + \beta(C + i*{strideC}),\text{ for i } \in \lbrack 0,batchCount - 1\rbrack\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are arrays of pointers to matrices stored in column-major format with dimensions \\(\text{op}(A\lbrack i\rbrack)\\) \\(m \times k\\) , \\(\text{op}(B\lbrack i\rbrack)\\) \\(k \times n\\) and \\(C\lbrack i\rbrack\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B\lbrack i\rbrack)\\) is defined similarly for matrix \\(B\lbrack i\rbrack\\) .

Note

\\(C\lbrack i\rbrack\\) matrices must not overlap, i.e. the individual gemm operations must be computable independently; otherwise, undefined behavior is expected.

On certain problem sizes, it might be advantageous to make multiple calls to [cublas<t>gemm()](#id8) in different CUDA streams, rather than use this API.

Note

In the table below, we use `A[i], B[i], C[i]` as notation for A, B and C matrices in the ith instance of the batch, implicitly assuming they are respectively offsets in number of elements `strideA, strideB, strideC` away from `A[i-1], B[i-1], C[i-1]`. The unit for the offset is number of elements and must not be zero .

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`transa` |  | input | Operation op(`A[i]`) that is non- or (conj.) transpose.  
`transb` |  | input | Operation op(`B[i]`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix op(`A[i]`) and `C[i]`.  
`n` |  | input | Number of columns of op(`B[i]`) and `C[i]`.  
`k` |  | input | Number of columns of op(`A[i]`) and rows of op(`B[i]`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ >* pointer to the A matrix corresponding to the first instance of the batch, with dimensions `lda x k` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `A[i]`.  
`strideA` |  | input | Value of type long long int that gives the offset in number of elements between `A[i]` and `A[i+1]`  
`B` | device | input | <_type_ >* pointer to the B matrix corresponding to the first instance of the batch, with dimensions `ldb x n` with `ldb >= max(1, k)` if `transb == CUBLAS_OP_N` and `ldb x k` with `ldb >= max(1,n)` otherwise.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store each matrix `B[i]`.  
`strideB` |  | input | Value of type long long int that gives the offset in number of elements between `B[i]` and `B[i+1]`  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0`, `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ >* pointer to the C matrix corresponding to the first instance of the batch, with dimensions `ldc x n` with `ldc >= max(1, m)`. Matrices `C[i]` should not overlap; otherwise, undefined behavior is expected.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store each matrix `C[i]`.  
`strideC` |  | input | Value of type long long int that gives the offset in number of elements between `C[i]` and `C[i+1]`  
`batchCount` |  | input | Number of GEMMs to perform in the batch.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `k < 0`, or
  * if `transa` and `transb` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda < max(1, m)` when `transa == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, k)` when `transb == CUBLAS_OP_N` and `ldb < max(1, n)` otherwise, or
  * if `ldc < max(1, m)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_ARCH_MISMATCH` | [cublasHgemmStridedBatched()](#cublas-t-gemmstridedbatched) is only supported for GPU with architecture capabilities equal or greater than 5.3  
  
###  2.7.5. cublas<t>gemmGroupedBatched() 
    
    
    cublasStatus_t cublasSgemmGroupedBatched(cublasHandle_t handle,
                                             const cublasOperation_t transa_array[],
                                             const cublasOperation_t transb_array[],
                                             const int m_array[],
                                             const int n_array[],
                                             const int k_array[],
                                             const float  alpha_array[],
                                             const float *const  Aarray[],
                                             const int lda_array[],
                                             const float *const  Barray[],
                                             const int ldb_array[],
                                             const float  beta_array[],
                                             float *const  Carray[],
                                             const int ldc_array[],
                                             int group_count,
                                             const int group_size[])
    cublasStatus_t cublasDgemmGroupedBatched(cublasHandle_t handle,
                                             const cublasOperation_t transa_array[],
                                             const cublasOperation_t transb_array[],
                                             const int m_array[],
                                             const int n_array[],
                                             const int k_array[],
                                             const double alpha_array[],
                                             const double *const Aarray[],
                                             const int lda_array[],
                                             const double *const Barray[],
                                             const int ldb_array[],
                                             const double beta_array[],
                                             double *const Carray[],
                                             const int ldc_array[],
                                             int group_count,
                                             const int group_size[])
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-matrix multiplication on groups of matrices. A given group is considered to be “uniform”, i.e. all instances have the same dimensions (m, n, k), leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. However, the dimensions, leading dimensions, transpositions, and scaling factors (alpha, beta) may vary between groups. The address of the input matrices and the output matrix of each instance of the batch are read from arrays of pointers passed to the function by the caller. This is functionally equivalent to the following:
    
    
    idx = 0;
    for i = 0:group_count - 1
        for j = 0:group_size[i] - 1
            gemm(transa_array[i], transb_array[i], m_array[i], n_array[i], k_array[i],
                 alpha_array[i], Aarray[idx], lda_array[i], Barray[idx], ldb_array[i],
                 beta_array[i], Carray[idx], ldc_array[i]);
            idx += 1;
        end
    end
    

where \\(\text{$\mathrm{alpha\\_array}$}\\) and \\(\text{$\mathrm{beta\\_array}$}\\) are arrays of scaling factors, and \\(\text{Aarray}\\), \\(\text{Barray}\\) and \\(\text{Carray}\\) are arrays of pointers to matrices stored in column-major format. For a given index, \\(\text{idx}\\), that is part of group \\(i\\), the dimensions are:

>   * \\(\text{op}(\text{Aarray}\lbrack\text{idx}\rbrack)\\): \\(\text{$\mathrm{m\\_array}$}\lbrack i\rbrack \times \text{$\mathrm{k\\_array}$}\lbrack i\rbrack\\)
> 
>   * \\(\text{op}(\text{Barray}\lbrack\text{idx}\rbrack)\\): \\(\text{$\mathrm{k\\_array}$}\lbrack i\rbrack \times \text{$\mathrm{n\\_array}$}\lbrack i\rbrack\\)
> 
>   * \\(\text{Carray}\lbrack\text{idx}\rbrack\\): \\(\text{$\mathrm{m\\_array}$}\lbrack i\rbrack \times \text{$\mathrm{n\\_array}$}\lbrack i\rbrack\\)
> 
> 


Note

This API takes arrays of two different lengths. The arrays of dimensions, leading dimensions, transpositions, and scaling factors are of length `group_count` and the arrays of matrices are of length `problem_count` where \\(\text{$\mathrm{problem\\_count}$} = \sum_{i = 0}^{\text{$\mathrm{group\\_count}$} - 1} \text{$\mathrm{group\\_size}$}\lbrack i\rbrack\\)

For matrix \\(A[\text{idx}]\\) in group \\(i\\)

\\(\text{op}(A[\text{idx}]) = \left\\{ \begin{matrix} A[\text{idx}] & {\text{if }\textsf{$\mathrm{transa\\_array}\lbrack i\rbrack$ == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A[\text{idx}]^{T} & {\text{if }\textsf{$\mathrm{transa\\_array}\lbrack i\rbrack$ == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A[\text{idx}]^{H} & {\text{if }\textsf{$\mathrm{transa\\_array}\lbrack i\rbrack$ == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B[\text{idx}])\\) is defined similarly for matrix \\(B[\text{idx}]\\) in group \\(i\\).

Note

\\(C\lbrack\text{idx}\rbrack\\) matrices must not overlap, that is, the individual gemm operations must be computable independently; otherwise, undefined behavior is expected.

On certain problem sizes, it might be advantageous to make multiple calls to [cublas<t>gemmBatched()](#id9) in different CUDA streams, rather than use this API.

Param. | Memory | In/out | Meaning | Array Length  
---|---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context. |   
`transa_array` | host | input | Operation op(`A[idx]`) that is non- or (conj.) transpose for each group. | group_count  
`transb_array` | host | input | Operation op(`B[idx]`) that is non- or (conj.) transpose for each group. | group_count  
`m_array` | host | input | Array containing the number of rows of matrix op(`A[idx]`) and `C[idx]` for each group. | group_count  
`n_array` | host | input | Array containing the number of columns of op(`B[idx]`) and `C[idx]` for each group. | group_count  
`k_array` | host | input | Array containing the number of columns of op(`A[idx]`) and rows of op(`B[idx]`) for each group. | group_count  
`alpha_array` | host | input | Array containing the <_type_ > scalar used for multiplication for each group. | group_count  
`Aarray` | device | input |  Array of pointers to <_type_ > array, with each array of dim. `lda[i] x k[i]` with `lda[i] >= max(1,m[i])` if `transa[i] == CUBLAS_OP_N` and `lda[i] x m[i]` with `lda[i] >= max(1,k[i])` otherwise. All pointers must meet certain alignment criteria. Please see below for details. | problem_count  
`lda_array` | host | input | Array containing the leading dimensions of two-dimensional arrays used to store each matrix `A[idx]` for each group. | group_count  
`Barray` | device | input |  Array of pointers to <_type_ > array, with each array of dim. `ldb[i] x n[i]` with `ldb[i] >= max(1,k[i])` if `transb[i] == CUBLAS_OP_N` and `ldb[i] x k[i]` with `ldb[i] >= max(1,n[i])` otherwise. All pointers must meet certain alignment criteria. Please see below for details. | problem_count  
`ldb_array` | host | input | Array containing the leading dimensions of two-dimensional arrays used to store each matrix `B[idx]` for each group. | group_count  
`beta_array` | host | input | Array containing the <_type_ > scalar used for multiplication for each group. | group_count  
`Carray` | device | in/out |  Array of pointers to <_type_ > array. It has dimensions `ldc[i] x n[i]` with `ldc[i] >= max(1,m[i])`. Matrices `C[idx]` should not overlap; otherwise, undefined behavior is expected. All pointers must meet certain alignment criteria. Please see below for details. | problem_count  
`ldc_array` | host | input | Array containing the leading dimensions of two-dimensional arrays used to store each matrix `C[idx]` for each group. | group_count  
`group_count` | host | input | Number of groups |   
`group_size` | host | input | Array containing the number of pointers contained in Aarray, Barray and Carray for each group. | group_count  
  
If math mode enables fast math modes when using [cublasSgemmGroupedBatched()](#cublas-t-gemmgroupedbatched), pointers (not the pointer arrays) placed in the GPU memory must be properly aligned to avoid misaligned memory access errors. Ideally all pointers are aligned to at least 16 Bytes. Otherwise it is required that they meet the following rule:

  * if `k % 4 == 0` then ensure `intptr_t(ptr) % 16 == 0`,


The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `transa_array`, `transb_array`, `m_array`, `n_array`, `k_array`, `alpha_array`, `lda_array`, `ldb_array`, `beta_array`, `ldc_array`, or `group_size` are NULL, or
  * if `group_count < 0`, or
  * if `m_array[i] < 0`, `n_array[i] < 0`, `k_array[i] < 0`, `group_size[i] < 0`, or
  * if `transa_array[i]` and `transb_array[i]` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda_array[i] < max(1, m_array[i])` if `transa_array[i] == CUBLAS_OP_N` and `lda_array[i] < max(1, k_array[i])` otherwise, or
  * if `ldb_array[i] < max(1, k_array[i])` if `transb_array[i] == CUBLAS_OP_N` and `ldb_array[i] < max(1, n_array[i])` otherwise, or
  * if `ldc_array[i] < max(1, m_array[i])`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_NOT_SUPPORTED` | The pointer mode is set to `CUBLAS_POINTER_MODE_DEVICE`  
  
###  2.7.6. cublas<t>symm() 
    
    
    cublasStatus_t cublasSsymm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               int m, int n,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *B, int ldb,
                               const float           *beta,
                               float           *C, int ldc)
    cublasStatus_t cublasDsymm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               int m, int n,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *B, int ldb,
                               const double          *beta,
                               double          *C, int ldc)
    cublasStatus_t cublasCsymm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               int m, int n,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *B, int ldb,
                               const cuComplex       *beta,
                               cuComplex       *C, int ldc)
    cublasStatus_t cublasZsymm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the symmetric matrix-matrix multiplication

\\(C = \left\\{ \begin{matrix} {\alpha AB + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {\alpha BA + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) is a symmetric matrix stored in lower or upper mode, \\(B\\) and \\(C\\) are \\(m \times n\\) matrices, and \\(\alpha\\) and \\(\beta\\) are scalars.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`side` |  | input | Indicates if matrix `A` is on the left or right of `B`.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`m` |  | input | Number of rows of matrix `C` and `B`, with matrix `A` sized accordingly.  
`n` |  | input | Number of columns of matrix `C` and `B`, with matrix `A` sized accordingly.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x m` with `lda >= max(1, m)` if `side == CUBLAS_SIDE_LEFT` and `lda x n` with `lda >= max(1, n)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, m)`.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0`, or
  * if `side` is not one of `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, m)` when `side == CUBLAS_SIDE_LEFT`, and `lda < max(1, n)` otherwise, or
  * if `ldb < max(1, m)`, or
  * if `ldc < max(1, m)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssymm()](http://www.netlib.org/blas/ssymm.f), [dsymm()](http://www.netlib.org/blas/dsymm.f), [csymm()](http://www.netlib.org/blas/csymm.f), [zsymm()](http://www.netlib.org/blas/zsymm.f)

###  2.7.7. cublas<t>syrk() 
    
    
    cublasStatus_t cublasSsyrk(cublasHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *beta,
                               float           *C, int ldc)
    cublasStatus_t cublasDsyrk(cublasHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *beta,
                               double          *C, int ldc)
    cublasStatus_t cublasCsyrk(cublasHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *beta,
                               cuComplex       *C, int ldc)
    cublasStatus_t cublasZsyrk(cublasHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the symmetric rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(A)^{T} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a symmetric matrix stored in lower or upper mode, and \\(A\\) is a matrix with dimensions \\(\text{op}(A)\\) \\(n \times k\\) . Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ \end{matrix} \right.\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or transpose.  
`n` |  | input | Number of rows of matrix op(`A`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `trans == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix A.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)` when `trans == CUBLAS_OP_N`, and `lda < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssyrk()](http://www.netlib.org/blas/ssyrk.f), [dsyrk()](http://www.netlib.org/blas/dsyrk.f), [csyrk()](http://www.netlib.org/blas/csyrk.f), [zsyrk()](http://www.netlib.org/blas/zsyrk.f)

###  2.7.8. cublas<t>syr2k() 
    
    
    cublasStatus_t cublasSsyr2k(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const float           *alpha,
                                const float           *A, int lda,
                                const float           *B, int ldb,
                                const float           *beta,
                                float           *C, int ldc)
    cublasStatus_t cublasDsyr2k(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const double          *alpha,
                                const double          *A, int lda,
                                const double          *B, int ldb,
                                const double          *beta,
                                double          *C, int ldc)
    cublasStatus_t cublasCsyr2k(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const cuComplex       *alpha,
                                const cuComplex       *A, int lda,
                                const cuComplex       *B, int ldb,
                                const cuComplex       *beta,
                                cuComplex       *C, int ldc)
    cublasStatus_t cublasZsyr2k(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *B, int ldb,
                                const cuDoubleComplex *beta,
                                cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the symmetric rank- \\(2k\\) update

\\(C = \alpha(\text{op}(A)\text{op}(B)^{T} + \text{op}(B)\text{op}(A)^{T}) + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a symmetric matrix stored in lower or upper mode, and \\(A\\) and \\(B\\) are matrices with dimensions \\(\text{op}(A)\\) \\(n \times k\\) and \\(\text{op}(B)\\) \\(n \times k\\) , respectively. Also, for matrix \\(A\\) and \\(B\\)

\\(\text{op(}A\text{) and op(}B\text{)} = \left\\{ \begin{matrix} {A\text{ and }B} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{T}\text{ and }B^{T}} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ \end{matrix} \right.\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or transpose.  
`n` |  | input | Number of rows of matrix op(`A`), op(`B`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`) and op(`B`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`B` | device | input | <_type_ > array of dimensions `ldb x k` with `ldb >= max(1, n)` if `transb == CUBLAS_OP_N` and `ldb x n` with `ldb >= max(1,k)` otherwise.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0`, then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, n)`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)` when `trans == CUBLAS_OP_N`, and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, n)` when `trans == CUBLAS_OP_N`, and `ldb < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssyr2k()](http://www.netlib.org/blas/ssyr2k.f), [dsyr2k()](http://www.netlib.org/blas/dsyr2k.f), [csyr2k()](http://www.netlib.org/blas/csyr2k.f), [zsyr2k()](http://www.netlib.org/blas/zsyr2k.f)

###  2.7.9. cublas<t>syrkx() 
    
    
    cublasStatus_t cublasSsyrkx(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const float           *alpha,
                                const float           *A, int lda,
                                const float           *B, int ldb,
                                const float           *beta,
                                float           *C, int ldc)
    cublasStatus_t cublasDsyrkx(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const double          *alpha,
                                const double          *A, int lda,
                                const double          *B, int ldb,
                                const double          *beta,
                                double          *C, int ldc)
    cublasStatus_t cublasCsyrkx(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const cuComplex       *alpha,
                                const cuComplex       *A, int lda,
                                const cuComplex       *B, int ldb,
                                const cuComplex       *beta,
                                cuComplex       *C, int ldc)
    cublasStatus_t cublasZsyrkx(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *B, int ldb,
                                const cuDoubleComplex *beta,
                                cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs a variation of the symmetric rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(B)^{T} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a symmetric matrix stored in lower or upper mode, and \\(A\\) and \\(B\\) are matrices with dimensions \\(\text{op}(A)\\) \\(n \times k\\) and \\(\text{op}(B)\\) \\(n \times k\\) , respectively. Also, for matrices \\(A\\) and \\(B\\)

\\(\text{op(}A\text{) and op(}B\text{)} = \left\\{ \begin{matrix} {A\text{ and }B} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{T}\text{ and }B^{T}} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ \end{matrix} \right.\\)

This routine can be used when B is in such way that the result is guaranteed to be symmetric. A usual example is when the matrix B is a scaled form of the matrix A: this is equivalent to B being the product of the matrix A and a diagonal matrix. For an efficient computation of the product of a regular matrix with a diagonal matrix, refer to the routine [cublas<t>dgmm()](#cublas-t-dgmm).

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or transpose.  
`n` |  | input | Number of rows of matrix op(`A`), op(`B`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`) and op(`B`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`B` | device | input | <_type_ > array of dimensions `ldb x k` with `ldb >= max(1, n)` if `transb == CUBLAS_OP_N` and `ldb x n` with `ldb >= max(1,k)` otherwise.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0`, then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, n)`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)` when `trans == CUBLAS_OP_N`, and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, n)` when `trans == CUBLAS_OP_N`, and `ldb < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssyrk()](http://www.netlib.org/blas/ssyrk.f), [dsyrk()](http://www.netlib.org/blas/dsyrk.f), [csyrk()](http://www.netlib.org/blas/csyrk.f), [zsyrk()](http://www.netlib.org/blas/zsyrk.f) and

[ssyr2k()](http://www.netlib.org/blas/ssyr2k.f), [dsyr2k()](http://www.netlib.org/blas/dsyr2k.f), [csyr2k()](http://www.netlib.org/blas/csyr2k.f), [zsyr2k()](http://www.netlib.org/blas/zsyr2k.f)

###  2.7.10. cublas<t>trmm() 
    
    
    cublasStatus_t cublasStrmm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *B, int ldb,
                               float                 *C, int ldc)
    cublasStatus_t cublasDtrmm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *B, int ldb,
                               double                *C, int ldc)
    cublasStatus_t cublasCtrmm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *B, int ldb,
                               cuComplex             *C, int ldc)
    cublasStatus_t cublasZtrmm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb,
                               cuDoubleComplex       *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the triangular matrix-matrix multiplication

\\(C = \left\\{ \begin{matrix} {\alpha\text{op}(A)B} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {\alpha B\text{op}(A)} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) is a triangular matrix stored in lower or upper mode with or without the main diagonal, \\(B\\) and \\(C\\) are \\(m \times n\\) matrix, and \\(\alpha\\) is a scalar. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Notice that in order to achieve better parallelism cuBLAS differs from the BLAS API only for this routine. The BLAS API assumes an in-place implementation (with results written back to B), while the cuBLAS API assumes an out-of-place implementation (with results written into C). The application can obtain the in-place functionality of BLAS in the cuBLAS API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`side` |  | input | Indicates if matrix `A` is on the left or right of `B`.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | Indicates if the elements on the main diagonal of matrix `A` are unity and should not be accessed.  
`m` |  | input | Number of rows of matrix `B`, with matrix `A` sized accordingly.  
`n` |  | input | Number of columns of matrix `B`, with matrix `A` sized accordingly.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication, if `alpha == 0` then `A` is not referenced and `B` does not have to be a valid input.  
`A` | device | input | <_type_ > array of dimension `lda x m` with `lda >= max(1, m)` if `side == CUBLAS_SIDE_LEFT` and `lda x n` with `lda >= max(1, n)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, m)`.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0`, `n < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `side` is not one of `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT`, or
  * if `lda < max(1, m)` if `side == CUBLAS_SIDE_LEFT`, and `lda < max(1, n)` otherwise, or
  * if `ldb < max(1, m)`, or
  * if `ldc < max(1, m)`, or
  * if `alpha` is NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[strmm()](http://www.netlib.org/blas/strmm.f), [dtrmm()](http://www.netlib.org/blas/dtrmm.f), [ctrmm()](http://www.netlib.org/blas/ctrmm.f), [ztrmm()](http://www.netlib.org/blas/ztrmm.f)

###  2.7.11. cublas<t>trsm() 
    
    
    cublasStatus_t cublasStrsm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const float           *alpha,
                               const float           *A, int lda,
                               float           *B, int ldb)
    cublasStatus_t cublasDtrsm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const double          *alpha,
                               const double          *A, int lda,
                               double          *B, int ldb)
    cublasStatus_t cublasCtrsm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               cuComplex       *B, int ldb)
    cublasStatus_t cublasZtrsm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               cuDoubleComplex *B, int ldb)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function solves the triangular linear system with multiple right-hand-sides

\\(\left\\{ \begin{matrix} {\text{op}(A)X = \alpha B} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {X\text{op}(A) = \alpha B} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) is a triangular matrix stored in lower or upper mode with or without the main diagonal, \\(X\\) and \\(B\\) are \\(m \times n\\) matrices, and \\(\alpha\\) is a scalar. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

The solution \\(X\\) overwrites the right-hand-sides \\(B\\) on exit.

No test for singularity or near-singularity is included in this function.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`side` |  | input | Indicates if matrix `A` is on the left or right of `X`.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | Indicates if the elements on the main diagonal of matrix `A` are unity and should not be accessed.  
`m` |  | input | Number of rows of matrix `B`, with matrix `A` sized accordingly.  
`n` |  | input | Number of columns of matrix `B`, with matrix `A` is sized accordingly.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication, if `alpha == 0` then `A` is not referenced and `B` does not have to be a valid input.  
`A` | device | input | <_type_ > array of dimension `lda x m` with `lda >= max(1, m)` if `side == CUBLAS_SIDE_LEFT` and `lda x n` with `lda >= max(1, n)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`B` | device | in/out | <_type_ > array. It has dimensions `ldb x n` with `ldb >= max(1, m)`.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0`, `n < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `side` is not one of `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT`, or
  * if `diag` is not one of `CUBLAS_DIAG_UNIT` and `CUBLAS_DIAG_NON_UNIT`, or
  * if `lda < max(1, m)` if `side == CUBLAS_SIDE_LEFT`, and `lda < max(1, n)` otherwise, or
  * if `ldb < max(1, m)`, or
  * if `alpha` is NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[strsm()](http://www.netlib.org/blas/strsm.f), [dtrsm()](http://www.netlib.org/blas/dtrsm.f), [ctrsm()](http://www.netlib.org/blas/ctrsm.f), [ztrsm()](http://www.netlib.org/blas/ztrsm.f)

###  2.7.12. cublas<t>trsmBatched() 
    
    
    cublasStatus_t cublasStrsmBatched( cublasHandle_t    handle,
                                       cublasSideMode_t  side,
                                       cublasFillMode_t  uplo,
                                       cublasOperation_t trans,
                                       cublasDiagType_t  diag,
                                       int m,
                                       int n,
                                       const float *alpha,
                                       const float *const A[],
                                       int lda,
                                       float *const B[],
                                       int ldb,
                                       int batchCount);
    cublasStatus_t cublasDtrsmBatched( cublasHandle_t    handle,
                                       cublasSideMode_t  side,
                                       cublasFillMode_t  uplo,
                                       cublasOperation_t trans,
                                       cublasDiagType_t  diag,
                                       int m,
                                       int n,
                                       const double *alpha,
                                       const double *const A[],
                                       int lda,
                                       double *const B[],
                                       int ldb,
                                       int batchCount);
    cublasStatus_t cublasCtrsmBatched( cublasHandle_t    handle,
                                       cublasSideMode_t  side,
                                       cublasFillMode_t  uplo,
                                       cublasOperation_t trans,
                                       cublasDiagType_t  diag,
                                       int m,
                                       int n,
                                       const cuComplex *alpha,
                                       const cuComplex *const A[],
                                       int lda,
                                       cuComplex *const B[],
                                       int ldb,
                                       int batchCount);
    cublasStatus_t cublasZtrsmBatched( cublasHandle_t    handle,
                                       cublasSideMode_t  side,
                                       cublasFillMode_t  uplo,
                                       cublasOperation_t trans,
                                       cublasDiagType_t  diag,
                                       int m,
                                       int n,
                                       const cuDoubleComplex *alpha,
                                       const cuDoubleComplex *const A[],
                                       int lda,
                                       cuDoubleComplex *const B[],
                                       int ldb,
                                       int batchCount);
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function solves an array of triangular linear systems with multiple right-hand-sides

\\(\left\\{ \begin{matrix} {\text{op}(A\lbrack i\rbrack)X\lbrack i\rbrack = \alpha B\lbrack i\rbrack} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {X\lbrack i\rbrack\text{op}(A\lbrack i\rbrack) = \alpha B\lbrack i\rbrack} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\lbrack i\rbrack\\) is a triangular matrix stored in lower or upper mode with or without the main diagonal, \\(X\lbrack i\rbrack\\) and \\(B\lbrack i\rbrack\\) are \\(m \times n\\) matrices, and \\(\alpha\\) is a scalar. Also, for matrix \\(A\\)

\\(\text{op}(A\lbrack i\rbrack) = \left\\{ \begin{matrix} {A\lbrack i\rbrack} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{T}\lbrack i\rbrack} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ {A^{H}\lbrack i\rbrack} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

The solution \\(X\lbrack i\rbrack\\) overwrites the right-hand-sides \\(B\lbrack i\rbrack\\) on exit.

No test for singularity or near-singularity is included in this function.

This function works for any sizes but is intended to be used for matrices of small sizes where the launch overhead is a significant factor. For bigger sizes, it might be advantageous to call `batchCount` times the regular [cublas<t>trsm()](#cublas-t-trsm) within a set of CUDA streams.

The current implementation is limited to devices with compute capability above or equal 2.0.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`side` |  | input | Indicates if matrix `A[i]` is on the left or right of `X[i]`.  
`uplo` |  | input | Indicates if matrix `A[i]` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A[i]`) that is non- or (conj.) transpose.  
`diag` |  | input | Indicates if the elements on the main diagonal of matrix `A[i]` are unity and should not be accessed.  
`m` |  | input | Number of rows of matrix `B[i]`, with matrix `A[i]` sized accordingly.  
`n` |  | input | Number of columns of matrix `B[i]`, with matrix `A[i]` is sized accordingly.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication, if `alpha == 0` then `A[i]` is not referenced and `B[i]` does not have to be a valid input.  
`A` | device | input | Array of pointers to <_type_ > array, with each array of dim. `lda x m` with `lda >= max(1, m)` if `side == CUBLAS_SIDE_LEFT` and `lda x n` with `lda >= max(1, n)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A[i]`.  
`B` | device | in/out | Array of pointers to <_type_ > array, with each array of dim. `ldb x n` with `ldb >= max(1, m)`. Matrices `B[i]` should not overlap; otherwise, undefined behavior is expected.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B[i]`.  
`batchCount` |  | input | Number of pointers contained in A and B.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0`, `n < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `side` is not one of `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT`, or
  * if `diag` is not one of `CUBLAS_DIAG_UNIT` and `CUBLAS_DIAG_NON_UNIT`, or
  * if `lda < max(1, m)` if `side == CUBLAS_SIDE_LEFT`, and `lda < max(1, n)` otherwise, or
  * if `ldb < max(1, m)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[strsm()](http://www.netlib.org/blas/strsm.f), [dtrsm()](http://www.netlib.org/blas/dtrsm.f), [ctrsm()](http://www.netlib.org/blas/ctrsm.f), [ztrsm()](http://www.netlib.org/blas/ztrsm.f)

###  2.7.13. cublas<t>hemm() 
    
    
    cublasStatus_t cublasChemm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               int m, int n,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *B, int ldb,
                               const cuComplex       *beta,
                               cuComplex       *C, int ldc)
    cublasStatus_t cublasZhemm(cublasHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               int m, int n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the Hermitian matrix-matrix multiplication

\\(C = \left\\{ \begin{matrix} {\alpha AB + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {\alpha BA + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) is a Hermitian matrix stored in lower or upper mode, \\(B\\) and \\(C\\) are \\(m \times n\\) matrices, and \\(\alpha\\) and \\(\beta\\) are scalars.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`side` |  | input | Indicates if matrix `A` is on the left or right of `B`.  
`uplo` |  | input | Indicates if matrix `A` lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.  
`m` |  | input | Number of rows of matrix `C` and `B`, with matrix `A` sized accordingly.  
`n` |  | input | Number of columns of matrix `C` and `B`, with matrix `A` sized accordingly.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x m` with `lda >= max(1, m)` if `side == CUBLAS_SIDE_LEFT` and `lda x n` with `lda >= max(1, n)` otherwise. The imaginary parts of the diagonal elements are assumed to be zero.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, m)`.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` |  | input | <_type_ > scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0`, or
  * if `side` is not one of `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, m)` when `side == CUBLAS_SIDE_LEFT`, and `lda < max(1, n)` otherwise, or
  * if `ldb < max(1, m)`, or
  * if `ldc < max(1, m)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[chemm()](http://www.netlib.org/blas/chemm.f), [zhemm()](http://www.netlib.org/blas/zhemm.f)

###  2.7.14. cublas<t>herk() 
    
    
    cublasStatus_t cublasCherk(cublasHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const float  *alpha,
                               const cuComplex       *A, int lda,
                               const float  *beta,
                               cuComplex       *C, int ldc)
    cublasStatus_t cublasZherk(cublasHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const double *alpha,
                               const cuDoubleComplex *A, int lda,
                               const double *beta,
                               cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the Hermitian rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(A)^{H} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a Hermitian matrix stored in lower or upper mode, and \\(A\\) is a matrix with dimensions \\(\text{op}(A)\\) \\(n \times k\\) . Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part is stored, the other Hermitian part is not referenced.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`n` |  | input | Number of rows of matrix op(`A`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`beta` |  | input | <_type_ > scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)` when `trans == CUBLAS_OP_N`, and `lda < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[cherk()](http://www.netlib.org/blas/cherk.f), [zherk()](http://www.netlib.org/blas/zherk.f)

###  2.7.15. cublas<t>her2k() 
    
    
    cublasStatus_t cublasCher2k(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const cuComplex       *alpha,
                                const cuComplex       *A, int lda,
                                const cuComplex       *B, int ldb,
                                const float  *beta,
                                cuComplex       *C, int ldc)
    cublasStatus_t cublasZher2k(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *B, int ldb,
                                const double *beta,
                                cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the Hermitian rank- \\(2k\\) update

\\(C = \alpha\text{op}(A)\text{op}(B)^{H} + \overset{ˉ}{\alpha}\text{op}(B)\text{op}(A)^{H} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a Hermitian matrix stored in lower or upper mode, and \\(A\\) and \\(B\\) are matrices with dimensions \\(\text{op}(A)\\) \\(n \times k\\) and \\(\text{op}(B)\\) \\(n \times k\\) , respectively. Also, for matrix \\(A\\) and \\(B\\)

\\(\text{op(}A\text{) and op(}B\text{)} = \left\\{ \begin{matrix} {A\text{ and }B} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{H}\text{ and }B^{H}} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part is stored, the other Hermitian part is not referenced.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`n` |  | input | Number of rows of matrix op(`A`), op(`B`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`) and op(`B`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x k` with `ldb >= max(1, n)` if `transb == CUBLAS_OP_N` and `ldb x n` with `ldb >= max(1,k)` otherwise.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)` when `trans == CUBLAS_OP_N`, and `lda < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[cher2k()](http://www.netlib.org/blas/cher2k.f), [zher2k()](http://www.netlib.org/blas/zher2k.f)

###  2.7.16. cublas<t>herkx() 
    
    
    cublasStatus_t cublasCherkx(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const cuComplex       *alpha,
                                const cuComplex       *A, int lda,
                                const cuComplex       *B, int ldb,
                                const float  *beta,
                                cuComplex       *C, int ldc)
    cublasStatus_t cublasZherkx(cublasHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                int n, int k,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *B, int ldb,
                                const double *beta,
                                cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs a variation of the Hermitian rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(B)^{H} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a Hermitian matrix stored in lower or upper mode, and \\(A\\) and \\(B\\) are matrices with dimensions \\(\text{op}(A)\\) \\(n \times k\\) and \\(\text{op}(B)\\) \\(n \times k\\) , respectively. Also, for matrix \\(A\\) and \\(B\\)

\\(\text{op(}A\text{) and op(}B\text{)} = \left\\{ \begin{matrix} {A\text{ and }B} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{H}\text{ and }B^{H}} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

This routine can be used when the matrix B is in such way that the result is guaranteed to be hermitian. An usual example is when the matrix B is a scaled form of the matrix A: this is equivalent to B being the product of the matrix A and a diagonal matrix. For an efficient computation of the product of a regular matrix with a diagonal matrix, refer to the routine [cublas<t>dgmm()](#cublas-t-dgmm).

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part is stored, the other Hermitian part is not referenced.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`n` |  | input | Number of rows of matrix op(`A`), op(`B`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`) and op(`B`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x k` with `ldb >= max(1, n)` if `transb == CUBLAS_OP_N` and `ldb x n` with `ldb >= max(1,k)` otherwise.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | Real scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)` when `trans == CUBLAS_OP_N`, and `lda < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[cherk()](http://www.netlib.org/blas/cherk.f), [zherk()](http://www.netlib.org/blas/zherk.f) and

[cher2k()](http://www.netlib.org/blas/cher2k.f), [zher2k()](http://www.netlib.org/blas/zher2k.f)


##  2.8. BLAS-like Extension 

This section describes the BLAS-extension functions that perform matrix-matrix operations.

###  2.8.1. cublas<t>geam() 
    
    
    cublasStatus_t cublasSgeam(cublasHandle_t handle,
                              cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n,
                              const float           *alpha,
                              const float           *A, int lda,
                              const float           *beta,
                              const float           *B, int ldb,
                              float           *C, int ldc)
    cublasStatus_t cublasDgeam(cublasHandle_t handle,
                              cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n,
                              const double          *alpha,
                              const double          *A, int lda,
                              const double          *beta,
                              const double          *B, int ldb,
                              double          *C, int ldc)
    cublasStatus_t cublasCgeam(cublasHandle_t handle,
                              cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n,
                              const cuComplex       *alpha,
                              const cuComplex       *A, int lda,
                              const cuComplex       *beta ,
                              const cuComplex       *B, int ldb,
                              cuComplex       *C, int ldc)
    cublasStatus_t cublasZgeam(cublasHandle_t handle,
                              cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *beta,
                              const cuDoubleComplex *B, int ldb,
                              cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-matrix addition/transposition

\\(C = \alpha\text{op}(A) + \beta\text{op}(B)\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are matrices stored in column-major format with dimensions \\(\text{op}(A)\\) \\(m \times n\\) , \\(\text{op}(B)\\) \\(m \times n\\) and \\(C\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B)\\) is defined similarly for matrix \\(B\\) .

The operation is out-of-place if C does not overlap A or B.

The in-place mode supports the following two operations,

\\(C = \alpha\text{*}C + \beta\text{op}(B)\\)

\\(C = \alpha\text{op}(A) + \beta\text{*}C\\)

For in-place mode, if `C == A`, `ldc == lda` and `transa == CUBLAS_OP_N`. If `C === B`, `ldc == ldb` and `transb == CUBLAS_OP_N`. If the user does not meet above requirements, `CUBLAS_STATUS_INVALID_VALUE` is returned.

The operation includes the following special cases:

the user can reset matrix C to zero by setting `*alpha = beta = 0`.

the user can transpose matrix A by setting `*alpha = 1 and *beta = 0`.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`transa` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`transb` |  | input | Operation op(`B`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix op(`A`) and `C`.  
`n` |  | input | Number of columns of matrix op(`B`) and `C`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication. If `*alpha == 0`, `A` does not have to be a valid input.  
`A` | device | input | <_type_ > array of dimensions `lda x n` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, n)` otherwise.  
`lda` |  | input | Leading dimension of two-dimensional array used to store the matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, m)` if `transb == CUBLAS_OP_N` and `ldb x m` with `ldb >= max(1,n)` otherwise.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `*beta == 0`, `B` does not have to be a valid input.  
`C` | device | output | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | Leading dimension of a two-dimensional array used to store the matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0`, or
  * if `transa` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `transb` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `lda < max(1, m)` when `transa == CUBLAS_OP_N`, and `lda < max(1, n)` otherwise, or
  * if `ldb < max(1, m)` if `transb == CUBLAS_OP_N`, and `ldb < max(1, n)` otherwise, or
  * if `ldc < max(1, m)`, or
  * if `A == C` and `(transa != CUBLAS_OP_N) || (lda != ldc)`, or
  * if `B == C` and `(transb != CUBLAS_OP_N) || (ldb != ldc)`, or
  * if `alpha` or `beta` are NULL

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
###  2.8.2. cublas<t>dgmm() 
    
    
    cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                              int m, int n,
                              const float           *A, int lda,
                              const float           *x, int incx,
                              float           *C, int ldc)
    cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                              int m, int n,
                              const double          *A, int lda,
                              const double          *x, int incx,
                              double          *C, int ldc)
    cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                              int m, int n,
                              const cuComplex       *A, int lda,
                              const cuComplex       *x, int incx,
                              cuComplex       *C, int ldc)
    cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                              int m, int n,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *x, int incx,
                              cuDoubleComplex *C, int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-matrix multiplication

\\(C = \left\\{ \begin{matrix} {A \times diag(X)} & {\text{if }\textsf{mode == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ {diag(X) \times A} & {\text{if }\textsf{mode == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) and \\(C\\) are matrices stored in column-major format with dimensions \\(m \times n\\) . \\(X\\) is a vector of size \\(n\\) if `mode == CUBLAS_SIDE_RIGHT` and of size \\(m\\) if `mode == CUBLAS_SIDE_LEFT`. \\(X\\) is gathered from one-dimensional array x with stride `incx`. The absolute value of `incx` is the stride and the sign of `incx` is direction of the stride. If `incx` is positive, then we forward x from the first element. Otherwise, we backward x from the last element. The formula of X is

\\(X\lbrack j\rbrack = \left\\{ \begin{matrix} {x\lbrack j \times incx\rbrack} & {\text{if }incx \geq 0} \\\ {x\lbrack(\chi - 1) \times |incx| - j \times |incx|\rbrack} & {\text{if }incx < 0} \\\ \end{matrix} \right.\\)

where \\(\chi = m\\) if `mode == CUBLAS_SIDE_LEFT` and \\(\chi = n\\) if `mode == CUBLAS_SIDE_RIGHT`.

Example 1: if the user wants to perform \\(diag(diag(B)) \times A\\) , then \\(incx = ldb + 1\\) where \\(ldb\\) is leading dimension of matrix `B`, either row-major or column-major.

Example 2: if the user wants to perform \\(\alpha \times A\\) , then there are two choices, either [cublas<t>geam()](#cublas-t-geam) with `*beta == 0` and `transa == CUBLAS_OP_N` or [cublas<t>dgmm()](#cublas-t-dgmm) with `incx == 0` and `x[0] == alpha`.

The operation is out-of-place. The in-place only works if `lda == ldc`.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`mode` |  | input | Left multiply if `mode == CUBLAS_SIDE_LEFT` or right multiply if `mode == CUBLAS_SIDE_RIGHT`  
`m` |  | input | Number of rows of matrix `A` and `C`.  
`n` |  | input | Number of columns of matrix `A` and `C`.  
`A` | device | input | <_type_ > array of dimensions `lda x n` with `lda >= max(1, m)`  
`lda` |  | input | Leading dimension of two-dimensional array used to store the matrix `A`.  
`x` | device | input | One-dimensional <_type_ > array of size `abs(incx) x m` if `mode == CUBLAS_SIDE_LEFT` and `abs(incx) x n` if `mode == CUBLAS_SIDE_RIGHT`  
`incx` |  | input | Stride of one-dimensional array `x`.  
`C` | device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | Leading dimension of a two-dimensional array used to store the matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0`, or
  * if `mode` is not one of `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT`, or
  * if `lda < max(1, m)`, or
  * if `ldc < max(1, m)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
###  2.8.3. cublas<t>getrfBatched() 
    
    
    cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle,
                                       int n,
                                       float *const Aarray[],
                                       int lda,
                                       int *PivotArray,
                                       int *infoArray,
                                       int batchSize);
    
    cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle,
                                       int n,
                                       double *const Aarray[],
                                       int lda,
                                       int *PivotArray,
                                       int *infoArray,
                                       int batchSize);
    
    cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle,
                                       int n,
                                       cuComplex *const Aarray[],
                                       int lda,
                                       int *PivotArray,
                                       int *infoArray,
                                       int batchSize);
    
    cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle,
                                       int n,
                                       cuDoubleComplex *const Aarray[],
                                       int lda,
                                       int *PivotArray,
                                       int *infoArray,
                                       int batchSize);
    

`Aarray` is an array of pointers to matrices stored in column-major format with dimensions `nxn` and leading dimension `lda`.

This function performs the LU factorization of each `Aarray[i]` for i = 0, …, `batchSize-1` by the following equation

\\(\text{P}\text{*}{Aarray}\lbrack i\rbrack = L\text{*}U\\)

where `P` is a permutation matrix which represents partial pivoting with row interchanges. `L` is a lower triangular matrix with unit diagonal and `U` is an upper triangular matrix.

Formally `P` is written by a product of permutation matrices `Pj`, for `j = 1,2,...,n`, say `P = P1 * P2 * P3 * .... * Pn`. `Pj` is a permutation matrix which interchanges two rows of vector x when performing `Pj*x`. `Pj` can be constructed by `j` element of `PivotArray[i]` by the following Matlab code
    
    
    // In Matlab PivotArray[i] is an array of base-1.
    // In C, PivotArray[i] is base-0.
    Pj = eye(n);
    swap Pj(j,:) and Pj(PivotArray[i][j]  ,:)
    

`L` and `U` are written back to original matrix `A`, and diagonal elements of `L` are discarded. The `L` and `U` can be constructed by the following Matlab code
    
    
    // A is a matrix of nxn after getrf.
    L = eye(n);
    for j = 1:n
        L(j+1:n,j) = A(j+1:n,j)
    end
    U = zeros(n);
    for i = 1:n
        U(i,i:n) = A(i,i:n)
    end
    

If matrix `A(=Aarray[i])` is singular, getrf still works and the value of `info(=infoArray[i])` reports first row index that LU factorization cannot proceed. If info is `k`, `U(k,k)` is zero. The equation `P*A == L*U` still holds, however `L` and `U` reconstruction needs a different Matlab code as follows:
    
    
    // A is a matrix of nxn after getrf.
    // info is k, which means U(k,k) is zero.
    L = eye(n);
    for j = 1:k-1
        L(j+1:n,j) = A(j+1:n,j)
    end
    U = zeros(n);
    for i = 1:k-1
        U(i,i:n) = A(i,i:n)
    end
    for i = k:n
        U(i,k:n) = A(i,k:n)
    end
    

This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor.

cublas<t>getrfBatched supports non-pivot LU factorization if `PivotArray` is NULL.

cublas<t>getrfBatched supports arbitrary dimension.

cublas<t>getrfBatched only supports compute capability 2.0 or above.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of rows and columns of `Aarray[i]`.  
`Aarray` | device | input/output | Array of pointers to <_type_ > array, with each array of dim. `n x n` with `lda >= max(1, n)`. Matrices `Aarray[i]` should not overlap; otherwise, undefined behavior is expected.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `Aarray[i]`.  
`PivotArray` | device | output | Array of size `n x batchSize` that contains the pivoting sequence of each factorization of `Aarray[i]` stored in a linear fashion. If `PivotArray` is NULL, pivoting is disabled.  
`infoArray` | device | output |  Array of size `batchSize` that info(=infoArray[i]) contains the information of factorization of `Aarray[i]`. If `info == 0`, the execution is successful. If `info = -j`, the `j`-th parameter had an illegal value. If `info = k`, `U(k, k) == 0`. The factorization has been completed, but U is exactly singular.  
`batchSize` |  | input | Number of pointers contained in A  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | The parameters `n < 0` or `batchSize < 0` or `lda <0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sgeqrf()](http://www.netlib.org/lapack/single/sgetrf.f), [dgeqrf()](http://www.netlib.org/lapack/double/dgetrf.f), [cgeqrf()](http://www.netlib.org/lapack/complex/cgetrf.f), [zgeqrf()](http://www.netlib.org/lapack/complex16/zgetrf.f)

###  2.8.4. cublas<t>getrsBatched() 
    
    
    cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int n,
                                       int nrhs,
                                       const float *const Aarray[],
                                       int lda,
                                       const int *devIpiv,
                                       float *const Barray[],
                                       int ldb,
                                       int *info,
                                       int batchSize);
    
    cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int n,
                                       int nrhs,
                                       const double *const Aarray[],
                                       int lda,
                                       const int *devIpiv,
                                       double *const Barray[],
                                       int ldb,
                                       int *info,
                                       int batchSize);
    
    cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int n,
                                       int nrhs,
                                       const cuComplex *const Aarray[],
                                       int lda,
                                       const int *devIpiv,
                                       cuComplex *const Barray[],
                                       int ldb,
                                       int *info,
                                       int batchSize);
    
    cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int n,
                                       int nrhs,
                                       const cuDoubleComplex *const Aarray[],
                                       int lda,
                                       const int *devIpiv,
                                       cuDoubleComplex *const Barray[],
                                       int ldb,
                                       int *info,
                                       int batchSize);
    

This function solves an array of systems of linear equations of the form:

\\(\text{op}(A\lbrack i \rbrack) X\lbrack i\rbrack = B\lbrack i\rbrack\\)

where \\(A\lbrack i\rbrack\\) is a matrix which has been LU factorized with pivoting, \\(X\lbrack i\rbrack\\) and \\(B\lbrack i\rbrack\\) are \\(n \times {nrhs}\\) matrices. Also, for matrix \\(A\\)

\\(\text{op}(A\lbrack i\rbrack) = \left\\{ \begin{matrix} {A\lbrack i\rbrack} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{T}\lbrack i\rbrack} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ {A^{H}\lbrack i\rbrack} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor.

[cublas<t>getrsBatched()](#cublas-t-getrsbatched) supports non-pivot LU factorization if `devIpiv` is NULL.

[cublas<t>getrsBatched()](#cublas-t-getrsbatched) supports arbitrary dimension.

[cublas<t>getrsBatched()](#cublas-t-getrsbatched) only supports compute capability 2.0 or above.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`n` |  | input | Number of rows and columns of `Aarray[i]`.  
`nrhs` |  | input | Number of columns of `Barray[i]`.  
`Aarray` | device | input | Array of pointers to <_type_ > array, with each array of dim. `n x n` with `lda >= max(1, n)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `Aarray[i]`.  
`devIpiv` | device | input | Array of size `n x batchSize` that contains the pivoting sequence of each factorization of `Aarray[i]` stored in a linear fashion. If `devIpiv` is NULL, pivoting for all `Aarray[i]` is ignored.  
`Barray` | device | input/output | Array of pointers to <_type_ > array, with each array of dim. `n x nrhs` with `ldb >= max(1, n)`. Matrices `Barray[i]` should not overlap; otherwise, undefined behavior is expected.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store each solution matrix `Barray[i]`.  
`info` | host | output |  If `info == 0`, the execution is successful. If `info = -j`, the `j`-th parameter had an illegal value.  
`batchSize` |  | input | Number of pointers contained in A  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `nrhs < 0`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `lda < max(1, n)`, or
  * if `ldb < max(1, n)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sgeqrs()](http://www.netlib.org/lapack/single/sgetrs.f), [dgeqrs()](http://www.netlib.org/lapack/double/dgetrs.f), [cgeqrs()](http://www.netlib.org/lapack/complex/cgetrs.f), [zgeqrs()](http://www.netlib.org/lapack/complex16/zgetrs.f)

###  2.8.5. cublas<t>getriBatched() 
    
    
    cublasStatus_t cublasSgetriBatched(cublasHandle_t handle,
                                       int n,
                                       const float *const Aarray[],
                                       int lda,
                                       int *PivotArray,
                                       float *const Carray[],
                                       int ldc,
                                       int *infoArray,
                                       int batchSize);
    
    cublasStatus_t cublasDgetriBatched(cublasHandle_t handle,
                                       int n,
                                       const double *const Aarray[],
                                       int lda,
                                       int *PivotArray,
                                       double *const Carray[],
                                       int ldc,
                                       int *infoArray,
                                       int batchSize);
    
    cublasStatus_t cublasCgetriBatched(cublasHandle_t handle,
                                       int n,
                                       const cuComplex *const Aarray[],
                                       int lda,
                                       int *PivotArray,
                                       cuComplex *const Carray[],
                                       int ldc,
                                       int *infoArray,
                                       int batchSize);
    
    cublasStatus_t cublasZgetriBatched(cublasHandle_t handle,
                                       int n,
                                       const cuDoubleComplex *const Aarray[],
                                       int lda,
                                       int *PivotArray,
                                       cuDoubleComplex *const Carray[],
                                       int ldc,
                                       int *infoArray,
                                       int batchSize);
    

`Aarray` and `Carray` are arrays of pointers to matrices stored in column-major format with dimensions `n*n` and leading dimension `lda` and `ldc` respectively.

This function performs the inversion of matrices `A[i]` for i = 0, …, `batchSize-1`.

Prior to calling cublas<t>getriBatched, the matrix `A[i]` must be factorized first using the routine cublas<t>getrfBatched. After the call of cublas<t>getrfBatched, the matrix pointing by `Aarray[i]` will contain the LU factors of the matrix `A[i]` and the vector pointing by `(PivotArray+i)` will contain the pivoting sequence.

Following the LU factorization, cublas<t>getriBatched uses forward and backward triangular solvers to complete inversion of matrices `A[i]` for i = 0, …, `batchSize-1`. The inversion is out-of-place, so memory space of Carray[i] cannot overlap memory space of Array[i].

Typically all parameters in cublas<t>getrfBatched would be passed into cublas<t>getriBatched. For example,
    
    
    // step 1: perform in-place LU decomposition, P*A = L*U.
    //      Aarray[i] is n*n matrix A[i]
        cublasDgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
    //      check infoArray[i] to see if factorization of A[i] is successful or not.
    //      Array[i] contains LU factorization of A[i]
    
    // step 2: perform out-of-place inversion, Carray[i] = inv(A[i])
        cublasDgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize);
    //      check infoArray[i] to see if inversion of A[i] is successful or not.
    

The user can check singularity from either cublas<t>getrfBatched or cublas<t>getriBatched.

This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor.

If cublas<t>getrfBatched is performed by non-pivoting, `PivotArray` of cublas<t>getriBatched should be NULL.

cublas<t>getriBatched supports arbitrary dimension.

cublas<t>getriBatched only supports compute capability 2.0 or above.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of rows and columns of `Aarray[i]`.  
`Aarray` | device | input | Array of pointers to <_type_ > array, with each array of dimension `n*n` with `lda >= max(1, n)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `Aarray[i]`.  
`PivotArray` | device | output | Array of size `n*batchSize` that contains the pivoting sequence of each factorization of `Aarray[i]` stored in a linear fashion. If `PivotArray` is NULL, pivoting is disabled.  
`Carray` | device | output | Array of pointers to <_type_ > array, with each array of dimension `n*n` with `ldc >= max(1, n)`. Matrices `Carray[i]` should not overlap; otherwise, undefined behavior is expected.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store each matrix `Carray[i]`.  
`infoArray` | device | output |  Array of size `batchSize` that info(=infoArray[i]) contains the information of inversion of `A[i]`. If `info == 0`, the execution is successful. If `info == k`, `U(k, k) == 0`. The U is exactly singular and the inversion failed.  
`batchSize` |  | input | Number of pointers contained in A  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `lda < 0` or `ldc < 0` or `batchSize < 0`, or
  * if `lda < n` or `ldc < n`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
###  2.8.6. cublas<t>matinvBatched() 
    
    
    cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle,
                                        int n,
                                        const float *const A[],
                                        int lda,
                                        float *const Ainv[],
                                        int lda_inv,
                                        int *info,
                                        int batchSize);
    
    cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle,
                                        int n,
                                        const double *const A[],
                                        int lda,
                                        double *const Ainv[],
                                        int lda_inv,
                                        int *info,
                                        int batchSize);
    
    cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle,
                                        int n,
                                        const cuComplex *const A[],
                                        int lda,
                                        cuComplex *const Ainv[],
                                        int lda_inv,
                                        int *info,
                                        int batchSize);
    
    cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle,
                                        int n,
                                        const cuDoubleComplex *const A[],
                                        int lda,
                                        cuDoubleComplex *const Ainv[],
                                        int lda_inv,
                                        int *info,
                                        int batchSize);
    

`A` and `Ainv` are arrays of pointers to matrices stored in column-major format with dimensions `n*n` and leading dimension `lda` and `lda_inv` respectively.

This function performs the inversion of matrices `A[i]` for i = 0, …, `batchSize-1`.

This function is a short cut of [cublas<t>getrfBatched()](#cublas-t-getrfbatched) plus [cublas<t>getriBatched()](#cublas-t-getribatched). However it doesn’t work if `n` is greater than 32. If not, the user has to go through [cublas<t>getrfBatched()](#cublas-t-getrfbatched) and [cublas<t>getriBatched()](#cublas-t-getribatched).

If the matrix `A[i]` is singular, then `info[i]` reports singularity, the same as [cublas<t>getrfBatched()](#cublas-t-getrfbatched).

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of rows and columns of `A[i]`.  
`A` | device | input | Array of pointers to <_type_ > array, with each array of dimension `n*n` with `lda >= max(1, n)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `A[i]`.  
`Ainv` | device | output | Array of pointers to <_type_ > array, with each array of dimension `n*n` with `lda_inv >= max(1, n)`. Matrices `Ainv[i]` should not overlap; otherwise, undefined behavior is expected.  
`lda_inv` |  | input | Leading dimension of two-dimensional array used to store each matrix `Ainv[i]`.  
`info` | device | output |  Array of size `batchSize` that info[i] contains the information of inversion of `A[i]`. If `info[i] == 0`, the execution is successful. If `info[i] == k`, then `U(k, k) == 0`. The U is exactly singular and the inversion failed.  
`batchSize` |  | input | Number of pointers contained in `A`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `lda < 0` or `lda_inv < 0` or `batchSize < 0`, or
  * if `lda < n` or `lda_inv < n`, or
  * if `n > 32`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
###  2.8.7. cublas<t>geqrfBatched() 
    
    
    cublasStatus_t cublasSgeqrfBatched( cublasHandle_t handle,
                                        int m,
                                        int n,
                                        float *const Aarray[],
                                        int lda,
                                        float *const TauArray[],
                                        int *info,
                                        int batchSize);
    
    cublasStatus_t cublasDgeqrfBatched( cublasHandle_t handle,
                                        int m,
                                        int n,
                                        double *const Aarray[],
                                        int lda,
                                        double *const TauArray[],
                                        int *info,
                                        int batchSize);
    
    cublasStatus_t cublasCgeqrfBatched( cublasHandle_t handle,
                                        int m,
                                        int n,
                                        cuComplex *const Aarray[],
                                        int lda,
                                        cuComplex *const TauArray[],
                                        int *info,
                                        int batchSize);
    
    cublasStatus_t cublasZgeqrfBatched( cublasHandle_t handle,
                                        int m,
                                        int n,
                                        cuDoubleComplex *const Aarray[],
                                        int lda,
                                        cuDoubleComplex *const TauArray[],
                                        int *info,
                                        int batchSize);
    

`Aarray` is an array of pointers to matrices stored in column-major format with dimensions `m x n` and leading dimension `lda`. `TauArray` is an array of pointers to vectors of dimension of at least `max (1, min(m, n)`.

This function performs the QR factorization of each `Aarray[i]` for `i = 0, ...,batchSize-1` using Householder reflections. Each matrix `Q[i]` is represented as a product of elementary reflectors and is stored in the lower part of each `Aarray[i]` as follows :
    
    
    Q[j] = H[j][1] H[j][2] . . . H[j](k), where k = min(m,n).
    

Each H[j][i] has the form
    
    
    H[j][i] = I - tau[j] * v * v'
    

where `tau[j]` is a real scalar, and `v` is a real vector with `v(1:i-1) = 0` and `v(i) = 1`; `v(i+1:m)` is stored on exit in `Aarray[j][i+1:m,i]`, and `tau` in `TauArray[j][i]`.

This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor.

cublas<t>geqrfBatched supports arbitrary dimension.

cublas<t>geqrfBatched only supports compute capability 2.0 or above.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`m` |  | input | Number of rows `Aarray[i]`.  
`n` |  | input | Number of columns of `Aarray[i]`.  
`Aarray` | device | input | Array of pointers to <_type_ > array, with each array of dim. `m x n` with `lda >= max(1, m)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `Aarray[i]`.  
`TauArray` | device | output | Array of pointers to <_type_ > vector, with each vector of dim. `max(1 ,min(m, n))`.  
`info` | host | output |  If `info == 0`, the parameters passed to the function are valid If `info < 0`, the parameter in position `-info` is invalid  
`batchSize` |  | input | Number of pointers contained in `Aarray`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `batchSize < 0`, or
  * if `lda < max(1, m)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sgeqrf()](http://www.netlib.org/lapack/single/sgeqrf.f), [dgeqrf()](http://www.netlib.org/lapack/double/dgeqrf.f), [cgeqrf()](http://www.netlib.org/lapack/complex/cgeqrf.f), [zgeqrf()](http://www.netlib.org/lapack/complex16/zgeqrf.f)

###  2.8.8. cublas<t>gelsBatched() 
    
    
    cublasStatus_t cublasSgelsBatched( cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int m,
                                       int n,
                                       int nrhs,
                                       float *const Aarray[],
                                       int lda,
                                       float *const Carray[],
                                       int ldc,
                                       int *info,
                                       int *devInfoArray,
                                       int batchSize );
    
    cublasStatus_t cublasDgelsBatched( cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int m,
                                       int n,
                                       int nrhs,
                                       double *const Aarray[],
                                       int lda,
                                       double *const Carray[],
                                       int ldc,
                                       int *info,
                                       int *devInfoArray,
                                       int batchSize );
    
    cublasStatus_t cublasCgelsBatched( cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int m,
                                       int n,
                                       int nrhs,
                                       cuComplex *const Aarray[],
                                       int lda,
                                       cuComplex *const Carray[],
                                       int ldc,
                                       int *info,
                                       int *devInfoArray,
                                       int batchSize );
    
    cublasStatus_t cublasZgelsBatched( cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int m,
                                       int n,
                                       int nrhs,
                                       cuDoubleComplex *const Aarray[],
                                       int lda,
                                       cuDoubleComplex *const Carray[],
                                       int ldc,
                                       int *info,
                                       int *devInfoArray,
                                       int batchSize );
    

`Aarray` is an array of pointers to matrices stored in column-major format. `Carray` is an array of pointers to matrices stored in column-major format.

This function find the least squares solution of a batch of overdetermined systems: it solves the least squares problem described as follows :
    
    
    minimize  || Carray[i] - Aarray[i]*Xarray[i] || , with i = 0, ...,batchSize-1
    

On exit, each `Aarray[i]` is overwritten with their QR factorization and each `Carray[i]` is overwritten with the least square solution

cublas<t>gelsBatched supports only the non-transpose operation and only solves over-determined systems (m >= n).

cublas<t>gelsBatched only supports compute capability 2.0 or above.

This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`trans` |  | input | Operation op(`Aarray[i]`) that is non- or (conj.) transpose. Only non-transpose operation is currently supported.  
`m` |  | input | Number of rows of each `Aarray[i]` and `Carray[i]` if `trans == CUBLAS_OP_N`, numbers of columns of each `Aarray[i]` otherwise (not supported currently).  
`n` |  | input | Number of columns of each `Aarray[i]` if `trans == CUBLAS_OP_N`, and number of rows of each `Aarray[i]` and `Carray[i]` otherwise (not supported currently).  
`nrhs` |  | input | Number of columns of each `Carray[i]`.  
`Aarray` | device | input/output | Array of pointers to <_type_ > array, with each array of dim. `m x n` with `lda >= max(1, m)` if `trans == CUBLAS_OP_N`, and `n x m` with `lda >= max(1, n)` otherwise (not supported currently). Matrices `Aarray[i]` should not overlap; otherwise, behavior is undefined.  
`lda` |  | input | Leading dimension of two-dimensional array used to store each matrix `Aarray[i]`.  
`Carray` | device | input/output | Array of pointers to <_type_ > array, with each array of dim. `m x nrhs` with `ldc >= max(1, m)` if `trans == CUBLAS_OP_N`, and `n x nrhs` with `lda >= max(1, n)` otherwise (not supported currently). Matrices `Carray[i]` should not overlap; otherwise, behavior is undefined.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store each matrix `Carray[i]`.  
`info` | host | output |  If `info == 0` the parameters passed to the function are valid If `info < 0` the parameter in position `-info` is invalid  
`devInfoArray` | device | output |  Optional array of integers of dimension batchsize. If non-null, every element of `devInfoArray[i] == V` has the following meaning: `V == 0` : the `i`-th problem was successfully solved `V > 0` : the `V`-th diagonal element of the `Aarray[i]` is zero. `Aarray[i]` does not have full rank.  
`batchSize` |  | input | Number of pointers contained in Aarray and Carray  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `nrhs < 0` or `batchSize < 0` or
  * if `lda < max(1, m)` or `ldc < max(1, m)`

  
`CUBLAS_STATUS_NOT_SUPPORTED` | The parameters `m <n` or `trans` is different from non-transpose.  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sgels()](http://www.netlib.org/lapack/single/sgels.f), [dgels()](http://www.netlib.org/lapack/double/dgels.f), [cgels()](http://www.netlib.org/lapack/complex/cgels.f), [zgels()](http://www.netlib.org/lapack/complex16/zgels.f)

###  2.8.9. cublas<t>tpttr() 
    
    
    cublasStatus_t cublasStpttr ( cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  const float *AP,
                                  float *A,
                                  int lda );
    
    cublasStatus_t cublasDtpttr ( cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  const double *AP,
                                  double *A,
                                  int lda );
    
    cublasStatus_t cublasCtpttr ( cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  const cuComplex *AP,
                                  cuComplex *A,
                                  int lda );
    
    cublasStatus_t cublasZtpttr ( cublasHandle_t handle,
                                  cublasFillMode_t uplo
                                  int n,
                                  const cuDoubleComplex *AP,
                                  cuDoubleComplex *A,
                                  int lda );
    

This function performs the conversion from the triangular packed format to the triangular format

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements of `AP` are copied into the lower triangular part of the triangular matrix `A` and the upper part of `A` is left untouched. If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements of `AP` are copied into the upper triangular part of the triangular matrix `A` and the lower part of `A` is left untouched.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `AP` contains lower or upper part of matrix `A`.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`AP` | device | input | <_type_ > array with \\(A\\) stored in packed format.  
`A` | device | output | <_type_ > array of dimensions `lda x n` , with `lda >= max(1, n)`. The opposite side of A is left untouched.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[stpttr()](http://www.netlib.org/lapack/explore-html/d7/d70/stpttr_8f.html), [dtpttr()](http://www.netlib.org/lapack/explore-html/df/d63/dtpttr_8f.html), [ctpttr()](http://www.netlib.org/lapack/explore-html/de/d13/ctpttr_8f.html), [ztpttr()](http://www.netlib.org/lapack/explore-html/d6/dbc/ztpttr_8f.html)

###  2.8.10. cublas<t>trttp() 
    
    
    cublasStatus_t cublasStrttp ( cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  const float *A,
                                  int lda,
                                  float *AP );
    
    cublasStatus_t cublasDtrttp ( cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  const double *A,
                                  int lda,
                                  double *AP );
    
    cublasStatus_t cublasCtrttp ( cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  const cuComplex *A,
                                  int lda,
                                  cuComplex *AP );
    
    cublasStatus_t cublasZtrttp ( cublasHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  cuDoubleComplex *AP );
    

This function performs the conversion from the triangular format to the triangular packed format

If `uplo == CUBLAS_FILL_MODE_LOWER` then the lower triangular part of the triangular matrix `A` is copied into the array `AP`. If `uplo == CUBLAS_FILL_MODE_UPPER` then then the upper triangular part of the triangular matrix `A` is copied into the array `AP`.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates which matrix `A` lower or upper part is referenced.  
`n` |  | input | Number of rows and columns of matrix `A`.  
`A` | device | input | <_type_ > array of dimensions `lda x n` , with `lda >= max(1, n)`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`AP` | device | output | <_type_ > array with `A` stored in packed format.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `lda < max(1, n)`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[strttp()](http://www.netlib.org/lapack/explore-html/d9/def/strttp_8f.html), [dtrttp()](http://www.netlib.org/lapack/explore-html/d0/daf/dtrttp_8f.html), [ctrttp()](http://www.netlib.org/lapack/explore-html/d7/d56/ctrttp_8f.html), [ztrttp()](http://www.netlib.org/lapack/explore-html/da/dc2/ztrttp_8f.html)

###  2.8.11. cublas<t>gemmEx() 
    
    
    cublasStatus_t cublasSgemmEx(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               int k,
                               const float    *alpha,
                               const void     *A,
                               cudaDataType_t Atype,
                               int lda,
                               const void     *B,
                               cudaDataType_t Btype,
                               int ldb,
                               const float    *beta,
                               void           *C,
                               cudaDataType_t Ctype,
                               int ldc)
    cublasStatus_t cublasCgemmEx(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               int k,
                               const cuComplex *alpha,
                               const void      *A,
                               cudaDataType_t  Atype,
                               int lda,
                               const void      *B,
                               cudaDataType_t  Btype,
                               int ldb,
                               const cuComplex *beta,
                               void            *C,
                               cudaDataType_t  Ctype,
                               int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an extension of [cublas<t>gemm()](#id8). In this function the input matrices and output matrices can have a lower precision but the computation is still done in the type `<t>`. For example, in the type `float` for [cublasSgemmEx()](#cublas-t-gemmex) and in the type `cuComplex` for [cublasCgemmEx()](#cublas-t-gemmex).

\\(C = \alpha\text{op}(A)\text{op}(B) + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are matrices stored in column-major format with dimensions \\(\text{op}(A)\\) \\(m \times k\\) , \\(\text{op}(B)\\) \\(k \times n\\) and \\(C\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B)\\) is defined similarly for matrix \\(B\\) .

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`transa` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`transb` |  | input | Operation op(`B`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix op(`A`) and `C`.  
`n` |  | input | Number of columns of matrix op(`B`) and `C`.  
`k` |  | input | Number of columns of op(`A`) and rows of op(`B`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimensions `lda x k` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, k)` otherwise.  
`Atype` |  | input | Enumerant specifying the datatype of matrix `A`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store the matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, k)` if `transb == CUBLAS_OP_N` and `ldb x k` with `ldb >= max(1,n)` otherwise.  
`Btype` |  | input | Enumerant specifying the datatype of matrix `B`.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0`, `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, m)`.  
`Ctype` |  | input | Enumerant specifying the datatype of matrix `C`.  
`ldc` |  | input | Leading dimension of a two-dimensional array used to store the matrix `C`.  
  
The matrix types combinations supported for [cublasSgemmEx()](#cublas-t-gemmex) are listed below:

C | A/B  
---|---  
`CUDA_R_16BF` | `CUDA_R_16BF`  
`CUDA_R_16F` | `CUDA_R_16F`  
`CUDA_R_32F` | `CUDA_R_8I`  
| `CUDA_R_16BF`  
| `CUDA_R_16F`  
| `CUDA_R_32F`  
  
The matrix types combinations supported for [cublasCgemmEx()](#cublas-t-gemmex) are listed below :

C | A/B  
---|---  
`CUDA_C_32F` | `CUDA_C_8I`  
| `CUDA_C_32F`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_ARCH_MISMATCH` | [cublasCgemmEx()](#cublas-t-gemmex) is only supported for GPU with architecture capabilities equal or greater than 5.0  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `Atype`, `Btype` and `Ctype` is not supported  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `k < 0`, or
  * if `transa` and `transb` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda < max(1, m)` when `transa == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, k)` when `transb == CUBLAS_OP_N` and `ldb < max(1, n)` otherwise, or
  * if `ldc < max(1, m)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sgemm()](http://www.netlib.org/blas/sgemm.f)

For more information about the numerical behavior of some GEMM algorithms, refer to the [GEMM Algorithms Numerical Behavior](#gemm-algorithms-numerical-behavior) section.

###  2.8.12. cublasGemmEx() 
    
    
    cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               int k,
                               const void    *alpha,
                               const void     *A,
                               cudaDataType_t Atype,
                               int lda,
                               const void     *B,
                               cudaDataType_t Btype,
                               int ldb,
                               const void    *beta,
                               void           *C,
                               cudaDataType_t Ctype,
                               int ldc,
                               cublasComputeType_t computeType,
                               cublasGemmAlgo_t algo)
    
    #if defined(__cplusplus)
    cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               int k,
                               const void     *alpha,
                               const void     *A,
                               cudaDataType   Atype,
                               int lda,
                               const void     *B,
                               cudaDataType   Btype,
                               int ldb,
                               const void     *beta,
                               void           *C,
                               cudaDataType   Ctype,
                               int ldc,
                               cudaDataType   computeType,
                               cublasGemmAlgo_t algo)
    #endif
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an extension of [cublas<t>gemm()](#id8) that allows the user to individually specify the data types for each of the A, B and C matrices, the precision of computation and the GEMM algorithm to be run. Supported combinations of arguments are listed further down in this section.

Note

The second variant of [cublasGemmEx()](#cublasgemmex) function is provided for backward compatibility with C++ applications code, where the `computeType` parameter is of `cudaDataType` instead of [cublasComputeType_t](#cublascomputetype-t). C applications would still compile with the updated function signature.

This function is only supported on devices with compute capability 5.0 or later.

\\(C = \alpha\text{op}(A)\text{op}(B) + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are matrices stored in column-major format with dimensions \\(\text{op}(A)\\) \\(m \times k\\) , \\(\text{op}(B)\\) \\(k \times n\\) and \\(C\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B)\\) is defined similarly for matrix \\(B\\) .

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`transa` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`transb` |  | input | Operation op(`B`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix op(`A`) and `C`.  
`n` |  | input | Number of columns of matrix op(`B`) and `C`.  
`k` |  | input | Number of columns of op(`A`) and rows of op(`B`).  
`alpha` | host or device | input | Scaling factor for A*B of the type that corresponds to the computeType and Ctype, see the table below for details.  
`A` | device | input | <_type_ > array of dimensions `lda x k` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, k)` otherwise.  
`Atype` |  | input | Enumerant specifying the datatype of matrix `A`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store the matrix `A`.  
`B` | device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, k)` if `transb == CUBLAS_OP_N` and `ldb x k` with `ldb >= max(1,n)` otherwise.  
`Btype` |  | input | Enumerant specifying the datatype of matrix `B`.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host or device | input | Scaling factor for C of the type that corresponds to the computeType and Ctype, see the table below for details. If `beta == 0`, `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, m)`.  
`Ctype` |  | input | Enumerant specifying the datatype of matrix `C`.  
`ldc` |  | input | Leading dimension of a two-dimensional array used to store the matrix `C`.  
`computeType` |  | input | Enumerant specifying the computation type.  
`algo` |  | input | Enumerant specifying the algorithm. See [cublasGemmAlgo_t](#cublasgemmalgo-t).  
  
[cublasGemmEx()](#cublasgemmex) supports the following Compute Type, Scale Type, Atype/Btype, and Ctype:

Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype  
---|---|---|---  
`CUBLAS_COMPUTE_16F` or `CUBLAS_COMPUTE_16F_PEDANTIC` | `CUDA_R_16F` | `CUDA_R_16F` | `CUDA_R_16F`  
`CUBLAS_COMPUTE_32I` or `CUBLAS_COMPUTE_32I_PEDANTIC` | `CUDA_R_32I` | `CUDA_R_8I` | `CUDA_R_32I`  
`CUBLAS_COMPUTE_32F` or `CUBLAS_COMPUTE_32F_PEDANTIC` | `CUDA_R_32F` | `CUDA_R_16BF` | `CUDA_R_16BF`  
`CUDA_R_16F` | `CUDA_R_16F`  
`CUDA_R_8I` | `CUDA_R_32F`  
`CUDA_R_16BF` | `CUDA_R_32F`  
`CUDA_R_16F` | `CUDA_R_32F`  
`CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_C_32F` | `CUDA_C_8I` | `CUDA_C_32F`  
`CUDA_C_32F` | `CUDA_C_32F`  
`CUBLAS_COMPUTE_32F_FAST_16F` or `CUBLAS_COMPUTE_32F_FAST_16BF` or `CUBLAS_COMPUTE_32F_FAST_TF32` or `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` | `CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_C_32F` | `CUDA_C_32F` | `CUDA_C_32F`  
`CUBLAS_COMPUTE_64F` or `CUBLAS_COMPUTE_64F_PEDANTIC` | `CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
`CUDA_C_64F` | `CUDA_C_64F` | `CUDA_C_64F`  
  
Note

`CUBLAS_COMPUTE_32I` and `CUBLAS_COMPUTE_32I_PEDANTIC` compute types are only supported with A, B being 4-byte aligned and lda, ldb being multiples of 4. For better performance, it is also recommended that IMMA kernels requirements for a regular data ordering listed [here](#cublasltmatmul-regular-imma-conditions) are met.

The possible error values returned by this function and their meanings are listed in the following table.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_ARCH_MISMATCH` | [cublasGemmEx()](#cublasgemmex) is only supported for GPU with architecture capabilities equal or greater than 5.0.  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `Atype`, `Btype` and `Ctype` or the algorithm, `algo` is not supported.  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `k < 0`, or
  * if `transa` and `transb` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda < max(1, m)` when `transa == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, k)` when `transb == CUBLAS_OP_N` and `ldb < max(1, n)` otherwise, or
  * if `ldc < max(1, m)`, or
  * if `alpha` or `beta` are NULL, or
  * if `C` is NULL when `beta` is not zero
  * if `Atype` or `Btype` or `Ctype` or `algo` are not supported

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU.  
  
Starting with release 11.2, using the typed functions instead of the extension functions (cublas**Ex()) helps in reducing the binary size when linking to static cuBLAS Library.

Also refer to: [sgemm.()](http://www.netlib.org/blas/sgemm.f)

For more information about the numerical behavior of some GEMM algorithms, refer to the [GEMM Algorithms Numerical Behavior](#gemm-algorithms-numerical-behavior) section.

###  2.8.13. cublasGemmBatchedEx() 
    
    
    cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const void    *alpha,
                                const void     *const Aarray[],
                                cudaDataType_t Atype,
                                int lda,
                                const void     *const Barray[],
                                cudaDataType_t Btype,
                                int ldb,
                                const void    *beta,
                                void           *const Carray[],
                                cudaDataType_t Ctype,
                                int ldc,
                                int batchCount,
                                cublasComputeType_t computeType,
                                cublasGemmAlgo_t algo)
    
    #if defined(__cplusplus)
    cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const void     *alpha,
                                const void     *const Aarray[],
                                cudaDataType   Atype,
                                int lda,
                                const void     *const Barray[],
                                cudaDataType   Btype,
                                int ldb,
                                const void     *beta,
                                void           *const Carray[],
                                cudaDataType   Ctype,
                                int ldc,
                                int batchCount,
                                cudaDataType   computeType,
                                cublasGemmAlgo_t algo)
    #endif
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an extension of [cublas<t>gemmBatched()](#id9) that performs the matrix-matrix multiplication of a batch of matrices and allows the user to individually specify the data types for each of the A, B and C matrix arrays, the precision of computation and the GEMM algorithm to be run. Like [cublas<t>gemmBatched()](#id9), the batch is considered to be “uniform”, i.e. all instances have the same dimensions (m, n, k), leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. The address of the input matrices and the output matrix of each instance of the batch are read from arrays of pointers passed to the function by the caller. Supported combinations of arguments are listed further down in this section.

Note

The second variant of [cublasGemmBatchedEx()](#cublasgemmbatchedex) function is provided for backward compatibility with C++ applications code, where the `computeType` parameter is of `cudaDataType` instead of [cublasComputeType_t](#cublascomputetype-t). C applications would still compile with the updated function signature.

\\(C\lbrack i\rbrack = \alpha\text{op}(A\lbrack i\rbrack)\text{op}(B\lbrack i\rbrack) + \beta C\lbrack i\rbrack,\text{ for i } \in \lbrack 0,batchCount - 1\rbrack\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are arrays of pointers to matrices stored in column-major format with dimensions \\(\text{op}(A\lbrack i\rbrack)\\) \\(m \times k\\) , \\(\text{op}(B\lbrack i\rbrack)\\) \\(k \times n\\) and \\(C\lbrack i\rbrack\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B\lbrack i\rbrack)\\) is defined similarly for matrix \\(B\lbrack i\rbrack\\) .

Note

\\(C\lbrack i\rbrack\\) matrices must not overlap, i.e. the individual gemm operations must be computable independently; otherwise, behavior is undefined.

On certain problem sizes, it might be advantageous to make multiple calls to [cublas<t>gemm()](#id8) in different CUDA streams, rather than use this API.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`transa` |  | input | Operation op(`Aarray[i]`) that is non- or (conj.) transpose.  
`transb` |  | input | Operation op(`Barray[i]`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix op(`Aarray[i]`) and `Carray[i]`.  
`n` |  | input | Number of columns of matrix op(`Barray[i]`) and `Carray[i]`.  
`k` |  | input | Number of columns of op(`Aarray[i]`) and rows of op(`Barray[i]`).  
`alpha` | host or device | input | Scaling factor for matrix products of the type that corresponds to the computeType and Ctype, see the table below for details.  
`Aarray` | device | input |  Array of pointers to <_Atype_ > array, with each array of dim. `lda x k` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, k)` otherwise. All pointers must meet certain alignment criteria. Please see below for details.  
`Atype` |  | input | Enumerant specifying the datatype of `Aarray`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store the matrix `Aarray[i]`.  
`Barray` | device | input |  Array of pointers to <_Btype_ > array, with each array of dim. `ldb x n` with `ldb >= max(1, k)` if `transb == CUBLAS_OP_N` and `ldb x k` with `ldb >= max(1,n)` otherwise. All pointers must meet certain alignment criteria. Please see below for details.  
`Btype` |  | input | Enumerant specifying the datatype of `Barray`.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `Barray[i]`.  
`beta` | host or device | input | Scaling factor for `Carray` of the type that corresponds to the computeType and Ctype, see the table below for details. If `beta == 0`, `Carray[i]` does not have to be a valid input.  
`Carray` | device | in/out |  Array of pointers to <_Ctype_ > array. It has dimensions `ldc x n` with `ldc >= max(1, m)`. Matrices `Carray[i]` should not overlap; otherwise, the behavior is undefined. All pointers must meet certain alignment criteria. Please see below for details.  
`Ctype` |  | input | Enumerant specifying the datatype of `Carray`.  
`ldc` |  | input | Leading dimension of a two-dimensional array used to store each matrix `Carray[i]`.  
`batchCount` |  | input | Number of pointers contained in `Aarray`, `Barray` and `Carray`.  
`computeType` |  | input | Enumerant specifying the computation type.  
`algo` |  | input | Enumerant specifying the algorithm. See [cublasGemmAlgo_t](#cublasgemmalgo-t).  
  
[cublasGemmBatchedEx()](#cublasgemmbatchedex) supports the following Compute Type, Scale Type, Atype/Btype, and Ctype:

Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype  
---|---|---|---  
`CUBLAS_COMPUTE_16F` or `CUBLAS_COMPUTE_16F_PEDANTIC` | `CUDA_R_16F` | `CUDA_R_16F` | `CUDA_R_16F`  
`CUBLAS_COMPUTE_32I` or `CUBLAS_COMPUTE_32I_PEDANTIC` | `CUDA_R_32I` | `CUDA_R_8I` | `CUDA_R_32I`  
`CUBLAS_COMPUTE_32F` or `CUBLAS_COMPUTE_32F_PEDANTIC` | `CUDA_R_32F` | `CUDA_R_16BF` | `CUDA_R_16BF`  
`CUDA_R_16F` | `CUDA_R_16F`  
`CUDA_R_8I` | `CUDA_R_32F`  
`CUDA_R_16BF` | `CUDA_R_32F`  
`CUDA_R_16F` | `CUDA_R_32F`  
`CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_C_32F` | `CUDA_C_8I` | `CUDA_C_32F`  
`CUDA_C_32F` | `CUDA_C_32F`  
`CUBLAS_COMPUTE_32F_FAST_16F` or `CUBLAS_COMPUTE_32F_FAST_16BF` or `CUBLAS_COMPUTE_32F_FAST_TF32` or `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` | `CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_C_32F` | `CUDA_C_32F` | `CUDA_C_32F`  
`CUBLAS_COMPUTE_64F` or `CUBLAS_COMPUTE_64F_PEDANTIC` | `CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
`CUDA_C_64F` | `CUDA_C_64F` | `CUDA_C_64F`  
  
If `Atype` is `CUDA_R_16F` or `CUDA_R_16BF`, or `computeType` is any of the `FAST` options, or when math mode or `algo` enable fast math modes, pointers (not the pointer arrays) placed in the GPU memory must be properly aligned to avoid misaligned memory access errors. Ideally all pointers are aligned to at least 16 Bytes. Otherwise it is recommended that they meet the following rule:

  * if `k % 8 == 0` then ensure `intptr_t(ptr) % 16 == 0`,

  * if `k % 2 == 0` then ensure `intptr_t(ptr) % 4 == 0`.


Note

Compute types `CUBLAS_COMPUTE_32I` and `CUBLAS_COMPUTE_32I_PEDANTIC` are only supported with all pointers `A[i]`, `B[i]` being 4-byte aligned and lda, ldb being multiples of 4. For a better performance, it is also recommended that IMMA kernels requirements for the regular data ordering listed [here](#cublasltmatmul-regular-imma-conditions) are met.

The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_ARCH_MISMATCH` | [cublasGemmBatchedEx()](#cublasgemmbatchedex) is only supported for GPU with architecture capabilities equal to or greater than 5.0.  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `Atype`, `Btype` and `Ctype` or the algorithm, `algo` is not supported.  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `k < 0`, or
  * if `transa` and `transb` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda < max(1, m)` when `transa == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, k)` when `transb == CUBLAS_OP_N` and `ldb < max(1, n)` otherwise, or
  * if `ldc < max(1, m)`, or
  * if `alpha` or `beta` are NULL, or
  * if `Atype` or `Btype` or `Ctype` or `algo` or `computeType` is not supported

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU.  
  
Also refer to: [sgemm.()](http://www.netlib.org/blas/sgemm.f)

###  2.8.14. cublasGemmStridedBatchedEx() 
    
    
    cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const void    *alpha,
                                const void     *A,
                                cudaDataType_t Atype,
                                int lda,
                                long long int strideA,
                                const void     *B,
                                cudaDataType_t Btype,
                                int ldb,
                                long long int strideB,
                                const void    *beta,
                                void           *C,
                                cudaDataType_t Ctype,
                                int ldc,
                                long long int strideC,
                                int batchCount,
                                cublasComputeType_t computeType,
                                cublasGemmAlgo_t algo)
    
    #if defined(__cplusplus)
    cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const void    *alpha,
                                const void     *A,
                                cudaDataType Atype,
                                int lda,
                                long long int strideA,
                                const void     *B,
                                cudaDataType Btype,
                                int ldb,
                                long long int strideB,
                                const void    *beta,
                                void           *C,
                                cudaDataType Ctype,
                                int ldc,
                                long long int strideC,
                                int batchCount,
                                cudaDataType computeType,
                                cublasGemmAlgo_t algo)
    #endif
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an extension of [cublas<t>gemmStridedBatched()](#id10) that performs the matrix-matrix multiplication of a batch of matrices and allows the user to individually specify the data types for each of the A, B and C matrices, the precision of computation and the GEMM algorithm to be run. Like [cublas<t>gemmStridedBatched()](#id10), the batch is considered to be “uniform”, i.e. all instances have the same dimensions (m, n, k), leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. Input matrices A, B and output matrix C for each instance of the batch are located at fixed offsets in number of elements from their locations in the previous instance. Pointers to A, B and C matrices for the first instance are passed to the function by the user along with the offsets in number of elements - strideA, strideB and strideC that determine the locations of input and output matrices in future instances.

Note

The second variant of [cublasGemmStridedBatchedEx()](#cublasgemmstridedbatchedex) function is provided for backward compatibility with C++ applications code, where the `computeType` parameter is of [cudaDataType_t](#cudadatatype-t) instead of [cublasComputeType_t](#cublascomputetype-t). C applications would still compile with the updated function signature.

\\(C + i*{strideC} = \alpha\text{op}(A + i*{strideA})\text{op}(B + i*{strideB}) + \beta(C + i*{strideC}),\text{ for i } \in \lbrack 0,batchCount - 1\rbrack\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are arrays of pointers to matrices stored in column-major format with dimensions \\(\text{op}(A\lbrack i\rbrack)\\) \\(m \times k\\) , \\(\text{op}(B\lbrack i\rbrack)\\) \\(k \times n\\) and \\(C\lbrack i\rbrack\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B\lbrack i\rbrack)\\) is defined similarly for matrix \\(B\lbrack i\rbrack\\) .

Note

\\(C\lbrack i\rbrack\\) matrices must not overlap, i.e. the individual gemm operations must be computable independently; otherwise, the behavior is undefined.

On certain problem sizes, it might be advantageous to make multiple calls to [cublas<t>gemm()](#id8) in different CUDA streams, rather than use this API.

Note

In the table below, we use `A[i], B[i], C[i]` as notation for A, B and C matrices in the ith instance of the batch, implicitly assuming they are respectively offsets in number of elements `strideA, strideB, strideC` away from `A[i-1], B[i-1], C[i-1]`. The unit for the offset is number of elements and must not be zero .

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`transa` |  | input | Operation op(`A[i]`) that is non- or (conj.) transpose.  
`transb` |  | input | Operation op(`B[i]`) that is non- or (conj.) transpose.  
`m` |  | input | Number of rows of matrix op(`A[i]`) and `C[i]`.  
`n` |  | input | Number of columns of matrix op(`B[i]`) and `C[i]`.  
`k` |  | input | Number of columns of op(`A[i]`) and rows of op(`B[i]`).  
`alpha` | host or device | input | Scaling factor for A*B of the <_Scale Type_ > that corresponds to the computeType and Ctype, see the table below for details.  
`A` | device | input | Pointer to <_Atype_ > matrix, A, corresponds to the first instance of the batch, with dimensions `lda x k` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, k)` otherwise.  
`Atype` |  | input | Enumerant specifying the datatype of `A`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store the matrix `A[i]`.  
`strideA` |  | input | Value of type long long int that gives the offset in number of elements between `A[i]` and `A[i+1]`.  
`B` | device | input | Pointer to <_Btype_ > matrix, B, corresponds to the first instance of the batch, with dimensions `ldb x n` with `ldb >= max(1, k)` if `transb == CUBLAS_OP_N` and `ldb x k` with `ldb >= max(1,n)` otherwise.  
`Btype` |  | input | Enumerant specifying the datatype of `B`.  
`ldb` |  | input | Leading dimension of two-dimensional array used to store matrix `B[i]`.  
`strideB` |  | input | Value of type long long int that gives the offset in number of elements between `B[i]` and `B[i+1]`.  
`beta` | host or device | input | Scaling factor for C of the <_Scale Type_ > that corresponds to the computeType and Ctype, see the table below for details. If `beta == 0`, `C[i]` does not have to be a valid input.  
`C` | device | in/out | Pointer to <_Ctype_ > matrix, C, corresponds to the first instance of the batch, with dimensions `ldc x n` with `ldc >= max(1, m)`. Matrices `C[i]` should not overlap; otherwise, undefined behavior is expected.  
`Ctype` |  | input | Enumerant specifying the datatype of `C`.  
`ldc` |  | input | Leading dimension of a two-dimensional array used to store each matrix `C[i]`.  
`strideC` |  | input | Value of type long long int that gives the offset in number of elements between `C[i]` and `C[i+1]`.  
`batchCount` |  | input | Number of GEMMs to perform in the batch.  
`computeType` |  | input | Enumerant specifying the computation type.  
`algo` |  | input | Enumerant specifying the algorithm. See [cublasGemmAlgo_t](#cublasgemmalgo-t).  
  
[cublasGemmStridedBatchedEx()](#cublasgemmstridedbatchedex) supports the following Compute Type, Scale Type, Atype/Btype, and Ctype:

Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype  
---|---|---|---  
`CUBLAS_COMPUTE_16F` or `CUBLAS_COMPUTE_16F_PEDANTIC` | `CUDA_R_16F` | `CUDA_R_16F` | `CUDA_R_16F`  
`CUBLAS_COMPUTE_32I` or `CUBLAS_COMPUTE_32I_PEDANTIC` | `CUDA_R_32I` | `CUDA_R_8I` | `CUDA_R_32I`  
`CUBLAS_COMPUTE_32F` or `CUBLAS_COMPUTE_32F_PEDANTIC` | `CUDA_R_32F` | `CUDA_R_16BF` | `CUDA_R_16BF`  
`CUDA_R_16F` | `CUDA_R_16F`  
`CUDA_R_8I` | `CUDA_R_32F`  
`CUDA_R_16BF` | `CUDA_R_32F`  
`CUDA_R_16F` | `CUDA_R_32F`  
`CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_C_32F` | `CUDA_C_8I` | `CUDA_C_32F`  
`CUDA_C_32F` | `CUDA_C_32F`  
`CUBLAS_COMPUTE_32F_FAST_16F` or `CUBLAS_COMPUTE_32F_FAST_16BF` or `CUBLAS_COMPUTE_32F_FAST_TF32` or `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` | `CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_C_32F` | `CUDA_C_32F` | `CUDA_C_32F`  
`CUBLAS_COMPUTE_64F` or `CUBLAS_COMPUTE_64F_PEDANTIC` | `CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
`CUDA_C_64F` | `CUDA_C_64F` | `CUDA_C_64F`  
  
Note

Compute types `CUBLAS_COMPUTE_32I` and `CUBLAS_COMPUTE_32I_PEDANTIC` are only supported with all pointers `A[i]`, `B[i]` being 4-byte aligned and lda, ldb being multiples of 4. For a better performance, it is also recommended that IMMA kernels requirements for the regular data ordering listed [here](#cublasltmatmul-regular-imma-conditions) are met.

The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_ARCH_MISMATCH` | [cublasGemmBatchedEx()](#cublasgemmbatchedex) is only supported for GPU with architecture capabilities equal or greater than 5.0.  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `Atype`, `Btype` and `Ctype` or the algorithm, `algo` is not supported.  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `m < 0` or `n < 0` or `k < 0`, or
  * if `transa` and `transb` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda < max(1, m)` when `transa == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldb < max(1, k)` when `transb == CUBLAS_OP_N` and `ldb < max(1, n)` otherwise, or
  * if `ldc < max(1, m)`, or
  * if `alpha` or `beta` are NULL, or
  * if `Atype` or `Btype` or `Ctype` or `algo` or `computeType` is not supported

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
Also refer to: [sgemm.()](http://www.netlib.org/blas/sgemm.f)

###  2.8.15. cublasGemmGroupedBatchedEx() 
    
    
    cublasStatus_t cublasGemmGroupedBatchedEx(cublasHandle_t handle,
                                const cublasOperation_t transa_array[],
                                const cublasOperation_t transb_array[],
                                const int m_array[],
                                const int n_array[],
                                const int k_array[],
                                const void    *alpha_array,
                                const void     *const Aarray[],
                                cudaDataType_t Atype,
                                const int lda_array[],
                                const void     *const Barray[],
                                cudaDataType_t Btype,
                                const int ldb_array[],
                                const void    *beta_array,
                                void           *const Carray[],
                                cudaDataType_t Ctype,
                                const int ldc_array[],
                                int group_count,
                                const int group_size[],
                                cublasComputeType_t computeType)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function performs the matrix-matrix multiplication on groups of matrices. A given group is considered to be “uniform”, i.e. all instances have the same dimensions (m, n, k), leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. However, the dimensions, leading dimensions, transpositions, and scaling factors (alpha, beta) may vary between groups. The address of the input matrices and the output matrix of each instance of the batch are read from arrays of pointers passed to the function by the caller. This is functionally equivalent to the following:
    
    
    idx = 0;
    for i = 0:group_count - 1
        for j = 0:group_size[i] - 1
            gemmEx(transa_array[i], transb_array[i], m_array[i], n_array[i], k_array[i],
                   alpha_array[i], Aarray[idx], Atype, lda_array[i], Barray[idx], Btype,
                   ldb_array[i], beta_array[i], Carray[idx], Ctype, ldc_array[i],
                   computeType, CUBLAS_GEMM_DEFAULT);
            idx += 1;
        end
    end
    

where \\(\text{$\mathrm{alpha\\_array}$}\\) and \\(\text{$\mathrm{beta\\_array}$}\\) are arrays of scaling factors, and \\(\text{Aarray}\\), \\(\text{Barray}\\) and \\(\text{Carray}\\) are arrays of pointers to matrices stored in column-major format. For a given index, \\(\text{idx}\\), that is part of group \\(i\\), the dimensions are:

>   * \\(\text{op}(\text{Aarray}\lbrack\text{idx}\rbrack)\\): \\(\text{$\mathrm{m\\_array}$}\lbrack i\rbrack \times \text{$\mathrm{k\\_array}$}\lbrack i\rbrack\\)
> 
>   * \\(\text{op}(\text{Barray}\lbrack\text{idx}\rbrack)\\): \\(\text{$\mathrm{k\\_array}$}\lbrack i\rbrack \times \text{$\mathrm{n\\_array}$}\lbrack i\rbrack\\)
> 
>   * \\(\text{Carray}\lbrack\text{idx}\rbrack\\): \\(\text{$\mathrm{m\\_array}$}\lbrack i\rbrack \times \text{$\mathrm{n\\_array}$}\lbrack i\rbrack\\)
> 
> 


Note

This API takes arrays of two different lengths. The arrays of dimensions, leading dimensions, transpositions, and scaling factors are of length `group_count` and the arrays of matrices are of length `problem_count` where \\(\text{$\mathrm{problem\\_count}$} = \sum_{i = 0}^{\text{$\mathrm{group\\_count}$} - 1} \text{$\mathrm{group\\_size}$}\lbrack i\rbrack\\)

For matrix \\(A[\text{idx}]\\) in group \\(i\\)

\\(\text{op}(A[\text{idx}]) = \left\\{ \begin{matrix} A[\text{idx}] & {\text{if }\textsf{$\mathrm{transa\\_array}\lbrack i\rbrack$ == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A[\text{idx}]^{T} & {\text{if }\textsf{$\mathrm{transa\\_array}\lbrack i\rbrack$ == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A[\text{idx}]^{H} & {\text{if }\textsf{$\mathrm{transa\\_array}\lbrack i\rbrack$ == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B[\text{idx}])\\) is defined similarly for matrix \\(B[\text{idx}]\\) in group \\(i\\).

Note

\\(C\lbrack\text{idx}\rbrack\\) matrices must not overlap, that is, the individual gemm operations must be computable independently; otherwise, undefined behavior is expected.

On certain problem sizes, it might be advantageous to make multiple calls to [cublasGemmBatchedEx()](#cublasgemmbatchedex) in different CUDA streams, rather than use this API.

Param. | Memory | In/out | Meaning | Array Length  
---|---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context. |   
`transa_array` | host | input | Array containing the operations, op(`A[idx]`), that is non- or (conj.) transpose for each group. | group_count  
`transb_array` | host | input | Array containing the operations, op(`B[idx]`), that is non- or (conj.) transpose for each group. | group_count  
`m_array` | host | input | Array containing the number of rows of matrix op(`A[idx]`) and `C[idx]` for each group. | group_count  
`n_array` | host | input | Array containing the number of columns of op(`B[idx]`) and `C[idx]` for each group. | group_count  
`k_array` | host | input | Array containing the number of columns of op(`A[idx]`) and rows of op(`B[idx]`) for each group. | group_count  
`alpha_array` | host | input | Array containing the <_Scale Type_ > scalar used for multiplication for each group. | group_count  
`Aarray` | device | input |  Array of pointers to <_Atype_ > array, with each array of dim. `lda[i] x k[i]` with `lda[i] >= max(1,m[i])` if `transa[i] == CUBLAS_OP_N` and `lda[i] x m[i]` with `lda[i] >= max(1,k[i])` otherwise. All pointers must meet certain alignment criteria. Please see below for details. | problem_count  
`Atype` |  | input | Enumerant specifying the datatype of `A`. |   
`lda_array` | host | input | Array containing the leading dimensions of two-dimensional arrays used to store each matrix `A[idx]` for each group. | group_count  
`Barray` | device | input |  Array of pointers to <_Btype_ > array, with each array of dim. `ldb[i] x n[i]` with `ldb[i] >= max(1,k[i])` if `transb[i] == CUBLAS_OP_N` and `ldb[i] x k[i]` with `ldb[i] >= max(1,n[i])` otherwise. All pointers must meet certain alignment criteria. Please see below for details. | problem_count  
`Btype` |  | input | Enumerant specifying the datatype of `B`. |   
`ldb_array` | host | input | Array containing the leading dimensions of two-dimensional arrays used to store each matrix `B[idx]` for each group. | group_count  
`beta_array` | host | input | Array containing the <_Scale Type_ > scalar used for multiplication for each group. | group_count  
`Carray` | device | in/out |  Array of pointers to <_Ctype_ > array. It has dimensions `ldc[i] x n[i]` with `ldc[i] >= max(1,m[i])`. Matrices `C[idx]` should not overlap; otherwise, undefined behavior is expected. All pointers must meet certain alignment criteria. Please see below for details. | problem_count  
`Ctype` |  | input | Enumerant specifying the datatype of `C`. |   
`ldc_array` | host | input | Array containing the leading dimensions of two-dimensional arrays used to store each matrix `C[idx]` for each group. | group_count  
`group_count` | host | input | Number of groups |   
`group_size` | host | input | Array containing the number of pointers contained in Aarray, Barray and Carray for each group. | group_count  
`computeType` |  | input | Enumerant specifying the computation type. |   
  
[cublasGemmGroupedBatchedEx()](#cublasgemmgroupedbatchedex) supports the following Compute Type, Scale Type, Atype/Btype, and Ctype:

Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype  
---|---|---|---  
`CUBLAS_COMPUTE_32F` | `CUDA_R_32F` | `CUDA_R_16BF` | `CUDA_R_16BF`  
`CUDA_R_16F` | `CUDA_R_16F`  
`CUDA_R_32F` | `CUDA_R_32F`  
`CUBLAS_COMPUTE_32F_PEDANTIC` | `CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUBLAS_COMPUTE_32F_FAST_TF32` | `CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUBLAS_COMPUTE_64F` or `CUBLAS_COMPUTE_64F_PEDANTIC` | `CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
  
If `Atype` is `CUDA_R_16F` or `CUDA_R_16BF` or if the `computeType` is any of the `FAST` options, pointers (not the pointer arrays) placed in the GPU memory must be properly aligned to avoid misaligned memory access errors. Ideally all pointers are aligned to at least 16 Bytes. Otherwise it is required that they meet the following rule:

  * if `(k * AtypeSize) % 16 == 0` then ensure `intptr_t(ptr) % 16 == 0`,

  * if `(k * AtypeSize) % 4 == 0` then ensure `intptr_t(ptr) % 4 == 0`.


The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `transa_array`, `transb_array`, `m_array`, `n_array`, `k_array`, `alpha_array`, `lda_array`, `ldb_array`, `beta_array`, `ldc_array`, or `group_size` are NULL, or
  * if `group_count < 0`, or
  * if `m_array[i] < 0`, `n_array[i] < 0`, `k_array[i] < 0`, `group_size[i] < 0`, or
  * if `transa_array[i]` and `transb_array[i]` are not one of `CUBLAS_OP_N`, `CUBLAS_OP_C`, `CUBLAS_OP_T`, or
  * if `lda_array[i] < max(1, m_array[i])` if `transa_array[i] == CUBLAS_OP_N` and `lda_array[i] < max(1, k_array[i])` otherwise, or
  * if `ldb_array[i] < max(1, k_array[i])` if `transb_array[i] == CUBLAS_OP_N` and `ldb_array[i] < max(1, n_array[i])` otherwise, or
  * if `ldc_array[i] < max(1, m_array[i])`

  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_NOT_SUPPORTED` | 

  * the pointer mode is set to `CUBLAS_POINTER_MODE_DEVICE`
  * `Atype` or `Btype` or `Ctype` or `computeType` are not supported

  
  
###  2.8.16. cublasCsyrkEx() 
    
    
    cublasStatus_t cublasCsyrkEx(cublasHandle_t handle,
                                 cublasFillMode_t uplo,
                                 cublasOperation_t trans,
                                 int n,
                                 int k,
                                 const cuComplex *alpha,
                                 const void      *A,
                                 cudaDataType    Atype,
                                 int lda,
                                 const cuComplex *beta,
                                 cuComplex       *C,
                                 cudaDataType    Ctype,
                                 int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an extension of [cublasCsyrk()](#cublas-t-syrk) where the input matrix and output matrix can have a lower precision but the computation is still done in the type `cuComplex`

This function performs the symmetric rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(A)^{T} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a symmetric matrix stored in lower or upper mode, and \\(A\\) is a matrix with dimensions \\(\text{op}(A)\\) \\(n \times k\\) . Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ \end{matrix} \right.\\)

Note

This routine is only supported on GPUs with architecture capabilities equal to or greater than 5.0

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or transpose.  
`n` |  | input | Number of rows of matrix op(`A`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `trans == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`Atype` |  | input | Enumerant specifying the datatype of matrix `A`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix A.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`.  
`Ctype` |  | input | Enumerant specifying the datatype of matrix `C`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The matrix types combinations supported for [cublasCsyrkEx()](#cublascsyrkex) are listed below:

A | C  
---|---  
`CUDA_C_8I` | `CUDA_C_32F`  
`CUDA_C_32F` | `CUDA_C_32F`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `lda < max(1, n)` if `trans == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `Atype` or `Ctype` are not supported

  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `Atype` and `Ctype` is not supported.  
`CUBLAS_STATUS_ARCH_MISMATCH` | The device has a compute capability lower than 5.0.  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU.  
  
For references please refer to NETLIB documentation:

[ssyrk()](http://www.netlib.org/blas/ssyrk.f), [dsyrk()](http://www.netlib.org/blas/dsyrk.f), [csyrk()](http://www.netlib.org/blas/csyrk.f), [zsyrk()](http://www.netlib.org/blas/zsyrk.f)

###  2.8.17. cublasCsyrk3mEx() 
    
    
    cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle,
                                   cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   int n,
                                   int k,
                                   const cuComplex *alpha,
                                   const void      *A,
                                   cudaDataType    Atype,
                                   int lda,
                                   const cuComplex *beta,
                                   cuComplex       *C,
                                   cudaDataType    Ctype,
                                   int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an extension of [cublasCsyrk()](#cublas-t-syrk) where the input matrix and output matrix can have a lower precision but the computation is still done in the type `cuComplex`. This routine is implemented using the Gauss complexity reduction algorithm which can lead to an increase in performance up to 25%

This function performs the symmetric rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(A)^{T} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a symmetric matrix stored in lower or upper mode, and \\(A\\) is a matrix with dimensions \\(\text{op}(A)\\) \\(n \times k\\) . Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ \end{matrix} \right.\\)

Note

This routine is only supported on GPUs with architecture capabilities equal to or greater than 5.0

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`trans` |  | input | Operation op(`A`) that is non- or transpose.  
`n` |  | input | Number of rows of matrix op(`A`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `trans == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`Atype` |  | input | Enumerant specifying the datatype of matrix `A`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix A.  
`beta` | host or device | input | <_type_ > scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`.  
`Ctype` |  | input | Enumerant specifying the datatype of matrix `C`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The matrix types combinations supported for [cublasCsyrk3mEx()](#cublascsyrk3mex) are listed below :

A | C  
---|---  
`CUDA_C_8I` | `CUDA_C_32F`  
`CUDA_C_32F` | `CUDA_C_32F`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `lda < max(1, n)` if `trans == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `Atype` or `Ctype` are not supported

  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `Atype` and `Ctype` is not supported.  
`CUBLAS_STATUS_ARCH_MISMATCH` | The device has a compute capability lower than 5.0.  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU.  
  
For references please refer to NETLIB documentation:

[ssyrk()](http://www.netlib.org/blas/ssyrk.f), [dsyrk()](http://www.netlib.org/blas/dsyrk.f), [csyrk()](http://www.netlib.org/blas/csyrk.f), [zsyrk()](http://www.netlib.org/blas/zsyrk.f)

###  2.8.18. cublasCherkEx() 
    
    
    cublasStatus_t cublasCherkEx(cublasHandle_t handle,
                               cublasFillMode_t uplo,
                               cublasOperation_t trans,
                               int n,
                               int k,
                               const float     *alpha,
                               const void      *A,
                               cudaDataType    Atype,
                               int lda,
                               const float    *beta,
                               cuComplex      *C,
                               cudaDataType   Ctype,
                               int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an extension of [cublasCherk()](#cublas-t-herk) where the input matrix and output matrix can have a lower precision but the computation is still done in the type `cuComplex`

This function performs the Hermitian rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(A)^{H} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a Hermitian matrix stored in lower or upper mode, and \\(A\\) is a matrix with dimensions \\(\text{op}(A)\\) \\(n \times k\\) . Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Note

This routine is only supported on GPUs with architecture capabilities equal to or greater than 5.0

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part is stored, the other Hermitian part is not referenced.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`n` |  | input | Number of rows of matrix op(`A`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`Atype` |  | input | Enumerant specifying the datatype of matrix `A`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`beta` |  | input | <_type_ > scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`Ctype` |  | input | Enumerant specifying the datatype of matrix `C`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The matrix types combinations supported for [cublasCherkEx()](#cublascherkex) are listed in the following table:

A | C  
---|---  
`CUDA_C_8I` | `CUDA_C_32F`  
`CUDA_C_32F` | `CUDA_C_32F`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `lda < max(1, n)` if `trans == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `Atype` or `Ctype` are not supported

  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `Atype` and `Ctype` is not supported.  
`CUBLAS_STATUS_ARCH_MISMATCH` | The device has a compute capability lower than 5.0.  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU.  
  
For references please refer to NETLIB documentation:

[cherk()](http://www.netlib.org/blas/cherk.f)

###  2.8.19. cublasCherk3mEx() 
    
    
    cublasStatus_t cublasCherk3mEx(cublasHandle_t handle,
                               cublasFillMode_t uplo,
                               cublasOperation_t trans,
                               int n,
                               int k,
                               const float     *alpha,
                               const void      *A,
                               cudaDataType    Atype,
                               int lda,
                               const float    *beta,
                               cuComplex      *C,
                               cudaDataType   Ctype,
                               int ldc)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an extension of [cublasCherk()](#cublas-t-herk) where the input matrix and output matrix can have a lower precision but the computation is still done in the type `cuComplex`. This routine is implemented using the Gauss complexity reduction algorithm which can lead to an increase in performance up to 25%

This function performs the Hermitian rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(A)^{H} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a Hermitian matrix stored in lower or upper mode, and \\(A\\) is a matrix with dimensions \\(\text{op}(A)\\) \\(n \times k\\) . Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Note

This routine is only supported on GPUs with architecture capabilities equal to or greater than 5.0

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`uplo` |  | input | Indicates if matrix `C` lower or upper part is stored, the other Hermitian part is not referenced.  
`trans` |  | input | Operation op(`A`) that is non- or (conj.) transpose.  
`n` |  | input | Number of rows of matrix op(`A`) and `C`.  
`k` |  | input | Number of columns of matrix op(`A`).  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`A` | device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `trans == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`Atype` |  | input | Enumerant specifying the datatype of matrix `A`.  
`lda` |  | input | Leading dimension of two-dimensional array used to store matrix `A`.  
`beta` |  | input | <_type_ > scalar used for multiplication. If `beta == 0` then `C` does not have to be a valid input.  
`C` | device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`Ctype` |  | input | Enumerant specifying the datatype of matrix `C`.  
`ldc` |  | input | Leading dimension of two-dimensional array used to store matrix `C`.  
  
The matrix types combinations supported for [cublasCherk3mEx()](#cublascherk3mex) are listed in the following table:

A | C  
---|---  
`CUDA_C_8I` | `CUDA_C_32F`  
`CUDA_C_32F` | `CUDA_C_32F`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `n < 0` or `k < 0`, or
  * if `uplo` is not one of `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, or
  * if `trans` is not one of `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, or
  * if `lda < max(1, n)` if `trans == CUBLAS_OP_N` and `lda < max(1, k)` otherwise, or
  * if `ldc < max(1, n)`, or
  * if `Atype` or `Ctype` are not supported

  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `Atype` and `Ctype` is not supported.  
`CUBLAS_STATUS_ARCH_MISMATCH` | The device has a compute capability lower than 5.0.  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU.  
  
For references please refer to NETLIB documentation:

[cherk()](http://www.netlib.org/blas/cherk.f)

###  2.8.20. cublasNrm2Ex() 
    
    
    cublasStatus_t  cublasNrm2Ex( cublasHandle_t handle,
                                  int n,
                                  const void *x,
                                  cudaDataType xType,
                                  int incx,
                                  void *result,
                                  cudaDataType resultType,
                                  cudaDataType executionType)
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an API generalization of the routine [cublas<t>nrm2()](#cublas-t-nrm2) where input data, output data and compute type can be specified independently.

This function computes the Euclidean norm of the vector `x`. The code uses a multiphase model of accumulation to avoid intermediate underflow and overflow, with the result being equivalent to \\(\sqrt{\sum_{i = 1}^{n}\left( {\mathbf{x}\lbrack j\rbrack \times \mathbf{x}\lbrack j\rbrack} \right)}\\) where \\(j = 1 + \left( {i - 1} \right)*\text{incx}\\) in exact arithmetic. Notice that the last equation reflects 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vector `x`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`xType` |  | input | Enumerant specifying the datatype of vector `x`.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`result` | host or device | output | The resulting norm, which is set to `0` if `n <= 0` or `incx <= 0`.  
`resultType` |  | input | Enumerant specifying the datatype of the `result`.  
`executionType` |  | input | Enumerant specifying the datatype in which the computation is executed.  
  
The datatypes combinations currently supported for [cublasNrm2Ex()](#cublasnrm2ex) are listed below :

x | result | execution  
---|---|---  
`CUDA_R_16F` | `CUDA_R_16F` | `CUDA_R_32F`  
`CUDA_R_16BF` | `CUDA_R_16BF` | `CUDA_R_32F`  
`CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_C_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
`CUDA_C_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_ALLOC_FAILED` | The reduction buffer could not be allocated  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `xType`, `resultType` and `executionType` is not supported  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_INVALID_VALUE` | 

  * If `xType` or `resultType` or `executionType` is not supported, or
  * if `result` is NULL

  
  
For references please refer to NETLIB documentation:

[snrm2()](http://www.netlib.org/blas/snrm2.f90), [dnrm2()](http://www.netlib.org/blas/dnrm2.f90), [scnrm2()](http://www.netlib.org/blas/scnrm2.f90), [dznrm2()](http://www.netlib.org/blas/dznrm2.f90)

###  2.8.21. cublasAxpyEx() 
    
    
    cublasStatus_t cublasAxpyEx (cublasHandle_t handle,
                                 int n,
                                 const void *alpha,
                                 cudaDataType alphaType,
                                 const void *x,
                                 cudaDataType xType,
                                 int incx,
                                 void *y,
                                 cudaDataType yType,
                                 int incy,
                                 cudaDataType executiontype);
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an API generalization of the routine [cublas<t>axpy()](#cublas-t-axpy) where input data, output data and compute type can be specified independently.

This function multiplies the vector `x` by the scalar \\(\alpha\\) and adds it to the vector `y` overwriting the latest vector with the result. Hence, the performed operation is \\(\mathbf{y}\lbrack j\rbrack = \alpha \times \mathbf{x}\lbrack k\rbrack + \mathbf{y}\lbrack j\rbrack\\) for \\(i = 1,\ldots,n\\) , \\(k = 1 + \left( {i - 1} \right)*\text{incx}\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incy}\\) . Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vector `x` and `y`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`alphaType` |  | input | Enumerant specifying the datatype of scalar `alpha`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`xType` |  | input | Enumerant specifying the datatype of vector `x`.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`yType` |  | input | Enumerant specifying the datatype of vector `y`.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`executionType` |  | input | Enumerant specifying the datatype in which the computation is executed.  
  
The datatypes combinations currently supported for [cublasAxpyEx()](#cublasaxpyex) are listed in the following table:

alpha | x | y | execution  
---|---|---|---  
`CUDA_R_32F` | `CUDA_R_16F` | `CUDA_R_16F` | `CUDA_R_32F`  
`CUDA_R_32F` | `CUDA_R_16BF` | `CUDA_R_16BF` | `CUDA_R_32F`  
`CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
`CUDA_C_32F` | `CUDA_C_32F` | `CUDA_C_32F` | `CUDA_C_32F`  
`CUDA_C_64F` | `CUDA_C_64F` | `CUDA_C_64F` | `CUDA_C_64F`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `xType`, `yType`, and `executionType` is not supported.  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU.  
`CUBLAS_STATUS_INVALID_VALUE` | `alphaType` or `xType` or `yType` or `executionType` is not supported.  
  
For references please refer to NETLIB documentation:

[saxpy()](http://www.netlib.org/blas/saxpy.f), [daxpy()](http://www.netlib.org/blas/daxpy.f), [caxpy()](http://www.netlib.org/blas/caxpy.f), [zaxpy()](http://www.netlib.org/blas/zaxpy.f)

###  2.8.22. cublasDotEx() 
    
    
    cublasStatus_t cublasDotEx (cublasHandle_t handle,
                                int n,
                                const void *x,
                                cudaDataType xType,
                                int incx,
                                const void *y,
                                cudaDataType yType,
                                int incy,
                                void *result,
                                cudaDataType resultType,
                                cudaDataType executionType);
    
    cublasStatus_t cublasDotcEx (cublasHandle_t handle,
                                 int n,
                                 const void *x,
                                 cudaDataType xType,
                                 int incx,
                                 const void *y,
                                 cudaDataType yType,
                                 int incy,
                                 void *result,
                                 cudaDataType resultType,
                                 cudaDataType executionType);
    

These functions support the [64-bit Integer Interface](#bit-integer-interface).

These functions are an API generalization of the routines [cublas<t>dot()](#id6) and [cublas<t>dotc()](#cublas-t-dot) where input data, output data and compute type can be specified independently. Note: [cublas<t>dotc()](#cublas-t-dot) is dot product conjugated, [cublas<t>dotu()](#cublas-t-dot) is dot product unconjugated.

This function computes the dot product of vectors `x` and `y`. Hence, the result is \\(\sum_{i = 1}^{n}\left( {\mathbf{x}\lbrack k\rbrack \times \mathbf{y}\lbrack j\rbrack} \right)\\) where \\(k = 1 + \left( {i - 1} \right)*\text{incx}\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incy}\\) . Notice that in the first equation the conjugate of the element of vector x should be used if the function name ends in character ‘c’ and that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vectors `x` and `y`.  
`x` | device | input | <_type_ > vector with `n` elements.  
`xType` |  | input | Enumerant specifying the datatype of vector `x`.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | input | <_type_ > vector with `n` elements.  
`yType` |  | input | Enumerant specifying the datatype of vector `y`.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`result` | host or device | output | The resulting dot product, which is set to `0` if `n <= 0`  
`resultType` |  | input | Enumerant specifying the datatype of the `result`.  
`executionType` |  | input | Enumerant specifying the datatype in which the computation is executed.  
  
The datatypes combinations currently supported for [cublasDotEx()](#cublasdotex) and [cublasDotcEx()](cublasDotEx\(\)) are listed below:

x | y | result | execution  
---|---|---|---  
`CUDA_R_16F` | `CUDA_R_16F` | `CUDA_R_16F` | `CUDA_R_32F`  
`CUDA_R_16BF` | `CUDA_R_16BF` | `CUDA_R_16BF` | `CUDA_R_32F`  
`CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
`CUDA_C_32F` | `CUDA_C_32F` | `CUDA_C_32F` | `CUDA_C_32F`  
`CUDA_C_64F` | `CUDA_C_64F` | `CUDA_C_64F` | `CUDA_C_64F`  
  
The possible error values returned by this function and their meanings are listed in the following table:

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully.  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized.  
`CUBLAS_STATUS_ALLOC_FAILED` | The reduction buffer could not be allocated.  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `xType`, `yType`, `resultType` and `executionType` is not supported.  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU.  
`CUBLAS_STATUS_INVALID_VALUE` | `xType` or `yType` or `resultType` or `executionType` is not supported.  
  
For references please refer to NETLIB documentation:

[sdot()](http://www.netlib.org/blas/sdot.f), [ddot()](http://www.netlib.org/blas/ddot.f), [cdotu()](http://www.netlib.org/blas/cdotu.f), [cdotc()](http://www.netlib.org/blas/cdotc.f), [zdotu()](http://www.netlib.org/blas/zdotu.f), [zdotc()](http://www.netlib.org/blas/zdotc.f)

###  2.8.23. cublasRotEx() 
    
    
    cublasStatus_t cublasRotEx(cublasHandle_t handle,
                               int n,
                               void *x,
                               cudaDataType xType,
                               int incx,
                               void *y,
                               cudaDataType yType,
                               int incy,
                               const void *c,  /* host or device pointer */
                               const void *s,
                               cudaDataType csType,
                               cudaDataType executiontype);
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function is an extension to the routine [cublas<t>rot()](#cublas-t-rot) where input data, output data, cosine/sine type, and compute type can be specified independently.

This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by \\(cos(alpha) = c\\), \\(sin(alpha) = s\\)):

\\(G = \begin{pmatrix} c & s \\\ {- s} & c \\\ \end{pmatrix}\\)

to vectors `x` and `y`.

Hence, the result is \\(\mathbf{x}\lbrack k\rbrack = c \times \mathbf{x}\lbrack k\rbrack + s \times \mathbf{y}\lbrack j\rbrack\\) and \\(\mathbf{y}\lbrack j\rbrack = - s \times \mathbf{x}\lbrack k\rbrack + c \times \mathbf{y}\lbrack j\rbrack\\) where \\(k = 1 + \left( {i - 1} \right)*\text{incx}\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incy}\\) . Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vectors `x` and `y`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`xType` |  | input | Enumerant specifying the datatype of vector `x`.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`y` | device | in/out | <_type_ > vector with `n` elements.  
`yType` |  | input | Enumerant specifying the datatype of vector `y`.  
`incy` |  | input | Stride between consecutive elements of `y`.  
`c` | host or device | input | Cosine element of the rotation matrix.  
`s` | host or device | input | Sine element of the rotation matrix.  
`csType` |  | input | Enumerant specifying the datatype of `c` and `s`.  
`executionType` |  | input | Enumerant specifying the datatype in which the computation is executed.  
  
The datatypes combinations currently supported for [cublasRotEx()](#cublasrotex) are listed below :

executionType | xType / yType | csType  
---|---|---  
`CUDA_R_32F` |  `CUDA_R_16BF` `CUDA_R_16F` `CUDA_R_32F` |  `CUDA_R_16BF` `CUDA_R_16F` `CUDA_R_32F`  
`CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
`CUDA_C_32F` |  `CUDA_C_32F` `CUDA_C_32F` |  `CUDA_R_32F` `CUDA_C_32F`  
`CUDA_C_64F` |  `CUDA_C_64F` `CUDA_C_64F` |  `CUDA_R_64F` `CUDA_C_64F`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[srot()](http://www.netlib.org/blas/srot.f), [drot()](http://www.netlib.org/blas/drot.f), [crot()](http://www.netlib.org/lapack/lapack_routine/crot.f), [csrot()](http://www.netlib.org/blas/csrot.f), [zrot()](http://www.netlib.org/lapack/lapack_routine/zrot.f), [zdrot()](http://www.netlib.org/blas/zdrot.f)

###  2.8.24. cublasScalEx() 
    
    
    cublasStatus_t  cublasScalEx(cublasHandle_t handle,
                                 int n,
                                 const void *alpha,
                                 cudaDataType alphaType,
                                 void *x,
                                 cudaDataType xType,
                                 int incx,
                                 cudaDataType executionType);
    

This function supports the [64-bit Integer Interface](#bit-integer-interface).

This function scales the vector `x` by the scalar \\(\alpha\\) and overwrites it with the result. Hence, the performed operation is \\(\mathbf{x}\lbrack j\rbrack = \alpha \times \mathbf{x}\lbrack j\rbrack\\) for \\(i = 1,\ldots,n\\) and \\(j = 1 + \left( {i - 1} \right)*\text{incx}\\) . Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | Handle to the cuBLAS library context.  
`n` |  | input | Number of elements in the vector `x`.  
`alpha` | host or device | input | <_type_ > scalar used for multiplication.  
`alphaType` |  | input | Enumerant specifying the datatype of scalar `alpha`.  
`x` | device | in/out | <_type_ > vector with `n` elements.  
`xType` |  | input | Enumerant specifying the datatype of vector `x`.  
`incx` |  | input | Stride between consecutive elements of `x`.  
`executionType` |  | input | Enumerant specifying the datatype in which the computation is executed.  
  
The datatypes combinations currently supported for [cublasScalEx()](#cublasscalex) are listed below :

alpha | x | execution  
---|---|---  
`CUDA_R_32F` | `CUDA_R_16F` | `CUDA_R_32F`  
`CUDA_R_32F` | `CUDA_R_16BF` | `CUDA_R_32F`  
`CUDA_R_32F` | `CUDA_R_32F` | `CUDA_R_32F`  
`CUDA_R_64F` | `CUDA_R_64F` | `CUDA_R_64F`  
`CUDA_C_32F` | `CUDA_C_32F` | `CUDA_C_32F`  
`CUDA_C_64F` | `CUDA_C_64F` | `CUDA_C_64F`  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | The operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | The library was not initialized  
`CUBLAS_STATUS_NOT_SUPPORTED` | The combination of the parameters `xType` and `executionType` is not supported  
`CUBLAS_STATUS_EXECUTION_FAILED` | The function failed to launch on the GPU  
`CUBLAS_STATUS_INVALID_VALUE` | `alphaType` or `xType` or `executionType` is not supported  
  
For references please refer to NETLIB documentation:

[sscal()](http://www.netlib.org/blas/sscal.f), [dscal()](http://www.netlib.org/blas/dscal.f), [csscal()](http://www.netlib.org/blas/csscal.f), [cscal()](http://www.netlib.org/blas/cscal.f), [zdscal()](http://www.netlib.org/blas/zdscal.f), [zscal()](http://www.netlib.org/blas/zscal.f)
