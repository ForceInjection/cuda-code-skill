# 4. Using the cuBLASXt API


##  4.1. General description   
  
The cuBLASXt API of cuBLAS exposes a multi-GPU capable host interface: when using this API the application only needs to allocate the required matrices on the host memory space. Additionally, the current implementation supports managed memory on Linux with GPU devices that have compute capability 6.x or greater but treats it as host memory. Managed memory is not supported on Windows. There are no restriction on the sizes of the matrices as long as they can fit into the host memory. The cuBLASXt API takes care of allocating the memory across the designated GPUs and dispatched the workload between them and finally retrieves the results back to the host. The cuBLASXt API supports only the compute-intensive BLAS3 routines (e.g matrix-matrix operations) where the PCI transfers back and forth from the GPU can be amortized. The cuBLASXt API has its own header file `cublasXt.h`.

Starting with release 8.0, cuBLASXt API allows any of the matrices to be located on a GPU device.

Note

When providing matrices allocated on the GPU using the Stream Ordered Memory Allocator, ensure visibility across all devices by using [cudaMemPoolSetAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g4210da54ee5a810945da586e00cb3019).

Note

The cuBLASXt API is only supported on 64-bit platforms.

###  4.1.1. Tiling design approach 

To be able to share the workload between multiple GPUs, the cuBLASXt API uses a tiling strategy : every matrix is divided in square tiles of user-controllable dimension BlockDim x BlockDim. The resulting matrix tiling defines the static scheduling policy : each resulting tile is affected to a GPU in a round robin fashion One CPU thread is created per GPU and is responsible to do the proper memory transfers and cuBLAS operations to compute all the tiles that it is responsible for. From a performance point of view, due to this static scheduling strategy, it is better that compute capabilities and PCI bandwidth are the same for every GPU. The figure below illustrates the tiles distribution between 3 GPUs. To compute the first tile G0 from C, the CPU thread 0 responsible of GPU0, have to load 3 tiles from the first row of A and tiles from the first column of B in a pipeline fashion in order to overlap memory transfer and computations and sum the results into the first tile G0 of C before to move on to the next tile G0.

![Example of cublasXt<t>gemm\(\) tiling for 3 Gpus](https://docs.nvidia.com/cuda/cublas/_images/cublasmg_gemm.jpg)

Example of [cublasXt<t>gemm()](#cublasxt-t-gemm) tiling for 3 Gpus

When the tile dimension is not an exact multiple of the dimensions of C, some tiles are partially filled on the right border or/and the bottom border. The current implementation does not pad the incomplete tiles but simply keep track of those incomplete tiles by doing the right reduced cuBLAS operations : this way, no extra computation is done. However it still can lead to some load unbalance when all GPUS do not have the same number of incomplete tiles to work on.

When one or more matrices are located on some GPU devices, the same tiling approach and workload sharing is applied. The memory transfers are in this case done between devices. However, when the computation of a tile and some data are located on the same GPU device, the memory transfer to/from the local data into tiles is bypassed and the GPU operates directly on the local data. This can lead to a significant performance increase, especially when only one GPU is used for the computation.

The matrices can be located on any GPU device, and do not have to be located on the same GPU device. Furthermore, the matrices can even be located on a GPU device that do not participate to the computation.

On the contrary of the cuBLAS API, even if all matrices are located on the same device, the cuBLASXt API is still a blocking API from the host point of view : the data results wherever located will be valid on the call return and no device synchronization is required.

###  4.1.2. Hybrid CPU-GPU computation 

In the case of very large problems, the cuBLASXt API offers the possibility to offload some of the computation to the host CPU. This feature can be setup with the routines [cublasXtSetCpuRoutine()](#cublasxtsetcpuroutine) and [cublasXtSetCpuRatio()](#cublasxtsetcpuratio) The workload affected to the CPU is put aside : it is simply a percentage of the resulting matrix taken from the bottom and the right side whichever dimension is bigger. The GPU tiling is done after that on the reduced resulting matrix.

If any of the matrices is located on a GPU device, the feature is ignored and all computation will be done only on the GPUs

This feature should be used with caution because it could interfere with the CPU threads responsible of feeding the GPUs.

Currently, only the routine [cublasXt<t>gemm()](#cublasxt-t-gemm) supports this feature.

###  4.1.3. Results reproducibility 

Currently all cuBLASXt API routines from a given toolkit version, generate the same bit-wise results when the following conditions are respected :

  * all GPUs participating to the computation have the same compute capabilities and the same number of SMs.

  * the tiles size is kept the same between run.

  * either the CPU hybrid computation is not used or the CPU Blas provided is also guaranteed to produce reproducible results.


##  4.2. cuBLASXt API Datatypes Reference 

###  4.2.1. cublasXtHandle_t 

The `cublasXtHandle_t` type is a pointer type to an opaque structure holding the cuBLASXt API context. The cuBLASXt API context must be initialized using [cublasXtCreate()](#cublasxtcreate) and the returned handle must be passed to all subsequent cuBLASXt API function calls. The context should be destroyed at the end using [cublasXtDestroy()](#cublasxtdestroy).

###  4.2.2. cublasXtOpType_t 

The `cublasOpType_t` enumerates the four possible types supported by BLAS routines. This enum is used as parameters of the routines `cublasXtSetCpuRoutine` and `cublasXtSetCpuRatio` to setup the hybrid configuration.

Value | Meaning  
---|---  
`CUBLASXT_FLOAT` | float or single precision type  
`CUBLASXT_DOUBLE` | double precision type  
`CUBLASXT_COMPLEX` | single precision complex  
`CUBLASXT_DOUBLECOMPLEX` | double precision complex  
  
###  4.2.3. cublasXtBlasOp_t 

The `cublasXtBlasOp_t` type enumerates the BLAS3 or BLAS-like routine supported by cuBLASXt API. This enum is used as parameters of the routines `cublasXtSetCpuRoutine` and `cublasXtSetCpuRatio` to setup the hybrid configuration.

Value | Meaning  
---|---  
`CUBLASXT_GEMM` | GEMM routine  
`CUBLASXT_SYRK` | SYRK routine  
`CUBLASXT_HERK` | HERK routine  
`CUBLASXT_SYMM` | SYMM routine  
`CUBLASXT_HEMM` | HEMM routine  
`CUBLASXT_TRSM` | TRSM routine  
`CUBLASXT_SYR2K` | SYR2K routine  
`CUBLASXT_HER2K` | HER2K routine  
`CUBLASXT_SPMM` | SPMM routine  
`CUBLASXT_SYRKX` | SYRKX routine  
`CUBLASXT_HERKX` | HERKX routine  
  
###  4.2.4. cublasXtPinningMemMode_t 

The type is used to enable or disable the Pinning Memory mode through the routine `cubasMgSetPinningMemMode`

Value | Meaning  
---|---  
`CUBLASXT_PINNING_DISABLED` | the Pinning Memory mode is disabled  
`CUBLASXT_PINNING_ENABLED` | the Pinning Memory mode is enabled


##  4.3. cuBLASXt API Helper Function Reference   
  
###  4.3.1. cublasXtCreate() 
    
    
    cublasStatus_t
    cublasXtCreate(cublasXtHandle_t *handle)
    

This function initializes the cuBLASXt API and creates a handle to an opaque structure holding the cuBLASXt API context. It allocates hardware resources on the host and device and must be called prior to making any other cuBLASXt API calls.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the initialization succeeded  
`CUBLAS_STATUS_ALLOC_FAILED` | the resources could not be allocated  
`CUBLAS_STATUS_NOT_SUPPORTED` | cuBLASXt API is only supported on 64-bit platform  
  
###  4.3.2. cublasXtDestroy() 
    
    
    cublasStatus_t
    cublasXtDestroy(cublasXtHandle_t handle)
    

This function releases hardware resources used by the cuBLASXt API context. The release of GPU resources may be deferred until the application exits. This function is usually the last call with a particular handle to the cuBLASXt API.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the shut down succeeded  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
  
###  4.3.3. cublasXtDeviceSelect() 
    
    
    cublasXtDeviceSelect(cublasXtHandle_t handle, int nbDevices, int deviceId[])
    

This function allows the user to provide the number of GPU devices and their respective Ids that will participate to the subsequent cuBLASXt API Math function calls. This function will create a cuBLAS context for every GPU provided in that list. Currently the device configuration is static and cannot be changed between Math function calls. In that regard, this function should be called only once after `cublasXtCreate`. To be able to run multiple configurations, multiple cuBLASXt API contexts should be created.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | User call was successful  
`CUBLAS_STATUS_INVALID_VALUE` | Access to at least one of the device could not be done or a cuBLAS context could not be created on at least one of the device  
`CUBLAS_STATUS_ALLOC_FAILED` | Some resources could not be allocated.  
  
###  4.3.4. cublasXtSetBlockDim() 
    
    
    cublasXtSetBlockDim(cublasXtHandle_t handle, int blockDim)
    

This function allows the user to set the block dimension used for the tiling of the matrices for the subsequent Math function calls. Matrices are split in square tiles of blockDim x blockDim dimension. This function can be called anytime and will take effect for the following Math function calls. The block dimension should be chosen in a way to optimize the math operation and to make sure that the PCI transfers are well overlapped with the computation.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the call has been successful  
`CUBLAS_STATUS_INVALID_VALUE` | blockDim <= 0  
  
###  4.3.5. cublasXtGetBlockDim() 
    
    
    cublasXtGetBlockDim(cublasXtHandle_t handle, int *blockDim)
    

This function allows the user to query the block dimension used for the tiling of the matrices.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the call has been successful  
  
###  4.3.6. cublasXtSetCpuRoutine() 
    
    
    cublasXtSetCpuRoutine(cublasXtHandle_t handle, cublasXtBlasOp_t blasOp, cublasXtOpType_t type, void *blasFunctor)
    

This function allows the user to provide a CPU implementation of the corresponding BLAS routine. This function can be used with the function [cublasXtSetCpuRatio()](#cublasxtsetcpuratio) to define an hybrid computation between the CPU and the GPUs. Currently the hybrid feature is only supported for the xGEMM routines.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the call has been successful  
`CUBLAS_STATUS_INVALID_VALUE` | blasOp or type define an invalid combination  
`CUBLAS_STATUS_NOT_SUPPORTED` | CPU-GPU Hybridization for that routine is not supported  
  
###  4.3.7. cublasXtSetCpuRatio() 
    
    
    cublasXtSetCpuRatio(cublasXtHandle_t handle, cublasXtBlasOp_t blasOp, cublasXtOpType_t type, float ratio )
    

This function allows the user to define the percentage of workload that should be done on a CPU in the context of an hybrid computation. This function can be used with the function [cublasXtSetCpuRoutine()](#cublasxtsetcpuroutine) to define an hybrid computation between the CPU and the GPUs. Currently the hybrid feature is only supported for the xGEMM routines.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the call has been successful  
`CUBLAS_STATUS_INVALID_VALUE` | blasOp or type define an invalid combination  
`CUBLAS_STATUS_NOT_SUPPORTED` | CPU-GPU Hybridization for that routine is not supported  
  
###  4.3.8. cublasXtSetPinningMemMode() 
    
    
    cublasXtSetPinningMemMode(cublasXtHandle_t handle, cublasXtPinningMemMode_t mode)
    

This function allows the user to enable or disable the Pinning Memory mode. When enabled, the matrices passed in subsequent cuBLASXt API calls will be pinned/unpinned using the CUDART routine `cudaHostRegister()` and `cudaHostUnregister()` respectively if the matrices are not already pinned. If a matrix happened to be pinned partially, it will also not be pinned. Pinning the memory improve PCI transfer performance and allows to overlap PCI memory transfer with computation. However pinning/unpinning the memory take some time which might not be amortized. It is advised that the user pins the memory on its own using `cudaMallocHost()` or `cudaHostRegister()` and unpin it when the computation sequence is completed. By default, the Pinning Memory mode is disabled.

Note

The Pinning Memory mode should not be enabled when matrices used for different calls to cuBLASXt API overlap. cuBLASXt determines that a matrix is pinned or not if the first address of that matrix is pinned using `cudaHostGetFlags()`, thus cannot know if the matrix is already partially pinned or not. This is especially true in multi-threaded application where memory could be partially or totally pinned or unpinned while another thread is accessing that memory.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the call has been successful  
`CUBLAS_STATUS_INVALID_VALUE` | the mode value is different from `CUBLASXT_PINNING_DISABLED` and `CUBLASXT_PINNING_ENABLED`  
  
###  4.3.9. cublasXtGetPinningMemMode() 
    
    
    cublasXtGetPinningMemMode(cublasXtHandle_t handle, cublasXtPinningMemMode_t *mode)
    

This function allows the user to query the Pinning Memory mode. By default, the Pinning Memory mode is disabled.

Return Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the call has been successful


##  4.4. cuBLASXt API Math Functions Reference   
  
In this chapter we describe the actual Linear Algebra routines that cuBLASXt API supports. We will use abbreviations <_type_ > for type and <_t_ > for the corresponding short type to make a more concise and clear presentation of the implemented functions. Unless otherwise specified <_type_ > and <_t_ > have the following meanings:

<_type_ > | <_t_ > | Meaning  
---|---|---  
`float` | ‘s’ or ‘S’ | real single-precision  
`double` | ‘d’ or ‘D’ | real double-precision  
`cuComplex` | ‘c’ or ‘C’ | complex single-precision  
`cuDoubleComplex` | ‘z’ or ‘Z’ | complex double-precision  
  
The abbreviation \\(\mathbf{Re}(\cdot)\\) and \\(\mathbf{Im}(\cdot)\\) will stand for the real and imaginary part of a number, respectively. Since imaginary part of a real number does not exist, we will consider it to be zero and can usually simply discard it from the equation where it is being used. Also, the \\(\bar{\alpha}\\) will denote the complex conjugate of \\(\alpha\\) .

In general throughout the documentation, the lower case Greek symbols \\(\alpha\\) and \\(\beta\\) will denote scalars, lower case English letters in bold type \\(\mathbf{x}\\) and \\(\mathbf{y}\\) will denote vectors and capital English letters \\(A\\) , \\(B\\) and \\(C\\) will denote matrices.

###  4.4.1. cublasXt<t>gemm() 
    
    
    cublasStatus_t cublasXtSgemm(cublasXtHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               size_t m, size_t n, size_t k,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *B, int ldb,
                               const float           *beta,
                               float           *C, int ldc)
    cublasStatus_t cublasXtDgemm(cublasXtHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *B, int ldb,
                               const double          *beta,
                               double          *C, int ldc)
    cublasStatus_t cublasXtCgemm(cublasXtHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *B, int ldb,
                               const cuComplex       *beta,
                               cuComplex       *C, int ldc)
    cublasStatus_t cublasXtZgemm(cublasXtHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *C, int ldc)
    

This function performs the matrix-matrix multiplication

\\(C = \alpha\text{op}(A)\text{op}(B) + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, and \\(A\\) , \\(B\\) and \\(C\\) are matrices stored in column-major format with dimensions \\(\text{op}(A)\\) \\(m \times k\\) , \\(\text{op}(B)\\) \\(k \times n\\) and \\(C\\) \\(m \times n\\) , respectively. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

and \\(\text{op}(B)\\) is defined similarly for matrix \\(B\\) .

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`transa` |  | input | operation op(`A`) that is non- or (conj.) transpose.  
`transb` |  | input | operation op(`B`) that is non- or (conj.) transpose.  
`m` |  | input | number of rows of matrix op(`A`) and `C`.  
`n` |  | input | number of columns of matrix op(`B`) and `C`.  
`k` |  | input | number of columns of op(`A`) and rows of op(`B`).  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`A` | host or device | input | <_type_ > array of dimensions `lda x k` with `lda >= max(1, m)` if `transa == CUBLAS_OP_N` and `lda x m` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store the matrix `A`.  
`B` | host or device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, k)` if `transb == CUBLAS_OP_N` and `ldb x k` with `ldb >= max(1, n)` otherwise.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host | input | <_type_ > scalar used for multiplication. If `beta == 0`, `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | leading dimension of a two-dimensional array used to store the matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `m,n,k<0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[sgemm()](http://www.netlib.org/blas/sgemm.f), [dgemm()](http://www.netlib.org/blas/dgemm.f), [cgemm()](http://www.netlib.org/blas/cgemm.f), [zgemm()](http://www.netlib.org/blas/zgemm.f)

###  4.4.2. cublasXt<t>hemm() 
    
    
    cublasStatus_t cublasXtChemm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               size_t m, size_t n,
                               const cuComplex       *alpha,
                               const cuComplex       *A, size_t lda,
                               const cuComplex       *B, size_t ldb,
                               const cuComplex       *beta,
                               cuComplex       *C, size_t ldc)
    cublasStatus_t cublasXtZhemm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               size_t m, size_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, size_t lda,
                               const cuDoubleComplex *B, size_t ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *C, size_t ldc)
    

This function performs the Hermitian matrix-matrix multiplication

\\(C = \left\\{ \begin{matrix} {\alpha AB + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {\alpha BA + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) is a Hermitian matrix stored in lower or upper mode, \\(B\\) and \\(C\\) are \\(m \times n\\) matrices, and \\(\alpha\\) and \\(\beta\\) are scalars.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`side` |  | input | indicates if matrix `A` is on the left or right of `B`.  
`uplo` |  | input | indicates if matrix `A` lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.  
`m` |  | input | number of rows of matrix `C` and `B`, with matrix `A` sized accordingly.  
`n` |  | input | number of columns of matrix `C` and `B`, with matrix `A` sized accordingly.  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`A` | host or device | input | <_type_ > array of dimension `lda x m` with `lda >= max(1, m)` if `side = CUBLAS_SIDE_LEFT` and `lda x n` with `lda >= max(1, n)` otherwise. The imaginary parts of the diagonal elements are assumed to be zero.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix `A`.  
`B` | host or device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, m)`.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host | input | <_type_ > scalar used for multiplication, if `beta == 0` then `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `m < 0` or `n < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[chemm()](http://www.netlib.org/blas/chemm.f), [zhemm()](http://www.netlib.org/blas/zhemm.f)

###  4.4.3. cublasXt<t>symm() 
    
    
    cublasStatus_t cublasXtSsymm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               size_t m, size_t n,
                               const float           *alpha,
                               const float           *A, size_t lda,
                               const float           *B, size_t ldb,
                               const float           *beta,
                               float           *C, size_t ldc)
    cublasStatus_t cublasXtDsymm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               size_t m, size_t n,
                               const double          *alpha,
                               const double          *A, size_t lda,
                               const double          *B, size_t ldb,
                               const double          *beta,
                               double          *C, size_t ldc)
    cublasStatus_t cublasXtCsymm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               size_t m, size_t n,
                               const cuComplex       *alpha,
                               const cuComplex       *A, size_t lda,
                               const cuComplex       *B, size_t ldb,
                               const cuComplex       *beta,
                               cuComplex       *C, size_t ldc)
    cublasStatus_t cublasXtZsymm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               size_t m, size_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, size_t lda,
                               const cuDoubleComplex *B, size_t ldb,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *C, size_t ldc)
    

This function performs the symmetric matrix-matrix multiplication

\\(C = \left\\{ \begin{matrix} {\alpha AB + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {\alpha BA + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) is a symmetric matrix stored in lower or upper mode, \\(A\\) and \\(A\\) are \\(m \times n\\) matrices, and \\(\alpha\\) and \\(\beta\\) are scalars.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`side` |  | input | indicates if matrix `A` is on the left or right of `B`.  
`uplo` |  | input | indicates if matrix `A` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`m` |  | input | number of rows of matrix `A` and `B`, with matrix `A` sized accordingly.  
`n` |  | input | number of columns of matrix `C` and `A`, with matrix `A` sized accordingly.  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`A` | host or device | input | <_type_ > array of dimension `lda x m` with `lda >= max(1, m)` if `side == CUBLAS_SIDE_LEFT` and `lda x n` with `lda >= max(1, n)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix `A`.  
`B` | host or device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, m)`.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host | input | <_type_ > scalar used for multiplication, if `beta == 0` then `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimension `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `m < 0` or `n < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssymm()](http://www.netlib.org/blas/ssymm.f), [dsymm()](http://www.netlib.org/blas/dsymm.f), [csymm()](http://www.netlib.org/blas/csymm.f), [zsymm()](http://www.netlib.org/blas/zsymm.f)

###  4.4.4. cublasXt<t>syrk() 
    
    
    cublasStatus_t cublasXtSsyrk(cublasXtHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *beta,
                               float           *C, int ldc)
    cublasStatus_t cublasXtDsyrk(cublasXtHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *beta,
                               double          *C, int ldc)
    cublasStatus_t cublasXtCsyrk(cublasXtHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const cuComplex       *alpha,
                               const cuComplex       *A, int lda,
                               const cuComplex       *beta,
                               cuComplex       *C, int ldc)
    cublasStatus_t cublasXtZsyrk(cublasXtHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *beta,
                               cuDoubleComplex *C, int ldc)
    

This function performs the symmetric rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(A)^{T} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a symmetric matrix stored in lower or upper mode, and \\(A\\) is a matrix with dimensions \\(\text{op}(A)\\) \\(n \times k\\) . Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ \end{matrix} \right.\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`uplo` |  | input | indicates if matrix `C` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`trans` |  | input | operation op(`A`) that is non- or transpose.  
`n` |  | input | number of rows of matrix op(`A`) and `C`.  
`k` |  | input | number of columns of matrix op(`A`).  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`A` | host or device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `trans == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix A.  
`beta` | host | input | <_type_ > scalar used for multiplication, if `beta == 0` then `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `n < 0` or `k < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssyrk()](http://www.netlib.org/blas/ssyrk.f), [dsyrk()](http://www.netlib.org/blas/dsyrk.f), [csyrk()](http://www.netlib.org/blas/csyrk.f), [zsyrk()](http://www.netlib.org/blas/zsyrk.f)

###  4.4.5. cublasXt<t>syr2k() 
    
    
    cublasStatus_t cublasXtSsyr2k(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const float           *alpha,
                                const float           *A, size_t lda,
                                const float           *B, size_t ldb,
                                const float           *beta,
                                float           *C, size_t ldc)
    cublasStatus_t cublasXtDsyr2k(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const double          *alpha,
                                const double          *A, size_t lda,
                                const double          *B, size_t ldb,
                                const double          *beta,
                                double          *C, size_t ldc)
    cublasStatus_t cublasXtCsyr2k(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const cuComplex       *alpha,
                                const cuComplex       *A, size_t lda,
                                const cuComplex       *B, size_t ldb,
                                const cuComplex       *beta,
                                cuComplex       *C, size_t ldc)
    cublasStatus_t cublasXtZsyr2k(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, size_t lda,
                                const cuDoubleComplex *B, size_t ldb,
                                const cuDoubleComplex *beta,
                                cuDoubleComplex *C, size_t ldc)
    

This function performs the symmetric rank- \\(2k\\) update

\\(C = \alpha(\text{op}(A)\text{op}(B)^{T} + \text{op}(B)\text{op}(A)^{T}) + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a symmetric matrix stored in lower or upper mode, and \\(A\\) and \\(B\\) are matrices with dimensions \\(\text{op}(A)\\) \\(n \times k\\) and \\(\text{op}(B)\\) \\(n \times k\\) , respectively. Also, for matrix \\(A\\) and \\(B\\)

\\(\text{op(}A\text{) and op(}B\text{)} = \left\\{ \begin{matrix} {A\text{ and }B} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{T}\text{ and }B^{T}} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ \end{matrix} \right.\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`uplo` |  | input | indicates if matrix `C` lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`trans` |  | input | operation op(`A`) that is non- or transpose.  
`n` |  | input | number of rows of matrix op(`A`), op(`B`) and `C`.  
`k` |  | input | number of columns of matrix op(`A`) and op(`B`).  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`A` | host or device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix `A`.  
`B` | host or device | input | <_type_ > array of dimensions `ldb x k` with `ldb >= max(1, n)` if `transb == CUBLAS_OP_N` and `ldb x n` with `ldb >= max(1, k)` otherwise.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host | input | <_type_ > scalar used for multiplication, if `beta == 0`, then `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, n)`.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `n < 0` or `k < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssyr2k()](http://www.netlib.org/blas/ssyr2k.f), [dsyr2k()](http://www.netlib.org/blas/dsyr2k.f), [csyr2k()](http://www.netlib.org/blas/csyr2k.f), [zsyr2k()](http://www.netlib.org/blas/zsyr2k.f)

###  4.4.6. cublasXt<t>syrkx() 
    
    
    cublasStatus_t cublasXtSsyrkx(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const float           *alpha,
                                const float           *A, size_t lda,
                                const float           *B, size_t ldb,
                                const float           *beta,
                                float           *C, size_t ldc)
    cublasStatus_t cublasXtDsyrkx(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const double          *alpha,
                                const double          *A, size_t lda,
                                const double          *B, size_t ldb,
                                const double          *beta,
                                double          *C, size_t ldc)
    cublasStatus_t cublasXtCsyrkx(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const cuComplex       *alpha,
                                const cuComplex       *A, size_t lda,
                                const cuComplex       *B, size_t ldb,
                                const cuComplex       *beta,
                                cuComplex       *C, size_t ldc)
    cublasStatus_t cublasXtZsyrkx(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, size_t lda,
                                const cuDoubleComplex *B, size_t ldb,
                                const cuDoubleComplex *beta,
                                cuDoubleComplex *C, size_t ldc)
    

This function performs a variation of the symmetric rank- \\(k\\) update

\\(C = \alpha(\text{op}(A)\text{op}(B)^{T} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a symmetric matrix stored in lower or upper mode, and \\(A\\) and \\(B\\) are matrices with dimensions \\(\text{op}(A)\\) \\(n \times k\\) and \\(\text{op}(B)\\) \\(n \times k\\) , respectively. Also, for matrix \\(A\\) and \\(B\\)

\\(\text{op(}A\text{) and op(}B\text{)} = \left\\{ \begin{matrix} {A\text{ and }B} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{T}\text{ and }B^{T}} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ \end{matrix} \right.\\)

This routine can be used when B is in such way that the result is guaranteed to be symmetric. A usual example is when the matrix B is a scaled form of the matrix A : this is equivalent to B being the product of the matrix A and a diagonal matrix.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`uplo` |  | input | indicates if matrix `C` lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`trans` |  | input | operation op(`A`) that is non- or transpose.  
`n` |  | input | number of rows of matrix op(`A`), op(`B`) and `C`.  
`k` |  | input | number of columns of matrix op(`A`) and op(`B`).  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`A` | host or device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix `A`.  
`B` | host or device | input | <_type_ > array of dimensions `ldb x k` with `ldb >= max(1, n)` if `transb == CUBLAS_OP_N` and `ldb x n` with `ldb >= max(1, k)` otherwise.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host | input | <_type_ > scalar used for multiplication, if `beta == 0`, then `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimensions `ldc x n` with `ldc >= max(1, n)`.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `n < 0` or `k < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssyrk()](http://www.netlib.org/blas/ssyrk.f), [dsyrk()](http://www.netlib.org/blas/dsyrk.f), [csyrk()](http://www.netlib.org/blas/csyrk.f), [zsyrk()](http://www.netlib.org/blas/zsyrk.f) and

[ssyr2k()](http://www.netlib.org/blas/ssyr2k.f), [dsyr2k()](http://www.netlib.org/blas/dsyr2k.f), [csyr2k()](http://www.netlib.org/blas/csyr2k.f), [zsyr2k()](http://www.netlib.org/blas/zsyr2k.f)

###  4.4.7. cublasXt<t>herk() 
    
    
    cublasStatus_t cublasXtCherk(cublasXtHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const float  *alpha,
                               const cuComplex       *A, int lda,
                               const float  *beta,
                               cuComplex       *C, int ldc)
    cublasStatus_t cublasXtZherk(cublasXtHandle_t handle,
                               cublasFillMode_t uplo, cublasOperation_t trans,
                               int n, int k,
                               const double *alpha,
                               const cuDoubleComplex *A, int lda,
                               const double *beta,
                               cuDoubleComplex *C, int ldc)
    

This function performs the Hermitian rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(A)^{H} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a Hermitian matrix stored in lower or upper mode, and \\(A\\) is a matrix with dimensions \\(\text{op}(A)\\) \\(n \times k\\) . Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`uplo` |  | input | indicates if matrix `C` lower or upper part is stored, the other Hermitian part is not referenced.  
`trans` |  | input | operation op(`A`) that is non- or (conj.) transpose.  
`n` |  | input | number of rows of matrix op(`A`) and `C`.  
`k` |  | input | number of columns of matrix op(`A`).  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`A` | host or device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix `A`.  
`beta` | host | input | <_type_ > scalar used for multiplication, if `beta == 0` then `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `n < 0` or `k < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[cherk()](http://www.netlib.org/blas/cherk.f), [zherk()](http://www.netlib.org/blas/zherk.f)

###  4.4.8. cublasXt<t>her2k() 
    
    
    cublasStatus_t cublasXtCher2k(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const cuComplex       *alpha,
                                const cuComplex       *A, size_t lda,
                                const cuComplex       *B, size_t ldb,
                                const float  *beta,
                                cuComplex       *C, size_t ldc)
    cublasStatus_t cublasXtZher2k(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, size_t lda,
                                const cuDoubleComplex *B, size_t ldb,
                                const double *beta,
                                cuDoubleComplex *C, size_t ldc)
    

This function performs the Hermitian rank- \\(2k\\) update

\\(C = \alpha\text{op}(A)\text{op}(B)^{H} + \overset{ˉ}{\alpha}\text{op}(B)\text{op}(A)^{H} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a Hermitian matrix stored in lower or upper mode, and \\(A\\) and \\(B\\) are matrices with dimensions \\(\text{op}(A)\\) \\(n \times k\\) and \\(\text{op}(B)\\) \\(n \times k\\) , respectively. Also, for matrix \\(A\\) and \\(B\\)

\\(\text{op(}A\text{) and op(}B\text{)} = \left\\{ \begin{matrix} {A\text{ and }B} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{H}\text{ and }B^{H}} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`uplo` |  | input | indicates if matrix `C` lower or upper part is stored, the other Hermitian part is not referenced.  
`trans` |  | input | operation op(`A`) that is non- or (conj.) transpose.  
`n` |  | input | number of rows of matrix op(`A`), op(`B`) and `C`.  
`k` |  | input | number of columns of matrix op(`A`) and op(`B`).  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`A` | host or device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix `A`.  
`B` | host or device | input | <_type_ > array of dimension `ldb x k` with `ldb >= max(1, n)` if `transb == CUBLAS_OP_N` and `ldb x n` with `ldb >= max(1, k)` otherwise.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host | input | <_type_ > scalar used for multiplication, if `beta == 0` then `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `n < 0` or `k < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[cher2k()](http://www.netlib.org/blas/cher2k.f), [zher2k()](http://www.netlib.org/blas/zher2k.f)

###  4.4.9. cublasXt<t>herkx() 
    
    
    cublasStatus_t cublasXtCherkx(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const cuComplex       *alpha,
                                const cuComplex       *A, size_t lda,
                                const cuComplex       *B, size_t ldb,
                                const float  *beta,
                                cuComplex       *C, size_t ldc)
    cublasStatus_t cublasXtZherkx(cublasXtHandle_t handle,
                                cublasFillMode_t uplo, cublasOperation_t trans,
                                size_t n, size_t k,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, size_t lda,
                                const cuDoubleComplex *B, size_t ldb,
                                const double *beta,
                                cuDoubleComplex *C, size_t ldc)
    

This function performs a variation of the Hermitian rank- \\(k\\) update

\\(C = \alpha\text{op}(A)\text{op}(B)^{H} + \beta C\\)

where \\(\alpha\\) and \\(\beta\\) are scalars, \\(C\\) is a Hermitian matrix stored in lower or upper mode, and \\(A\\) and \\(B\\) are matrices with dimensions \\(\text{op}(A)\\) \\(n \times k\\) and \\(\text{op}(B)\\) \\(n \times k\\) , respectively. Also, for matrix \\(A\\) and \\(B\\)

\\(\text{op(}A\text{) and op(}B\text{)} = \left\\{ \begin{matrix} {A\text{ and }B} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ {A^{H}\text{ and }B^{H}} & {\text{if }\textsf{trans == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

This routine can be used when the matrix B is in such way that the result is guaranteed to be hermitian. A usual example is when the matrix B is a scaled form of the matrix A : this is equivalent to B being the product of the matrix A and a diagonal matrix.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`uplo` |  | input | indicates if matrix `C` lower or upper part is stored, the other Hermitian part is not referenced.  
`trans` |  | input | operation op(`A`) that is non- or (conj.) transpose.  
`n` |  | input | number of rows of matrix op(`A`), op(`B`) and `C`.  
`k` |  | input | number of columns of matrix op(`A`) and op(`B`).  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`A` | host or device | input | <_type_ > array of dimension `lda x k` with `lda >= max(1, n)` if `transa == CUBLAS_OP_N` and `lda x n` with `lda >= max(1, k)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix `A`.  
`B` | host or device | input | <_type_ > array of dimension `ldb x k` with `ldb >= max(1, n)` if `transb == CUBLAS_OP_N` and `ldb x n` with `ldb >= max(1, k)` otherwise.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host | input | real scalar used for multiplication, if `beta == 0` then `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimension `ldc x n`, with `ldc >= max(1, n)`. The imaginary parts of the diagonal elements are assumed and set to zero.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `n < 0` or `k < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[cherk()](http://www.netlib.org/blas/cherk.f), [zherk()](http://www.netlib.org/blas/zherk.f) and

[cher2k()](http://www.netlib.org/blas/cher2k.f), [zher2k()](http://www.netlib.org/blas/zher2k.f)

###  4.4.10. cublasXt<t>trsm() 
    
    
    cublasStatus_t cublasXtStrsm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasXtDiagType_t diag,
                               size_t m, size_t n,
                               const float           *alpha,
                               const float           *A, size_t lda,
                               float           *B, size_t ldb)
    cublasStatus_t cublasXtDtrsm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasXtDiagType_t diag,
                               size_t m, size_t n,
                               const double          *alpha,
                               const double          *A, size_t lda,
                               double          *B, size_t ldb)
    cublasStatus_t cublasXtCtrsm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasXtDiagType_t diag,
                               size_t m, size_t n,
                               const cuComplex       *alpha,
                               const cuComplex       *A, size_t lda,
                               cuComplex       *B, size_t ldb)
    cublasStatus_t cublasXtZtrsm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasXtDiagType_t diag,
                               size_t m, size_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, size_t lda,
                               cuDoubleComplex *B, size_t ldb)
    

This function solves the triangular linear system with multiple right-hand-sides

\\(\left\\{ \begin{matrix} {\text{op}(A)X = \alpha B} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {X\text{op}(A) = \alpha B} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) is a triangular matrix stored in lower or upper mode with or without the main diagonal, \\(X\\) and \\(B\\) are \\(m \times n\\) matrices, and \\(\alpha\\) is a scalar. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

The solution \\(X\\) overwrites the right-hand-sides \\(B\\) on exit.

No test for singularity or near-singularity is included in this function.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`side` |  | input | indicates if matrix `A` is on the left or right of `X`.  
`uplo` |  | input | indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | indicates if the elements on the main diagonal of matrix `A` are unity and should not be accessed.  
`m` |  | input | number of rows of matrix `B`, with matrix `A` sized accordingly.  
`n` |  | input | number of columns of matrix `B`, with matrix `A` is sized accordingly.  
`alpha` | host | input | <_type_ > scalar used for multiplication, if `alpha == 0` then `A` is not referenced and `B` does not have to be a valid input.  
`A` | host or device | input | <_type_ > array of dimension `lda x m` with `lda >= max(1, m)` if `side == CUBLAS_SIDE_LEFT` and `lda x n` with `lda >= max(1, n)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix `A`.  
`B` | host or device | in/out | <_type_ > array. It has dimensions `ldb x n` with `ldb >= max(1, m)`.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `m < 0` or `n < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[strsm()](http://www.netlib.org/blas/strsm.f), [dtrsm()](http://www.netlib.org/blas/dtrsm.f), [ctrsm()](http://www.netlib.org/blas/ctrsm.f), [ztrsm()](http://www.netlib.org/blas/ztrsm.f)

###  4.4.11. cublasXt<t>trmm() 
    
    
    cublasStatus_t cublasXtStrmm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               size_t m, size_t n,
                               const float           *alpha,
                               const float           *A, size_t lda,
                               const float           *B, size_t ldb,
                               float                 *C, size_t ldc)
    cublasStatus_t cublasXtDtrmm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               size_t m, size_t n,
                               const double          *alpha,
                               const double          *A, size_t lda,
                               const double          *B, size_t ldb,
                               double                *C, size_t ldc)
    cublasStatus_t cublasXtCtrmm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               size_t m, size_t n,
                               const cuComplex       *alpha,
                               const cuComplex       *A, size_t lda,
                               const cuComplex       *B, size_t ldb,
                               cuComplex             *C, size_t ldc)
    cublasStatus_t cublasXtZtrmm(cublasXtHandle_t handle,
                               cublasSideMode_t side, cublasFillMode_t uplo,
                               cublasOperation_t trans, cublasDiagType_t diag,
                               size_t m, size_t n,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, size_t lda,
                               const cuDoubleComplex *B, size_t ldb,
                               cuDoubleComplex       *C, size_t ldc)
    

This function performs the triangular matrix-matrix multiplication

\\(C = \left\\{ \begin{matrix} {\alpha\text{op}(A)B} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {\alpha B\text{op}(A)} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) is a triangular matrix stored in lower or upper mode with or without the main diagonal, \\(B\\) and \\(C\\) are \\(m \times n\\) matrix, and \\(\alpha\\) is a scalar. Also, for matrix \\(A\\)

\\(\text{op}(A) = \left\\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_N}$}} \\\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_T}$}} \\\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\\_OP\\_C}$}} \\\ \end{matrix} \right.\\)

Notice that in order to achieve better parallelism, similarly to the cublas API, cuBLASXt API differs from the BLAS API for this routine. The BLAS API assumes an in-place implementation (with results written back to B), while the cuBLASXt API assumes an out-of-place implementation (with results written into C). The application can still obtain the in-place functionality of BLAS in the cuBLASXt API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`side` |  | input | indicates if matrix `A` is on the left or right of `B`.  
`uplo` |  | input | indicates if matrix `A` lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.  
`trans` |  | input | operation op(`A`) that is non- or (conj.) transpose.  
`diag` |  | input | indicates if the elements on the main diagonal of matrix `A` are unity and should not be accessed.  
`m` |  | input | number of rows of matrix `B`, with matrix `A` sized accordingly.  
`n` |  | input | number of columns of matrix `B`, with matrix `A` sized accordingly.  
`alpha` | host | input | <_type_ > scalar used for multiplication, if `alpha` then `A` is not referenced and `B` does not have to be a valid input.  
`A` | host or device | input | <_type_ > array of dimension `lda x m` with `lda >= max(1, m)` if `side == CUBLAS_SIDE_LEFT` and `lda x n` with `lda >= max(1, n)` otherwise.  
`lda` |  | input | leading dimension of two-dimensional array used to store matrix `A`.  
`B` | host or device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, m)`.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
`C` | host or device | in/out | <_type_ > array of dimension `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `m < 0` or `n < 0`  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[strmm()](http://www.netlib.org/blas/strmm.f), [dtrmm()](http://www.netlib.org/blas/dtrmm.f), [ctrmm()](http://www.netlib.org/blas/ctrmm.f), [ztrmm()](http://www.netlib.org/blas/ztrmm.f)

###  4.4.12. cublasXt<t>spmm() 
    
    
    cublasStatus_t cublasXtSspmm( cublasXtHandle_t handle,
                                    cublasSideMode_t side,
                                    cublasFillMode_t uplo,
                                    size_t m,
                                    size_t n,
                                    const float *alpha,
                                    const float *AP,
                                    const float *B,
                                    size_t ldb,
                                    const float *beta,
                                    float *C,
                                    size_t ldc );
    
    cublasStatus_t cublasXtDspmm( cublasXtHandle_t handle,
                                    cublasSideMode_t side,
                                    cublasFillMode_t uplo,
                                    size_t m,
                                    size_t n,
                                    const double *alpha,
                                    const double *AP,
                                    const double *B,
                                    size_t ldb,
                                    const double *beta,
                                    double *C,
                                    size_t ldc );
    
    cublasStatus_t cublasXtCspmm( cublasXtHandle_t handle,
                                    cublasSideMode_t side,
                                    cublasFillMode_t uplo,
                                    size_t m,
                                    size_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *AP,
                                    const cuComplex *B,
                                    size_t ldb,
                                    const cuComplex *beta,
                                    cuComplex *C,
                                    size_t ldc );
    
    cublasStatus_t cublasXtZspmm( cublasXtHandle_t handle,
                                    cublasSideMode_t side,
                                    cublasFillMode_t uplo,
                                    size_t m,
                                    size_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *AP,
                                    const cuDoubleComplex *B,
                                    size_t ldb,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *C,
                                    size_t ldc );
    

This function performs the symmetric packed matrix-matrix multiplication

\\(C = \left\\{ \begin{matrix} {\alpha AB + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_LEFT}$}} \\\ {\alpha BA + \beta C} & {\text{if }\textsf{side == $\mathrm{CUBLAS\\_SIDE\\_RIGHT}$}} \\\ \end{matrix} \right.\\)

where \\(A\\) is a \\(n \times n\\) symmetric matrix stored in packed format, \\(B\\) and \\(C\\) are \\(m \times n\\) matrices, and \\(\alpha\\) and \\(\beta\\) are scalars.

If `uplo == CUBLAS_FILL_MODE_LOWER` then the elements in the lower triangular part of the symmetric matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+((2*n-j+1)*j)/2]` for \\(j = 1,\ldots,n\\) and \\(i \geq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

If `uplo == CUBLAS_FILL_MODE_UPPER` then the elements in the upper triangular part of the symmetric matrix \\(A\\) are packed together column by column without gaps, so that the element \\(A(i,j)\\) is stored in the memory location `AP[i+(j*(j+1))/2]` for \\(j = 1,\ldots,n\\) and \\(i \leq j\\) . Consequently, the packed format requires only \\(\frac{n(n + 1)}{2}\\) elements for storage.

Note

The packed matrix AP must be located on the host or managed memory whereas the other matrices can be located on the host or any GPU device

Param. | Memory | In/out | Meaning  
---|---|---|---  
`handle` |  | input | handle to the cuBLASXt API context.  
`side` |  | input | indicates if matrix `A` is on the left or right of `B`.  
`uplo` |  | input | indicates if matrix `A` lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements.  
`m` |  | input | number of rows of matrix `A` and `B`, with matrix `A` sized accordingly.  
`n` |  | input | number of columns of matrix `C` and `A`, with matrix `A` sized accordingly.  
`alpha` | host | input | <_type_ > scalar used for multiplication.  
`AP` | host | input | <_type_ > array with \\(A\\) stored in packed format.  
`B` | host or device | input | <_type_ > array of dimension `ldb x n` with `ldb >= max(1, m)`.  
`ldb` |  | input | leading dimension of two-dimensional array used to store matrix `B`.  
`beta` | host | input | <_type_ > scalar used for multiplication, if `beta == 0` then `C` does not have to be a valid input.  
`C` | host or device | in/out | <_type_ > array of dimension `ldc x n` with `ldc >= max(1, m)`.  
`ldc` |  | input | leading dimension of two-dimensional array used to store matrix `C`.  
  
The possible error values returned by this function and their meanings are listed below.

Error Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_INVALID_VALUE` | the parameters `m < 0` or `n < 0`  
`CUBLAS_STATUS_NOT_SUPPORTED` | the matrix AP is located on a GPU device  
`CUBLAS_STATUS_EXECUTION_FAILED` | the function failed to launch on the GPU  
  
For references please refer to NETLIB documentation:

[ssymm()](http://www.netlib.org/blas/ssymm.f), [dsymm()](http://www.netlib.org/blas/dsymm.f), [csymm()](http://www.netlib.org/blas/csymm.f), [zsymm()](http://www.netlib.org/blas/zsymm.f)
