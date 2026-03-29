# 6. Using the cuBLAS Legacy API


This section does not provide a full reference of each Legacy API datatype and entry point. Instead, it describes how to use the API, especially where this is different from the regular cuBLAS API.


Note that in this section, all references to the “cuBLAS Library” refer to the Legacy cuBLAS API only.


Warning

The legacy cuBLAS API is deprecated and will be removed in future release.


##  6.1. Error Status 

The `cublasStatus` type is used for function status returns. The cuBLAS Library helper functions return status directly, while the status of core functions can be retrieved using `cublasGetError()`. Notice that reading the error status via `cublasGetError()`, resets the internal error state to `CUBLAS_STATUS_SUCCESS`. Currently, the following values are defined:

Value | Meaning  
---|---  
`CUBLAS_STATUS_SUCCESS` | the operation completed successfully  
`CUBLAS_STATUS_NOT_INITIALIZED` | the library was not initialized  
`CUBLAS_STATUS_ALLOC_FAILED` | the resource allocation failed  
`CUBLAS_STATUS_INVALID_VALUE` | an invalid numerical value was used as an argument  
`CUBLAS_STATUS_ARCH_MISMATCH` | an absent device architectural feature is required  
`CUBLAS_STATUS_MAPPING_ERROR` | an access to GPU memory space failed  
`CUBLAS_STATUS_EXECUTION_FAILED` | the GPU program failed to execute  
`CUBLAS_STATUS_INTERNAL_ERROR` | an internal operation failed  
`CUBLAS_STATUS_NOT_SUPPORTED` | the feature required is not supported  
  
This legacy type corresponds to type [cublasStatus_t](#cublasstatus-t) in the cuBLAS library API.


##  6.2. Initialization and Shutdown 

The functions `cublasInit()` and `cublasShutdown()` are used to initialize and shutdown the cuBLAS library. It is recommended for `cublasInit()` to be called before any other function is invoked. It allocates hardware resources on the GPU device that is currently bound to the host thread from which it was invoked.

The legacy initialization and shutdown functions are similar to the cuBLAS library API routines [cublasCreate()](#cublascreate) and [cublasDestroy()](#cublasdestroy).


##  6.3. Thread Safety 

The legacy API is not thread safe when used with multiple host threads and devices. It is recommended to be used only when utmost compatibility with Fortran is required and when a single host thread is used to setup the library and make all the functions calls.


##  6.4. Memory Management 

The memory used by the legacy cuBLAS library API is allocated and released using functions `cublasAlloc()` and `cublasFree()`, respectively. These functions create and destroy an object in the GPU memory space capable of holding an array of `n` elements, where each element requires `elemSize` bytes of storage. Please see the legacy cuBLAS API header file “cublas.h” for the prototypes of these functions.

The function `cublasAlloc()` is a wrapper around the function `cudaMalloc()`, therefore device pointers returned by `cublasAlloc()` can be passed to any CUDA™ device kernel functions. However, these device pointers can not be dereferenced in the host code. The function `cublasFree()` is a wrapper around the function `cudaFree()`.


##  6.5. Scalar Parameters 

In the legacy cuBLAS API, scalar parameters are passed by value from the host. Also, the few functions that do return a scalar result, such as dot() and nrm2(), return the resulting value on the host, and hence these routines will wait for kernel execution on the device to complete before returning, which makes parallelism with streams impractical. However, the majority of functions do not return any value, in order to be more compatible with Fortran and the existing BLAS libraries.


##  6.6. Helper Functions 

In this section we list the helper functions provided by the legacy cuBLAS API and their functionality. For the exact prototypes of these functions please refer to the legacy cuBLAS API header file “cublas.h”.

Helper function | Meaning  
---|---  
`cublasInit()` | initialize the library  
`cublasShutdown()` | shuts down the library  
`cublasGetError()` | retrieves the error status of the library  
`cublasSetKernelStream()` | sets the stream to be used by the library  
`cublasAlloc()` | allocates the device memory for the library  
`cublasFree()` | releases the device memory allocated for the library  
`cublasSetVector()` | copies a vector `x` on the host to a vector on the GPU  
`cublasGetVector()` | copies a vector `x` on the GPU to a vector on the host  
`cublasSetMatrix()` | copies a \\(m \times n\\) tile from a matrix on the host to the GPU  
`cublasGetMatrix()` | copies a \\(m \times n\\) tile from a matrix on the GPU to the host  
`cublasSetVectorAsync()` | similar to `cublasSetVector()`, but the copy is asynchronous  
`cublasGetVectorAsync()` | similar to `cublasGetVector()`, but the copy is asynchronous  
`cublasSetMatrixAsync()` | similar to `cublasSetMatrix()`, but the copy is asynchronous  
`cublasGetMatrixAsync()` | similar to `cublasGetMatrix()`, but the copy is asynchronous


##  6.7. Level-1,2,3 Functions   
  
The Level-1,2,3 cuBLAS functions (also called core functions) have the same name and behavior as the ones listed in the chapters 3, 4 and 5 in this document. Please refer to the legacy cuBLAS API header file “cublas.h” for their exact prototype. Also, the next section talks a bit more about the differences between the legacy and the cuBLAS API prototypes, more specifically how to convert the function calls from one API to another.


##  6.8. Converting Legacy to the cuBLAS API 

There are a few general rules that can be used to convert from legacy to the cuBLAS API:

  * Exchange the header file “cublas.h” for “cublas_v2.h”.

  * Exchange the type `cublasStatus` for [cublasStatus_t](#cublasstatus-t).

  * Exchange the function `cublasSetKernelStream()` for [cublasSetStream()](#cublassetstream).

  * Exchange the function `cublasAlloc()` and `cublasFree()` for `cudaMalloc()` and `cudaFree()`, respectively. Notice that `cudaMalloc()` expects the size of the allocated memory to be provided in bytes (usually simply provide `n x elemSize` to allocate `n` elements, each of size `elemSize` bytes).

  * Declare the `cublasHandle_t` cuBLAS library handle.

  * Initialize the handle using [cublasCreate()](#cublascreate). Also, release the handle once finished using [cublasDestroy()](#cublasdestroy).

  * Add the handle as the first parameter to all the cuBLAS library function calls.

  * Change the scalar parameters to be passed by reference, instead of by value (usually simply adding “&” symbol in C/C++ is enough, because the parameters are passed by reference on the host by _default_). However, note that if the routine is running asynchronously, then the variable holding the scalar parameter cannot be changed until the kernels that the routine dispatches are completed. See the CUDA C++ Programming Guide for a detailed discussion of how to use streams.

  * Change the parameter characters `N` or `n` (non-transpose operation), `T` or `t` (transpose operation) and `C` or `c` (conjugate transpose operation) to `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_C`, respectively.

  * Change the parameter characters `L` or `l` (lower part filled) and `U` or `u` (upper part filled) to `CUBLAS_FILL_MODE_LOWER` and `CUBLAS_FILL_MODE_UPPER`, respectively.

  * Change the parameter characters `N` or `n` (non-unit diagonal) and `U` or `u` (unit diagonal) to `CUBLAS_DIAG_NON_UNIT` and `CUBLAS_DIAG_UNIT`, respectively.

  * Change the parameter characters `L` or `l` (left side) and `R` or `r` (right side) to `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT`, respectively.

  * If the legacy API function returns a scalar value, add an extra scalar parameter of the same type passed by reference, as the last parameter to the same function.

  * Instead of using `cublasGetError()`, use the return value of the function itself to check for errors.

  * Finally, please use the function prototypes in the header files `cublas.h` and `cublas_v2.h` to check the code for correctness.


##  6.9. Examples 

For sample code references that use the legacy cuBLAS API please see the two examples below. They show an application written in C using the legacy cuBLAS library API with two indexing styles (Example A.1. “Application Using C and cuBLAS: 1-based indexing” and Example A.2. “Application Using C and cuBLAS: 0-based Indexing”). This application is analogous to the one using the cuBLAS library API that is shown in the Introduction chapter.

Example A.1. Application Using C and cuBLAS: 1-based indexing
    
    
    //-----------------------------------------------------------
    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include "cublas.h"
    #define M 6
    #define N 5
    #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
    
    static __inline__ void modify (float *m, int ldm, int n, int p, int q, float alpha, float beta){
        cublasSscal (n-q+1, alpha, &m[IDX2F(p,q,ldm)], ldm);
        cublasSscal (ldm-p+1, beta, &m[IDX2F(p,q,ldm)], 1);
    }
    
    int main (void){
        int i, j;
        cublasStatus stat;
        float* devPtrA;
        float* a = 0;
        a = (float *)malloc (M * N * sizeof (*a));
        if (!a) {
            printf ("host memory allocation failed");
            return EXIT_FAILURE;
        }
        for (j = 1; j <= N; j++) {
            for (i = 1; i <= M; i++) {
                a[IDX2F(i,j,M)] = (float)((i-1) * M + j);
            }
        }
        cublasInit();
        stat = cublasAlloc (M*N, sizeof(*a), (void**)&devPtrA);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("device memory allocation failed");
            cublasShutdown();
            return EXIT_FAILURE;
        }
        stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data download failed");
            cublasFree (devPtrA);
            cublasShutdown();
            return EXIT_FAILURE;
        }
        modify (devPtrA, M, N, 2, 3, 16.0f, 12.0f);
        stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data upload failed");
            cublasFree (devPtrA);
            cublasShutdown();
            return EXIT_FAILURE;
        }
        cublasFree (devPtrA);
        cublasShutdown();
        for (j = 1; j <= N; j++) {
            for (i = 1; i <= M; i++) {
                printf ("%7.0f", a[IDX2F(i,j,M)]);
            }
            printf ("\n");
        }
        free(a);
        return EXIT_SUCCESS;
    }
    

Example A.2. Application Using C and cuBLAS: 0-based indexing
    
    
    //-----------------------------------------------------------
    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include "cublas.h"
    #define M 6
    #define N 5
    #define IDX2C(i,j,ld) (((j)*(ld))+(i))
    
    static __inline__ void modify (float *m, int ldm, int n, int p, int q, float alpha, float beta){
        cublasSscal (n-q, alpha, &m[IDX2C(p,q,ldm)], ldm);
        cublasSscal (ldm-p, beta, &m[IDX2C(p,q,ldm)], 1);
    }
    
    int main (void){
        int i, j;
        cublasStatus stat;
        float* devPtrA;
        float* a = 0;
        a = (float *)malloc (M * N * sizeof (*a));
        if (!a) {
            printf ("host memory allocation failed");
            return EXIT_FAILURE;
        }
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
            }
        }
        cublasInit();
        stat = cublasAlloc (M*N, sizeof(*a), (void**)&devPtrA);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("device memory allocation failed");
            cublasShutdown();
            return EXIT_FAILURE;
        }
        stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data download failed");
            cublasFree (devPtrA);
            cublasShutdown();
            return EXIT_FAILURE;
        }
        modify (devPtrA, M, N, 1, 2, 16.0f, 12.0f);
        stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("data upload failed");
            cublasFree (devPtrA);
            cublasShutdown();
            return EXIT_FAILURE;
        }
        cublasFree (devPtrA);
        cublasShutdown();
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                printf ("%7.0f", a[IDX2C(i,j,M)]);
            }
            printf ("\n");
        }
        free(a);
        return EXIT_SUCCESS;
    }
    
