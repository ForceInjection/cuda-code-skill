# 15.14. __nv_fp6x2_e2m3

**Source:** [struct____nv__fp6x2__e2m3.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp6x2__e2m3.html)

---

#  15.14. __nv_fp6x2_e2m3[](#nv-fp6x2-e2m3 "Permalink to this headline")

struct __nv_fp6x2_e2m3[](#_CPPv415__nv_fp6x2_e2m3 "Permalink to this definition")  

    

[__nv_fp6x2_e2m3](#struct____nv__fp6x2__e2m3) datatype 

This structure implements the datatype for handling two `fp6` floating-point numbers of `e2m3` kind each.

The structure implements converting constructors and operators. 

Public Functions

__host__ __device__ inline __nv_fp6x2_e2m3()[](#_CPPv4N15__nv_fp6x2_e2m315__nv_fp6x2_e2m3Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp6x2_e2m3(const [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") f)[](#_CPPv4N15__nv_fp6x2_e2m315__nv_fp6x2_e2m3EK7__half2 "Permalink to this definition")  

    

Constructor from `[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6x2_e2m3(const [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162") f)[](#_CPPv4N15__nv_fp6x2_e2m315__nv_fp6x2_e2m3EK14__nv_bfloat162 "Permalink to this definition")  

    

Constructor from `[__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6x2_e2m3(const double2 f)[](#_CPPv4N15__nv_fp6x2_e2m315__nv_fp6x2_e2m3EK7double2 "Permalink to this definition")  

    

Constructor from `double2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6x2_e2m3(const float2 f)[](#_CPPv4N15__nv_fp6x2_e2m315__nv_fp6x2_e2m3EK6float2 "Permalink to this definition")  

    

Constructor from `float2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

Public Members

[__nv_fp6x2_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html#_CPPv420__nv_fp6x2_storage_t "__nv_fp6x2_storage_t") __x[](#_CPPv4N15__nv_fp6x2_e2m33__xE "Permalink to this definition")  

    

Storage variable contains the vector of two `fp6` floating-point data values.