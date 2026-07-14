# 15.16. __nv_fp6x4_e2m3

**Source:** [struct____nv__fp6x4__e2m3.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp6x4__e2m3.html)

---

#  15.16. __nv_fp6x4_e2m3[](#nv-fp6x4-e2m3 "Permalink to this headline")

struct __nv_fp6x4_e2m3[](#_CPPv415__nv_fp6x4_e2m3 "Permalink to this definition")  

    

[__nv_fp6x4_e2m3](#struct____nv__fp6x4__e2m3) datatype 

This structure implements the datatype for handling four `fp6` floating-point numbers of `e2m3` kind each.

The structure implements converting constructors and operators. 

Public Functions

__host__ __device__ inline __nv_fp6x4_e2m3()[](#_CPPv4N15__nv_fp6x4_e2m315__nv_fp6x4_e2m3Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp6x4_e2m3(const [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") flo, const [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") fhi)[](#_CPPv4N15__nv_fp6x4_e2m315__nv_fp6x4_e2m3EK7__half2K7__half2 "Permalink to this definition")  

    

Constructor from a pair of `[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2)` data type values, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6x4_e2m3(const [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162") flo, const [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162") fhi)[](#_CPPv4N15__nv_fp6x4_e2m315__nv_fp6x4_e2m3EK14__nv_bfloat162K14__nv_bfloat162 "Permalink to this definition")  

    

Constructor from a pair of `[__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)` data type values, relies on `__NV_SATFINITE` behavior for out-of-range values. 

inline explicit __NV_SILENCE_DEPRECATION_BEGIN __host__ __device__ __nv_fp6x4_e2m3(const double4 f)[](#_CPPv4N15__nv_fp6x4_e2m315__nv_fp6x4_e2m3EK7double4 "Permalink to this definition")  

    

Constructor from `double4` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

inline explicit __NV_SILENCE_DEPRECATION_END __host__ __device__ __nv_fp6x4_e2m3(const double4_16a f)[](#_CPPv4N15__nv_fp6x4_e2m315__nv_fp6x4_e2m3EK11double4_16a "Permalink to this definition")  

    

Constructor from `double4_16a` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6x4_e2m3(const double4_32a f)[](#_CPPv4N15__nv_fp6x4_e2m315__nv_fp6x4_e2m3EK11double4_32a "Permalink to this definition")  

    

Constructor from `double4_32a` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6x4_e2m3(const float4 f)[](#_CPPv4N15__nv_fp6x4_e2m315__nv_fp6x4_e2m3EK6float4 "Permalink to this definition")  

    

Constructor from `float4` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

Public Members

[__nv_fp6x4_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html#_CPPv420__nv_fp6x4_storage_t "__nv_fp6x4_storage_t") __x[](#_CPPv4N15__nv_fp6x4_e2m33__xE "Permalink to this definition")  

    

Storage variable contains the vector of four `fp6` floating-point data values.