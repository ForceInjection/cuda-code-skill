# 15.22. __nv_fp8x2_e5m2

**Source:** [struct____nv__fp8x2__e5m2.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8x2__e5m2.html)

---

#  15.22. __nv_fp8x2_e5m2[](#nv-fp8x2-e5m2 "Permalink to this headline")

struct __nv_fp8x2_e5m2[](#_CPPv415__nv_fp8x2_e5m2 "Permalink to this definition")  

    

[__nv_fp8x2_e5m2](#struct____nv__fp8x2__e5m2) datatype 

This structure implements the datatype for handling two `fp8` floating-point numbers of `e5m2` kind each: with 1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits.

The structure implements converting constructors and operators. 

Public Functions

__nv_fp8x2_e5m2() = default[](#_CPPv4N15__nv_fp8x2_e5m215__nv_fp8x2_e5m2Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp8x2_e5m2(const [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") f)[](#_CPPv4N15__nv_fp8x2_e5m215__nv_fp8x2_e5m2EK7__half2 "Permalink to this definition")  

    

Constructor from `[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8x2_e5m2(const [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162") f)[](#_CPPv4N15__nv_fp8x2_e5m215__nv_fp8x2_e5m2EK14__nv_bfloat162 "Permalink to this definition")  

    

Constructor from `[__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8x2_e5m2(const double2 f)[](#_CPPv4N15__nv_fp8x2_e5m215__nv_fp8x2_e5m2EK7double2 "Permalink to this definition")  

    

Constructor from `double2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8x2_e5m2(const float2 f)[](#_CPPv4N15__nv_fp8x2_e5m215__nv_fp8x2_e5m2EK6float2 "Permalink to this definition")  

    

Constructor from `float2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit operator [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2")() const[](#_CPPv4NK15__nv_fp8x2_e5m2cv7__half2Ev "Permalink to this definition")  

    

Conversion operator to `[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2)` data type. 

__host__ __device__ inline explicit operator float2() const[](#_CPPv4NK15__nv_fp8x2_e5m2cv6float2Ev "Permalink to this definition")  

    

Conversion operator to `float2` data type. 

Public Members

[__nv_fp8x2_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#_CPPv420__nv_fp8x2_storage_t "__nv_fp8x2_storage_t") __x[](#_CPPv4N15__nv_fp8x2_e5m23__xE "Permalink to this definition")  

    

Storage variable contains the vector of two `fp8` floating-point data values.