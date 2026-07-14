# 15.12. __nv_fp6_e2m3

**Source:** [struct____nv__fp6__e2m3.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp6__e2m3.html)

---

#  15.12. __nv_fp6_e2m3[](#nv-fp6-e2m3 "Permalink to this headline")

struct __nv_fp6_e2m3[](#_CPPv413__nv_fp6_e2m3 "Permalink to this definition")  

    

[__nv_fp6_e2m3](#struct____nv__fp6__e2m3) datatype 

This structure implements the datatype for storing `fp6` floating-point numbers of `e2m3` kind: with 1 sign, 2 exponent, 1 implicit and 3 explicit mantissa bits. This encoding does not support Inf/NaN.

The structure implements converting constructors and operators. 

Public Functions

__host__ __device__ inline __nv_fp6_e2m3()[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") f)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EK6__half "Permalink to this definition")  

    

Constructor from `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") f)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EK13__nv_bfloat16 "Permalink to this definition")  

    

Constructor from `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const double f)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKd "Permalink to this definition")  

    

Constructor from `double` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const float f)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKf "Permalink to this definition")  

    

Constructor from `float` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const int val)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKi "Permalink to this definition")  

    

Constructor from `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const long int val)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKl "Permalink to this definition")  

    

Constructor from `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const long long int val)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKx "Permalink to this definition")  

    

Constructor from `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const short int val)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKs "Permalink to this definition")  

    

Constructor from `short` `int` data type. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const unsigned int val)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKj "Permalink to this definition")  

    

Constructor from `unsigned` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const unsigned long int val)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKm "Permalink to this definition")  

    

Constructor from `unsigned` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const unsigned long long int val)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKy "Permalink to this definition")  

    

Constructor from `unsigned` `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e2m3(const unsigned short int val)[](#_CPPv4N13__nv_fp6_e2m313__nv_fp6_e2m3EKt "Permalink to this definition")  

    

Constructor from `unsigned` `short` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

Public Members

[__nv_fp6_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html#_CPPv418__nv_fp6_storage_t "__nv_fp6_storage_t") __x[](#_CPPv4N13__nv_fp6_e2m33__xE "Permalink to this definition")  

    

Storage variable contains the `fp6` floating-point data.