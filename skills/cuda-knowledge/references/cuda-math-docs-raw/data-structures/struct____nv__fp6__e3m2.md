# 15.13. __nv_fp6_e3m2

**Source:** [struct____nv__fp6__e3m2.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp6__e3m2.html)

---

#  15.13. __nv_fp6_e3m2[](#nv-fp6-e3m2 "Permalink to this headline")

struct __nv_fp6_e3m2[](#_CPPv413__nv_fp6_e3m2 "Permalink to this definition")  

    

[__nv_fp6_e3m2](#struct____nv__fp6__e3m2) datatype 

This structure implements the datatype for handling `fp6` floating-point numbers of `e3m2` kind: with 1 sign, 3 exponent, 1 implicit and 2 explicit mantissa bits. This encoding does not support Inf/NaN.

The structure implements converting constructors and operators. 

Public Functions

__host__ __device__ inline __nv_fp6_e3m2()[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") f)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EK6__half "Permalink to this definition")  

    

Constructor from `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") f)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EK13__nv_bfloat16 "Permalink to this definition")  

    

Constructor from `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const double f)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKd "Permalink to this definition")  

    

Constructor from `double` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const float f)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKf "Permalink to this definition")  

    

Constructor from `float` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const int val)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKi "Permalink to this definition")  

    

Constructor from `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const long int val)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKl "Permalink to this definition")  

    

Constructor from `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const long long int val)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKx "Permalink to this definition")  

    

Constructor from `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const short int val)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKs "Permalink to this definition")  

    

Constructor from `short` `int` data type. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const unsigned int val)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKj "Permalink to this definition")  

    

Constructor from `unsigned` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const unsigned long int val)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKm "Permalink to this definition")  

    

Constructor from `unsigned` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const unsigned long long int val)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKy "Permalink to this definition")  

    

Constructor from `unsigned` `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp6_e3m2(const unsigned short int val)[](#_CPPv4N13__nv_fp6_e3m213__nv_fp6_e3m2EKt "Permalink to this definition")  

    

Constructor from `unsigned` `short` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

Public Members

[__nv_fp6_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html#_CPPv418__nv_fp6_storage_t "__nv_fp6_storage_t") __x[](#_CPPv4N13__nv_fp6_e3m23__xE "Permalink to this definition")  

    

Storage variable contains the `fp6` floating-point data.