# 15.9. __nv_fp4_e2m1

**Source:** [struct____nv__fp4__e2m1.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp4__e2m1.html)

---

#  15.9. __nv_fp4_e2m1[](#nv-fp4-e2m1 "Permalink to this headline")

struct __nv_fp4_e2m1[](#_CPPv413__nv_fp4_e2m1 "Permalink to this definition")  

    

[__nv_fp4_e2m1](#struct____nv__fp4__e2m1) datatype 

This structure implements the datatype for handling `fp4` floating-point numbers of `e2m1` kind: with 1 sign, 2 exponent, 1 implicit and 1 explicit mantissa bits. This encoding does not support Inf/NaN.

The structure implements converting constructors and operators. 

Public Functions

__host__ __device__ inline __nv_fp4_e2m1()[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") f)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EK6__half "Permalink to this definition")  

    

Constructor from `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") f)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EK13__nv_bfloat16 "Permalink to this definition")  

    

Constructor from `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const double f)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKd "Permalink to this definition")  

    

Constructor from `double` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const float f)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKf "Permalink to this definition")  

    

Constructor from `float` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const int val)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKi "Permalink to this definition")  

    

Constructor from `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const long int val)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKl "Permalink to this definition")  

    

Constructor from `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const long long int val)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKx "Permalink to this definition")  

    

Constructor from `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const short int val)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKs "Permalink to this definition")  

    

Constructor from `short` `int` data type. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const unsigned int val)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKj "Permalink to this definition")  

    

Constructor from `unsigned` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const unsigned long int val)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKm "Permalink to this definition")  

    

Constructor from `unsigned` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const unsigned long long int val)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKy "Permalink to this definition")  

    

Constructor from `unsigned` `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp4_e2m1(const unsigned short int val)[](#_CPPv4N13__nv_fp4_e2m113__nv_fp4_e2m1EKt "Permalink to this definition")  

    

Constructor from `unsigned` `short` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

Public Members

[__nv_fp4_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html#_CPPv418__nv_fp4_storage_t "__nv_fp4_storage_t") __x[](#_CPPv4N13__nv_fp4_e2m13__xE "Permalink to this definition")  

    

Storage variable contains the `fp4` floating-point data.