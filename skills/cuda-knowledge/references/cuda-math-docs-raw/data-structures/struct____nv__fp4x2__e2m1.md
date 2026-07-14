# 15.10. __nv_fp4x2_e2m1

**Source:** [struct____nv__fp4x2__e2m1.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp4x2__e2m1.html)

---

#  15.10. __nv_fp4x2_e2m1[](#nv-fp4x2-e2m1 "Permalink to this headline")

struct __nv_fp4x2_e2m1[](#_CPPv415__nv_fp4x2_e2m1 "Permalink to this definition")  

    

[__nv_fp4x2_e2m1](#struct____nv__fp4x2__e2m1) datatype 

This structure implements the datatype for handling two `fp4` floating-point numbers of `e2m1` kind each.

The structure implements converting constructors and operators. 

Public Functions

__host__ __device__ inline __nv_fp4x2_e2m1()[](#_CPPv4N15__nv_fp4x2_e2m115__nv_fp4x2_e2m1Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp4x2_e2m1(const [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") f)[](#_CPPv4N15__nv_fp4x2_e2m115__nv_fp4x2_e2m1EK7__half2 "Permalink to this definition")  

    

Constructor from `[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp4x2_e2m1(const [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162") f)[](#_CPPv4N15__nv_fp4x2_e2m115__nv_fp4x2_e2m1EK14__nv_bfloat162 "Permalink to this definition")  

    

Constructor from `[__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp4x2_e2m1(const double2 f)[](#_CPPv4N15__nv_fp4x2_e2m115__nv_fp4x2_e2m1EK7double2 "Permalink to this definition")  

    

Constructor from `double2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp4x2_e2m1(const float2 f)[](#_CPPv4N15__nv_fp4x2_e2m115__nv_fp4x2_e2m1EK6float2 "Permalink to this definition")  

    

Constructor from `float2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

Public Members

[__nv_fp4x2_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html#_CPPv420__nv_fp4x2_storage_t "__nv_fp4x2_storage_t") __x[](#_CPPv4N15__nv_fp4x2_e2m13__xE "Permalink to this definition")  

    

Storage variable contains the vector of two `fp4` floating-point data values.