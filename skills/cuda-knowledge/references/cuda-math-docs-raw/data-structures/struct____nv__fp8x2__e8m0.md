# 15.23. __nv_fp8x2_e8m0

**Source:** [struct____nv__fp8x2__e8m0.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8x2__e8m0.html)

---

#  15.23. __nv_fp8x2_e8m0[](#nv-fp8x2-e8m0 "Permalink to this headline")

struct __nv_fp8x2_e8m0[](#_CPPv415__nv_fp8x2_e8m0 "Permalink to this definition")  

    

[__nv_fp8x2_e8m0](#struct____nv__fp8x2__e8m0) datatype 

This structure implements the datatype for storage and operations on the vector of two scale factors of `e8m0` kind each. 

Public Functions

__nv_fp8x2_e8m0() = default[](#_CPPv4N15__nv_fp8x2_e8m015__nv_fp8x2_e8m0Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp8x2_e8m0(const [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") f)[](#_CPPv4N15__nv_fp8x2_e8m015__nv_fp8x2_e8m0EK7__half2 "Permalink to this definition")  

    

Constructor from `[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8x2_e8m0(const [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162") f)[](#_CPPv4N15__nv_fp8x2_e8m015__nv_fp8x2_e8m0EK14__nv_bfloat162 "Permalink to this definition")  

    

Constructor from `[__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8x2_e8m0(const double2 f)[](#_CPPv4N15__nv_fp8x2_e8m015__nv_fp8x2_e8m0EK7double2 "Permalink to this definition")  

    

Constructor from `double2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8x2_e8m0(const float2 f)[](#_CPPv4N15__nv_fp8x2_e8m015__nv_fp8x2_e8m0EK6float2 "Permalink to this definition")  

    

Constructor from `float2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit operator [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2")() const[](#_CPPv4NK15__nv_fp8x2_e8m0cv7__half2Ev "Permalink to this definition")  

    

Conversion operator to `[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2)` data type. 

__host__ __device__ inline explicit operator [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162")() const[](#_CPPv4NK15__nv_fp8x2_e8m0cv14__nv_bfloat162Ev "Permalink to this definition")  

    

Conversion operator to `[__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)` data type. 

__host__ __device__ inline explicit operator float2() const[](#_CPPv4NK15__nv_fp8x2_e8m0cv6float2Ev "Permalink to this definition")  

    

Conversion operator to `float2` data type. 

Public Members

[__nv_fp8x2_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#_CPPv420__nv_fp8x2_storage_t "__nv_fp8x2_storage_t") __x[](#_CPPv4N15__nv_fp8x2_e8m03__xE "Permalink to this definition")  

    

Storage variable contains the vector of two scale factor values.