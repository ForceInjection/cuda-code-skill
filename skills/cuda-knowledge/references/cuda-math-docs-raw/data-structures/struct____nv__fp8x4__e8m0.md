# 15.26. __nv_fp8x4_e8m0

**Source:** [struct____nv__fp8x4__e8m0.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8x4__e8m0.html)

---

#  15.26. __nv_fp8x4_e8m0[](#nv-fp8x4-e8m0 "Permalink to this headline")

struct __nv_fp8x4_e8m0[](#_CPPv415__nv_fp8x4_e8m0 "Permalink to this definition")  

    

[__nv_fp8x4_e8m0](#struct____nv__fp8x4__e8m0) datatype 

This structure implements the datatype for storage and operations on the vector of scale factors of `e8m0` kind each. 

Public Functions

__nv_fp8x4_e8m0() = default[](#_CPPv4N15__nv_fp8x4_e8m015__nv_fp8x4_e8m0Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp8x4_e8m0(const [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") flo, const [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") fhi)[](#_CPPv4N15__nv_fp8x4_e8m015__nv_fp8x4_e8m0EK7__half2K7__half2 "Permalink to this definition")  

    

Constructor from a pair of `[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2)` data type values, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8x4_e8m0(const [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162") flo, const [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162") fhi)[](#_CPPv4N15__nv_fp8x4_e8m015__nv_fp8x4_e8m0EK14__nv_bfloat162K14__nv_bfloat162 "Permalink to this definition")  

    

Constructor from a pair of `[__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)` data type values, relies on `__NV_SATFINITE` behavior for out-of-range values. 

inline explicit __NV_SILENCE_DEPRECATION_BEGIN __host__ __device__ __nv_fp8x4_e8m0(const double4 f)[](#_CPPv4N15__nv_fp8x4_e8m015__nv_fp8x4_e8m0EK7double4 "Permalink to this definition")  

    

Constructor from `double4` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

inline explicit __NV_SILENCE_DEPRECATION_END __host__ __device__ __nv_fp8x4_e8m0(const double4_16a f)[](#_CPPv4N15__nv_fp8x4_e8m015__nv_fp8x4_e8m0EK11double4_16a "Permalink to this definition")  

    

Constructor from `double4_16a` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8x4_e8m0(const double4_32a f)[](#_CPPv4N15__nv_fp8x4_e8m015__nv_fp8x4_e8m0EK11double4_32a "Permalink to this definition")  

    

Constructor from `double4_32a` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8x4_e8m0(const float4 f)[](#_CPPv4N15__nv_fp8x4_e8m015__nv_fp8x4_e8m0EK6float4 "Permalink to this definition")  

    

Constructor from `float4` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit operator float4() const[](#_CPPv4NK15__nv_fp8x4_e8m0cv6float4Ev "Permalink to this definition")  

    

Conversion operator to `float4` vector data type. 

Public Members

[__nv_fp8x4_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#_CPPv420__nv_fp8x4_storage_t "__nv_fp8x4_storage_t") __x[](#_CPPv4N15__nv_fp8x4_e8m03__xE "Permalink to this definition")  

    

Storage variable contains the vector of four scale factor values.