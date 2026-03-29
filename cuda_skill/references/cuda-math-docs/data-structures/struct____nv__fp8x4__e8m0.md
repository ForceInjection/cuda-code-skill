# 15.26. __nv_fp8x4_e8m0

**Source:** struct____nv__fp8x4__e8m0.html


#  15.26. __nv_fp8x4_e8m0

struct __nv_fp8x4_e8m0



__nv_fp8x4_e8m0 datatype

This structure implements the datatype for storage and operations on the vector of scale factors of `e8m0` kind each.

Public Functions

__nv_fp8x4_e8m0() = default



Constructor by default.

__host__ __device__ inline explicit __nv_fp8x4_e8m0(const __half2 flo, const __half2 fhi)



Constructor from a pair of `__half2` data type values, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8x4_e8m0(const __nv_bfloat162 flo, const __nv_bfloat162 fhi)



Constructor from a pair of `__nv_bfloat162` data type values, relies on `__NV_SATFINITE` behavior for out-of-range values.

inline explicit __NV_SILENCE_DEPRECATION_BEGIN __host__ __device__ __nv_fp8x4_e8m0(const double4 f)



Constructor from `double4` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

inline explicit __NV_SILENCE_DEPRECATION_END __host__ __device__ __nv_fp8x4_e8m0(const double4_16a f)



Constructor from `double4_16a` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8x4_e8m0(const double4_32a f)



Constructor from `double4_32a` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8x4_e8m0(const float4 f)



Constructor from `float4` vector data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit operator float4() const



Conversion operator to `float4` vector data type.

Public Members

__nv_fp8x4_storage_t __x



Storage variable contains the vector of four scale factor values.