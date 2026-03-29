# 15.23. __nv_fp8x2_e8m0

**Source:** struct____nv__fp8x2__e8m0.html


#  15.23. __nv_fp8x2_e8m0

struct __nv_fp8x2_e8m0



__nv_fp8x2_e8m0 datatype

This structure implements the datatype for storage and operations on the vector of two scale factors of `e8m0` kind each.

Public Functions

__nv_fp8x2_e8m0() = default



Constructor by default.

__host__ __device__ inline explicit __nv_fp8x2_e8m0(const __half2 f)



Constructor from `__half2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8x2_e8m0(const __nv_bfloat162 f)



Constructor from `__nv_bfloat162` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8x2_e8m0(const double2 f)



Constructor from `double2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8x2_e8m0(const float2 f)



Constructor from `float2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit operator __half2() const



Conversion operator to `__half2` data type.

__host__ __device__ inline explicit operator __nv_bfloat162() const



Conversion operator to `__nv_bfloat162` data type.

__host__ __device__ inline explicit operator float2() const



Conversion operator to `float2` data type.

Public Members

__nv_fp8x2_storage_t __x



Storage variable contains the vector of two scale factor values.