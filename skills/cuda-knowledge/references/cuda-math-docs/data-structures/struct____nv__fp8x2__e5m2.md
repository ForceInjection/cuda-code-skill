# 15.22. __nv_fp8x2_e5m2

**Source:** struct____nv__fp8x2__e5m2.html


#  15.22. __nv_fp8x2_e5m2

struct __nv_fp8x2_e5m2



__nv_fp8x2_e5m2 datatype

This structure implements the datatype for handling two `fp8` floating-point numbers of `e5m2` kind each: with 1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits.

The structure implements converting constructors and operators.

Public Functions

__nv_fp8x2_e5m2() = default



Constructor by default.

__host__ __device__ inline explicit __nv_fp8x2_e5m2(const __half2 f)



Constructor from `__half2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8x2_e5m2(const __nv_bfloat162 f)



Constructor from `__nv_bfloat162` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8x2_e5m2(const double2 f)



Constructor from `double2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8x2_e5m2(const float2 f)



Constructor from `float2` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit operator __half2() const



Conversion operator to `__half2` data type.

__host__ __device__ inline explicit operator float2() const



Conversion operator to `float2` data type.

Public Members

__nv_fp8x2_storage_t __x



Storage variable contains the vector of two `fp8` floating-point data values.