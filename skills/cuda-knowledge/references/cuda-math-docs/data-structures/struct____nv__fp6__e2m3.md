# 15.12. __nv_fp6_e2m3

**Source:** struct____nv__fp6__e2m3.html


#  15.12. __nv_fp6_e2m3

struct __nv_fp6_e2m3



__nv_fp6_e2m3 datatype

This structure implements the datatype for storing `fp6` floating-point numbers of `e2m3` kind: with 1 sign, 2 exponent, 1 implicit and 3 explicit mantissa bits. This encoding does not support Inf/NaN.

The structure implements converting constructors and operators.

Public Functions

__host__ __device__ inline __nv_fp6_e2m3()



Constructor by default.

__host__ __device__ inline explicit __nv_fp6_e2m3(const __half f)



Constructor from `__half` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode.

__host__ __device__ inline explicit __nv_fp6_e2m3(const __nv_bfloat16 f)



Constructor from `__nv_bfloat16` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode.

__host__ __device__ inline explicit __nv_fp6_e2m3(const double f)



Constructor from `double` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode.

__host__ __device__ inline explicit __nv_fp6_e2m3(const float f)



Constructor from `float` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode.

__host__ __device__ inline explicit __nv_fp6_e2m3(const int val)



Constructor from `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp6_e2m3(const long int val)



Constructor from `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp6_e2m3(const long long int val)



Constructor from `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp6_e2m3(const short int val)



Constructor from `short` `int` data type.

__host__ __device__ inline explicit __nv_fp6_e2m3(const unsigned int val)



Constructor from `unsigned` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp6_e2m3(const unsigned long int val)



Constructor from `unsigned` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp6_e2m3(const unsigned long long int val)



Constructor from `unsigned` `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp6_e2m3(const unsigned short int val)



Constructor from `unsigned` `short` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

Public Members

__nv_fp6_storage_t __x



Storage variable contains the `fp6` floating-point data.