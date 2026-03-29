# 15.9. __nv_fp4_e2m1

**Source:** struct____nv__fp4__e2m1.html


#  15.9. __nv_fp4_e2m1

struct __nv_fp4_e2m1



__nv_fp4_e2m1 datatype

This structure implements the datatype for handling `fp4` floating-point numbers of `e2m1` kind: with 1 sign, 2 exponent, 1 implicit and 1 explicit mantissa bits. This encoding does not support Inf/NaN.

The structure implements converting constructors and operators.

Public Functions

__host__ __device__ inline __nv_fp4_e2m1()



Constructor by default.

__host__ __device__ inline explicit __nv_fp4_e2m1(const __half f)



Constructor from `__half` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode.

__host__ __device__ inline explicit __nv_fp4_e2m1(const __nv_bfloat16 f)



Constructor from `__nv_bfloat16` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode.

__host__ __device__ inline explicit __nv_fp4_e2m1(const double f)



Constructor from `double` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode.

__host__ __device__ inline explicit __nv_fp4_e2m1(const float f)



Constructor from `float` data type, relies on `__NV_SATFINITE` behavior for out-of-range values and `cudaRoundNearest` rounding mode.

__host__ __device__ inline explicit __nv_fp4_e2m1(const int val)



Constructor from `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp4_e2m1(const long int val)



Constructor from `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp4_e2m1(const long long int val)



Constructor from `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp4_e2m1(const short int val)



Constructor from `short` `int` data type.

__host__ __device__ inline explicit __nv_fp4_e2m1(const unsigned int val)



Constructor from `unsigned` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp4_e2m1(const unsigned long int val)



Constructor from `unsigned` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp4_e2m1(const unsigned long long int val)



Constructor from `unsigned` `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp4_e2m1(const unsigned short int val)



Constructor from `unsigned` `short` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

Public Members

__nv_fp4_storage_t __x



Storage variable contains the `fp4` floating-point data.