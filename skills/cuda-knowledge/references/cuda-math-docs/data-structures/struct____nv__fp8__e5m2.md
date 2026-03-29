# 15.19. __nv_fp8_e5m2

**Source:** struct____nv__fp8__e5m2.html


#  15.19. __nv_fp8_e5m2

struct __nv_fp8_e5m2



__nv_fp8_e5m2 datatype

This structure implements the datatype for handling `fp8` floating-point numbers of `e5m2` kind: with 1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits.

The structure implements converting constructors and operators.

Public Functions

__nv_fp8_e5m2() = default



Constructor by default.

__host__ __device__ inline explicit __nv_fp8_e5m2(const __half f)



Constructor from `__half` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const __nv_bfloat16 f)



Constructor from `__nv_bfloat16` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const double f)



Constructor from `double` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const float f)



Constructor from `float` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const int val)



Constructor from `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const long int val)



Constructor from `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const long long int val)



Constructor from `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const short int val)



Constructor from `short` `int` data type.

__host__ __device__ inline explicit __nv_fp8_e5m2(const unsigned int val)



Constructor from `unsigned` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const unsigned long int val)



Constructor from `unsigned` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const unsigned long long int val)



Constructor from `unsigned` `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit __nv_fp8_e5m2(const unsigned short int val)



Constructor from `unsigned` `short` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values.

__host__ __device__ inline explicit operator __half() const



Conversion operator to `__half` data type.

__host__ __device__ inline explicit operator __nv_bfloat16() const



Conversion operator to `__nv_bfloat16` data type.

__host__ __device__ inline explicit operator bool() const



Conversion operator to `bool` data type.

+0 and -0 inputs convert to `false`. Non-zero inputs convert to `true`.

__host__ __device__ inline explicit operator char() const



Conversion operator to an implementation defined `char` data type.

Detects signedness of the `char` type and proceeds accordingly, see further details in signed and unsigned char operators.

Clamps inputs to the output range. `NaN` inputs convert to `zero`.

__host__ __device__ inline explicit operator double() const



Conversion operator to `double` data type.

__host__ __device__ inline explicit operator float() const



Conversion operator to `float` data type.

__host__ __device__ inline explicit operator int() const



Conversion operator to `int` data type.

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`.

__host__ __device__ inline explicit operator long int() const



Conversion operator to `long` `int` data type.

Clamps too large inputs to the output range. `NaN` inputs convert to `zero` if output type is 32-bit. `NaN` inputs convert to `0x8000000000000000ULL` if output type is 64-bit.

__host__ __device__ inline explicit operator long long int() const



Conversion operator to `long` `long` `int` data type.

Clamps too large inputs to the output range. `NaN` inputs convert to `0x8000000000000000LL`.

__host__ __device__ inline explicit operator short int() const



Conversion operator to `short` `int` data type.

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`.

__host__ __device__ inline explicit operator signed char() const



Conversion operator to `signed` `char` data type.

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`.

__host__ __device__ inline explicit operator unsigned char() const



Conversion operator to `unsigned` `char` data type.

Clamps negative and too large inputs to the output range. `NaN` inputs convert to `zero`.

__host__ __device__ inline explicit operator unsigned int() const



Conversion operator to `unsigned` `int` data type.

Clamps negative and too large inputs to the output range. `NaN` inputs convert to `zero`.

__host__ __device__ inline explicit operator unsigned long int() const



Conversion operator to `unsigned` `long` `int` data type.

Clamps negative and too large inputs to the output range. `NaN` inputs convert to `zero` if output type is 32-bit. `NaN` inputs convert to `0x8000000000000000ULL` if output type is 64-bit.

__host__ __device__ inline explicit operator unsigned long long int() const



Conversion operator to `unsigned` `long` `long` `int` data type.

Clamps negative and too large inputs to the output range. `NaN` inputs convert to `0x8000000000000000ULL`.

__host__ __device__ inline explicit operator unsigned short int() const



Conversion operator to `unsigned` `short` `int` data type.

Clamps negative and too large inputs to the output range. `NaN` inputs convert to `zero`.

Public Members

__nv_fp8_storage_t __x



Storage variable contains the `fp8` floating-point data.