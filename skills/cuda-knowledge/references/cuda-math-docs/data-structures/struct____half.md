# 15.1. __half

**Source:** struct____half.html


#  15.1. __half

struct __half



__half data type

This structure implements the datatype for storing half-precision floating-point numbers. The structure implements assignment, arithmetic and comparison operators, and type conversions. 16 bits are being used in total: 1 sign bit, 5 bits for the exponent, and the significand is being stored in 10 bits. The total precision is 11 bits. There are 15361 representable numbers within the interval [0.0, 1.0], endpoints included. On average we have log10(2**11) ~ 3.311 decimal digits.

The objective here is to provide IEEE754-compliant implementation of `binary16` type and arithmetic with limitations due to device HW not supporting floating-point exceptions.

Public Functions

__half() = default



Constructor by default.

Emtpy default constructor, result is uninitialized.

__host__ __device__ inline constexpr __half(const __half_raw &hr)



Constructor from `__half_raw`.

__host__ __device__ explicit __half(const __nv_bfloat16 f)



Construct `__half` from `__nv_bfloat16` input using default round-to-nearest-even rounding mode.

Need to include the header file `cuda_bf16.h`

__host__ __device__ inline __half(const double f)



Construct `__half` from `double` input using default round-to-nearest-even rounding mode.

See also

__double2half(double) for further details.

__host__ __device__ inline __half(const float f)



Construct `__half` from `float` input using default round-to-nearest-even rounding mode.

See also

__float2half(float) for further details.

__host__ __device__ inline __half(const int val)



Construct `__half` from `int` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __half(const long long val)



Construct `__half` from `long` `long` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __half(const long val)



Construct `__half` from `long` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __half(const short val)



Construct `__half` from `short` integer input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __half(const unsigned int val)



Construct `__half` from `unsigned` `int` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __half(const unsigned long long val)



Construct `__half` from `unsigned` `long` `long` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __half(const unsigned long val)



Construct `__half` from `unsigned` `long` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __half(const unsigned short val)



Construct `__half` from `unsigned` `short` integer input using default round-to-nearest-even rounding mode.

__host__ __device__ operator __half_raw() const



Type cast to `__half_raw` operator.

__host__ __device__ operator __half_raw() volatile const



Type cast to `__half_raw` operator with `volatile` input.

__host__ __device__ inline operator bool() const



Conversion operator to `bool` data type.

+0 and -0 inputs convert to `false`. Non-zero inputs convert to `true`.

__host__ __device__ operator char() const



Conversion operator to an implementation defined `char` data type.

Using round-toward-zero rounding mode.

Detects signedness of the `char` type and proceeds accordingly, see further details in __half2char_rz(__half) and __half2uchar_rz(__half).

__host__ __device__ operator float() const



Type cast to `float` operator.

__host__ __device__ operator int() const



Conversion operator to `int` data type.

Using round-toward-zero rounding mode.

See also

__half2int_rz(__half) for further details.

__host__ __device__ operator long() const



Conversion operator to `long` data type.

Using round-toward-zero rounding mode.

Detects size of the `long` type and proceeds accordingly, see further details in __half2int_rz(__half) and __half2ll_rz(__half).

__host__ __device__ operator long long() const



Conversion operator to `long` `long` data type.

Using round-toward-zero rounding mode.

See also

__half2ll_rz(__half) for further details.

__host__ __device__ operator short() const



Conversion operator to `short` data type.

Using round-toward-zero rounding mode.

See also

__half2short_rz(__half) for further details.

__host__ __device__ operator signed char() const



Conversion operator to `signed` `char` data type.

Using round-toward-zero rounding mode.

See also

__half2char_rz(__half) for further details.

__host__ __device__ operator unsigned char() const



Conversion operator to `unsigned` `char` data type.

Using round-toward-zero rounding mode.

See also

__half2uchar_rz(__half) for further details.

__host__ __device__ operator unsigned int() const



Conversion operator to `unsigned` `int` data type.

Using round-toward-zero rounding mode.

See also

__half2uint_rz(__half) for further details.

__host__ __device__ operator unsigned long() const



Conversion operator to `unsigned` `long` data type.

Using round-toward-zero rounding mode.

Detects size of the `unsigned` `long` type and proceeds accordingly, see further details in __half2uint_rz(__half) and __half2ull_rz(__half).

__host__ __device__ operator unsigned long long() const



Conversion operator to `unsigned` `long` `long` data type.

Using round-toward-zero rounding mode.

See also

__half2ull_rz(__half) for further details.

__host__ __device__ operator unsigned short() const



Conversion operator to `unsigned` `short` data type.

Using round-toward-zero rounding mode.

See also

__half2ushort_rz(__half) for further details.

__host__ __device__ __half &operator=(const __half_raw &hr)



Assignment operator from `__half_raw`.

__host__ __device__ volatile __half &operator=(const __half_raw &hr) volatile



Assignment operator from `__half_raw` to `volatile` `__half`.

__host__ __device__ __half &operator=(const double f)



Type cast to `__half` assignment operator from `double` input using default round-to-nearest-even rounding mode.

See also

__double2half(double) for further details.

__host__ __device__ __half &operator=(const float f)



Type cast to `__half` assignment operator from `float` input using default round-to-nearest-even rounding mode.

See also

__float2half(float) for further details.

__host__ __device__ __half &operator=(const int val)



Type cast from `int` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __half &operator=(const long long val)



Type cast from `long` `long` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __half &operator=(const short val)



Type cast from `short` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __half &operator=(const unsigned int val)



Type cast from `unsigned` `int` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __half &operator=(const unsigned long long val)



Type cast from `unsigned` `long` `long` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __half &operator=(const unsigned short val)



Type cast from `unsigned` `short` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ volatile __half &operator=(volatile const __half_raw &hr) volatile



Assignment operator from `volatile` `__half_raw` to `volatile` `__half`.