# 15.5. __nv_bfloat16

**Source:** struct____nv__bfloat16.html


#  15.5. __nv_bfloat16

struct __nv_bfloat16



nv_bfloat16 datatype

This structure implements the datatype for storing nv_bfloat16 floating-point numbers. The structure implements assignment operators and type conversions. 16 bits are being used in total: 1 sign bit, 8 bits for the exponent, and the significand is being stored in 7 bits. The total precision is 8 bits.

Public Functions

__nv_bfloat16() = default



Constructor by default.

Emtpy default constructor, result is uninitialized.

__host__ __device__ inline explicit __nv_bfloat16(const __half f)



Construct `__nv_bfloat16` from `__half` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline constexpr __nv_bfloat16(const __nv_bfloat16_raw &hr)



Constructor from `__nv_bfloat16_raw`.

__host__ __device__ inline __nv_bfloat16(const double f)



Construct `__nv_bfloat16` from `double` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __nv_bfloat16(const float f)



Construct `__nv_bfloat16` from `float` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __nv_bfloat16(const long val)



Construct `__nv_bfloat16` from `long` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __nv_bfloat16(const unsigned long val)



Construct `__nv_bfloat16` from `unsigned` `long` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __nv_bfloat16(int val)



Construct `__nv_bfloat16` from `int` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __nv_bfloat16(long long val)



Construct `__nv_bfloat16` from `long` `long` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __nv_bfloat16(short val)



Construct `__nv_bfloat16` from `short` integer input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __nv_bfloat16(unsigned int val)



Construct `__nv_bfloat16` from `unsigned` `int` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __nv_bfloat16(unsigned long long val)



Construct `__nv_bfloat16` from `unsigned` `long` `long` input using default round-to-nearest-even rounding mode.

__host__ __device__ inline __nv_bfloat16(unsigned short val)



Construct `__nv_bfloat16` from `unsigned` `short` integer input using default round-to-nearest-even rounding mode.

__host__ __device__ operator __nv_bfloat16_raw() const



Type cast to `__nv_bfloat16_raw` operator.

__host__ __device__ operator __nv_bfloat16_raw() volatile const



Type cast to `__nv_bfloat16_raw` operator with `volatile` input.

__host__ __device__ inline operator bool() const



Conversion operator to `bool` data type.

+0 and -0 inputs convert to `false`. Non-zero inputs convert to `true`.

__host__ __device__ operator char() const



Conversion operator to an implementation defined `char` data type.

Using round-toward-zero rounding mode.

Detects signedness of the `char` type and proceeds accordingly, see further details in signed and unsigned char operators.

__host__ __device__ operator float() const



Type cast to `float` operator.

__host__ __device__ operator int() const



Conversion operator to `int` data type.

Using round-toward-zero rounding mode.

See __bfloat162int_rz(__nv_bfloat16) for further details

__host__ __device__ operator long() const



Conversion operator to `long` data type.

Using round-toward-zero rounding mode.

__host__ __device__ operator long long() const



Conversion operator to `long` `long` data type.

Using round-toward-zero rounding mode.

See __bfloat162ll_rz(__nv_bfloat16) for further details

__host__ __device__ operator short() const



Conversion operator to `short` data type.

Using round-toward-zero rounding mode.

See __bfloat162short_rz(__nv_bfloat16) for further details

__host__ __device__ operator signed char() const



Conversion operator to `signed` `char` data type.

Using round-toward-zero rounding mode.

See __bfloat162char_rz(__nv_bfloat16) for further details

__host__ __device__ operator unsigned char() const



Conversion operator to `unsigned` `char` data type.

Using round-toward-zero rounding mode.

See __bfloat162uchar_rz(__nv_bfloat16) for further details

__host__ __device__ operator unsigned int() const



Conversion operator to `unsigned` `int` data type.

Using round-toward-zero rounding mode.

See __bfloat162uint_rz(__nv_bfloat16) for further details

__host__ __device__ operator unsigned long() const



Conversion operator to `unsigned` `long` data type.

Using round-toward-zero rounding mode.

__host__ __device__ operator unsigned long long() const



Conversion operator to `unsigned` `long` `long` data type.

Using round-toward-zero rounding mode.

See __bfloat162ull_rz(__nv_bfloat16) for further details

__host__ __device__ operator unsigned short() const



Conversion operator to `unsigned` `short` data type.

Using round-toward-zero rounding mode.

See __bfloat162ushort_rz(__nv_bfloat16) for further details

__host__ __device__ __nv_bfloat16 &operator=(const __nv_bfloat16_raw &hr)



Assignment operator from `__nv_bfloat16_raw`.

__host__ __device__ volatile __nv_bfloat16 &operator=(const __nv_bfloat16_raw &hr) volatile



Assignment operator from `__nv_bfloat16_raw` to `volatile` `__nv_bfloat16`.

__host__ __device__ __nv_bfloat16 &operator=(const double f)



Type cast to `__nv_bfloat16` assignment operator from `double` input using default round-to-nearest-even rounding mode.

__host__ __device__ __nv_bfloat16 &operator=(const float f)



Type cast to `__nv_bfloat16` assignment operator from `float` input using default round-to-nearest-even rounding mode.

__host__ __device__ volatile __nv_bfloat16 &operator=(volatile const __nv_bfloat16_raw &hr) volatile



Assignment operator from `volatile` `__nv_bfloat16_raw` to `volatile` `__nv_bfloat16`.

__host__ __device__ __nv_bfloat16 &operator=(int val)



Type cast from `int` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __nv_bfloat16 &operator=(long long val)



Type cast from `long` `long` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __nv_bfloat16 &operator=(short val)



Type cast from `short` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __nv_bfloat16 &operator=(unsigned int val)



Type cast from `unsigned` `int` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __nv_bfloat16 &operator=(unsigned long long val)



Type cast from `unsigned` `long` `long` assignment operator, using default round-to-nearest-even rounding mode.

__host__ __device__ __nv_bfloat16 &operator=(unsigned short val)



Type cast from `unsigned` `short` assignment operator, using default round-to-nearest-even rounding mode.