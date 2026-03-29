# 11. Type Casting Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__CAST.html


#  11\. Type Casting Intrinsics

This section describes type casting intrinsic functions that are only supported in device code.

To use these functions, you do not need to include any additional header file in your program.

Functions

__device__ float __double2float_rd(double x)


Convert a double to a float in round-down mode.

__device__ float __double2float_rn(double x)


Convert a double to a float in round-to-nearest-even mode.

__device__ float __double2float_ru(double x)


Convert a double to a float in round-up mode.

__device__ float __double2float_rz(double x)


Convert a double to a float in round-towards-zero mode.

__device__ int __double2hiint(double x)


Reinterpret high 32 bits in a double as a signed integer.

__device__ int __double2int_rd(double x)


Convert a double to a signed int in round-down mode.

__device__ int __double2int_rn(double x)


Convert a double to a signed int in round-to-nearest-even mode.

__device__ int __double2int_ru(double x)


Convert a double to a signed int in round-up mode.

__device__ int __double2int_rz(double x)


Convert a double to a signed int in round-towards-zero mode.

__device__ long long int __double2ll_rd(double x)


Convert a double to a signed 64-bit int in round-down mode.

__device__ long long int __double2ll_rn(double x)


Convert a double to a signed 64-bit int in round-to-nearest-even mode.

__device__ long long int __double2ll_ru(double x)


Convert a double to a signed 64-bit int in round-up mode.

__device__ long long int __double2ll_rz(double x)


Convert a double to a signed 64-bit int in round-towards-zero mode.

__device__ int __double2loint(double x)


Reinterpret low 32 bits in a double as a signed integer.

__device__ unsigned int __double2uint_rd(double x)


Convert a double to an unsigned int in round-down mode.

__device__ unsigned int __double2uint_rn(double x)


Convert a double to an unsigned int in round-to-nearest-even mode.

__device__ unsigned int __double2uint_ru(double x)


Convert a double to an unsigned int in round-up mode.

__device__ unsigned int __double2uint_rz(double x)


Convert a double to an unsigned int in round-towards-zero mode.

__device__ unsigned long long int __double2ull_rd(double x)


Convert a double to an unsigned 64-bit int in round-down mode.

__device__ unsigned long long int __double2ull_rn(double x)


Convert a double to an unsigned 64-bit int in round-to-nearest-even mode.

__device__ unsigned long long int __double2ull_ru(double x)


Convert a double to an unsigned 64-bit int in round-up mode.

__device__ unsigned long long int __double2ull_rz(double x)


Convert a double to an unsigned 64-bit int in round-towards-zero mode.

__device__ long long int __double_as_longlong(double x)


Reinterpret bits in a double as a 64-bit signed integer.

__device__ int __float2int_rd(float x)


Convert a float to a signed integer in round-down mode.

__device__ int __float2int_rn(float x)


Convert a float to a signed integer in round-to-nearest-even mode.

__device__ int __float2int_ru(float)


Convert a float to a signed integer in round-up mode.

__device__ int __float2int_rz(float x)


Convert a float to a signed integer in round-towards-zero mode.

__device__ long long int __float2ll_rd(float x)


Convert a float to a signed 64-bit integer in round-down mode.

__device__ long long int __float2ll_rn(float x)


Convert a float to a signed 64-bit integer in round-to-nearest-even mode.

__device__ long long int __float2ll_ru(float x)


Convert a float to a signed 64-bit integer in round-up mode.

__device__ long long int __float2ll_rz(float x)


Convert a float to a signed 64-bit integer in round-towards-zero mode.

__device__ unsigned int __float2uint_rd(float x)


Convert a float to an unsigned integer in round-down mode.

__device__ unsigned int __float2uint_rn(float x)


Convert a float to an unsigned integer in round-to-nearest-even mode.

__device__ unsigned int __float2uint_ru(float x)


Convert a float to an unsigned integer in round-up mode.

__device__ unsigned int __float2uint_rz(float x)


Convert a float to an unsigned integer in round-towards-zero mode.

__device__ unsigned long long int __float2ull_rd(float x)


Convert a float to an unsigned 64-bit integer in round-down mode.

__device__ unsigned long long int __float2ull_rn(float x)


Convert a float to an unsigned 64-bit integer in round-to-nearest-even mode.

__device__ unsigned long long int __float2ull_ru(float x)


Convert a float to an unsigned 64-bit integer in round-up mode.

__device__ unsigned long long int __float2ull_rz(float x)


Convert a float to an unsigned 64-bit integer in round-towards-zero mode.

__device__ int __float_as_int(float x)


Reinterpret bits in a float as a signed integer.

__device__ unsigned int __float_as_uint(float x)


Reinterpret bits in a float as a unsigned integer.

__device__ double __hiloint2double(int hi, int lo)


Reinterpret high and low 32-bit integer values as a double.

__device__ double __int2double_rn(int x)


Convert a signed int to a double.

__device__ float __int2float_rd(int x)


Convert a signed integer to a float in round-down mode.

__device__ float __int2float_rn(int x)


Convert a signed integer to a float in round-to-nearest-even mode.

__device__ float __int2float_ru(int x)


Convert a signed integer to a float in round-up mode.

__device__ float __int2float_rz(int x)


Convert a signed integer to a float in round-towards-zero mode.

__device__ float __int_as_float(int x)


Reinterpret bits in an integer as a float.

__device__ double __ll2double_rd(long long int x)


Convert a signed 64-bit int to a double in round-down mode.

__device__ double __ll2double_rn(long long int x)


Convert a signed 64-bit int to a double in round-to-nearest-even mode.

__device__ double __ll2double_ru(long long int x)


Convert a signed 64-bit int to a double in round-up mode.

__device__ double __ll2double_rz(long long int x)


Convert a signed 64-bit int to a double in round-towards-zero mode.

__device__ float __ll2float_rd(long long int x)


Convert a signed integer to a float in round-down mode.

__device__ float __ll2float_rn(long long int x)


Convert a signed 64-bit integer to a float in round-to-nearest-even mode.

__device__ float __ll2float_ru(long long int x)


Convert a signed integer to a float in round-up mode.

__device__ float __ll2float_rz(long long int x)


Convert a signed integer to a float in round-towards-zero mode.

__device__ double __longlong_as_double(long long int x)


Reinterpret bits in a 64-bit signed integer as a double.

__device__ double __uint2double_rn(unsigned int x)


Convert an unsigned int to a double.

__device__ float __uint2float_rd(unsigned int x)


Convert an unsigned integer to a float in round-down mode.

__device__ float __uint2float_rn(unsigned int x)


Convert an unsigned integer to a float in round-to-nearest-even mode.

__device__ float __uint2float_ru(unsigned int x)


Convert an unsigned integer to a float in round-up mode.

__device__ float __uint2float_rz(unsigned int x)


Convert an unsigned integer to a float in round-towards-zero mode.

__device__ float __uint_as_float(unsigned int x)


Reinterpret bits in an unsigned integer as a float.

__device__ double __ull2double_rd(unsigned long long int x)


Convert an unsigned 64-bit int to a double in round-down mode.

__device__ double __ull2double_rn(unsigned long long int x)


Convert an unsigned 64-bit int to a double in round-to-nearest-even mode.

__device__ double __ull2double_ru(unsigned long long int x)


Convert an unsigned 64-bit int to a double in round-up mode.

__device__ double __ull2double_rz(unsigned long long int x)


Convert an unsigned 64-bit int to a double in round-towards-zero mode.

__device__ float __ull2float_rd(unsigned long long int x)


Convert an unsigned integer to a float in round-down mode.

__device__ float __ull2float_rn(unsigned long long int x)


Convert an unsigned integer to a float in round-to-nearest-even mode.

__device__ float __ull2float_ru(unsigned long long int x)


Convert an unsigned integer to a float in round-up mode.

__device__ float __ull2float_rz(unsigned long long int x)


Convert an unsigned integer to a float in round-towards-zero mode.

##  11.1. Functions

__device__ float __double2float_rd(double x)



Convert a double to a float in round-down mode.

Convert the double-precision floating-point value `x` to a single-precision floating-point value in round-down (to negative infinity) mode.

Returns


Returns converted value.

__device__ float __double2float_rn(double x)



Convert a double to a float in round-to-nearest-even mode.

Convert the double-precision floating-point value `x` to a single-precision floating-point value in round-to-nearest-even mode.

Returns


Returns converted value.

__device__ float __double2float_ru(double x)



Convert a double to a float in round-up mode.

Convert the double-precision floating-point value `x` to a single-precision floating-point value in round-up (to positive infinity) mode.

Returns


Returns converted value.

__device__ float __double2float_rz(double x)



Convert a double to a float in round-towards-zero mode.

Convert the double-precision floating-point value `x` to a single-precision floating-point value in round-towards-zero mode.

Returns


Returns converted value.

__device__ int __double2hiint(double x)



Reinterpret high 32 bits in a double as a signed integer.

Reinterpret the high 32 bits in the double-precision floating-point value `x` as a signed integer.

Returns


Returns reinterpreted value.

__device__ int __double2int_rd(double x)



Convert a double to a signed int in round-down mode.

Convert the double-precision floating-point value `x` to a signed integer value in round-down (to negative infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ int __double2int_rn(double x)



Convert a double to a signed int in round-to-nearest-even mode.

Convert the double-precision floating-point value `x` to a signed integer value in round-to-nearest-even mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ int __double2int_ru(double x)



Convert a double to a signed int in round-up mode.

Convert the double-precision floating-point value `x` to a signed integer value in round-up (to positive infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ int __double2int_rz(double x)



Convert a double to a signed int in round-towards-zero mode.

Convert the double-precision floating-point value `x` to a signed integer value in round-towards-zero mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ long long int __double2ll_rd(double x)



Convert a double to a signed 64-bit int in round-down mode.

Convert the double-precision floating-point value `x` to a signed 64-bit integer value in round-down (to negative infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ long long int __double2ll_rn(double x)



Convert a double to a signed 64-bit int in round-to-nearest-even mode.

Convert the double-precision floating-point value `x` to a signed 64-bit integer value in round-to-nearest-even mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ long long int __double2ll_ru(double x)



Convert a double to a signed 64-bit int in round-up mode.

Convert the double-precision floating-point value `x` to a signed 64-bit integer value in round-up (to positive infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ long long int __double2ll_rz(double x)



Convert a double to a signed 64-bit int in round-towards-zero mode.

Convert the double-precision floating-point value `x` to a signed 64-bit integer value in round-towards-zero mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ int __double2loint(double x)



Reinterpret low 32 bits in a double as a signed integer.

Reinterpret the low 32 bits in the double-precision floating-point value `x` as a signed integer.

Returns


Returns reinterpreted value.

__device__ unsigned int __double2uint_rd(double x)



Convert a double to an unsigned int in round-down mode.

Convert the double-precision floating-point value `x` to an unsigned integer value in round-down (to negative infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned int __double2uint_rn(double x)



Convert a double to an unsigned int in round-to-nearest-even mode.

Convert the double-precision floating-point value `x` to an unsigned integer value in round-to-nearest-even mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned int __double2uint_ru(double x)



Convert a double to an unsigned int in round-up mode.

Convert the double-precision floating-point value `x` to an unsigned integer value in round-up (to positive infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned int __double2uint_rz(double x)



Convert a double to an unsigned int in round-towards-zero mode.

Convert the double-precision floating-point value `x` to an unsigned integer value in round-towards-zero mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned long long int __double2ull_rd(double x)



Convert a double to an unsigned 64-bit int in round-down mode.

Convert the double-precision floating-point value `x` to an unsigned 64-bit integer value in round-down (to negative infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned long long int __double2ull_rn(double x)



Convert a double to an unsigned 64-bit int in round-to-nearest-even mode.

Convert the double-precision floating-point value `x` to an unsigned 64-bit integer value in round-to-nearest-even mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned long long int __double2ull_ru(double x)



Convert a double to an unsigned 64-bit int in round-up mode.

Convert the double-precision floating-point value `x` to an unsigned 64-bit integer value in round-up (to positive infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned long long int __double2ull_rz(double x)



Convert a double to an unsigned 64-bit int in round-towards-zero mode.

Convert the double-precision floating-point value `x` to an unsigned 64-bit integer value in round-towards-zero mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ long long int __double_as_longlong(double x)



Reinterpret bits in a double as a 64-bit signed integer.

Reinterpret the bits in the double-precision floating-point value `x` as a signed 64-bit integer.

Returns


Returns reinterpreted value.

__device__ int __float2int_rd(float x)



Convert a float to a signed integer in round-down mode.

Convert the single-precision floating-point value `x` to a signed integer in round-down (to negative infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ int __float2int_rn(float x)



Convert a float to a signed integer in round-to-nearest-even mode.

Convert the single-precision floating-point value `x` to a signed integer in round-to-nearest-even mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ int __float2int_ru(float)



Convert a float to a signed integer in round-up mode.

Convert the single-precision floating-point value `x` to a signed integer in round-up (to positive infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ int __float2int_rz(float x)



Convert a float to a signed integer in round-towards-zero mode.

Convert the single-precision floating-point value `x` to a signed integer in round-towards-zero mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ long long int __float2ll_rd(float x)



Convert a float to a signed 64-bit integer in round-down mode.

Convert the single-precision floating-point value `x` to a signed 64-bit integer in round-down (to negative infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ long long int __float2ll_rn(float x)



Convert a float to a signed 64-bit integer in round-to-nearest-even mode.

Convert the single-precision floating-point value `x` to a signed 64-bit integer in round-to-nearest-even mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ long long int __float2ll_ru(float x)



Convert a float to a signed 64-bit integer in round-up mode.

Convert the single-precision floating-point value `x` to a signed 64-bit integer in round-up (to positive infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ long long int __float2ll_rz(float x)



Convert a float to a signed 64-bit integer in round-towards-zero mode.

Convert the single-precision floating-point value `x` to a signed 64-bit integer in round-towards-zero mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned int __float2uint_rd(float x)



Convert a float to an unsigned integer in round-down mode.

Convert the single-precision floating-point value `x` to an unsigned integer in round-down (to negative infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned int __float2uint_rn(float x)



Convert a float to an unsigned integer in round-to-nearest-even mode.

Convert the single-precision floating-point value `x` to an unsigned integer in round-to-nearest-even mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned int __float2uint_ru(float x)



Convert a float to an unsigned integer in round-up mode.

Convert the single-precision floating-point value `x` to an unsigned integer in round-up (to positive infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned int __float2uint_rz(float x)



Convert a float to an unsigned integer in round-towards-zero mode.

Convert the single-precision floating-point value `x` to an unsigned integer in round-towards-zero mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned long long int __float2ull_rd(float x)



Convert a float to an unsigned 64-bit integer in round-down mode.

Convert the single-precision floating-point value `x` to an unsigned 64-bit integer in round-down (to negative infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned long long int __float2ull_rn(float x)



Convert a float to an unsigned 64-bit integer in round-to-nearest-even mode.

Convert the single-precision floating-point value `x` to an unsigned 64-bit integer in round-to-nearest-even mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned long long int __float2ull_ru(float x)



Convert a float to an unsigned 64-bit integer in round-up mode.

Convert the single-precision floating-point value `x` to an unsigned 64-bit integer in round-up (to positive infinity) mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ unsigned long long int __float2ull_rz(float x)



Convert a float to an unsigned 64-bit integer in round-towards-zero mode.

Convert the single-precision floating-point value `x` to an unsigned 64-bit integer in round-towards-zero mode.

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined.

Returns


Returns converted value.

__device__ int __float_as_int(float x)



Reinterpret bits in a float as a signed integer.

Reinterpret the bits in the single-precision floating-point value `x` as a signed integer.

Returns


Returns reinterpreted value.

__device__ unsigned int __float_as_uint(float x)



Reinterpret bits in a float as a unsigned integer.

Reinterpret the bits in the single-precision floating-point value `x` as a unsigned integer.

Returns


Returns reinterpreted value.

__device__ double __hiloint2double(int hi, int lo)



Reinterpret high and low 32-bit integer values as a double.

Reinterpret the integer value of `hi` as the high 32 bits of a double-precision floating-point value and the integer value of `lo` as the low 32 bits of the same double-precision floating-point value.

Returns


Returns reinterpreted value.

__device__ double __int2double_rn(int x)



Convert a signed int to a double.

Convert the signed integer value `x` to a double-precision floating-point value.

Returns


Returns converted value.

__device__ float __int2float_rd(int x)



Convert a signed integer to a float in round-down mode.

Convert the signed integer value `x` to a single-precision floating-point value in round-down (to negative infinity) mode.

Returns


Returns converted value.

__device__ float __int2float_rn(int x)



Convert a signed integer to a float in round-to-nearest-even mode.

Convert the signed integer value `x` to a single-precision floating-point value in round-to-nearest-even mode.

Returns


Returns converted value.

__device__ float __int2float_ru(int x)



Convert a signed integer to a float in round-up mode.

Convert the signed integer value `x` to a single-precision floating-point value in round-up (to positive infinity) mode.

Returns


Returns converted value.

__device__ float __int2float_rz(int x)



Convert a signed integer to a float in round-towards-zero mode.

Convert the signed integer value `x` to a single-precision floating-point value in round-towards-zero mode.

Returns


Returns converted value.

__device__ float __int_as_float(int x)



Reinterpret bits in an integer as a float.

Reinterpret the bits in the signed integer value `x` as a single-precision floating-point value.

Returns


Returns reinterpreted value.

__device__ double __ll2double_rd(long long int x)



Convert a signed 64-bit int to a double in round-down mode.

Convert the signed 64-bit integer value `x` to a double-precision floating-point value in round-down (to negative infinity) mode.

Returns


Returns converted value.

__device__ double __ll2double_rn(long long int x)



Convert a signed 64-bit int to a double in round-to-nearest-even mode.

Convert the signed 64-bit integer value `x` to a double-precision floating-point value in round-to-nearest-even mode.

Returns


Returns converted value.

__device__ double __ll2double_ru(long long int x)



Convert a signed 64-bit int to a double in round-up mode.

Convert the signed 64-bit integer value `x` to a double-precision floating-point value in round-up (to positive infinity) mode.

Returns


Returns converted value.

__device__ double __ll2double_rz(long long int x)



Convert a signed 64-bit int to a double in round-towards-zero mode.

Convert the signed 64-bit integer value `x` to a double-precision floating-point value in round-towards-zero mode.

Returns


Returns converted value.

__device__ float __ll2float_rd(long long int x)



Convert a signed integer to a float in round-down mode.

Convert the signed integer value `x` to a single-precision floating-point value in round-down (to negative infinity) mode.

Returns


Returns converted value.

__device__ float __ll2float_rn(long long int x)



Convert a signed 64-bit integer to a float in round-to-nearest-even mode.

Convert the signed 64-bit integer value `x` to a single-precision floating-point value in round-to-nearest-even mode.

Returns


Returns converted value.

__device__ float __ll2float_ru(long long int x)



Convert a signed integer to a float in round-up mode.

Convert the signed integer value `x` to a single-precision floating-point value in round-up (to positive infinity) mode.

Returns


Returns converted value.

__device__ float __ll2float_rz(long long int x)



Convert a signed integer to a float in round-towards-zero mode.

Convert the signed integer value `x` to a single-precision floating-point value in round-towards-zero mode.

Returns


Returns converted value.

__device__ double __longlong_as_double(long long int x)



Reinterpret bits in a 64-bit signed integer as a double.

Reinterpret the bits in the 64-bit signed integer value `x` as a double-precision floating-point value.

Returns


Returns reinterpreted value.

__device__ double __uint2double_rn(unsigned int x)



Convert an unsigned int to a double.

Convert the unsigned integer value `x` to a double-precision floating-point value.

Returns


Returns converted value.

__device__ float __uint2float_rd(unsigned int x)



Convert an unsigned integer to a float in round-down mode.

Convert the unsigned integer value `x` to a single-precision floating-point value in round-down (to negative infinity) mode.

Returns


Returns converted value.

__device__ float __uint2float_rn(unsigned int x)



Convert an unsigned integer to a float in round-to-nearest-even mode.

Convert the unsigned integer value `x` to a single-precision floating-point value in round-to-nearest-even mode.

Returns


Returns converted value.

__device__ float __uint2float_ru(unsigned int x)



Convert an unsigned integer to a float in round-up mode.

Convert the unsigned integer value `x` to a single-precision floating-point value in round-up (to positive infinity) mode.

Returns


Returns converted value.

__device__ float __uint2float_rz(unsigned int x)



Convert an unsigned integer to a float in round-towards-zero mode.

Convert the unsigned integer value `x` to a single-precision floating-point value in round-towards-zero mode.

Returns


Returns converted value.

__device__ float __uint_as_float(unsigned int x)



Reinterpret bits in an unsigned integer as a float.

Reinterpret the bits in the unsigned integer value `x` as a single-precision floating-point value.

Returns


Returns reinterpreted value.

__device__ double __ull2double_rd(unsigned long long int x)



Convert an unsigned 64-bit int to a double in round-down mode.

Convert the unsigned 64-bit integer value `x` to a double-precision floating-point value in round-down (to negative infinity) mode.

Returns


Returns converted value.

__device__ double __ull2double_rn(unsigned long long int x)



Convert an unsigned 64-bit int to a double in round-to-nearest-even mode.

Convert the unsigned 64-bit integer value `x` to a double-precision floating-point value in round-to-nearest-even mode.

Returns


Returns converted value.

__device__ double __ull2double_ru(unsigned long long int x)



Convert an unsigned 64-bit int to a double in round-up mode.

Convert the unsigned 64-bit integer value `x` to a double-precision floating-point value in round-up (to positive infinity) mode.

Returns


Returns converted value.

__device__ double __ull2double_rz(unsigned long long int x)



Convert an unsigned 64-bit int to a double in round-towards-zero mode.

Convert the unsigned 64-bit integer value `x` to a double-precision floating-point value in round-towards-zero mode.

Returns


Returns converted value.

__device__ float __ull2float_rd(unsigned long long int x)



Convert an unsigned integer to a float in round-down mode.

Convert the unsigned integer value `x` to a single-precision floating-point value in round-down (to negative infinity) mode.

Returns


Returns converted value.

__device__ float __ull2float_rn(unsigned long long int x)



Convert an unsigned integer to a float in round-to-nearest-even mode.

Convert the unsigned integer value `x` to a single-precision floating-point value in round-to-nearest-even mode.

Returns


Returns converted value.

__device__ float __ull2float_ru(unsigned long long int x)



Convert an unsigned integer to a float in round-up mode.

Convert the unsigned integer value `x` to a single-precision floating-point value in round-up (to positive infinity) mode.

Returns


Returns converted value.

__device__ float __ull2float_rz(unsigned long long int x)



Convert an unsigned integer to a float in round-towards-zero mode.

Convert the unsigned integer value `x` to a single-precision floating-point value in round-towards-zero mode.

Returns


Returns converted value.