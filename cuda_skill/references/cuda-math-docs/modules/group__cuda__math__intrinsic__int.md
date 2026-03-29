# 13. Integer Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__INT.html


#  13\. Integer Intrinsics

This section describes integer intrinsic functions.

All of these functions are supported in device code. For some of the functions, host-specific implementations are also provided. For example, see `__nv_bswap16()`. To use these functions, you do not need to include any additional header file in your program.

Functions

__device__ unsigned int __brev(unsigned int x)


Reverse the bit order of a 32-bit unsigned integer.

__device__ unsigned long long int __brevll(unsigned long long int x)


Reverse the bit order of a 64-bit unsigned integer.

__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s)


Return selected bytes from two 32-bit unsigned integers.

__device__ __CLZ_RETURN_TYPE __clz(__CLZ_PARAMETER_TYPE x)


Return the number of consecutive high-order zero bits in a 32-bit integer.

__device__ __CLZ_RETURN_TYPE __clzll(__CLZLL_PARAMETER_TYPE x)


Count the number of consecutive high-order zero bits in a 64-bit integer.

__device__ int __dp2a_hi(int srcA, int srcB, int c)


Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the upper half of the second input.

__device__ unsigned int __dp2a_hi(unsigned int srcA, unsigned int srcB, unsigned int c)


Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the upper half of the second input.

__device__ unsigned int __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned int c)


Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the upper half of the second input.

__device__ int __dp2a_hi(short2 srcA, char4 srcB, int c)


Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the upper half of the second input.

__device__ unsigned int __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned int c)


Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the lower half of the second input.

__device__ int __dp2a_lo(short2 srcA, char4 srcB, int c)


Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the lower half of the second input.

__device__ unsigned int __dp2a_lo(unsigned int srcA, unsigned int srcB, unsigned int c)


Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the lower half of the second input.

__device__ int __dp2a_lo(int srcA, int srcB, int c)


Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the lower half of the second input.

__device__ unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c)


Four-way `unsigned` `int8` dot product with `unsigned` `int32` accumulate.

__device__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c)


Four-way `unsigned` `int8` dot product with `unsigned` `int32` accumulate.

__device__ int __dp4a(int srcA, int srcB, int c)


Four-way `signed` `int8` dot product with `int32` accumulate.

__device__ int __dp4a(char4 srcA, char4 srcB, int c)


Four-way `signed` `int8` dot product with `int32` accumulate.

__device__ int __ffs(int x)


Find the position of the least significant bit set to 1 in a 32-bit integer.

__device__ int __ffsll(long long int x)


Find the position of the least significant bit set to 1 in a 64-bit integer.

__device__ unsigned __fns(unsigned mask, unsigned base, int offset)


Find the position of the n-th set to 1 bit in a 32-bit integer.

__device__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)


Concatenate `hi` : `lo` , shift left by `shift` & 31 bits, return the most significant 32 bits.

__device__ unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)


Concatenate `hi` : `lo` , shift left by min( `shift` , 32) bits, return the most significant 32 bits.

__device__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)


Concatenate `hi` : `lo` , shift right by `shift` & 31 bits, return the least significant 32 bits.

__device__ unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)


Concatenate `hi` : `lo` , shift right by min( `shift` , 32) bits, return the least significant 32 bits.

__device__ int __hadd(int x, int y)


Compute average of signed input arguments, avoiding overflow in the intermediate sum.

__device__ int __mul24(int x, int y)


Calculate the least significant 32 bits of the product of the least significant 24 bits of two integers.

__device__ long long int __mul64hi(long long int x, long long int y)


Calculate the most significant 64 bits of the product of the two 64-bit integers.

__device__ int __mulhi(int x, int y)


Calculate the most significant 32 bits of the product of the two 32-bit integers.

__host__ __device__ unsigned short __nv_bswap16(unsigned short x)


Reverse the order of bytes of the 16-bit unsigned integer.

__host__ __device__ unsigned int __nv_bswap32(unsigned int x)


Reverse the order of bytes of the 32-bit unsigned integer.

__host__ __device__ unsigned long long __nv_bswap64(unsigned long long x)


Reverse the order of bytes of the 64-bit unsigned integer.

__device__ int __popc(unsigned int x)


Count the number of bits that are set to 1 in a 32-bit integer.

__device__ int __popcll(unsigned long long int x)


Count the number of bits that are set to 1 in a 64-bit integer.

__device__ int __rhadd(int x, int y)


Compute rounded average of signed input arguments, avoiding overflow in the intermediate sum.

__device__ unsigned int __sad(int x, int y, unsigned int z)


Calculate \\(|x - y| + z\\) , the sum of absolute difference.

__device__ unsigned int __uhadd(unsigned int x, unsigned int y)


Compute average of unsigned input arguments, avoiding overflow in the intermediate sum.

__device__ unsigned int __umul24(unsigned int x, unsigned int y)


Calculate the least significant 32 bits of the product of the least significant 24 bits of two unsigned integers.

__device__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y)


Calculate the most significant 64 bits of the product of the two 64 unsigned bit integers.

__device__ unsigned int __umulhi(unsigned int x, unsigned int y)


Calculate the most significant 32 bits of the product of the two 32-bit unsigned integers.

__device__ unsigned int __urhadd(unsigned int x, unsigned int y)


Compute rounded average of unsigned input arguments, avoiding overflow in the intermediate sum.

__device__ unsigned int __usad(unsigned int x, unsigned int y, unsigned int z)


Calculate \\(|x - y| + z\\) , the sum of absolute difference.

##  13.1. Functions

__device__ unsigned int __brev(unsigned int x)



Reverse the bit order of a 32-bit unsigned integer.

Reverses the bit order of the 32-bit unsigned integer `x`.

Returns


Returns the bit-reversed value of `x`. i.e. bit N of the return value corresponds to bit 31-N of `x`.

__device__ unsigned long long int __brevll(unsigned long long int x)



Reverse the bit order of a 64-bit unsigned integer.

Reverses the bit order of the 64-bit unsigned integer `x`.

Returns


Returns the bit-reversed value of `x`. i.e. bit N of the return value corresponds to bit 63-N of `x`.

__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s)



Return selected bytes from two 32-bit unsigned integers.

Create 8-byte source

  * uint64_t `tmp64` = ((uint64_t)`y` << 32) | `x`;


Extract selector bits

  * `selector0` = (`s` >> 0) & 0x7;

  * `selector1` = (`s` >> 4) & 0x7;

  * `selector2` = (`s` >> 8) & 0x7;

  * `selector3` = (`s` >> 12) & 0x7;


Return 4 selected bytes from 8-byte source:

  * `res`[07:00] = `tmp64`[`selector0`];

  * `res`[15:08] = `tmp64`[`selector1`];

  * `res`[23:16] = `tmp64`[`selector2`];

  * `res`[31:24] = `tmp64`[`selector3`];


Returns


Returns a 32-bit integer consisting of four bytes from eight input bytes provided in the two input integers `x` and `y`, as specified by a selector, `s`.

__device__ __CLZ_RETURN_TYPE __clz(__CLZ_PARAMETER_TYPE x)



Return the number of consecutive high-order zero bits in a 32-bit integer.

Count the number of consecutive leading zero bits, starting at the most significant bit (bit 31) of `x`.

To accomodate to ACLE builtins

  * on ARM64 with GCC 11.4 or later as the host compiler, __CLZ_RETURN_TYPE is ‘unsigned int’ and __CLZ_PARAMETER_TYPE is ‘unsigned int’.

  * for all other cases, __CLZ_RETURN_TYPE is ‘int’ and __CLZ_PARAMETER_TYPE is ‘int’.


Returns


Returns a value between 0 and 32 inclusive representing the number of zero bits.

__device__ __CLZ_RETURN_TYPE __clzll(__CLZLL_PARAMETER_TYPE x)



Count the number of consecutive high-order zero bits in a 64-bit integer.

Count the number of consecutive leading zero bits, starting at the most significant bit (bit 63) of `x`.

To accomodate to ACLE builtins

  * on ARM64 with GCC 11.4 or later as the host compiler, __CLZ_RETURN_TYPE is ‘unsigned int’ and __CLZLL_PARAMETER_TYPE is ‘unsigned long int’.

  * for all other cases, __CLZ_RETURN_TYPE is ‘int’ and __CLZLL_PARAMETER_TYPE is ‘long long int’.


Returns


Returns a value between 0 and 64 inclusive representing the number of zero bits.

__device__ int __dp2a_hi(int srcA, int srcB, int c)



Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the upper half of the second input.

Extracts two packed 16-bit integers from `scrA` and two packed 8-bit integers from the upper 16 bits of `srcB`, then creates two pairwise 8x16 products and adds them together to a signed 32-bit integer `c`.

__device__ unsigned int __dp2a_hi(unsigned int srcA, unsigned int srcB, unsigned int c)



Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the upper half of the second input.

Extracts two packed 16-bit integers from `scrA` and two packed 8-bit integers from the upper 16 bits of `srcB`, then creates two pairwise 8x16 products and adds them together to an unsigned 32-bit integer `c`.

__device__ unsigned int __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned int c)



Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the upper half of the second input.

Takes two packed 16-bit integers from `scrA` vector and two packed 8-bit integers from the upper 16 bits of `srcB` vector, then creates two pairwise 8x16 products and adds them together to an unsigned 32-bit integer `c`.

__device__ int __dp2a_hi(short2 srcA, char4 srcB, int c)



Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the upper half of the second input.

Takes two packed 16-bit integers from `scrA` vector and two packed 8-bit integers from the upper 16 bits of `srcB` vector, then creates two pairwise 8x16 products and adds them together to a signed 32-bit integer `c`.

__device__ unsigned int __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned int c)



Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the lower half of the second input.

Takes two packed 16-bit integers from `scrA` vector and two packed 8-bit integers from the lower 16 bits of `srcB` vector, then creates two pairwise 8x16 products and adds them together to an unsigned 32-bit integer `c`.

__device__ int __dp2a_lo(short2 srcA, char4 srcB, int c)



Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the lower half of the second input.

Takes two packed 16-bit integers from `scrA` vector and two packed 8-bit integers from the lower 16 bits of `srcB` vector, then creates two pairwise 8x16 products and adds them together to a signed 32-bit integer `c`.

__device__ unsigned int __dp2a_lo(unsigned int srcA, unsigned int srcB, unsigned int c)



Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the lower half of the second input.

Extracts two packed 16-bit integers from `scrA` and two packed 8-bit integers from the lower 16 bits of `srcB`, then creates two pairwise 8x16 products and adds them together to an unsigned 32-bit integer `c`.

__device__ int __dp2a_lo(int srcA, int srcB, int c)



Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the lower half of the second input.

Extracts two packed 16-bit integers from `scrA` and two packed 8-bit integers from the lower 16 bits of `srcB`, then creates two pairwise 8x16 products and adds them together to a signed 32-bit integer `c`.

__device__ unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c)



Four-way `unsigned` `int8` dot product with `unsigned` `int32` accumulate.

Takes four pairs of packed byte-sized integers from `scrA` and `srcB` vectors, then creates four pairwise products and adds them together to an unsigned 32-bit integer `c`.

__device__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c)



Four-way `unsigned` `int8` dot product with `unsigned` `int32` accumulate.

Extracts four pairs of packed byte-sized integers from `scrA` and `srcB`, then creates four pairwise products and adds them together to an unsigned 32-bit integer `c`.

__device__ int __dp4a(int srcA, int srcB, int c)



Four-way `signed` `int8` dot product with `int32` accumulate.

Extracts four pairs of packed byte-sized integers from `scrA` and `srcB`, then creates four pairwise products and adds them together to a signed 32-bit integer `c`.

__device__ int __dp4a(char4 srcA, char4 srcB, int c)



Four-way `signed` `int8` dot product with `int32` accumulate.

Takes four pairs of packed byte-sized integers from `scrA` and `srcB` vectors, then creates four pairwise products and adds them together to a signed 32-bit integer `c`.

__device__ int __ffs(int x)



Find the position of the least significant bit set to 1 in a 32-bit integer.

Find the position of the first (least significant) bit set to 1 in `x`, where the least significant bit position is 1.

Returns


Returns a value between 0 and 32 inclusive representing the position of the first bit set.

  * __ffs(0) returns 0.


__device__ int __ffsll(long long int x)



Find the position of the least significant bit set to 1 in a 64-bit integer.

Find the position of the first (least significant) bit set to 1 in `x`, where the least significant bit position is 1.

Returns


Returns a value between 0 and 64 inclusive representing the position of the first bit set.

  * __ffsll(0) returns 0.


__device__ unsigned __fns(unsigned mask, unsigned base, int offset)



Find the position of the n-th set to 1 bit in a 32-bit integer.

Given a 32-bit value `mask` and an integer value `base` (between 0 and 31), find the n-th (given by `offset`) set bit in `mask` from the `base` bit. If not found, return 0xFFFFFFFF.

See also <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns> for more information.

Returns


Returns a value between 0 and 32 inclusive representing the position of the n-th set bit.

  * parameter `base` must be <=31, otherwise behavior is undefined.


__device__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)



Concatenate `hi` : `lo`, shift left by `shift` & 31 bits, return the most significant 32 bits.

Shift the 64-bit value formed by concatenating argument `lo` and `hi` left by the amount specified by the argument `shift`. Argument `lo` holds bits 31:0 and argument `hi` holds bits 63:32 of the 64-bit source value. The source is shifted left by the wrapped value of `shift` (`shift` & 31). The most significant 32-bits of the result are returned.

Returns


Returns the most significant 32 bits of the shifted 64-bit value.

__device__ unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)



Concatenate `hi` : `lo`, shift left by min(`shift`, 32) bits, return the most significant 32 bits.

Shift the 64-bit value formed by concatenating argument `lo` and `hi` left by the amount specified by the argument `shift`. Argument `lo` holds bits 31:0 and argument `hi` holds bits 63:32 of the 64-bit source value. The source is shifted left by the clamped value of `shift` (min(`shift`, 32)). The most significant 32-bits of the result are returned.

Returns


Returns the most significant 32 bits of the shifted 64-bit value.

__device__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)



Concatenate `hi` : `lo`, shift right by `shift` & 31 bits, return the least significant 32 bits.

Shift the 64-bit value formed by concatenating argument `lo` and `hi` right by the amount specified by the argument `shift`. Argument `lo` holds bits 31:0 and argument `hi` holds bits 63:32 of the 64-bit source value. The source is shifted right by the wrapped value of `shift` (`shift` & 31). The least significant 32-bits of the result are returned.

Returns


Returns the least significant 32 bits of the shifted 64-bit value.

__device__ unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)



Concatenate `hi` : `lo`, shift right by min(`shift`, 32) bits, return the least significant 32 bits.

Shift the 64-bit value formed by concatenating argument `lo` and `hi` right by the amount specified by the argument `shift`. Argument `lo` holds bits 31:0 and argument `hi` holds bits 63:32 of the 64-bit source value. The source is shifted right by the clamped value of `shift` (min(`shift`, 32)). The least significant 32-bits of the result are returned.

Returns


Returns the least significant 32 bits of the shifted 64-bit value.

__device__ int __hadd(int x, int y)



Compute average of signed input arguments, avoiding overflow in the intermediate sum.

Compute average of signed input arguments `x` and `y` as ( `x` \+ `y` ) >> 1, avoiding overflow in the intermediate sum.

Returns


Returns a signed integer value representing the signed average value of the two inputs.

__device__ int __mul24(int x, int y)



Calculate the least significant 32 bits of the product of the least significant 24 bits of two integers.

Calculate the least significant 32 bits of the product of the least significant 24 bits of `x` and `y`. The high order 8 bits of `x` and `y` are ignored.

Returns


Returns the least significant 32 bits of the product `x` * `y`.

__device__ long long int __mul64hi(long long int x, long long int y)



Calculate the most significant 64 bits of the product of the two 64-bit integers.

Calculate the most significant 64 bits of the 128-bit product `x` * `y`, where `x` and `y` are 64-bit integers.

Returns


Returns the most significant 64 bits of the product `x` * `y`.

__device__ int __mulhi(int x, int y)



Calculate the most significant 32 bits of the product of the two 32-bit integers.

Calculate the most significant 32 bits of the 64-bit product `x` * `y`, where `x` and `y` are 32-bit integers.

Returns


Returns the most significant 32 bits of the product `x` * `y`.

__host__ __device__ unsigned short __nv_bswap16(unsigned short x)



Reverse the order of bytes of the 16-bit unsigned integer.

Reverse the order of bytes of `x` . Only supported in MSVC and other host compilers which define the `__GNUC__` macro, such as GCC and CLANG.

Returns


Returns `x` with the order of bytes reversed.

__host__ __device__ unsigned int __nv_bswap32(unsigned int x)



Reverse the order of bytes of the 32-bit unsigned integer.

Reverse the order of bytes of `x` . Only supported in MSVC and other host compilers which define the `__GNUC__` macro, such as GCC and CLANG.

Returns


Returns `x` with the order of bytes reversed.

__host__ __device__ unsigned long long __nv_bswap64(unsigned long long x)



Reverse the order of bytes of the 64-bit unsigned integer.

Reverse the order of bytes of `x` . Only supported in MSVC and other host compilers which define the `__GNUC__` macro, such as GCC and CLANG.

Returns


Returns `x` with the order of bytes reversed.

__device__ int __popc(unsigned int x)



Count the number of bits that are set to 1 in a 32-bit integer.

Count the number of bits that are set to 1 in `x`.

Returns


Returns a value between 0 and 32 inclusive representing the number of set bits.

__device__ int __popcll(unsigned long long int x)



Count the number of bits that are set to 1 in a 64-bit integer.

Count the number of bits that are set to 1 in `x`.

Returns


Returns a value between 0 and 64 inclusive representing the number of set bits.

__device__ int __rhadd(int x, int y)



Compute rounded average of signed input arguments, avoiding overflow in the intermediate sum.

Compute average of signed input arguments `x` and `y` as ( `x` \+ `y` \+ 1 ) >> 1, avoiding overflow in the intermediate sum.

Returns


Returns a signed integer value representing the signed rounded average value of the two inputs.

__device__ unsigned int __sad(int x, int y, unsigned int z)



Calculate \\( |x - y| + z \\) , the sum of absolute difference.

Calculate \\( |x - y| + z \\) , the 32-bit sum of the third argument `z` plus and the absolute value of the difference between the first argument, `x`, and second argument, `y`.

Inputs `x` and `y` are signed 32-bit integers, input `z` is a 32-bit unsigned integer.

Returns


Returns \\( |x - y| + z \\).

__device__ unsigned int __uhadd(unsigned int x, unsigned int y)



Compute average of unsigned input arguments, avoiding overflow in the intermediate sum.

Compute average of unsigned input arguments `x` and `y` as ( `x` \+ `y` ) >> 1, avoiding overflow in the intermediate sum.

Returns


Returns an unsigned integer value representing the unsigned average value of the two inputs.

__device__ unsigned int __umul24(unsigned int x, unsigned int y)



Calculate the least significant 32 bits of the product of the least significant 24 bits of two unsigned integers.

Calculate the least significant 32 bits of the product of the least significant 24 bits of `x` and `y`. The high order 8 bits of `x` and `y` are ignored.

Returns


Returns the least significant 32 bits of the product `x` * `y`.

__device__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y)



Calculate the most significant 64 bits of the product of the two 64 unsigned bit integers.

Calculate the most significant 64 bits of the 128-bit product `x` * `y`, where `x` and `y` are 64-bit unsigned integers.

Returns


Returns the most significant 64 bits of the product `x` * `y`.

__device__ unsigned int __umulhi(unsigned int x, unsigned int y)



Calculate the most significant 32 bits of the product of the two 32-bit unsigned integers.

Calculate the most significant 32 bits of the 64-bit product `x` * `y`, where `x` and `y` are 32-bit unsigned integers.

Returns


Returns the most significant 32 bits of the product `x` * `y`.

__device__ unsigned int __urhadd(unsigned int x, unsigned int y)



Compute rounded average of unsigned input arguments, avoiding overflow in the intermediate sum.

Compute average of unsigned input arguments `x` and `y` as ( `x` \+ `y` \+ 1 ) >> 1, avoiding overflow in the intermediate sum.

Returns


Returns an unsigned integer value representing the unsigned rounded average value of the two inputs.

__device__ unsigned int __usad(unsigned int x, unsigned int y, unsigned int z)



Calculate \\( |x - y| + z \\) , the sum of absolute difference.

Calculate \\( |x - y| + z \\) , the 32-bit sum of the third argument `z` plus and the absolute value of the difference between the first argument, `x`, and second argument, `y`.

Inputs `x`, `y`, and `z` are unsigned 32-bit integers.

Returns


Returns \\( |x - y| + z \\).