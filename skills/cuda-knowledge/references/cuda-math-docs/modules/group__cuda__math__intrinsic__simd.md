# 14. SIMD Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__SIMD.html


#  14\. SIMD Intrinsics

This section describes SIMD intrinsic functions that are only supported in device code.

To use these functions, you do not need to include any additional header file in your program.

Functions

__device__ unsigned int __vabs2(unsigned int a)


Computes per-halfword absolute value: |a|.

__device__ unsigned int __vabs4(unsigned int a)


Computes per-byte absolute value: |a|.

__device__ unsigned int __vabsdiffs2(unsigned int a, unsigned int b)


Computes per-halfword absolute difference of signed integer: |a - b|.

__device__ unsigned int __vabsdiffs4(unsigned int a, unsigned int b)


Computes per-byte absolute difference of signed integer: |a - b|.

__device__ unsigned int __vabsdiffu2(unsigned int a, unsigned int b)


Computes per-halfword absolute difference of unsigned integer: |a - b|.

__device__ unsigned int __vabsdiffu4(unsigned int a, unsigned int b)


Computes per-byte absolute difference of unsigned integer: |a - b|.

__device__ unsigned int __vabsss2(unsigned int a)


Computes per-halfword absolute value with signed saturation: |a|.

__device__ unsigned int __vabsss4(unsigned int a)


Computes per-byte absolute value with signed saturation: |a|.

__device__ unsigned int __vadd2(unsigned int a, unsigned int b)


Performs per-halfword (un)signed addition, with wrap-around: a + b.

__device__ unsigned int __vadd4(unsigned int a, unsigned int b)


Performs per-byte (un)signed addition: a + b.

__device__ unsigned int __vaddss2(unsigned int a, unsigned int b)


Performs per-halfword addition with signed saturation: a + b.

__device__ unsigned int __vaddss4(unsigned int a, unsigned int b)


Performs per-byte addition with signed saturation: a + b.

__device__ unsigned int __vaddus2(unsigned int a, unsigned int b)


Performs per-halfword addition with unsigned saturation: a + b.

__device__ unsigned int __vaddus4(unsigned int a, unsigned int b)


Performs per-byte addition with unsigned saturation: a + b.

__device__ unsigned int __vavgs2(unsigned int a, unsigned int b)


Performs per-halfword signed rounded average computation.

__device__ unsigned int __vavgs4(unsigned int a, unsigned int b)


Computes per-byte signed rounded average.

__device__ unsigned int __vavgu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned rounded average computation.

__device__ unsigned int __vavgu4(unsigned int a, unsigned int b)


Performs per-byte unsigned rounded average.

__device__ unsigned int __vcmpeq2(unsigned int a, unsigned int b)


Performs per-halfword (un)signed comparison: a == b ? 0xffff : 0.

__device__ unsigned int __vcmpeq4(unsigned int a, unsigned int b)


Performs per-byte (un)signed comparison: a == b ? 0xff : 0.

__device__ unsigned int __vcmpges2(unsigned int a, unsigned int b)


Performs per-halfword signed comparison: a >= b ? 0xffff : 0.

__device__ unsigned int __vcmpges4(unsigned int a, unsigned int b)


Performs per-byte signed comparison: a >= b ? 0xff : 0.

__device__ unsigned int __vcmpgeu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned comparison: a >= b ? 0xffff : 0.

__device__ unsigned int __vcmpgeu4(unsigned int a, unsigned int b)


Performs per-byte unsigned comparison: a >= b ? 0xff : 0.

__device__ unsigned int __vcmpgts2(unsigned int a, unsigned int b)


Performs per-halfword signed comparison: a > b ? 0xffff : 0.

__device__ unsigned int __vcmpgts4(unsigned int a, unsigned int b)


Performs per-byte signed comparison: a > b ? 0xff : 0.

__device__ unsigned int __vcmpgtu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned comparison: a > b ? 0xffff : 0.

__device__ unsigned int __vcmpgtu4(unsigned int a, unsigned int b)


Performs per-byte unsigned comparison: a > b ? 0xff : 0.

__device__ unsigned int __vcmples2(unsigned int a, unsigned int b)


Performs per-halfword signed comparison: a <= b ? 0xffff : 0.

__device__ unsigned int __vcmples4(unsigned int a, unsigned int b)


Performs per-byte signed comparison: a <= b ? 0xff : 0.

__device__ unsigned int __vcmpleu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned comparison: a <= b ? 0xffff : 0.

__device__ unsigned int __vcmpleu4(unsigned int a, unsigned int b)


Performs per-byte unsigned comparison: a <= b ? 0xff : 0.

__device__ unsigned int __vcmplts2(unsigned int a, unsigned int b)


Performs per-halfword signed comparison: a < b ? 0xffff : 0.

__device__ unsigned int __vcmplts4(unsigned int a, unsigned int b)


Performs per-byte signed comparison: a < b ? 0xff : 0.

__device__ unsigned int __vcmpltu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned comparison: a < b ? 0xffff : 0.

__device__ unsigned int __vcmpltu4(unsigned int a, unsigned int b)


Performs per-byte unsigned comparison: a < b ? 0xff : 0.

__device__ unsigned int __vcmpne2(unsigned int a, unsigned int b)


Performs per-halfword (un)signed comparison: a != b ? 0xffff : 0.

__device__ unsigned int __vcmpne4(unsigned int a, unsigned int b)


Performs per-byte (un)signed comparison: a != b ? 0xff : 0.

__device__ unsigned int __vhaddu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned average computation.

__device__ unsigned int __vhaddu4(unsigned int a, unsigned int b)


Computes per-byte unsigned average.

__host__ __device__ unsigned int __viaddmax_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword max(a + b, c)

__host__ __device__ unsigned int __viaddmax_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword max(max(a + b, c), 0)

__host__ __device__ int __viaddmax_s32(const int a, const int b, const int c)


Computes max(a + b, c)

__host__ __device__ int __viaddmax_s32_relu(const int a, const int b, const int c)


Computes max(max(a + b, c), 0)

__host__ __device__ unsigned int __viaddmax_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword max(a + b, c)

__host__ __device__ unsigned int __viaddmax_u32(const unsigned int a, const unsigned int b, const unsigned int c)


Computes max(a + b, c)

__host__ __device__ unsigned int __viaddmin_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword min(a + b, c)

__host__ __device__ unsigned int __viaddmin_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword max(min(a + b, c), 0)

__host__ __device__ int __viaddmin_s32(const int a, const int b, const int c)


Computes min(a + b, c)

__host__ __device__ int __viaddmin_s32_relu(const int a, const int b, const int c)


Computes max(min(a + b, c), 0)

__host__ __device__ unsigned int __viaddmin_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword min(a + b, c)

__host__ __device__ unsigned int __viaddmin_u32(const unsigned int a, const unsigned int b, const unsigned int c)


Computes min(a + b, c)

__host__ __device__ unsigned int __vibmax_s16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)


Performs per-halfword max(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a >= b).

__host__ __device__ int __vibmax_s32(const int a, const int b, bool *const pred)


Computes max(a, b), also sets the value pointed to by pred to (a >= b).

__host__ __device__ unsigned int __vibmax_u16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)


Performs per-halfword max(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a >= b).

__host__ __device__ unsigned int __vibmax_u32(const unsigned int a, const unsigned int b, bool *const pred)


Computes max(a, b), also sets the value pointed to by pred to (a >= b).

__host__ __device__ unsigned int __vibmin_s16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)


Performs per-halfword min(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a <= b).

__host__ __device__ int __vibmin_s32(const int a, const int b, bool *const pred)


Computes min(a, b), also sets the value pointed to by pred to (a <= b).

__host__ __device__ unsigned int __vibmin_u16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)


Performs per-halfword min(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a <= b).

__host__ __device__ unsigned int __vibmin_u32(const unsigned int a, const unsigned int b, bool *const pred)


Computes min(a, b), also sets the value pointed to by pred to (a <= b).

__host__ __device__ unsigned int __vimax3_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword max(max(a, b), c)

__host__ __device__ unsigned int __vimax3_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword max(max(max(a, b), c), 0)

__host__ __device__ int __vimax3_s32(const int a, const int b, const int c)


Computes max(max(a, b), c)

__host__ __device__ int __vimax3_s32_relu(const int a, const int b, const int c)


Computes max(max(max(a, b), c), 0)

__host__ __device__ unsigned int __vimax3_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword max(max(a, b), c)

__host__ __device__ unsigned int __vimax3_u32(const unsigned int a, const unsigned int b, const unsigned int c)


Computes max(max(a, b), c)

__host__ __device__ unsigned int __vimax_s16x2_relu(const unsigned int a, const unsigned int b)


Performs per-halfword max(max(a, b), 0)

__host__ __device__ int __vimax_s32_relu(const int a, const int b)


Computes max(max(a, b), 0)

__host__ __device__ unsigned int __vimin3_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword min(min(a, b), c)

__host__ __device__ unsigned int __vimin3_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword max(min(min(a, b), c), 0)

__host__ __device__ int __vimin3_s32(const int a, const int b, const int c)


Computes min(min(a, b), c)

__host__ __device__ int __vimin3_s32_relu(const int a, const int b, const int c)


Computes max(min(min(a, b), c), 0)

__host__ __device__ unsigned int __vimin3_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)


Performs per-halfword min(min(a, b), c)

__host__ __device__ unsigned int __vimin3_u32(const unsigned int a, const unsigned int b, const unsigned int c)


Computes min(min(a, b), c)

__host__ __device__ unsigned int __vimin_s16x2_relu(const unsigned int a, const unsigned int b)


Performs per-halfword max(min(a, b), 0)

__host__ __device__ int __vimin_s32_relu(const int a, const int b)


Computes max(min(a, b), 0)

__device__ unsigned int __vmaxs2(unsigned int a, unsigned int b)


Performs per-halfword signed maximum computation.

__device__ unsigned int __vmaxs4(unsigned int a, unsigned int b)


Computes per-byte signed maximum.

__device__ unsigned int __vmaxu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned maximum computation.

__device__ unsigned int __vmaxu4(unsigned int a, unsigned int b)


Computes per-byte unsigned maximum.

__device__ unsigned int __vmins2(unsigned int a, unsigned int b)


Performs per-halfword signed minimum computation.

__device__ unsigned int __vmins4(unsigned int a, unsigned int b)


Computes per-byte signed minimum.

__device__ unsigned int __vminu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned minimum computation.

__device__ unsigned int __vminu4(unsigned int a, unsigned int b)


Computes per-byte unsigned minimum.

__device__ unsigned int __vneg2(unsigned int a)


Computes per-halfword negation.

__device__ unsigned int __vneg4(unsigned int a)


Performs per-byte negation.

__device__ unsigned int __vnegss2(unsigned int a)


Computes per-halfword negation with signed saturation.

__device__ unsigned int __vnegss4(unsigned int a)


Performs per-byte negation with signed saturation.

__device__ unsigned int __vsads2(unsigned int a, unsigned int b)


Performs per-halfword sum of absolute difference of signed.

__device__ unsigned int __vsads4(unsigned int a, unsigned int b)


Computes per-byte sum of abs difference of signed.

__device__ unsigned int __vsadu2(unsigned int a, unsigned int b)


Computes per-halfword sum of abs diff of unsigned.

__device__ unsigned int __vsadu4(unsigned int a, unsigned int b)


Computes per-byte sum of abs difference of unsigned.

__device__ unsigned int __vseteq2(unsigned int a, unsigned int b)


Performs per-halfword (un)signed comparison: returns 1 if both parts compare equal.

__device__ unsigned int __vseteq4(unsigned int a, unsigned int b)


Performs per-byte (un)signed comparison: returns 1 if all 4 pairs compare equal.

__device__ unsigned int __vsetges2(unsigned int a, unsigned int b)


Performs per-halfword signed comparison: returns 1 if both parts compare greater than or equal.

__device__ unsigned int __vsetges4(unsigned int a, unsigned int b)


Performs per-byte signed comparison: returns 1 if all 4 pairs compare greater than or equal.

__device__ unsigned int __vsetgeu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned comparison: returns 1 if both parts compare greater than or equal.

__device__ unsigned int __vsetgeu4(unsigned int a, unsigned int b)


Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare greater than or equal.

__device__ unsigned int __vsetgts2(unsigned int a, unsigned int b)


Performs per-halfword signed comparison: returns 1 if both parts compare greater than.

__device__ unsigned int __vsetgts4(unsigned int a, unsigned int b)


Performs per-byte signed comparison: returns 1 if all 4 pairs compare greater than.

__device__ unsigned int __vsetgtu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned comparison: returns 1 if both parts compare greater than.

__device__ unsigned int __vsetgtu4(unsigned int a, unsigned int b)


Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare greater than.

__device__ unsigned int __vsetles2(unsigned int a, unsigned int b)


Performs per-halfword unsigned comparison: returns 1 if both parts compare less than or equal.

__device__ unsigned int __vsetles4(unsigned int a, unsigned int b)


Performs per-byte signed comparison: returns 1 if all 4 pairs compare less than or equal.

__device__ unsigned int __vsetleu2(unsigned int a, unsigned int b)


Performs per-halfword signed comparison: returns 1 if both parts compare less than or equal.

__device__ unsigned int __vsetleu4(unsigned int a, unsigned int b)


Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare less than or equal.

__device__ unsigned int __vsetlts2(unsigned int a, unsigned int b)


Performs per-halfword signed comparison: returns 1 if both parts compare less than.

__device__ unsigned int __vsetlts4(unsigned int a, unsigned int b)


Performs per-byte signed comparison: returns 1 if all 4 pairs compare less than.

__device__ unsigned int __vsetltu2(unsigned int a, unsigned int b)


Performs per-halfword unsigned comparison: returns 1 if both parts compare less than.

__device__ unsigned int __vsetltu4(unsigned int a, unsigned int b)


Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare less than.

__device__ unsigned int __vsetne2(unsigned int a, unsigned int b)


Performs per-halfword (un)signed comparison: returns 1 if both parts compare not equal.

__device__ unsigned int __vsetne4(unsigned int a, unsigned int b)


Performs per-byte (un)signed comparison: returns 1 if all 4 pairs compare not equal.

__device__ unsigned int __vsub2(unsigned int a, unsigned int b)


Performs per-halfword (un)signed subtraction, with wrap-around: a - b.

__device__ unsigned int __vsub4(unsigned int a, unsigned int b)


Performs per-byte subtraction: a - b.

__device__ unsigned int __vsubss2(unsigned int a, unsigned int b)


Performs per-halfword (un)signed subtraction, with signed saturation: a - b.

__device__ unsigned int __vsubss4(unsigned int a, unsigned int b)


Performs per-byte subtraction with signed saturation: a - b.

__device__ unsigned int __vsubus2(unsigned int a, unsigned int b)


Performs per-halfword subtraction with unsigned saturation: a - b.

__device__ unsigned int __vsubus4(unsigned int a, unsigned int b)


Performs per-byte subtraction with unsigned saturation: a - b.

##  14.1. Functions

__device__ unsigned int __vabs2(unsigned int a)



Computes per-halfword absolute value: |a|.

Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes, then computes absolute value for each of parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vabs4(unsigned int a)



Computes per-byte absolute value: |a|.

Splits argument by bytes. Computes absolute value of each byte. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vabsdiffs2(unsigned int a, unsigned int b)



Computes per-halfword absolute difference of signed integer: |a - b|.

Splits 4 bytes of each into 2 parts, each consisting of 2 bytes. For corresponding parts function computes absolute difference. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vabsdiffs4(unsigned int a, unsigned int b)



Computes per-byte absolute difference of signed integer: |a - b|.

Splits 4 bytes of each into 4 parts, each consisting of 1 byte. For corresponding parts function computes absolute difference. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vabsdiffu2(unsigned int a, unsigned int b)



Computes per-halfword absolute difference of unsigned integer: |a - b|.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes absolute difference. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vabsdiffu4(unsigned int a, unsigned int b)



Computes per-byte absolute difference of unsigned integer: |a - b|.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes absolute difference. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vabsss2(unsigned int a)



Computes per-halfword absolute value with signed saturation: |a|.

Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes, then computes absolute value with signed saturation for each of parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vabsss4(unsigned int a)



Computes per-byte absolute value with signed saturation: |a|.

Splits 4 bytes of argument into 4 parts, each consisting of 1 byte, then computes absolute value with signed saturation for each of parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vadd2(unsigned int a, unsigned int b)



Performs per-halfword (un)signed addition, with wrap-around: a + b.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then performs unsigned addition on corresponding parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vadd4(unsigned int a, unsigned int b)



Performs per-byte (un)signed addition: a + b.

Splits ‘a’ into 4 bytes, then performs unsigned addition on each of these bytes with the corresponding byte from ‘b’, ignoring overflow. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vaddss2(unsigned int a, unsigned int b)



Performs per-halfword addition with signed saturation: a + b.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then performs addition with signed saturation on corresponding parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vaddss4(unsigned int a, unsigned int b)



Performs per-byte addition with signed saturation: a + b.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte, then performs addition with signed saturation on corresponding parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vaddus2(unsigned int a, unsigned int b)



Performs per-halfword addition with unsigned saturation: a + b.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then performs addition with unsigned saturation on corresponding parts.

Returns


Returns computed value.

__device__ unsigned int __vaddus4(unsigned int a, unsigned int b)



Performs per-byte addition with unsigned saturation: a + b.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte, then performs addition with unsigned saturation on corresponding parts.

Returns


Returns computed value.

__device__ unsigned int __vavgs2(unsigned int a, unsigned int b)



Performs per-halfword signed rounded average computation.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then computes signed rounded average of corresponding parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vavgs4(unsigned int a, unsigned int b)



Computes per-byte signed rounded average.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. then computes signed rounded average of corresponding parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vavgu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned rounded average computation.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then computes unsigned rounded average of corresponding parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vavgu4(unsigned int a, unsigned int b)



Performs per-byte unsigned rounded average.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. then computes unsigned rounded average of corresponding parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vcmpeq2(unsigned int a, unsigned int b)



Performs per-halfword (un)signed comparison: a == b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if they are equal, and 0000 otherwise. For example __vcmpeq2(0x1234aba5, 0x1234aba6) returns 0xffff0000.

Returns


Returns 0xffff computed value.

__device__ unsigned int __vcmpeq4(unsigned int a, unsigned int b)



Performs per-byte (un)signed comparison: a == b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if they are equal, and 00 otherwise. For example __vcmpeq4(0x1234aba5, 0x1234aba6) returns 0xffffff00.

Returns


Returns 0xff if a = b, else returns 0.

__device__ unsigned int __vcmpges2(unsigned int a, unsigned int b)



Performs per-halfword signed comparison: a >= b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part >= ‘b’ part, and 0000 otherwise. For example __vcmpges2(0x1234aba5, 0x1234aba6) returns 0xffff0000.

Returns


Returns 0xffff if a >= b, else returns 0.

__device__ unsigned int __vcmpges4(unsigned int a, unsigned int b)



Performs per-byte signed comparison: a >= b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part >= ‘b’ part, and 00 otherwise. For example __vcmpges4(0x1234aba5, 0x1234aba6) returns 0xffffff00.

Returns


Returns 0xff if a >= b, else returns 0.

__device__ unsigned int __vcmpgeu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned comparison: a >= b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part >= ‘b’ part, and 0000 otherwise. For example __vcmpgeu2(0x1234aba5, 0x1234aba6) returns 0xffff0000.

Returns


Returns 0xffff if a >= b, else returns 0.

__device__ unsigned int __vcmpgeu4(unsigned int a, unsigned int b)



Performs per-byte unsigned comparison: a >= b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part >= ‘b’ part, and 00 otherwise. For example __vcmpgeu4(0x1234aba5, 0x1234aba6) returns 0xffffff00.

Returns


Returns 0xff if a >= b, else returns 0.

__device__ unsigned int __vcmpgts2(unsigned int a, unsigned int b)



Performs per-halfword signed comparison: a > b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part > ‘b’ part, and 0000 otherwise. For example __vcmpgts2(0x1234aba5, 0x1234aba6) returns 0x00000000.

Returns


Returns 0xffff if a > b, else returns 0.

__device__ unsigned int __vcmpgts4(unsigned int a, unsigned int b)



Performs per-byte signed comparison: a > b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part > ‘b’ part, and 00 otherwise. For example __vcmpgts4(0x1234aba5, 0x1234aba6) returns 0x00000000.

Returns


Returns 0xff if a > b, else returns 0.

__device__ unsigned int __vcmpgtu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned comparison: a > b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part > ‘b’ part, and 0000 otherwise. For example __vcmpgtu2(0x1234aba5, 0x1234aba6) returns 0x00000000.

Returns


Returns 0xffff if a > b, else returns 0.

__device__ unsigned int __vcmpgtu4(unsigned int a, unsigned int b)



Performs per-byte unsigned comparison: a > b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part > ‘b’ part, and 00 otherwise. For example __vcmpgtu4(0x1234aba5, 0x1234aba6) returns 0x00000000.

Returns


Returns 0xff if a > b, else returns 0.

__device__ unsigned int __vcmples2(unsigned int a, unsigned int b)



Performs per-halfword signed comparison: a <= b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part <= ‘b’ part, and 0000 otherwise. For example __vcmples2(0x1234aba5, 0x1234aba6) returns 0xffffffff.

Returns


Returns 0xffff if a <= b, else returns 0.

__device__ unsigned int __vcmples4(unsigned int a, unsigned int b)



Performs per-byte signed comparison: a <= b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part <= ‘b’ part, and 00 otherwise. For example __vcmples4(0x1234aba5, 0x1234aba6) returns 0xffffffff.

Returns


Returns 0xff if a <= b, else returns 0.

__device__ unsigned int __vcmpleu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned comparison: a <= b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part <= ‘b’ part, and 0000 otherwise. For example __vcmpleu2(0x1234aba5, 0x1234aba6) returns 0xffffffff.

Returns


Returns 0xffff if a <= b, else returns 0.

__device__ unsigned int __vcmpleu4(unsigned int a, unsigned int b)



Performs per-byte unsigned comparison: a <= b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part <= ‘b’ part, and 00 otherwise. For example __vcmpleu4(0x1234aba5, 0x1234aba6) returns 0xffffffff.

Returns


Returns 0xff if a <= b, else returns 0.

__device__ unsigned int __vcmplts2(unsigned int a, unsigned int b)



Performs per-halfword signed comparison: a < b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part < ‘b’ part, and 0000 otherwise. For example __vcmplts2(0x1234aba5, 0x1234aba6) returns 0x0000ffff.

Returns


Returns 0xffff if a < b, else returns 0.

__device__ unsigned int __vcmplts4(unsigned int a, unsigned int b)



Performs per-byte signed comparison: a < b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part < ‘b’ part, and 00 otherwise. For example __vcmplts4(0x1234aba5, 0x1234aba6) returns 0x000000ff.

Returns


Returns 0xff if a < b, else returns 0.

__device__ unsigned int __vcmpltu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned comparison: a < b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part < ‘b’ part, and 0000 otherwise. For example __vcmpltu2(0x1234aba5, 0x1234aba6) returns 0x0000ffff.

Returns


Returns 0xffff if a < b, else returns 0.

__device__ unsigned int __vcmpltu4(unsigned int a, unsigned int b)



Performs per-byte unsigned comparison: a < b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part < ‘b’ part, and 00 otherwise. For example __vcmpltu4(0x1234aba5, 0x1234aba6) returns 0x000000ff.

Returns


Returns 0xff if a < b, else returns 0.

__device__ unsigned int __vcmpne2(unsigned int a, unsigned int b)



Performs per-halfword (un)signed comparison: a != b ? 0xffff : 0.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part != ‘b’ part, and 0000 otherwise. For example __vcmplts2(0x1234aba5, 0x1234aba6) returns 0x0000ffff.

Returns


Returns 0xffff if a != b, else returns 0.

__device__ unsigned int __vcmpne4(unsigned int a, unsigned int b)



Performs per-byte (un)signed comparison: a != b ? 0xff : 0.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part != ‘b’ part, and 00 otherwise. For example __vcmplts4(0x1234aba5, 0x1234aba6) returns 0x000000ff.

Returns


Returns 0xff if a != b, else returns 0.

__device__ unsigned int __vhaddu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned average computation.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then computes unsigned average of corresponding parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vhaddu4(unsigned int a, unsigned int b)



Computes per-byte unsigned average.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. then computes unsigned average of corresponding parts. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ unsigned int __viaddmax_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword max(a + b, c)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs an add and compare: max(a_part + b_part), c_part) Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ unsigned int __viaddmax_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword max(max(a + b, c), 0)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs an add, followed by a max with relu: max(max(a_part + b_part), c_part), 0) Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ int __viaddmax_s32(const int a, const int b, const int c)



Computes max(a + b, c)

Calculates the sum of signed integers `a` and `b` and takes the max with `c`.

Returns


Returns computed value.

__host__ __device__ int __viaddmax_s32_relu(const int a, const int b, const int c)



Computes max(max(a + b, c), 0)

Calculates the sum of signed integers `a` and `b` and takes the max with `c`. If the result is less than `0` then `0` is returned.

Returns


Returns computed value.

__host__ __device__ unsigned int __viaddmax_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword max(a + b, c)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs an add and compare: max(a_part + b_part), c_part) Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ unsigned int __viaddmax_u32(const unsigned int a, const unsigned int b, const unsigned int c)



Computes max(a + b, c)

Calculates the sum of unsigned integers `a` and `b` and takes the max with `c`.

Returns


Returns computed value.

__host__ __device__ unsigned int __viaddmin_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword min(a + b, c)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs an add and compare: min(a_part + b_part), c_part) Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ unsigned int __viaddmin_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword max(min(a + b, c), 0)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs an add, followed by a min with relu: max(min(a_part + b_part), c_part), 0) Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ int __viaddmin_s32(const int a, const int b, const int c)



Computes min(a + b, c)

Calculates the sum of signed integers `a` and `b` and takes the min with `c`.

Returns


Returns computed value.

__host__ __device__ int __viaddmin_s32_relu(const int a, const int b, const int c)



Computes max(min(a + b, c), 0)

Calculates the sum of signed integers `a` and `b` and takes the min with `c`. If the result is less than `0` then `0` is returned.

Returns


Returns computed value.

__host__ __device__ unsigned int __viaddmin_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword min(a + b, c)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs an add and compare: min(a_part + b_part), c_part) Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ unsigned int __viaddmin_u32(const unsigned int a, const unsigned int b, const unsigned int c)



Computes min(a + b, c)

Calculates the sum of unsigned integers `a` and `b` and takes the min with `c`.

Returns


Returns computed value.

__host__ __device__ unsigned int __vibmax_s16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)



Performs per-halfword max(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a >= b).

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a maximum ( = max(a_part, b_part) ). Partial results are recombined and returned as unsigned int. Sets the value pointed to by `pred_hi` to the value (a_high_part >= b_high_part). Sets the value pointed to by `pred_lo` to the value (a_low_part >= b_low_part).

Returns


Returns computed values.

__host__ __device__ int __vibmax_s32(const int a, const int b, bool *const pred)



Computes max(a, b), also sets the value pointed to by pred to (a >= b).

Calculates the maximum of `a` and `b` of two signed ints. Also sets the value pointed to by `pred` to the value (a >= b).

Returns


Returns computed values.

__host__ __device__ unsigned int __vibmax_u16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)



Performs per-halfword max(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a >= b).

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs a maximum ( = max(a_part, b_part) ). Partial results are recombined and returned as unsigned int. Sets the value pointed to by `pred_hi` to the value (a_high_part >= b_high_part). Sets the value pointed to by `pred_lo` to the value (a_low_part >= b_low_part).

Returns


Returns computed values.

__host__ __device__ unsigned int __vibmax_u32(const unsigned int a, const unsigned int b, bool *const pred)



Computes max(a, b), also sets the value pointed to by pred to (a >= b).

Calculates the maximum of `a` and `b` of two unsigned ints. Also sets the value pointed to by `pred` to the value (a >= b).

Returns


Returns computed values.

__host__ __device__ unsigned int __vibmin_s16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)



Performs per-halfword min(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a <= b).

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a maximum ( = max(a_part, b_part) ). Partial results are recombined and returned as unsigned int. Sets the value pointed to by `pred_hi` to the value (a_high_part <= b_high_part). Sets the value pointed to by `pred_lo` to the value (a_low_part <= b_low_part).

Returns


Returns computed values.

__host__ __device__ int __vibmin_s32(const int a, const int b, bool *const pred)



Computes min(a, b), also sets the value pointed to by pred to (a <= b).

Calculates the minimum of `a` and `b` of two signed ints. Also sets the value pointed to by `pred` to the value (a <= b).

Returns


Returns computed values.

__host__ __device__ unsigned int __vibmin_u16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)



Performs per-halfword min(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a <= b).

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs a maximum ( = max(a_part, b_part) ). Partial results are recombined and returned as unsigned int. Sets the value pointed to by `pred_hi` to the value (a_high_part <= b_high_part). Sets the value pointed to by `pred_lo` to the value (a_low_part <= b_low_part).

Returns


Returns computed values.

__host__ __device__ unsigned int __vibmin_u32(const unsigned int a, const unsigned int b, bool *const pred)



Computes min(a, b), also sets the value pointed to by pred to (a <= b).

Calculates the minimum of `a` and `b` of two unsigned ints. Also sets the value pointed to by `pred` to the value (a <= b).

Returns


Returns computed values.

__host__ __device__ unsigned int __vimax3_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword max(max(a, b), c)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a 3-way max ( = max(max(a_part, b_part), c_part) ). Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ unsigned int __vimax3_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword max(max(max(a, b), c), 0)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a three-way max with relu ( = max(a_part, b_part, c_part, 0) ). Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ int __vimax3_s32(const int a, const int b, const int c)



Computes max(max(a, b), c)

Calculates the 3-way max of signed integers `a`, `b` and `c`.

Returns


Returns computed value.

__host__ __device__ int __vimax3_s32_relu(const int a, const int b, const int c)



Computes max(max(max(a, b), c), 0)

Calculates the maximum of three signed ints, if this is less than `0` then `0` is returned.

Returns


Returns computed value.

__host__ __device__ unsigned int __vimax3_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword max(max(a, b), c)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs a 3-way max ( = max(max(a_part, b_part), c_part) ). Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ unsigned int __vimax3_u32(const unsigned int a, const unsigned int b, const unsigned int c)



Computes max(max(a, b), c)

Calculates the 3-way max of unsigned integers `a`, `b` and `c`.

Returns


Returns computed value.

__host__ __device__ unsigned int __vimax_s16x2_relu(const unsigned int a, const unsigned int b)



Performs per-halfword max(max(a, b), 0)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a max with relu ( = max(a_part, b_part, 0) ). Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ int __vimax_s32_relu(const int a, const int b)



Computes max(max(a, b), 0)

Calculates the maximum of `a` and `b` of two signed ints, if this is less than `0` then `0` is returned.

Returns


Returns computed value.

__host__ __device__ unsigned int __vimin3_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword min(min(a, b), c)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a 3-way min ( = min(min(a_part, b_part), c_part) ). Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ unsigned int __vimin3_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword max(min(min(a, b), c), 0)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a three-way min with relu ( = max(min(a_part, b_part, c_part), 0) ). Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ int __vimin3_s32(const int a, const int b, const int c)



Computes min(min(a, b), c)

Calculates the 3-way min of signed integers `a`, `b` and `c`.

Returns


Returns computed value.

__host__ __device__ int __vimin3_s32_relu(const int a, const int b, const int c)



Computes max(min(min(a, b), c), 0)

Calculates the minimum of three signed ints, if this is less than `0` then `0` is returned.

Returns


Returns computed value.

__host__ __device__ unsigned int __vimin3_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)



Performs per-halfword min(min(a, b), c)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs a 3-way min ( = min(min(a_part, b_part), c_part) ). Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ unsigned int __vimin3_u32(const unsigned int a, const unsigned int b, const unsigned int c)



Computes min(min(a, b), c)

Calculates the 3-way min of unsigned integers `a`, `b` and `c`.

Returns


Returns computed value.

__host__ __device__ unsigned int __vimin_s16x2_relu(const unsigned int a, const unsigned int b)



Performs per-halfword max(min(a, b), 0)

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a min with relu ( = max(min(a_part, b_part), 0) ). Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__host__ __device__ int __vimin_s32_relu(const int a, const int b)



Computes max(min(a, b), 0)

Calculates the minimum of `a` and `b` of two signed ints, if this is less than `0` then `0` is returned.

Returns


Returns computed value.

__device__ unsigned int __vmaxs2(unsigned int a, unsigned int b)



Performs per-halfword signed maximum computation.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes signed maximum. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vmaxs4(unsigned int a, unsigned int b)



Computes per-byte signed maximum.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes signed maximum. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vmaxu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned maximum computation.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes unsigned maximum. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vmaxu4(unsigned int a, unsigned int b)



Computes per-byte unsigned maximum.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes unsigned maximum. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vmins2(unsigned int a, unsigned int b)



Performs per-halfword signed minimum computation.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes signed minimum. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vmins4(unsigned int a, unsigned int b)



Computes per-byte signed minimum.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes signed minimum. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vminu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned minimum computation.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes unsigned minimum. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vminu4(unsigned int a, unsigned int b)



Computes per-byte unsigned minimum.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes unsigned minimum. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vneg2(unsigned int a)



Computes per-halfword negation.

Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes. For each part function computes negation. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vneg4(unsigned int a)



Performs per-byte negation.

Splits 4 bytes of argument into 4 parts, each consisting of 1 byte. For each part function computes negation. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vnegss2(unsigned int a)



Computes per-halfword negation with signed saturation.

Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes. For each part function computes negation. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vnegss4(unsigned int a)



Performs per-byte negation with signed saturation.

Splits 4 bytes of argument into 4 parts, each consisting of 1 byte. For each part function computes negation. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vsads2(unsigned int a, unsigned int b)



Performs per-halfword sum of absolute difference of signed.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes absolute difference and sum it up. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vsads4(unsigned int a, unsigned int b)



Computes per-byte sum of abs difference of signed.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes absolute difference and sum it up. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vsadu2(unsigned int a, unsigned int b)



Computes per-halfword sum of abs diff of unsigned.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes absolute differences and returns sum of those differences.

Returns


Returns computed value.

__device__ unsigned int __vsadu4(unsigned int a, unsigned int b)



Computes per-byte sum of abs difference of unsigned.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes absolute differences and returns sum of those differences.

Returns


Returns computed value.

__device__ unsigned int __vseteq2(unsigned int a, unsigned int b)



Performs per-halfword (un)signed comparison: returns 1 if both parts compare equal.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part == ‘b’ part. If both equalities are satisfied, function returns 1.

Returns


Returns 1 if a = b, else returns 0.

__device__ unsigned int __vseteq4(unsigned int a, unsigned int b)



Performs per-byte (un)signed comparison: returns 1 if all 4 pairs compare equal.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part == ‘b’ part. If both equalities are satisfied, function returns 1.

Returns


Returns 1 if a = b, else returns 0.

__device__ unsigned int __vsetges2(unsigned int a, unsigned int b)



Performs per-halfword signed comparison: returns 1 if both parts compare greater than or equal.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part >= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a >= b, else returns 0.

__device__ unsigned int __vsetges4(unsigned int a, unsigned int b)



Performs per-byte signed comparison: returns 1 if all 4 pairs compare greater than or equal.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part >= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a >= b, else returns 0.

__device__ unsigned int __vsetgeu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned comparison: returns 1 if both parts compare greater than or equal.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part >= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a >= b, else returns 0.

__device__ unsigned int __vsetgeu4(unsigned int a, unsigned int b)



Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare greater than or equal.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part >= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a >= b, else returns 0.

__device__ unsigned int __vsetgts2(unsigned int a, unsigned int b)



Performs per-halfword signed comparison: returns 1 if both parts compare greater than.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part > ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a > b, else returns 0.

__device__ unsigned int __vsetgts4(unsigned int a, unsigned int b)



Performs per-byte signed comparison: returns 1 if all 4 pairs compare greater than.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part > ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a > b, else returns 0.

__device__ unsigned int __vsetgtu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned comparison: returns 1 if both parts compare greater than.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part > ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a > b, else returns 0.

__device__ unsigned int __vsetgtu4(unsigned int a, unsigned int b)



Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare greater than.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part > ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a > b, else returns 0.

__device__ unsigned int __vsetles2(unsigned int a, unsigned int b)



Performs per-halfword unsigned comparison: returns 1 if both parts compare less than or equal.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a <= b, else returns 0.

__device__ unsigned int __vsetles4(unsigned int a, unsigned int b)



Performs per-byte signed comparison: returns 1 if all 4 pairs compare less than or equal.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a <= b, else returns 0.

__device__ unsigned int __vsetleu2(unsigned int a, unsigned int b)



Performs per-halfword signed comparison: returns 1 if both parts compare less than or equal.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a <= b, else returns 0.

__device__ unsigned int __vsetleu4(unsigned int a, unsigned int b)



Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare less than or equal.

Splits 4 bytes of each argument into 4 part, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a <= b, else returns 0.

__device__ unsigned int __vsetlts2(unsigned int a, unsigned int b)



Performs per-halfword signed comparison: returns 1 if both parts compare less than.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a < b, else returns 0.

__device__ unsigned int __vsetlts4(unsigned int a, unsigned int b)



Performs per-byte signed comparison: returns 1 if all 4 pairs compare less than.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a < b, else returns 0.

__device__ unsigned int __vsetltu2(unsigned int a, unsigned int b)



Performs per-halfword unsigned comparison: returns 1 if both parts compare less than.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a < b, else returns 0.

__device__ unsigned int __vsetltu4(unsigned int a, unsigned int b)



Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare less than.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1.

Returns


Returns 1 if a < b, else returns 0.

__device__ unsigned int __vsetne2(unsigned int a, unsigned int b)



Performs per-halfword (un)signed comparison: returns 1 if both parts compare not equal.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part != ‘b’ part. If both conditions are satisfied, function returns 1.

Returns


Returns 1 if a != b, else returns 0.

__device__ unsigned int __vsetne4(unsigned int a, unsigned int b)



Performs per-byte (un)signed comparison: returns 1 if all 4 pairs compare not equal.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part != ‘b’ part. If both conditions are satisfied, function returns 1.

Returns


Returns 1 if a != b, else returns 0.

__device__ unsigned int __vsub2(unsigned int a, unsigned int b)



Performs per-halfword (un)signed subtraction, with wrap-around: a - b.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs subtraction. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vsub4(unsigned int a, unsigned int b)



Performs per-byte subtraction: a - b.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs subtraction. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vsubss2(unsigned int a, unsigned int b)



Performs per-halfword (un)signed subtraction, with signed saturation: a - b.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs subtraction with signed saturation. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vsubss4(unsigned int a, unsigned int b)



Performs per-byte subtraction with signed saturation: a - b.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs subtraction with signed saturation. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vsubus2(unsigned int a, unsigned int b)



Performs per-halfword subtraction with unsigned saturation: a - b.

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs subtraction with unsigned saturation. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.

__device__ unsigned int __vsubus4(unsigned int a, unsigned int b)



Performs per-byte subtraction with unsigned saturation: a - b.

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs subtraction with unsigned saturation. Partial results are recombined and returned as unsigned int.

Returns


Returns computed value.