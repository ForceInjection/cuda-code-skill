# 7. Single Precision Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__SINGLE.html


#  7\. Single Precision Intrinsics

This section describes single precision intrinsic functions that are only supported in device code.

To use these functions, you do not need to include any additional header file in your program.

Functions

__device__ float __cosf(float x)


Calculate the fast approximate cosine of the input argument.

__device__ float __exp10f(float x)


Calculate the fast approximate base 10 exponential of the input argument.

__device__ float __expf(float x)


Calculate the fast approximate base \\(e\\) exponential of the input argument.

__device__ float2 __fadd2_rd(float2 x, float2 y)


Compute vector add operation \\(x + y\\) in round-down mode.

__device__ float2 __fadd2_rn(float2 x, float2 y)


Compute vector add operation \\(x + y\\) in round-to-nearest-even mode.

__device__ float2 __fadd2_ru(float2 x, float2 y)


Compute vector add operation \\(x + y\\) in round-up mode.

__device__ float2 __fadd2_rz(float2 x, float2 y)


Compute vector add operation \\(x + y\\) in round-towards-zero mode.

__device__ float __fadd_rd(float x, float y)


Add two floating-point values in round-down mode.

__device__ float __fadd_rn(float x, float y)


Add two floating-point values in round-to-nearest-even mode.

__device__ float __fadd_ru(float x, float y)


Add two floating-point values in round-up mode.

__device__ float __fadd_rz(float x, float y)


Add two floating-point values in round-towards-zero mode.

__device__ float __fdiv_rd(float x, float y)


Divide two floating-point values in round-down mode.

__device__ float __fdiv_rn(float x, float y)


Divide two floating-point values in round-to-nearest-even mode.

__device__ float __fdiv_ru(float x, float y)


Divide two floating-point values in round-up mode.

__device__ float __fdiv_rz(float x, float y)


Divide two floating-point values in round-towards-zero mode.

__device__ float __fdividef(float x, float y)


Calculate the fast approximate division of the input arguments.

__device__ float2 __ffma2_rd(float2 x, float2 y, float2 z)


Compute vector fused multiply-add operation \\(x \times y + z\\) in round-down mode.

__device__ float2 __ffma2_rn(float2 x, float2 y, float2 z)


Compute vector fused multiply-add operation \\(x \times y + z\\) in round-to-nearest-even mode.

__device__ float2 __ffma2_ru(float2 x, float2 y, float2 z)


Compute vector fused multiply-add operation \\(x \times y + z\\) in round-up mode.

__device__ float2 __ffma2_rz(float2 x, float2 y, float2 z)


Compute vector fused multiply-add operation \\(x \times y + z\\) in round-towards-zero mode.

__device__ float __fmaf_ieee_rd(float x, float y, float z)


Compute fused multiply-add operation in round-down mode, ignore `-ftz=true` compiler flag.

__device__ float __fmaf_ieee_rn(float x, float y, float z)


Compute fused multiply-add operation in round-to-nearest-even mode, ignore `-ftz=true` compiler flag.

__device__ float __fmaf_ieee_ru(float x, float y, float z)


Compute fused multiply-add operation in round-up mode, ignore `-ftz=true` compiler flag.

__device__ float __fmaf_ieee_rz(float x, float y, float z)


Compute fused multiply-add operation in round-towards-zero mode, ignore `-ftz=true` compiler flag.

__device__ float __fmaf_rd(float x, float y, float z)


Compute \\(x \times y + z\\) as a single operation, in round-down mode.

__device__ float __fmaf_rn(float x, float y, float z)


Compute \\(x \times y + z\\) as a single operation, in round-to-nearest-even mode.

__device__ float __fmaf_ru(float x, float y, float z)


Compute \\(x \times y + z\\) as a single operation, in round-up mode.

__device__ float __fmaf_rz(float x, float y, float z)


Compute \\(x \times y + z\\) as a single operation, in round-towards-zero mode.

__device__ float2 __fmul2_rd(float2 x, float2 y)


Compute vector multiply operation \\(x \times y\\) in round-down mode.

__device__ float2 __fmul2_rn(float2 x, float2 y)


Compute vector multiply operation \\(x \times y\\) in round-to-nearest-even mode.

__device__ float2 __fmul2_ru(float2 x, float2 y)


Compute vector multiply operation \\(x \times y\\) in round-up mode.

__device__ float2 __fmul2_rz(float2 x, float2 y)


Compute vector multiply operation \\(x \times y\\) in round-towards-zero mode.

__device__ float __fmul_rd(float x, float y)


Multiply two floating-point values in round-down mode.

__device__ float __fmul_rn(float x, float y)


Multiply two floating-point values in round-to-nearest-even mode.

__device__ float __fmul_ru(float x, float y)


Multiply two floating-point values in round-up mode.

__device__ float __fmul_rz(float x, float y)


Multiply two floating-point values in round-towards-zero mode.

__device__ float __frcp_rd(float x)


Compute \\(\frac{1}{x}\\) in round-down mode.

__device__ float __frcp_rn(float x)


Compute \\(\frac{1}{x}\\) in round-to-nearest-even mode.

__device__ float __frcp_ru(float x)


Compute \\(\frac{1}{x}\\) in round-up mode.

__device__ float __frcp_rz(float x)


Compute \\(\frac{1}{x}\\) in round-towards-zero mode.

__device__ float __frsqrt_rn(float x)


Compute \\(1/\sqrt{x}\\) in round-to-nearest-even mode.

__device__ float __fsqrt_rd(float x)


Compute \\(\sqrt{x}\\) in round-down mode.

__device__ float __fsqrt_rn(float x)


Compute \\(\sqrt{x}\\) in round-to-nearest-even mode.

__device__ float __fsqrt_ru(float x)


Compute \\(\sqrt{x}\\) in round-up mode.

__device__ float __fsqrt_rz(float x)


Compute \\(\sqrt{x}\\) in round-towards-zero mode.

__device__ float __fsub_rd(float x, float y)


Subtract two floating-point values in round-down mode.

__device__ float __fsub_rn(float x, float y)


Subtract two floating-point values in round-to-nearest-even mode.

__device__ float __fsub_ru(float x, float y)


Subtract two floating-point values in round-up mode.

__device__ float __fsub_rz(float x, float y)


Subtract two floating-point values in round-towards-zero mode.

__device__ float __log10f(float x)


Calculate the fast approximate base 10 logarithm of the input argument.

__device__ float __log2f(float x)


Calculate the fast approximate base 2 logarithm of the input argument.

__device__ float __logf(float x)


Calculate the fast approximate base \\(e\\) logarithm of the input argument.

__device__ float __powf(float x, float y)


Calculate the fast approximate of \\(x^y\\) .

__device__ float __saturatef(float x)


Clamp the input argument to [+0.0, 1.0].

__device__ void __sincosf(float x, float *sptr, float *cptr)


Calculate the fast approximate of sine and cosine of the first input argument.

__device__ float __sinf(float x)


Calculate the fast approximate sine of the input argument.

__device__ float __tanf(float x)


Calculate the fast approximate tangent of the input argument.

__device__ float __tanhf(float x)


Calculate the fast approximate hyperbolic tangent of the input argument.

##  7.1. Functions

__device__ float __cosf(float x)



Calculate the fast approximate cosine of the input argument.

Calculate the fast approximate cosine of the input argument `x`, measured in radians.

See also

cosf() for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns the approximate cosine of `x`.

__device__ float __exp10f(float x)



Calculate the fast approximate base 10 exponential of the input argument.

Calculate the fast approximate base 10 exponential of the input argument `x`, \\( 10^x \\).

See also

exp10f() for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns an approximation to \\( 10^x \\).

__device__ float __expf(float x)



Calculate the fast approximate base \\( e \\) exponential of the input argument.

Calculate the fast approximate base \\( e \\) exponential of the input argument `x`, \\( e^x \\).

See also

expf() for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns an approximation to \\( e^x \\).

__device__ float2 __fadd2_rd(float2 x, float2 y)



Compute vector add operation \\( x + y \\) in round-down mode.

Numeric behavior per component is the same as __fadd_rd().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float2 __fadd2_rn(float2 x, float2 y)



Compute vector add operation \\( x + y \\) in round-to-nearest-even mode.

Numeric behavior per component is the same as __fadd_rn().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float2 __fadd2_ru(float2 x, float2 y)



Compute vector add operation \\( x + y \\) in round-up mode.

Numeric behavior per component is the same as __fadd_ru().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float2 __fadd2_rz(float2 x, float2 y)



Compute vector add operation \\( x + y \\) in round-towards-zero mode.

Numeric behavior per component is the same as __fadd_rz().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float __fadd_rd(float x, float y)



Add two floating-point values in round-down mode.

Compute the sum of `x` and `y` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` \+ `y`.

  * __fadd_rd(`x`, `y`) is equivalent to __fadd_rd(`y`, `x`).

  * __fadd_rd(`x`, \\( \pm\infty \\)) returns \\( \pm\infty \\) for finite `x`.

  * __fadd_rd( \\( \pm\infty \\), \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * __fadd_rd( \\( \pm\infty \\), \\( \mp\infty \\)) returns NaN.

  * __fadd_rd( \\( \pm 0 \\), \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __fadd_rd(`x`, `-x`) returns \\( -0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fadd_rn(float x, float y)



Add two floating-point values in round-to-nearest-even mode.

Compute the sum of `x` and `y` in round-to-nearest-even rounding mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` \+ `y`.

  * __fadd_rn(`x`, `y`) is equivalent to __fadd_rn(`y`, `x`).

  * __fadd_rn(`x`, \\( \pm\infty \\)) returns \\( \pm\infty \\) for finite `x`.

  * __fadd_rn( \\( \pm\infty \\), \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * __fadd_rn( \\( \pm\infty \\), \\( \mp\infty \\)) returns NaN.

  * __fadd_rn( \\( \pm 0 \\), \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __fadd_rn(`x`, `-x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fadd_ru(float x, float y)



Add two floating-point values in round-up mode.

Compute the sum of `x` and `y` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` \+ `y`.

  * __fadd_ru(`x`, `y`) is equivalent to __fadd_ru(`y`, `x`).

  * __fadd_ru(`x`, \\( \pm\infty \\)) returns \\( \pm\infty \\) for finite `x`.

  * __fadd_ru( \\( \pm\infty \\), \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * __fadd_ru( \\( \pm\infty \\), \\( \mp\infty \\)) returns NaN.

  * __fadd_ru( \\( \pm 0 \\), \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __fadd_ru(`x`, `-x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fadd_rz(float x, float y)



Add two floating-point values in round-towards-zero mode.

Compute the sum of `x` and `y` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` \+ `y`.

  * __fadd_rz(`x`, `y`) is equivalent to __fadd_rz(`y`, `x`).

  * __fadd_rz(`x`, \\( \pm\infty \\)) returns \\( \pm\infty \\) for finite `x`.

  * __fadd_rz( \\( \pm\infty \\), \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * __fadd_rz( \\( \pm\infty \\), \\( \mp\infty \\)) returns NaN.

  * __fadd_rz( \\( \pm 0 \\), \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __fadd_rz(`x`, `-x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fdiv_rd(float x, float y)



Divide two floating-point values in round-down mode.

Divide two floating-point values `x` by `y` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns `x` / `y`.

  * sign of the quotient `x` / `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __fdiv_rd( \\( \pm 0 \\), \\( \pm 0 \\)) returns NaN.

  * __fdiv_rd( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __fdiv_rd(`x`, \\( \pm\infty \\)) returns \\( 0 \\) of appropriate sign for finite `x`.

  * __fdiv_rd( \\( \pm\infty \\), `y`) returns \\( \infty \\) of appropriate sign for finite `y`.

  * __fdiv_rd(`x`, \\( \pm 0 \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __fdiv_rd( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for `y` \\( \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fdiv_rn(float x, float y)



Divide two floating-point values in round-to-nearest-even mode.

Divide two floating-point values `x` by `y` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns `x` / `y`.

  * sign of the quotient `x` / `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __fdiv_rn( \\( \pm 0 \\), \\( \pm 0 \\)) returns NaN.

  * __fdiv_rn( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __fdiv_rn(`x`, \\( \pm\infty \\)) returns \\( 0 \\) of appropriate sign for finite `x`.

  * __fdiv_rn( \\( \pm\infty \\), `y`) returns \\( \infty \\) of appropriate sign for finite `y`.

  * __fdiv_rn(`x`, \\( \pm 0 \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __fdiv_rn( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for `y` \\( \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fdiv_ru(float x, float y)



Divide two floating-point values in round-up mode.

Divide two floating-point values `x` by `y` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns `x` / `y`.

  * sign of the quotient `x` / `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __fdiv_ru( \\( \pm 0 \\), \\( \pm 0 \\)) returns NaN.

  * __fdiv_ru( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __fdiv_ru(`x`, \\( \pm\infty \\)) returns \\( 0 \\) of appropriate sign for finite `x`.

  * __fdiv_ru( \\( \pm\infty \\), `y`) returns \\( \infty \\) of appropriate sign for finite `y`.

  * __fdiv_ru(`x`, \\( \pm 0 \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __fdiv_ru( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for `y` \\( \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fdiv_rz(float x, float y)



Divide two floating-point values in round-towards-zero mode.

Divide two floating-point values `x` by `y` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns `x` / `y`.

  * sign of the quotient `x` / `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __fdiv_rz( \\( \pm 0 \\), \\( \pm 0 \\)) returns NaN.

  * __fdiv_rz( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __fdiv_rz(`x`, \\( \pm\infty \\)) returns \\( 0 \\) of appropriate sign for finite `x`.

  * __fdiv_rz( \\( \pm\infty \\), `y`) returns \\( \infty \\) of appropriate sign for finite `y`.

  * __fdiv_rz(`x`, \\( \pm 0 \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __fdiv_rz( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for `y` \\( \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fdividef(float x, float y)



Calculate the fast approximate division of the input arguments.

Calculate the fast approximate division of `x` by `y`.

See also

__fdiv_rn() for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns `x` / `y`.

  * __fdividef( \\( \infty \\) , `y`) returns NaN for \\( 2^{126} < |y| < 2^{128} \\).

  * __fdividef(`x`, `y`) returns 0 for \\( 2^{126} < |y| < 2^{128} \\) and finite \\( x \\).


__device__ float2 __ffma2_rd(float2 x, float2 y, float2 z)



Compute vector fused multiply-add operation \\( x \times y + z \\) in round-down mode.

Numeric behavior per component is the same as __fmaf_rd().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float2 __ffma2_rn(float2 x, float2 y, float2 z)



Compute vector fused multiply-add operation \\( x \times y + z \\) in round-to-nearest-even mode.

Numeric behavior per component is the same as __fmaf_rn().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float2 __ffma2_ru(float2 x, float2 y, float2 z)



Compute vector fused multiply-add operation \\( x \times y + z \\) in round-up mode.

Numeric behavior per component is the same as __fmaf_ru().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float2 __ffma2_rz(float2 x, float2 y, float2 z)



Compute vector fused multiply-add operation \\( x \times y + z \\) in round-towards-zero mode.

Numeric behavior per component is the same as __fmaf_rz().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float __fmaf_ieee_rd(float x, float y, float z)



Compute fused multiply-add operation in round-down mode, ignore `-ftz=true` compiler flag.

Behavior is the same as __fmaf_rd(`x`, `y`, `z`), the difference is in handling denormalized inputs and outputs: `-ftz` compiler flag has no effect.

__device__ float __fmaf_ieee_rn(float x, float y, float z)



Compute fused multiply-add operation in round-to-nearest-even mode, ignore `-ftz=true` compiler flag.

Behavior is the same as __fmaf_rn(`x`, `y`, `z`), the difference is in handling denormalized inputs and outputs: `-ftz` compiler flag has no effect.

__device__ float __fmaf_ieee_ru(float x, float y, float z)



Compute fused multiply-add operation in round-up mode, ignore `-ftz=true` compiler flag.

Behavior is the same as __fmaf_ru(`x`, `y`, `z`), the difference is in handling denormalized inputs and outputs: `-ftz` compiler flag has no effect.

__device__ float __fmaf_ieee_rz(float x, float y, float z)



Compute fused multiply-add operation in round-towards-zero mode, ignore `-ftz=true` compiler flag.

Behavior is the same as __fmaf_rz(`x`, `y`, `z`), the difference is in handling denormalized inputs and outputs: `-ftz` compiler flag has no effect.

__device__ float __fmaf_rd(float x, float y, float z)



Compute \\( x \times y + z \\) as a single operation, in round-down mode.

Computes the value of \\( x \times y + z \\) as a single ternary operation, rounding the result once in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * __fmaf_rd( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * __fmaf_rd( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * __fmaf_rd(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * __fmaf_rd(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * __fmaf_rd(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fmaf_rd(`x`, `y`, \\( \mp 0 \\)) returns \\( -0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fmaf_rd(`x`, `y`, `z`) returns \\( -0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fmaf_rn(float x, float y, float z)



Compute \\( x \times y + z \\) as a single operation, in round-to-nearest-even mode.

Computes the value of \\( x \times y + z \\) as a single ternary operation, rounding the result once in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * __fmaf_rn( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * __fmaf_rn( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * __fmaf_rn(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * __fmaf_rn(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * __fmaf_rn(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fmaf_rn(`x`, `y`, \\( \mp 0 \\)) returns \\( +0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fmaf_rn(`x`, `y`, `z`) returns \\( +0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fmaf_ru(float x, float y, float z)



Compute \\( x \times y + z \\) as a single operation, in round-up mode.

Computes the value of \\( x \times y + z \\) as a single ternary operation, rounding the result once in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * __fmaf_ru( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * __fmaf_ru( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * __fmaf_ru(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * __fmaf_ru(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * __fmaf_ru(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fmaf_ru(`x`, `y`, \\( \mp 0 \\)) returns \\( +0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fmaf_ru(`x`, `y`, `z`) returns \\( +0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fmaf_rz(float x, float y, float z)



Compute \\( x \times y + z \\) as a single operation, in round-towards-zero mode.

Computes the value of \\( x \times y + z \\) as a single ternary operation, rounding the result once in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * __fmaf_rz( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * __fmaf_rz( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * __fmaf_rz(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * __fmaf_rz(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * __fmaf_rz(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fmaf_rz(`x`, `y`, \\( \mp 0 \\)) returns \\( +0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fmaf_rz(`x`, `y`, `z`) returns \\( +0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float2 __fmul2_rd(float2 x, float2 y)



Compute vector multiply operation \\( x \times y \\) in round-down mode.

Numeric behavior per component is the same as __fmul_rd().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float2 __fmul2_rn(float2 x, float2 y)



Compute vector multiply operation \\( x \times y \\) in round-to-nearest-even mode.

Numeric behavior per component is the same as __fmul_rn().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float2 __fmul2_ru(float2 x, float2 y)



Compute vector multiply operation \\( x \times y \\) in round-up mode.

Numeric behavior per component is the same as __fmul_ru().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float2 __fmul2_rz(float2 x, float2 y)



Compute vector multiply operation \\( x \times y \\) in round-towards-zero mode.

Numeric behavior per component is the same as __fmul_rz().

Note

This intrinsic requires compute capability >= 10.0.

Note

The vector variants may not always provide better performance.

__device__ float __fmul_rd(float x, float y)



Multiply two floating-point values in round-down mode.

Compute the product of `x` and `y` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` * `y`.

  * sign of the product `x` * `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __fmul_rd(`x`, `y`) is equivalent to __fmul_rd(`y`, `x`).

  * __fmul_rd(`x`, \\( \pm\infty \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __fmul_rd( \\( \pm 0 \\), \\( \pm\infty \\)) returns NaN.

  * __fmul_rd( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for finite `y`.

  * If either argument is NaN, NaN is returned.


__device__ float __fmul_rn(float x, float y)



Multiply two floating-point values in round-to-nearest-even mode.

Compute the product of `x` and `y` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` * `y`.

  * sign of the product `x` * `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __fmul_rn(`x`, `y`) is equivalent to __fmul_rn(`y`, `x`).

  * __fmul_rn(`x`, \\( \pm\infty \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __fmul_rn( \\( \pm 0 \\), \\( \pm\infty \\)) returns NaN.

  * __fmul_rn( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for finite `y`.

  * If either argument is NaN, NaN is returned.


__device__ float __fmul_ru(float x, float y)



Multiply two floating-point values in round-up mode.

Compute the product of `x` and `y` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` * `y`.

  * sign of the product `x` * `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __fmul_ru(`x`, `y`) is equivalent to __fmul_ru(`y`, `x`).

  * __fmul_ru(`x`, \\( \pm\infty \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __fmul_ru( \\( \pm 0 \\), \\( \pm\infty \\)) returns NaN.

  * __fmul_ru( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for finite `y`.

  * If either argument is NaN, NaN is returned.


__device__ float __fmul_rz(float x, float y)



Multiply two floating-point values in round-towards-zero mode.

Compute the product of `x` and `y` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` * `y`.

  * sign of the product `x` * `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __fmul_rz(`x`, `y`) is equivalent to __fmul_rz(`y`, `x`).

  * __fmul_rz(`x`, \\( \pm\infty \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __fmul_rz( \\( \pm 0 \\), \\( \pm\infty \\)) returns NaN.

  * __fmul_rz( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for finite `y`.

  * If either argument is NaN, NaN is returned.


__device__ float __frcp_rd(float x)



Compute \\( \frac{1}{x} \\) in round-down mode.

Compute the reciprocal of `x` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns \\( \frac{1}{x} \\).

  * __frcp_rd( \\( \pm 0 \\)) returns \\( \pm\infty \\).

  * __frcp_rd( \\( \pm\infty \\)) returns \\( \pm 0 \\).

  * __frcp_rd(NaN) returns NaN.


__device__ float __frcp_rn(float x)



Compute \\( \frac{1}{x} \\) in round-to-nearest-even mode.

Compute the reciprocal of `x` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns \\( \frac{1}{x} \\).

  * __frcp_rn( \\( \pm 0 \\)) returns \\( \pm\infty \\).

  * __frcp_rn( \\( \pm\infty \\)) returns \\( \pm 0 \\).

  * __frcp_rn(NaN) returns NaN.


__device__ float __frcp_ru(float x)



Compute \\( \frac{1}{x} \\) in round-up mode.

Compute the reciprocal of `x` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns \\( \frac{1}{x} \\).

  * __frcp_ru( \\( \pm 0 \\)) returns \\( \pm\infty \\).

  * __frcp_ru( \\( \pm\infty \\)) returns \\( \pm 0 \\).

  * __frcp_ru(NaN) returns NaN.


__device__ float __frcp_rz(float x)



Compute \\( \frac{1}{x} \\) in round-towards-zero mode.

Compute the reciprocal of `x` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns \\( \frac{1}{x} \\).

  * __frcp_rz( \\( \pm 0 \\)) returns \\( \pm\infty \\).

  * __frcp_rz( \\( \pm\infty \\)) returns \\( \pm 0 \\).

  * __frcp_rz(NaN) returns NaN.


__device__ float __frsqrt_rn(float x)



Compute \\( 1/\sqrt{x} \\) in round-to-nearest-even mode.

Compute the reciprocal square root of `x` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns \\( 1/\sqrt{x} \\).

  * __frsqrt_rn( \\( \pm 0 \\)) returns \\( \pm\infty \\).

  * __frsqrt_rn( \\( +\infty \\)) returns \\( +0 \\).

  * __frsqrt_rn(`x`) returns NaN for `x` < 0.

  * __frsqrt_rn(NaN) returns NaN.


__device__ float __fsqrt_rd(float x)



Compute \\( \sqrt{x} \\) in round-down mode.

Compute the square root of `x` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns \\( \sqrt{x} \\).

  * __fsqrt_rd( \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __fsqrt_rd( \\( +\infty \\)) returns \\( +\infty \\).

  * __fsqrt_rd(`x`) returns NaN for `x` < 0.

  * __fsqrt_rd(NaN) returns NaN.


__device__ float __fsqrt_rn(float x)



Compute \\( \sqrt{x} \\) in round-to-nearest-even mode.

Compute the square root of `x` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns \\( \sqrt{x} \\).

  * __fsqrt_rn( \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __fsqrt_rn( \\( +\infty \\)) returns \\( +\infty \\).

  * __fsqrt_rn(`x`) returns NaN for `x` < 0.

  * __fsqrt_rn(NaN) returns NaN.


__device__ float __fsqrt_ru(float x)



Compute \\( \sqrt{x} \\) in round-up mode.

Compute the square root of `x` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns \\( \sqrt{x} \\).

  * __fsqrt_ru( \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __fsqrt_ru( \\( +\infty \\)) returns \\( +\infty \\).

  * __fsqrt_ru(`x`) returns NaN for `x` < 0.

  * __fsqrt_ru(NaN) returns NaN.


__device__ float __fsqrt_rz(float x)



Compute \\( \sqrt{x} \\) in round-towards-zero mode.

Compute the square root of `x` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns \\( \sqrt{x} \\).

  * __fsqrt_rz( \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __fsqrt_rz( \\( +\infty \\)) returns \\( +\infty \\).

  * __fsqrt_rz(`x`) returns NaN for `x` < 0.

  * __fsqrt_rz(NaN) returns NaN.


__device__ float __fsub_rd(float x, float y)



Subtract two floating-point values in round-down mode.

Compute the difference of `x` and `y` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` \- `y`.

  * __fsub_rd( \\( \pm\infty \\), `y`) returns \\( \pm\infty \\) for finite `y`.

  * __fsub_rd(`x`, \\( \pm\infty \\)) returns \\( \mp\infty \\) for finite `x`.

  * __fsub_rd( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __fsub_rd( \\( \pm\infty \\), \\( \mp\infty \\)) returns \\( \pm\infty \\).

  * __fsub_rd( \\( \pm 0 \\), \\( \mp 0 \\)) returns \\( \pm 0 \\).

  * __fsub_rd(`x`, `x`) returns \\( -0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fsub_rn(float x, float y)



Subtract two floating-point values in round-to-nearest-even mode.

Compute the difference of `x` and `y` in round-to-nearest-even rounding mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` \- `y`.

  * __fsub_rn( \\( \pm\infty \\), `y`) returns \\( \pm\infty \\) for finite `y`.

  * __fsub_rn(`x`, \\( \pm\infty \\)) returns \\( \mp\infty \\) for finite `x`.

  * __fsub_rn( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __fsub_rn( \\( \pm\infty \\), \\( \mp\infty \\)) returns \\( \pm\infty \\).

  * __fsub_rn( \\( \pm 0 \\), \\( \mp 0 \\)) returns \\( \pm 0 \\).

  * __fsub_rn(`x`, `x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fsub_ru(float x, float y)



Subtract two floating-point values in round-up mode.

Compute the difference of `x` and `y` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` \- `y`.

  * __fsub_ru( \\( \pm\infty \\), `y`) returns \\( \pm\infty \\) for finite `y`.

  * __fsub_ru(`x`, \\( \pm\infty \\)) returns \\( \mp\infty \\) for finite `x`.

  * __fsub_ru( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __fsub_ru( \\( \pm\infty \\), \\( \mp\infty \\)) returns \\( \pm\infty \\).

  * __fsub_ru( \\( \pm 0 \\), \\( \mp 0 \\)) returns \\( \pm 0 \\).

  * __fsub_ru(`x`, `x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __fsub_rz(float x, float y)



Subtract two floating-point values in round-towards-zero mode.

Compute the difference of `x` and `y` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

This operation will never be merged into a single multiply-add instruction.

Returns


Returns `x` \- `y`.

  * __fsub_rz( \\( \pm\infty \\), `y`) returns \\( \pm\infty \\) for finite `y`.

  * __fsub_rz(`x`, \\( \pm\infty \\)) returns \\( \mp\infty \\) for finite `x`.

  * __fsub_rz( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __fsub_rz( \\( \pm\infty \\), \\( \mp\infty \\)) returns \\( \pm\infty \\).

  * __fsub_rz( \\( \pm 0 \\), \\( \mp 0 \\)) returns \\( \pm 0 \\).

  * __fsub_rz(`x`, `x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float __log10f(float x)



Calculate the fast approximate base 10 logarithm of the input argument.

Calculate the fast approximate base 10 logarithm of the input argument `x`.

See also

log10f() for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns an approximation to \\( \log_{10}(x) \\).

__device__ float __log2f(float x)



Calculate the fast approximate base 2 logarithm of the input argument.

Calculate the fast approximate base 2 logarithm of the input argument `x`.

See also

log2f() for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns an approximation to \\( \log_2(x) \\).

__device__ float __logf(float x)



Calculate the fast approximate base \\( e \\) logarithm of the input argument.

Calculate the fast approximate base \\( e \\) logarithm of the input argument `x`.

See also

logf() for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns an approximation to \\( \log_e(x) \\).

__device__ float __powf(float x, float y)



Calculate the fast approximate of \\( x^y \\).

Calculate the fast approximate of `x`, the first input argument, raised to the power of `y`, the second input argument, \\( x^y \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns an approximation to \\( x^y \\).

__device__ float __saturatef(float x)



Clamp the input argument to [+0.0, 1.0].

Clamp the input argument `x` to be within the interval [+0.0, 1.0].

Returns


  * __saturatef(`x`) returns +0 if \\( x \le 0 \\).

  * __saturatef(`x`) returns 1 if \\( x \ge 1 \\).

  * __saturatef(`x`) returns `x` if \\( 0 < x < 1 \\).

  * __saturatef(NaN) returns +0.


__device__ void __sincosf(float x, float *sptr, float *cptr)



Calculate the fast approximate of sine and cosine of the first input argument.

Calculate the fast approximate of sine and cosine of the first input argument `x` (measured in radians). The results for sine and cosine are written into the second argument, `sptr`, and, respectively, third argument, `cptr`.

See also

__sinf() and __cosf().

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Denorm input/output is flushed to sign preserving 0.0.

__device__ float __sinf(float x)



Calculate the fast approximate sine of the input argument.

Calculate the fast approximate sine of the input argument `x`, measured in radians.

See also

sinf() for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Output in the denormal range is flushed to sign preserving 0.0.

Returns


Returns the approximate sine of `x`.

__device__ float __tanf(float x)



Calculate the fast approximate tangent of the input argument.

Calculate the fast approximate tangent of the input argument `x`, measured in radians.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

The result is computed as the fast divide of __sinf() by __cosf(). Denormal output is flushed to sign-preserving 0.0.

Returns


Returns the approximate tangent of `x`.

__device__ float __tanhf(float x)



Calculate the fast approximate hyperbolic tangent of the input argument.

Calculate the fast approximate hyperbolic tangent of the input argument `x`, measured in radians.

See also

tanhf() for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Returns


Returns the approximate hyperbolic tangent of `x`.