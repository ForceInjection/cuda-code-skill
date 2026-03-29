# 10. FP128 Quad Precision Mathematical Functions

**Source:** group__CUDA__MATH__QUAD.html


#  10\. FP128 Quad Precision Mathematical Functions

This section describes quad precision mathematical functions.

To use these functions, include the header file `device_fp128_functions.h` in your program.

Functions declared here have `__nv_fp128_` prefix to distinguish them from other global namespace symbols.

Note that FP128 CUDA Math functions are only available to device programs on platforms where host compiler supports the basic quad precision datatype `__float128` or `_Float128`.

Every FP128 CUDA Math function name is overloaded to support either of these host-compiler-specific types, whenever the types are available. See for example:


    #ifdef __FLOAT128_CPP_SPELLING_ENABLED__
        __float128 __nv_fp128_sqrt(__float128 x);
    #endif
    #ifdef __FLOAT128_C_SPELLING_ENABLED__
        _Float128 __nv_fp128_sqrt(_Float128 x);
    #endif


Note

FP128 device computations require compute capability >= 10.0.

Functions

__device__ __float128 __nv_fp128_acos(__float128 x)


Calculate \\(\cos^{-1}{x}\\) , the arc cosine of input argument.

__device__ __float128 __nv_fp128_acosh(__float128 x)


Calculate \\(\cosh^{-1}{x}\\) , the nonnegative inverse hyperbolic cosine of the input argument.

__device__ __float128 __nv_fp128_add(__float128 x, __float128 y)


Compute \\(x + y\\) , the sum of the two floating-point inputs using round-to-nearest-even rounding mode.

__device__ __float128 __nv_fp128_asin(__float128 x)


Calculate \\(\sin^{-1}{x}\\) , the arc sine of input argument.

__device__ __float128 __nv_fp128_asinh(__float128 x)


Calculate \\(\sinh^{-1}{x}\\) , the inverse hyperbolic sine of the input argument.

__device__ __float128 __nv_fp128_atan(__float128 x)


Calculate \\(\tan^{-1}{x}\\) , the arc tangent of input argument.

__device__ __float128 __nv_fp128_atanh(__float128 x)


Calculate \\(\tanh^{-1}{x}\\) , the inverse hyperbolic tangent of the input argument.

__device__ __float128 __nv_fp128_ceil(__float128 x)


Calculate \\(\lceil x \rceil\\) , the smallest integer greater than or equal to `x` .

__device__ __float128 __nv_fp128_copysign(__float128 x, __float128 y)


Create value with the magnitude of the first agument `x` , and the sign of the second argument `y` .

__device__ __float128 __nv_fp128_cos(__float128 x)


Calculate \\(\cos{x}\\) , the cosine of input argument (measured in radians).

__device__ __float128 __nv_fp128_cosh(__float128 x)


Calculate \\(\cosh{x}\\) , the hyperbolic cosine of the input argument.

__device__ __float128 __nv_fp128_div(__float128 x, __float128 y)


Compute \\(\frac{x}{y}\\) , the quotient of the two floating-point inputs using round-to-nearest-even rounding mode.

__device__ __float128 __nv_fp128_exp(__float128 x)


Calculate \\(e^x\\) , the base \\(e\\) exponential of the input argument.

__device__ __float128 __nv_fp128_exp10(__float128 x)


Calculate \\(10^x\\) , the base 10 exponential of the input argument.

__device__ __float128 __nv_fp128_exp2(__float128 x)


Calculate \\(2^x\\) , the base 2 exponential of the input argument.

__device__ __float128 __nv_fp128_expm1(__float128 x)


Calculate \\(e^x - 1\\) , the base e exponential of the input argument, minus 1.

__device__ __float128 __nv_fp128_fabs(__float128 x)


Calculate \\(|x|\\) , the absolute value of the input argument.

__device__ __float128 __nv_fp128_fdim(__float128 x, __float128 y)


Compute the positive difference between `x` and `y` .

__device__ __float128 __nv_fp128_floor(__float128 x)


Calculate \\(\lfloor x \rfloor\\) , the largest integer less than or equal to `x` .

__device__ __float128 __nv_fp128_fma(__float128 x, __float128 y, __float128 c)


Compute \\(x \times y + z\\) as a single operation using round-to-nearest-even rounding mode.

__device__ __float128 __nv_fp128_fmax(__float128 x, __float128 y)


Determine the maximum numeric value of the arguments.

__device__ __float128 __nv_fp128_fmin(__float128 x, __float128 y)


Determine the minimum numeric value of the arguments.

__device__ __float128 __nv_fp128_fmod(__float128 x, __float128 y)


Calculate the floating-point remainder of `x` / `y` .

__device__ __float128 __nv_fp128_frexp(__float128 x, int *nptr)


Extract mantissa and exponent of the floating-point input argument.

__device__ __float128 __nv_fp128_hypot(__float128 x, __float128 y)


Calculate \\(\sqrt{x^2+y^2}\\) , the square root of the sum of squares of two arguments.

__device__ int __nv_fp128_ilogb(__float128 x)


Compute the unbiased integer exponent of the input argument.

__device__ int __nv_fp128_isnan(__float128 x)


Determine whether the input argument is a NaN.

__device__ int __nv_fp128_isunordered(__float128 x, __float128 y)


Determine whether the pair of inputs is unordered.

__device__ __float128 __nv_fp128_ldexp(__float128 x, int exp)


Calculate the value of \\(x\cdot 2^{exp}\\) .

__device__ __float128 __nv_fp128_log(__float128 x)


Calculate \\(\log_{e}{x}\\) , the base \\(e\\) logarithm of the input argument.

__device__ __float128 __nv_fp128_log10(__float128 x)


Calculate \\(\log_{10}{x}\\) , the base 10 logarithm of the input argument.

__device__ __float128 __nv_fp128_log1p(__float128 x)


Calculate the value of \\(\log_{e}(1+x)\\) .

__device__ __float128 __nv_fp128_log2(__float128 x)


Calculate \\(\log_{2}{x}\\) , the base 2 logarithm of the input argument.

__device__ __float128 __nv_fp128_modf(__float128 x, __float128 *iptr)


Break down the input argument into fractional and integral parts.

__device__ __float128 __nv_fp128_mul(__float128 x, __float128 y)


Compute \\(x \cdot y\\) , the product of the two floating-point inputs using round-to-nearest-even rounding mode.

__device__ __float128 __nv_fp128_pow(__float128 x, __float128 y)


Calculate the value of \\(x^{y}\\) , first argument to the power of second argument.

__device__ __float128 __nv_fp128_remainder(__float128 x, __float128 y)


Compute the floating-point remainder function.

__device__ __float128 __nv_fp128_rint(__float128 x)


Round to nearest integer value in floating-point format, with halfway cases rounded to the nearest even integer value.

__device__ __float128 __nv_fp128_round(__float128 x)


Round to nearest integer value in floating-point format, with halfway cases rounded away from zero.

__device__ __float128 __nv_fp128_sin(__float128 x)


Calculate \\(\sin{x}\\) , the sine of input argument (measured in radians).

__device__ __float128 __nv_fp128_sinh(__float128 x)


Calculate \\(\sinh{x}\\) , the hyperbolic sine of the input argument.

__device__ __float128 __nv_fp128_sqrt(__float128 x)


Calculate \\(\sqrt{x}\\) , the square root of the input argument.

__device__ __float128 __nv_fp128_sub(__float128 x, __float128 y)


Compute \\(x - y\\) , the difference of the two floating-point inputs using round-to-nearest-even rounding mode.

__device__ __float128 __nv_fp128_tan(__float128 x)


Calculate \\(\tan{x}\\) , the tangent of input argument (measured in radians).

__device__ __float128 __nv_fp128_tanh(__float128 x)


Calculate \\(\tanh{x}\\) , the hyperbolic tangent of the input argument.

__device__ __float128 __nv_fp128_trunc(__float128 x)


Truncate input argument to the integral part.

##  10.1. Functions

__device__ __float128 __nv_fp128_acos(__float128 x)



Calculate \\( \cos^{-1}{x} \\), the arc cosine of input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The principal value of the arc cosine of the input argument `x`. Result will be in radians, in the interval [0, \\( \pi \\) ] for `x` inside [-1, +1].

  * __nv_fp128_acos(1) returns +0.

  * __nv_fp128_acos(`x`) returns NaN for `x` outside [-1, +1].

  * __nv_fp128_acos(NaN) returns NaN.


__device__ __float128 __nv_fp128_acosh(__float128 x)



Calculate \\( \cosh^{-1}{x} \\), the nonnegative inverse hyperbolic cosine of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


Result will be in the interval [0, \\( +\infty \\) ].

  * __nv_fp128_acosh(1) returns 0.

  * __nv_fp128_acosh(`x`) returns NaN for `x` in the interval  \\( -\infty \\) , 1).

  * __nv_fp128_acosh( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_acosh(NaN) returns NaN.


__device__ __float128 __nv_fp128_add(__float128 x, __float128 y)[



Compute \\( x + y \\), the sum of the two floating-point inputs using round-to-nearest-even rounding mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


Returns `x` \+ `y`.

  * __nv_fp128_add(`x`, `y`) is equivalent to __nv_fp128_add(`y`, `x`).

  * __nv_fp128_add(`x`, \\( \pm\infty \\)) returns \\( \pm\infty \\) for finite `x`.

  * __nv_fp128_add( \\( \pm\infty \\), \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * __nv_fp128_add( \\( \pm\infty \\), \\( \mp\infty \\)) returns NaN.

  * __nv_fp128_add( \\( \pm 0 \\), \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __nv_fp128_add(`x`, `-x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ __float128 __nv_fp128_asin(__float128 x)



Calculate \\( \sin^{-1}{x} \\), the arc sine of input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The principal value of the arc sine of the input argument `x`. Result will be in radians, in the interval [- \\( \pi/2 \\) , + \\( \pi/2 \\) ] for `x` inside [-1, +1].

  * __nv_fp128_asin( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_asin(`x`) returns NaN for `x` outside [-1, +1].

  * __nv_fp128_asin(NaN) returns NaN.


__device__ __float128 __nv_fp128_asinh(__float128 x)



Calculate \\( \sinh^{-1}{x} \\), the inverse hyperbolic sine of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_asinh( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_asinh( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * __nv_fp128_asinh(NaN) returns NaN.


__device__ __float128 __nv_fp128_atan(__float128 x)



Calculate \\( \tan^{-1}{x} \\), the arc tangent of input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The principal value of the arc tangent of the input argument `x`. Result will be in radians, in the interval [- \\( \pi/2 \\) , + \\( \pi/2 \\) ].

  * __nv_fp128_atan( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_atan( \\( \pm \infty \\) ) returns \\( \pm \pi \\) /2.

  * __nv_fp128_atan(NaN) returns NaN.


__device__ __float128 __nv_fp128_atanh(__float128 x)



Calculate \\( \tanh^{-1}{x} \\), the inverse hyperbolic tangent of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_atanh( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_atanh( \\( \pm 1 \\) ) returns \\( \pm \infty \\).

  * __nv_fp128_atanh(`x`) returns NaN for `x` outside interval [-1, 1].

  * __nv_fp128_atanh(NaN) returns NaN.


__device__ __float128 __nv_fp128_ceil(__float128 x)



Calculate \\( \lceil x \rceil \\), the smallest integer greater than or equal to `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


\\( \lceil x \rceil \\) expressed as a floating-point number.

  * __nv_fp128_ceil( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * __nv_fp128_ceil( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_ceil(NaN) returns NaN.


__device__ __float128 __nv_fp128_copysign(__float128 x, __float128 y)



Create value with the magnitude of the first agument `x`, and the sign of the second argument `y`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * copysign(`NaN`, `y`) returns a `NaN` with the sign of `y`.


__device__ __float128 __nv_fp128_cos(__float128 x)



Calculate \\( \cos{x} \\), the cosine of input argument (measured in radians).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


\\( \cos{x} \\).

  * __nv_fp128_cos( \\( \pm 0 \\) ) returns \\( 1 \\).

  * __nv_fp128_cos( \\( \pm \infty \\) ) returns NaN.

  * __nv_fp128_cos(NaN) returns NaN.


__device__ __float128 __nv_fp128_cosh(__float128 x)



Calculate \\( \cosh{x} \\), the hyperbolic cosine of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_cosh( \\( \pm 0 \\) ) returns 1.

  * __nv_fp128_cosh( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_cosh(NaN) returns NaN.


__device__ __float128 __nv_fp128_div(__float128 x, __float128 y)



Compute \\( \frac{x}{y} \\), the quotient of the two floating-point inputs using round-to-nearest-even rounding mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * sign of the quotient `x` / `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __nv_fp128_div( \\( \pm 0 \\), \\( \pm 0 \\)) returns NaN.

  * __nv_fp128_div( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __nv_fp128_div(`x`, \\( \pm\infty \\)) returns \\( 0 \\) of appropriate sign for finite `x`.

  * __nv_fp128_div( \\( \pm\infty \\), `y`) returns \\( \infty \\) of appropriate sign for finite `y`.

  * __nv_fp128_div(`x`, \\( \pm 0 \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __nv_fp128_div( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for `y` \\( \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ __float128 __nv_fp128_exp(__float128 x)



Calculate \\( e^x \\), the base \\( e \\) exponential of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_exp( \\( \pm 0 \\) ) returns 1.

  * __nv_fp128_exp( \\( -\infty \\) ) returns +0.

  * __nv_fp128_exp( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_exp(NaN) returns NaN.


__device__ __float128 __nv_fp128_exp10(__float128 x)



Calculate \\( 10^x \\), the base 10 exponential of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_exp10( \\( \pm 0 \\) ) returns 1.

  * __nv_fp128_exp10( \\( -\infty \\) ) returns +0.

  * __nv_fp128_exp10( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_exp10(NaN) returns NaN.


__device__ __float128 __nv_fp128_exp2(__float128 x)



Calculate \\( 2^x \\), the base 2 exponential of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_exp2( \\( \pm 0 \\) ) returns 1.

  * ex__nv_fp128_exp2p2f( \\( -\infty \\) ) returns +0.

  * __nv_fp128_exp2( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_exp2(NaN) returns NaN.


__device__ __float128 __nv_fp128_expm1(__float128 x)



Calculate \\( e^x - 1 \\), the base e exponential of the input argument, minus 1.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_expm1( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_expm1( \\( -\infty \\) ) returns -1.

  * __nv_fp128_expm1( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_expm1(NaN) returns NaN.


__device__ __float128 __nv_fp128_fabs(__float128 x)



Calculate \\( |x| \\), the absolute value of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_fabs( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_fabs( \\( \pm 0 \\) ) returns +0.

  * __nv_fp128_fabs(NaN) returns an unspecified NaN.


__device__ __float128 __nv_fp128_fdim(__float128 x, __float128 y)



Compute the positive difference between `x` and `y`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_fdim(`x`, `y`) returns `x` \- `y` if \\( x > y \\).

  * __nv_fp128_fdim(`x`, `y`) returns +0 if \\( x \leq y \\).

  * If either argument is NaN, NaN is returned.


__device__ __float128 __nv_fp128_floor(__float128 x)



Calculate \\( \lfloor x \rfloor \\), the largest integer less than or equal to `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


\\( \lfloor x \rfloor \\) expressed as a floating-point number.

  * __nv_fp128_floor( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * __nv_fp128_floor( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_floor(NaN) returns NaN.


__device__ __float128 __nv_fp128_fma(__float128 x, __float128 y, __float128 c)



Compute \\( x \times y + z \\) as a single operation using round-to-nearest-even rounding mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The value of \\( x \times y + z \\) as a single ternary operation, rounded once using round-to-nearest, ties-to-even rounding mode.

  * __nv_fp128_fma( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * __nv_fp128_fma( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * __nv_fp128_fma(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * __nv_fp128_fma(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * __nv_fp128_fma(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __nv_fp128_fma(`x`, `y`, \\( \mp 0 \\)) returns \\( +0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __nv_fp128_fma(`x`, `y`, `z`) returns \\( +0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ __float128 __nv_fp128_fmax(__float128 x, __float128 y)



Determine the maximum numeric value of the arguments.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The maximum numeric value of the arguments `x` and `y`. Treats NaN arguments as missing data.

  * If both arguments are NaN, returns NaN.

  * If one argument is NaN, returns the numeric argument.


__device__ __float128 __nv_fp128_fmin(__float128 x, __float128 y)



Determine the minimum numeric value of the arguments.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The minimum numeric value of the arguments `x` and `y`. Treats NaN arguments as missing data.

  * If both arguments are NaN, returns NaN.

  * If one argument is NaN, returns the numeric argument.


__device__ __float128 __nv_fp128_fmod(__float128 x, __float128 y)



Calculate the floating-point remainder of `x` / `y`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The floating-point remainder of the division operation `x` / `y` calculated by this function is exactly the value `x - n*y`, where `n` is `x` / `y` with its fractional part truncated.

  * The computed value will have the same sign as `x`, and its magnitude will be less than the magnitude of `y`.

  * __nv_fp128_fmod( \\( \pm 0 \\) , `y`) returns \\( \pm 0 \\) if `y` is not zero.

  * __nv_fp128_fmod(`x`, \\( \pm \infty \\) ) returns `x` if `x` is finite.

  * __nv_fp128_fmod(`x`, `y`) returns NaN if `x` is \\( \pm\infty \\) or `y` is zero.

  * If either argument is NaN, NaN is returned.


__device__ __float128 __nv_fp128_frexp(__float128 x, int *nptr)



Extract mantissa and exponent of the floating-point input argument.

Decompose the floating-point value `x` into a component `m` for the normalized fraction element and an integral term `n` for the exponent. The absolute value of `m` will be greater than or equal to 0.5 and less than 1.0 or it will be equal to 0; \\( x = m\cdot 2^n \\). The integer exponent `n` will be stored in the location to which `nptr` points.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The fractional component `m`.

  * __nv_fp128_frexp( \\( \pm 0 \\) , `nptr`) returns \\( \pm 0 \\) and stores zero in the location pointed to by `nptr`.

  * __nv_fp128_frexp( \\( \pm \infty \\) , `nptr`) returns \\( \pm \infty \\) and stores an unspecified value in the location to which `nptr` points.

  * __nv_fp128_frexp(NaN, `y`) returns a NaN and stores an unspecified value in the location to which `nptr` points.


__device__ __float128 __nv_fp128_hypot(__float128 x, __float128 y)



Calculate \\( \sqrt{x^2+y^2} \\), the square root of the sum of squares of two arguments.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The length of the hypotenuse of a right triangle whose two sides have lengths \\( |x| \\) and \\( |y| \\) without undue overflow or underflow.

  * __nv_fp128_hypot(`x`,`y`), __nv_fp128_hypot(`y`,`x`), and __nv_fp128_hypot(`x`, `-y`) are equivalent.

  * __nv_fp128_hypot(`x`, \\( \pm 0 \\) ) is equivalent to __nv_fp128_fabs(`x`).

  * __nv_fp128_hypot( \\( \pm \infty \\) ,`y`) returns \\( +\infty \\), even if `y` is a NaN.

  * __nv_fp128_hypot(NaN, `y`) returns NaN, when `y` is not \\( \pm\infty \\).


__device__ int __nv_fp128_ilogb(__float128 x)



Compute the unbiased integer exponent of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * If successful, returns the unbiased exponent of the argument.

  * __nv_fp128_ilogb( \\( \pm 0 \\) ) returns `INT_MIN`.

  * __nv_fp128_ilogb(NaN) returns `INT_MIN`.

  * __nv_fp128_ilogb( \\( \pm \infty \\) ) returns `INT_MAX`.

  * Note: above behavior does not take into account `FP_ILOGB0` nor `FP_ILOGBNAN`.


__device__ int __nv_fp128_isnan(__float128 x)



Determine whether the input argument is a NaN.

Note

FP128 device computations require compute capability >= 10.0.

Returns


A nonzero value if and only if `x` is a NaN value.

__device__ int __nv_fp128_isunordered(__float128 x, __float128 y)



Determine whether the pair of inputs is unordered.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * nonzero value if at least one of input values is a NaN.

  * zero otherwise


__device__ __float128 __nv_fp128_ldexp(__float128 x, int exp)



Calculate the value of \\( x\cdot 2^{exp} \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_ldexp( \\( \pm 0 \\) , `exp`) returns \\( \pm 0 \\).

  * __nv_fp128_ldexp(`x`, 0) returns `x`.

  * __nv_fp128_ldexp( \\( \pm \infty \\) , `exp`) returns \\( \pm \infty \\).

  * __nv_fp128_ldexp(NaN, `exp`) returns NaN.


__device__ __float128 __nv_fp128_log(__float128 x)



Calculate \\( \log_{e}{x} \\), the base \\( e \\) logarithm of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_log( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * __nv_fp128_log(1) returns +0.

  * __nv_fp128_log(`x`) returns NaN for `x` < 0.

  * __nv_fp128_log( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_log(NaN) returns NaN.


__device__ __float128 __nv_fp128_log10(__float128 x)



Calculate \\( \log_{10}{x} \\), the base 10 logarithm of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_log10( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * __nv_fp128_log10(1) returns +0.

  * __nv_fp128_log10(`x`) returns NaN for `x` < 0.

  * __nv_fp128_log10( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_log10(NaN) returns NaN.


__device__ __float128 __nv_fp128_log1p(__float128 x)



Calculate the value of \\( \log_{e}(1+x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_log1p( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_log1p(-1) returns \\( -\infty \\).

  * __nv_fp128_log1p(`x`) returns NaN for `x` < -1.

  * __nv_fp128_log1p( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_log1p(NaN) returns NaN.


__device__ __float128 __nv_fp128_log2(__float128 x)



Calculate \\( \log_{2}{x} \\), the base 2 logarithm of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_log2( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * __nv_fp128_log2(1) returns +0.

  * __nv_fp128_log2(`x`) returns NaN for `x` < 0.

  * __nv_fp128_log2( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_log2(NaN) returns NaN.


__device__ __float128 __nv_fp128_modf(__float128 x, __float128 *iptr)



Break down the input argument into fractional and integral parts.

Break down the argument `x` into fractional and integral parts. The integral part is stored in floating-point format in the location to which `iptr` points. Fractional and integral parts are given the same sign as the argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_modf( \\( \pm x \\) , `iptr`) returns a result with the same sign as `x`.

  * __nv_fp128_modf( \\( \pm \infty \\) , `iptr`) returns \\( \pm 0 \\) and stores \\( \pm \infty \\) in the object pointed to by `iptr`.

  * __nv_fp128_modf(NaN, `iptr`) stores a NaN in the object pointed to by `iptr` and returns a NaN.


__device__ __float128 __nv_fp128_mul(__float128 x, __float128 y)



Compute \\( x \cdot y \\), the product of the two floating-point inputs using round-to-nearest-even rounding mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


Returns `x` * `y`.

  * sign of the product `x` * `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __nv_fp128_mul(`x`, `y`) is equivalent to __nv_fp128_mul(`y`, `x`).

  * __nv_fp128_mul(`x`, \\( \pm\infty \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __nv_fp128_mul( \\( \pm 0 \\), \\( \pm\infty \\)) returns NaN.

  * __nv_fp128_mul( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for finite `y`.

  * If either argument is NaN, NaN is returned.


__device__ __float128 __nv_fp128_pow(__float128 x, __float128 y)



Calculate the value of \\( x^{y} \\), first argument to the power of second argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_pow( \\( \pm 0 \\) , `y`) returns \\( \pm \infty \\) for `y` an odd integer less than 0.

  * __nv_fp128_pow( \\( \pm 0 \\) , `y`) returns \\( +\infty \\) for `y` less than 0 and not an odd integer.

  * __nv_fp128_pow( \\( \pm 0 \\) , `y`) returns \\( \pm 0 \\) for `y` an odd integer greater than 0.

  * __nv_fp128_pow( \\( \pm 0 \\) , `y`) returns +0 for `y` > 0 and not an odd integer.

  * __nv_fp128_pow(-1, \\( \pm \infty \\) ) returns 1.

  * __nv_fp128_pow(+1, `y`) returns 1 for any `y`, even a NaN.

  * __nv_fp128_pow(`x`, \\( \pm 0 \\) ) returns 1 for any `x`, even a NaN.

  * __nv_fp128_pow(`x`, `y`) returns a NaN for finite `x` < 0 and finite non-integer `y`.

  * __nv_fp128_pow(`x`, \\( -\infty \\) ) returns \\( +\infty \\) for \\( | x | < 1 \\).

  * __nv_fp128_pow(`x`, \\( -\infty \\) ) returns +0 for \\( | x | > 1 \\).

  * __nv_fp128_pow(`x`, \\( +\infty \\) ) returns +0 for \\( | x | < 1 \\).

  * __nv_fp128_pow(`x`, \\( +\infty \\) ) returns \\( +\infty \\) for \\( | x | > 1 \\).

  * __nv_fp128_pow( \\( -\infty \\) , `y`) returns -0 for `y` an odd integer less than 0.

  * __nv_fp128_pow( \\( -\infty \\) , `y`) returns +0 for `y` < 0 and not an odd integer.

  * __nv_fp128_pow( \\( -\infty \\) , `y`) returns \\( -\infty \\) for `y` an odd integer greater than 0.

  * __nv_fp128_pow( \\( -\infty \\) , `y`) returns \\( +\infty \\) for `y` > 0 and not an odd integer.

  * __nv_fp128_pow( \\( +\infty \\) , `y`) returns +0 for `y` < 0.

  * __nv_fp128_pow( \\( +\infty \\) , `y`) returns \\( +\infty \\) for `y` > 0.

  * __nv_fp128_pow(`x`, `y`) returns NaN if either `x` or `y` or both are NaN and `x` \\( \neq \\) +1 and `y` \\( \neq\pm 0 \\).


__device__ __float128 __nv_fp128_remainder(__float128 x, __float128 y)



Compute the floating-point remainder function.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


The floating-point remainder `r` of dividing `x` by `y` for nonzero `y` is defined as \\( r = x - n y \\). The value `n` is the integer value nearest \\( \frac{x}{y} \\). In the halfway cases when \\( | n -\frac{x}{y} | = \frac{1}{2} \\) , the even `n` value is chosen.

  * __nv_fp128_remainder(`x`, \\( \pm 0 \\) ) returns NaN.

  * __nv_fp128_remainder( \\( \pm \infty \\) , `y`) returns NaN.

  * __nv_fp128_remainder(`x`, \\( \pm \infty \\) ) returns `x` for finite `x`.

  * If either argument is NaN, NaN is returned.


__device__ __float128 __nv_fp128_rint(__float128 x)



Round to nearest integer value in floating-point format, with halfway cases rounded to the nearest even integer value.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_rint( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_rint( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * __nv_fp128_rint(NaN) returns NaN.


__device__ __float128 __nv_fp128_round(__float128 x)



Round to nearest integer value in floating-point format, with halfway cases rounded away from zero.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_round( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_round( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * __nv_fp128_round(NaN) returns NaN.


__device__ __float128 __nv_fp128_sin(__float128 x)



Calculate \\( \sin{x} \\), the sine of input argument (measured in radians).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


\\( \sin{x} \\).

  * __nv_fp128_sin( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_sin( \\( \pm \infty \\) ) returns NaN.

  * __nv_fp128_sin(NaN) returns NaN.


__device__ __float128 __nv_fp128_sinh(__float128 x)



Calculate \\( \sinh{x} \\), the hyperbolic sine of the input argument.

Calculate \\( \sinh{x} \\), the hyperbolic sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_sinhinh( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_sinh( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * __nv_fp128_sinh(NaN) returns NaN.


__device__ __float128 __nv_fp128_sqrt(__float128 x)



Calculate \\( \sqrt{x} \\), the square root of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


\\( \sqrt{x} \\).

  * __nv_fp128_sqrt( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_sqrt( \\( +\infty \\) ) returns \\( +\infty \\).

  * __nv_fp128_sqrt(`x`) returns NaN if `x` is less than 0.

  * __nv_fp128_sqrt(NaN) returns NaN.


__device__ __float128 __nv_fp128_sub(__float128 x, __float128 y)



Compute \\( x - y \\), the difference of the two floating-point inputs using round-to-nearest-even rounding mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


Returns `x` \- `y`.

  * __nv_fp128_sub( \\( \pm\infty \\), `y`) returns \\( \pm\infty \\) for finite `y`.

  * __nv_fp128_sub(`x`, \\( \pm\infty \\)) returns \\( \mp\infty \\) for finite `x`.

  * __nv_fp128_sub( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __nv_fp128_sub( \\( \pm\infty \\), \\( \mp\infty \\)) returns \\( \pm\infty \\).

  * __nv_fp128_sub( \\( \pm 0 \\), \\( \mp 0 \\)) returns \\( \pm 0 \\).

  * __nv_fp128_sub(`x`, `x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ __float128 __nv_fp128_tan(__float128 x)



Calculate \\( \tan{x} \\), the tangent of input argument (measured in radians).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


\\( \tan{x} \\).

  * __nv_fp128_tan( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_tan( \\( \pm \infty \\) ) returns NaN.

  * __nv_fp128_tan(NaN) returns NaN.


__device__ __float128 __nv_fp128_tanh(__float128 x)



Calculate \\( \tanh{x} \\), the hyperbolic tangent of the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


  * __nv_fp128_tanh( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_tanh( \\( \pm \infty \\) ) returns \\( \pm 1 \\).

  * __nv_fp128_tanh(NaN) returns NaN.


__device__ __float128 __nv_fp128_trunc(__float128 x)



Truncate input argument to the integral part.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Quad-Precision Floating-Point Functions section.

Note

FP128 device computations require compute capability >= 10.0.

Returns


Rounded `x` to the nearest integer value in floating-point format, that does not exceed `x` in magnitude.

  * __nv_fp128_trunc( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * __nv_fp128_trunc( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * __nv_fp128_trunc(NaN) returns NaN.