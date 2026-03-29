# 6. Single Precision Mathematical Functions

**Source:** group__CUDA__MATH__SINGLE.html


#  6\. Single Precision Mathematical Functions

This section describes single precision mathematical functions.

To use these functions, you do not need to include any additional header file in your program.

Functions

__device__ float acosf(float x)


Calculate the arc cosine of the input argument.

__device__ float acoshf(float x)


Calculate the nonnegative inverse hyperbolic cosine of the input argument.

__device__ float asinf(float x)


Calculate the arc sine of the input argument.

__device__ float asinhf(float x)


Calculate the inverse hyperbolic sine of the input argument.

__device__ float atan2f(float y, float x)


Calculate the arc tangent of the ratio of first and second input arguments.

__device__ float atanf(float x)


Calculate the arc tangent of the input argument.

__device__ float atanhf(float x)


Calculate the inverse hyperbolic tangent of the input argument.

__device__ float cbrtf(float x)


Calculate the cube root of the input argument.

__device__ float ceilf(float x)


Calculate ceiling of the input argument.

__device__ float copysignf(float x, float y)


Create value with given magnitude, copying sign of second value.

__device__ float cosf(float x)


Calculate the cosine of the input argument.

__device__ float coshf(float x)


Calculate the hyperbolic cosine of the input argument.

__device__ float cospif(float x)


Calculate the cosine of the input argument \\(\times \pi\\) .

__device__ float cyl_bessel_i0f(float x)


Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.

__device__ float cyl_bessel_i1f(float x)


Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.

__device__ float erfcf(float x)


Calculate the complementary error function of the input argument.

__device__ float erfcinvf(float x)


Calculate the inverse complementary error function of the input argument.

__device__ float erfcxf(float x)


Calculate the scaled complementary error function of the input argument.

__device__ float erff(float x)


Calculate the error function of the input argument.

__device__ float erfinvf(float x)


Calculate the inverse error function of the input argument.

__device__ float exp10f(float x)


Calculate the base 10 exponential of the input argument.

__device__ float exp2f(float x)


Calculate the base 2 exponential of the input argument.

__device__ float expf(float x)


Calculate the base \\(e\\) exponential of the input argument.

__device__ float expm1f(float x)


Calculate the base \\(e\\) exponential of the input argument, minus 1.

__device__ float fabsf(float x)


Calculate the absolute value of its argument.

__device__ float fdimf(float x, float y)


Compute the positive difference between `x` and `y` .

__device__ float fdividef(float x, float y)


Divide two floating-point values.

__device__ float floorf(float x)


Calculate the largest integer less than or equal to `x` .

__device__ float fmaf(float x, float y, float z)


Compute \\(x \times y + z\\) as a single operation.

__device__ float fmaxf(float x, float y)


Determine the maximum numeric value of the arguments.

__device__ float fminf(float x, float y)


Determine the minimum numeric value of the arguments.

__device__ float fmodf(float x, float y)


Calculate the floating-point remainder of `x` / `y` .

__device__ float frexpf(float x, int *nptr)


Extract mantissa and exponent of a floating-point value.

__device__ float hypotf(float x, float y)


Calculate the square root of the sum of squares of two arguments.

__device__ int ilogbf(float x)


Compute the unbiased integer exponent of the argument.

__device__ __RETURN_TYPE isfinite(float a)


Determine whether argument is finite.

__device__ __RETURN_TYPE isinf(float a)


Determine whether argument is infinite.

__device__ __RETURN_TYPE isnan(float a)


Determine whether argument is a NaN.

__device__ float j0f(float x)


Calculate the value of the Bessel function of the first kind of order 0 for the input argument.

__device__ float j1f(float x)


Calculate the value of the Bessel function of the first kind of order 1 for the input argument.

__device__ float jnf(int n, float x)


Calculate the value of the Bessel function of the first kind of order n for the input argument.

__device__ float ldexpf(float x, int exp)


Calculate the value of \\(x\cdot 2^{exp}\\) .

__device__ float lgammaf(float x)


Calculate the natural logarithm of the absolute value of the gamma function of the input argument.

__device__ long long int llrintf(float x)


Round input to nearest integer value.

__device__ long long int llroundf(float x)


Round to nearest integer value.

__device__ float log10f(float x)


Calculate the base 10 logarithm of the input argument.

__device__ float log1pf(float x)


Calculate the value of \\(\log_{e}(1+x)\\) .

__device__ float log2f(float x)


Calculate the base 2 logarithm of the input argument.

__device__ float logbf(float x)


Calculate the floating-point representation of the exponent of the input argument.

__device__ float logf(float x)


Calculate the natural logarithm of the input argument.

__device__ long int lrintf(float x)


Round input to nearest integer value.

__device__ long int lroundf(float x)


Round to nearest integer value.

__device__ float max(const float a, const float b)


Calculate the maximum value of the input `float` arguments.

__device__ float min(const float a, const float b)


Calculate the minimum value of the input `float` arguments.

__device__ float modff(float x, float *iptr)


Break down the input argument into fractional and integral parts.

__device__ float nanf(const char *tagp)


Returns "Not a Number" value.

__device__ float nearbyintf(float x)


Round the input argument to the nearest integer.

__device__ float nextafterf(float x, float y)


Return next representable single-precision floating-point value after argument `x` in the direction of `y` .

__device__ float norm3df(float a, float b, float c)


Calculate the square root of the sum of squares of three coordinates of the argument.

__device__ float norm4df(float a, float b, float c, float d)


Calculate the square root of the sum of squares of four coordinates of the argument.

__device__ float normcdff(float x)


Calculate the standard normal cumulative distribution function.

__device__ float normcdfinvf(float x)


Calculate the inverse of the standard normal cumulative distribution function.

__device__ float normf(int dim, float const *p)


Calculate the square root of the sum of squares of any number of coordinates.

__device__ float powf(float x, float y)


Calculate the value of first argument to the power of second argument.

__device__ float rcbrtf(float x)


Calculate reciprocal cube root function.

__device__ float remainderf(float x, float y)


Compute single-precision floating-point remainder.

__device__ float remquof(float x, float y, int *quo)


Compute single-precision floating-point remainder and part of quotient.

__device__ float rhypotf(float x, float y)


Calculate one over the square root of the sum of squares of two arguments.

__device__ float rintf(float x)


Round input to nearest integer value in floating-point.

__device__ float rnorm3df(float a, float b, float c)


Calculate one over the square root of the sum of squares of three coordinates.

__device__ float rnorm4df(float a, float b, float c, float d)


Calculate one over the square root of the sum of squares of four coordinates.

__device__ float rnormf(int dim, float const *p)


Calculate the reciprocal of square root of the sum of squares of any number of coordinates.

__device__ float roundf(float x)


Round to nearest integer value in floating-point.

__device__ float rsqrtf(float x)


Calculate the reciprocal of the square root of the input argument.

__device__ float scalblnf(float x, long int n)


Scale floating-point input by integer power of two.

__device__ float scalbnf(float x, int n)


Scale floating-point input by integer power of two.

__device__ __RETURN_TYPE signbit(float a)


Return the sign bit of the input.

__device__ void sincosf(float x, float *sptr, float *cptr)


Calculate the sine and cosine of the first input argument.

__device__ void sincospif(float x, float *sptr, float *cptr)


Calculate the sine and cosine of the first input argument \\(\times \pi\\) .

__device__ float sinf(float x)


Calculate the sine of the input argument.

__device__ float sinhf(float x)


Calculate the hyperbolic sine of the input argument.

__device__ float sinpif(float x)


Calculate the sine of the input argument \\(\times \pi\\) .

__device__ float sqrtf(float x)


Calculate the square root of the input argument.

__device__ float tanf(float x)


Calculate the tangent of the input argument.

__device__ float tanhf(float x)


Calculate the hyperbolic tangent of the input argument.

__device__ float tgammaf(float x)


Calculate the gamma function of the input argument.

__device__ float truncf(float x)


Truncate input argument to the integral part.

__device__ float y0f(float x)


Calculate the value of the Bessel function of the second kind of order 0 for the input argument.

__device__ float y1f(float x)


Calculate the value of the Bessel function of the second kind of order 1 for the input argument.

__device__ float ynf(int n, float x)


Calculate the value of the Bessel function of the second kind of order n for the input argument.

##  6.1. Functions

__device__ float acosf(float x)



Calculate the arc cosine of the input argument.

Calculate the principal value of the arc cosine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Result will be in radians, in the interval [0, \\( \pi \\) ] for `x` inside [-1, +1].

  * acosf(1) returns +0.

  * acosf(`x`) returns NaN for `x` outside [-1, +1].

  * acosf(NaN) returns NaN.


__device__ float acoshf(float x)



Calculate the nonnegative inverse hyperbolic cosine of the input argument.

Calculate the nonnegative inverse hyperbolic cosine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Result will be in the interval [0, \\( +\infty \\) ].

  * acoshf(1) returns 0.

  * acoshf(`x`) returns NaN for `x` in the interval  \\( -\infty \\) , 1).

  * acoshf( \\( +\infty \\) ) returns \\( +\infty \\).

  * acoshf(NaN) returns NaN.


__device__ float asinf(float x)[



Calculate the arc sine of the input argument.

Calculate the principal value of the arc sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Result will be in radians, in the interval [- \\( \pi/2 \\) , + \\( \pi/2 \\) ] for `x` inside [-1, +1].

  * asinf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * asinf(`x`) returns NaN for `x` outside [-1, +1].

  * asinf(NaN) returns NaN.


__device__ float asinhf(float x)



Calculate the inverse hyperbolic sine of the input argument.

Calculate the inverse hyperbolic sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * asinhf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * asinhf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * asinhf(NaN) returns NaN.


__device__ float atan2f(float y, float x)



Calculate the arc tangent of the ratio of first and second input arguments.

Calculate the principal value of the arc tangent of the ratio of first and second input arguments `y` / `x`. The quadrant of the result is determined by the signs of inputs `y` and `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Result will be in radians, in the interval [- \\( \pi \\) , + \\( \pi \\) ].

  * atan2f( \\( \pm 0 \\) , -0) returns \\( \pm \pi \\).

  * atan2f( \\( \pm 0 \\) , +0) returns \\( \pm 0 \\).

  * atan2f( \\( \pm 0 \\) , `x`) returns \\( \pm \pi \\) for `x` < 0.

  * atan2f( \\( \pm 0 \\) , `x`) returns \\( \pm 0 \\) for `x` > 0.

  * atan2f(`y`, \\( \pm 0 \\) ) returns \\( -\pi \\) /2 for `y` < 0.

  * atan2f(`y`, \\( \pm 0 \\) ) returns \\( \pi \\) /2 for `y` > 0.

  * atan2f( \\( \pm y \\) , \\( -\infty \\) ) returns \\( \pm \pi \\) for finite `y` > 0.

  * atan2f( \\( \pm y \\) , \\( +\infty \\) ) returns \\( \pm 0 \\) for finite `y` > 0.

  * atan2f( \\( \pm \infty \\) , `x`) returns \\( \pm \pi \\) /2 for finite `x`.

  * atan2f( \\( \pm \infty \\) , \\( -\infty \\) ) returns \\( \pm 3\pi \\) /4.

  * atan2f( \\( \pm \infty \\) , \\( +\infty \\) ) returns \\( \pm \pi \\) /4.

  * If either argument is NaN, NaN is returned.


__device__ float atanf(float x)



Calculate the arc tangent of the input argument.

Calculate the principal value of the arc tangent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Result will be in radians, in the interval [- \\( \pi/2 \\) , + \\( \pi/2 \\) ].

  * atanf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * atanf( \\( \pm \infty \\) ) returns \\( \pm \pi \\) /2.

  * atanf(NaN) returns NaN.


__device__ float atanhf(float x)



Calculate the inverse hyperbolic tangent of the input argument.

Calculate the inverse hyperbolic tangent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * atanhf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * atanhf( \\( \pm 1 \\) ) returns \\( \pm \infty \\).

  * atanhf(`x`) returns NaN for `x` outside interval [-1, 1].

  * atanhf(NaN) returns NaN.


__device__ float cbrtf(float x)



Calculate the cube root of the input argument.

Calculate the cube root of `x`, \\( x^{1/3} \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns \\( x^{1/3} \\).

  * cbrtf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * cbrtf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * cbrtf(NaN) returns NaN.


__device__ float ceilf(float x)



Calculate ceiling of the input argument.

Compute the smallest integer value not less than `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns \\( \lceil x \rceil \\) expressed as a floating-point number.

  * ceilf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * ceilf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * ceilf(NaN) returns NaN.


__device__ float copysignf(float x, float y)



Create value with given magnitude, copying sign of second value.

Create a floating-point value with the magnitude `x` and the sign of `y`.

Returns


  * a value with the magnitude of `x` and the sign of `y`.

  * copysignf(`NaN`, `y`) returns a `NaN` with the sign of `y`.


__device__ float cosf(float x)



Calculate the cosine of the input argument.

Calculate the cosine of the input argument `x` (measured in radians).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


  * cosf( \\( \pm 0 \\) ) returns 1.

  * cosf( \\( \pm \infty \\) ) returns NaN.

  * cosf(NaN) returns NaN.


__device__ float coshf(float x)



Calculate the hyperbolic cosine of the input argument.

Calculate the hyperbolic cosine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * coshf( \\( \pm 0 \\) ) returns 1.

  * coshf( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * coshf(NaN) returns NaN.


__device__ float cospif(float x)



Calculate the cosine of the input argument \\( \times \pi \\).

Calculate the cosine of `x` \\( \times \pi \\) (measured in radians), where `x` is the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * cospif( \\( \pm 0 \\) ) returns 1.

  * cospif( \\( \pm \infty \\) ) returns NaN.

  * cospif(NaN) returns NaN.


__device__ float cyl_bessel_i0f(float x)



Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.

Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument `x`, \\( I_0(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the value of the regular modified cylindrical Bessel function of order 0.

  * cyl_bessel_i0f( \\( \pm 0 \\)) returns +1.

  * cyl_bessel_i0f( \\( \pm\infty \\)) returns \\( +\infty \\).

  * cyl_bessel_i0f(NaN) returns NaN.


__device__ float cyl_bessel_i1f(float x)



Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.

Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument `x`, \\( I_1(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the value of the regular modified cylindrical Bessel function of order 1.

  * cyl_bessel_i1f( \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * cyl_bessel_i1f( \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * cyl_bessel_i1f(NaN) returns NaN.


__device__ float erfcf(float x)



Calculate the complementary error function of the input argument.

Calculate the complementary error function of the input argument `x`, 1 - erf(`x`).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * erfcf( \\( -\infty \\) ) returns 2.

  * erfcf( \\( +\infty \\) ) returns +0.

  * erfcf(NaN) returns NaN.


__device__ float erfcinvf(float x)



Calculate the inverse complementary error function of the input argument.

Calculate the inverse complementary error function \\( \operatorname{erfc}^{-1} \\) (`x`), of the input argument `x` in the interval [0, 2].

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * erfcinvf( \\( \pm 0 \\) ) returns \\( +\infty \\).

  * erfcinvf(2) returns \\( -\infty \\).

  * erfcinvf(`x`) returns NaN for `x` outside [0, 2].

  * erfcinvf(NaN) returns NaN.


__device__ float erfcxf(float x)



Calculate the scaled complementary error function of the input argument.

Calculate the scaled complementary error function of the input argument `x`, \\( e^{x^2}\cdot \operatorname{erfc}(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * erfcxf( \\( -\infty \\) ) returns \\( +\infty \\).

  * erfcxf( \\( +\infty \\) ) returns +0.

  * erfcxf(NaN) returns NaN.


__device__ float erff(float x)



Calculate the error function of the input argument.

Calculate the value of the error function for the input argument `x`, \\( \frac{2}{\sqrt \pi} \int_0^x e^{-t^2} dt \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * erff( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * erff( \\( \pm \infty \\) ) returns \\( \pm 1 \\).

  * erff(NaN) returns NaN.


__device__ float erfinvf(float x)



Calculate the inverse error function of the input argument.

Calculate the inverse error function \\( \operatorname{erf}^{-1} \\) (`x`), of the input argument `x` in the interval [-1, 1].

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * erfinvf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * erfinvf(1) returns \\( +\infty \\).

  * erfinvf(-1) returns \\( -\infty \\).

  * erfinvf(`x`) returns NaN for `x` outside [-1, +1].

  * erfinvf(NaN) returns NaN.


__device__ float exp10f(float x)



Calculate the base 10 exponential of the input argument.

Calculate \\( 10^x \\) , the base 10 exponential of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


  * exp10f( \\( \pm 0 \\) ) returns 1.

  * exp10f( \\( -\infty \\) ) returns +0.

  * exp10f( \\( +\infty \\) ) returns \\( +\infty \\).

  * exp10f(NaN) returns NaN.


__device__ float exp2f(float x)



Calculate the base 2 exponential of the input argument.

Calculate \\( 2^x \\) , the base 2 exponential of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * exp2f( \\( \pm 0 \\) ) returns 1.

  * exp2f( \\( -\infty \\) ) returns +0.

  * exp2f( \\( +\infty \\) ) returns \\( +\infty \\).

  * exp2f(NaN) returns NaN.


__device__ float expf(float x)



Calculate the base \\( e \\) exponential of the input argument.

Calculate \\( e^x \\) , the base \\( e \\) exponential of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


  * expf( \\( \pm 0 \\) ) returns 1.

  * expf( \\( -\infty \\) ) returns +0.

  * expf( \\( +\infty \\) ) returns \\( +\infty \\).

  * expf(NaN) returns NaN.


__device__ float expm1f(float x)



Calculate the base \\( e \\) exponential of the input argument, minus 1.

Calculate \\( e^x \\) -1, the base \\( e \\) exponential of the input argument `x`, minus 1.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * expm1f( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * expm1f( \\( -\infty \\) ) returns -1.

  * expm1f( \\( +\infty \\) ) returns \\( +\infty \\).

  * expm1f(NaN) returns NaN.


__device__ float fabsf(float x)



Calculate the absolute value of its argument.

Calculate the absolute value of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the absolute value of its argument.

  * fabsf( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * fabsf( \\( \pm 0 \\) ) returns +0.

  * fabsf(NaN) returns an unspecified NaN.


__device__ float fdimf(float x, float y)



Compute the positive difference between `x` and `y`.

Compute the positive difference between `x` and `y`. The positive difference is `x` \- `y` when `x` > `y` and +0 otherwise.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the positive difference between `x` and `y`.

  * fdimf(`x`, `y`) returns `x` \- `y` if `x` > `y`.

  * fdimf(`x`, `y`) returns +0 if `x` \\( \leq \\) `y`.

  * If either argument is NaN, NaN is returned.


__device__ float fdividef(float x, float y)



Divide two floating-point values.

Compute `x` divided by `y`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


Returns `x` / `y`.

  * Follows the regular division operation behavior by default.

  * If `-use_fast_math` is specified and is not amended by an explicit `-prec_div=true`, uses __fdividef() for higher performance


__device__ float floorf(float x)



Calculate the largest integer less than or equal to `x`.

Calculate the largest integer value which is less than or equal to `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns \\( \lfloor x \rfloor \\) expressed as a floating-point number.

  * floorf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * floorf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * floorf(NaN) returns NaN.


__device__ float fmaf(float x, float y, float z)



Compute \\( x \times y + z \\) as a single operation.

Compute the value of \\( x \times y + z \\) as a single ternary operation. After computing the value to infinite precision, the value is rounded once using round-to-nearest, ties-to-even rounding mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * fmaf( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * fmaf( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * fmaf(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * fmaf(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * fmaf(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * fmaf(`x`, `y`, \\( \mp 0 \\)) returns \\( +0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * fmaf(`x`, `y`, `z`) returns \\( +0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ float fmaxf(float x, float y)



Determine the maximum numeric value of the arguments.

Determines the maximum numeric value of the arguments `x` and `y`. Treats NaN arguments as missing data. If one argument is a NaN and the other is legitimate numeric value, the numeric value is chosen.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the maximum numeric values of the arguments `x` and `y`.

  * If both arguments are NaN, returns NaN.

  * If one argument is NaN, returns the numeric argument.


__device__ float fminf(float x, float y)



Determine the minimum numeric value of the arguments.

Determines the minimum numeric value of the arguments `x` and `y`. Treats NaN arguments as missing data. If one argument is a NaN and the other is legitimate numeric value, the numeric value is chosen.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the minimum numeric value of the arguments `x` and `y`.

  * If both arguments are NaN, returns NaN.

  * If one argument is NaN, returns the numeric argument.


__device__ float fmodf(float x, float y)



Calculate the floating-point remainder of `x` / `y`.

Calculate the floating-point remainder of `x` / `y`. The floating-point remainder of the division operation `x` / `y` calculated by this function is exactly the value `x - n*y`, where `n` is `x` / `y` with its fractional part truncated. The computed value will have the same sign as `x`, and its magnitude will be less than the magnitude of `y`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * Returns the floating-point remainder of `x` / `y`.

  * fmodf( \\( \pm 0 \\) , `y`) returns \\( \pm 0 \\) if `y` is not zero.

  * fmodf(`x`, \\( \pm \infty \\) ) returns `x` if `x` is finite.

  * fmodf(`x`, `y`) returns NaN if `x` is \\( \pm\infty \\) or `y` is zero.

  * If either argument is NaN, NaN is returned.


__device__ float frexpf(float x, int *nptr)



Extract mantissa and exponent of a floating-point value.

Decomposes the floating-point value `x` into a component `m` for the normalized fraction element and another term `n` for the exponent. The absolute value of `m` will be greater than or equal to 0.5 and less than 1.0 or it will be equal to 0; \\( x = m\cdot 2^n \\). The integer exponent `n` will be stored in the location to which `nptr` points.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the fractional component `m`.

  * frexpf( \\( \pm 0 \\) , `nptr`) returns \\( \pm 0 \\) and stores zero in the location pointed to by `nptr`.

  * frexpf( \\( \pm \infty \\) , `nptr`) returns \\( \pm \infty \\) and stores an unspecified value in the location to which `nptr` points.

  * frexpf(NaN, `y`) returns a NaN and stores an unspecified value in the location to which `nptr` points.


__device__ float hypotf(float x, float y)



Calculate the square root of the sum of squares of two arguments.

Calculates the length of the hypotenuse of a right triangle whose two sides have lengths `x` and `y` without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the length of the hypotenuse \\( \sqrt{x^2+y^2} \\).

  * hypotf(`x`,`y`), hypotf(`y`,`x`), and hypotf(`x`, `-y`) are equivalent.

  * hypotf(`x`, \\( \pm 0 \\) ) is equivalent to fabsf(`x`).

  * hypotf( \\( \pm \infty \\) ,`y`) returns \\( +\infty \\) , even if `y` is a NaN.

  * hypotf(NaN, `y`) returns NaN, when `y` is not \\( \pm\infty \\).


__device__ int ilogbf(float x)



Compute the unbiased integer exponent of the argument.

Calculates the unbiased integer exponent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * If successful, returns the unbiased exponent of the argument.

  * ilogbf( \\( \pm 0 \\) ) returns `INT_MIN`.

  * ilogbf(NaN) returns `INT_MIN`.

  * ilogbf( \\( \pm \infty \\) ) returns `INT_MAX`.

  * Note: above behavior does not take into account `FP_ILOGB0` nor `FP_ILOGBNAN`.


__device__ __RETURN_TYPE isfinite(float a)



Determine whether argument is finite.

Determine whether the floating-point value `a` is a finite value (zero, subnormal, or normal and not infinity or NaN).

Returns


  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is a finite value.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is a finite value.


__device__ __RETURN_TYPE isinf(float a)



Determine whether argument is infinite.

Determine whether the floating-point value `a` is an infinite value (positive or negative).

Returns


  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is an infinite value.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is an infinite value.


__device__ __RETURN_TYPE isnan(float a)



Determine whether argument is a NaN.

Determine whether the floating-point value `a` is a NaN.

Returns


  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is a NaN value.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is a NaN value.


__device__ float j0f(float x)



Calculate the value of the Bessel function of the first kind of order 0 for the input argument.

Calculate the value of the Bessel function of the first kind of order 0 for the input argument `x`, \\( J_0(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the first kind of order 0.

  * j0f( \\( \pm \infty \\) ) returns +0.

  * j0f(NaN) returns NaN.


__device__ float j1f(float x)



Calculate the value of the Bessel function of the first kind of order 1 for the input argument.

Calculate the value of the Bessel function of the first kind of order 1 for the input argument `x`, \\( J_1(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the first kind of order 1.

  * j1f( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * j1f( \\( \pm \infty \\) ) returns \\( \pm 0 \\).

  * j1f(NaN) returns NaN.


__device__ float jnf(int n, float x)



Calculate the value of the Bessel function of the first kind of order n for the input argument.

Calculate the value of the Bessel function of the first kind of order `n` for the input argument `x`, \\( J_n(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the first kind of order `n`.

  * jnf(`n`, NaN) returns NaN.

  * jnf(`n`, `x`) returns NaN for `n` < 0.

  * jnf(`n`, \\( +\infty \\) ) returns +0.


__device__ float ldexpf(float x, int exp)



Calculate the value of \\( x\cdot 2^{exp} \\).

Calculate the value of \\( x\cdot 2^{exp} \\) of the input arguments `x` and `exp`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * ldexpf(`x`, `exp`) is equivalent to scalbnf(`x`, `exp`).


__device__ float lgammaf(float x)



Calculate the natural logarithm of the absolute value of the gamma function of the input argument.

Calculate the natural logarithm of the absolute value of the gamma function of the input argument `x`, namely the value of \\( \log_{e}\left|\Gamma(x)\right| \\)

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * lgammaf(1) returns +0.

  * lgammaf(2) returns +0.

  * lgammaf(`x`) returns \\( +\infty \\) if `x` \\( \leq \\) 0 and `x` is an integer.

  * lgammaf( \\( -\infty \\) ) returns \\( +\infty \\).

  * lgammaf( \\( +\infty \\) ) returns \\( +\infty \\).

  * lgammaf(NaN) returns NaN.


__device__ long long int llrintf(float x)



Round input to nearest integer value.

Round `x` to the nearest integer value, with halfway cases rounded to the nearest even integer value. If the result is outside the range of the return type, the behavior is undefined.

Returns


Returns rounded integer value.

__device__ long long int llroundf(float x)



Round to nearest integer value.

Round `x` to the nearest integer value, with halfway cases rounded away from zero. If the result is outside the range of the return type, the behavior is undefined.

Note

This function may be slower than alternate rounding methods. See llrintf().

Returns


Returns rounded integer value.

__device__ float log10f(float x)



Calculate the base 10 logarithm of the input argument.

Calculate the base 10 logarithm of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


  * log10f( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * log10f(1) returns +0.

  * log10f(`x`) returns NaN for `x` < 0.

  * log10f( \\( +\infty \\) ) returns \\( +\infty \\).

  * log10f(NaN) returns NaN.


__device__ float log1pf(float x)



Calculate the value of \\( \log_{e}(1+x) \\).

Calculate the value of \\( \log_{e}(1+x) \\) of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * log1pf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * log1pf(-1) returns \\( -\infty \\).

  * log1pf(`x`) returns NaN for `x` < -1.

  * log1pf( \\( +\infty \\) ) returns \\( +\infty \\).

  * log1pf(NaN) returns NaN.


__device__ float log2f(float x)



Calculate the base 2 logarithm of the input argument.

Calculate the base 2 logarithm of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


  * log2f( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * log2f(1) returns +0.

  * log2f(`x`) returns NaN for `x` < 0.

  * log2f( \\( +\infty \\) ) returns \\( +\infty \\).

  * log2f(NaN) returns NaN.


__device__ float logbf(float x)



Calculate the floating-point representation of the exponent of the input argument.

Calculate the floating-point representation of the exponent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * logbf( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * logbf( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * logbf(NaN) returns NaN.


__device__ float logf(float x)



Calculate the natural logarithm of the input argument.

Calculate the natural logarithm of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


  * logf( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * logf(1) returns +0.

  * logf(`x`) returns NaN for `x` < 0.

  * logf( \\( +\infty \\) ) returns \\( +\infty \\).

  * logf(NaN) returns NaN.


__device__ long int lrintf(float x)



Round input to nearest integer value.

Round `x` to the nearest integer value, with halfway cases rounded to the nearest even integer value. If the result is outside the range of the return type, the behavior is undefined.

Returns


Returns rounded integer value.

__device__ long int lroundf(float x)



Round to nearest integer value.

Round `x` to the nearest integer value, with halfway cases rounded away from zero. If the result is outside the range of the return type, the behavior is undefined.

Note

This function may be slower than alternate rounding methods. See lrintf().

Returns


Returns rounded integer value.

__device__ float max(const float a, const float b)



Calculate the maximum value of the input `float` arguments.

Calculate the maximum value of the arguments `a` and `b`. Behavior is equivalent to fmaxf() function.

Note, this is different from `std:`: specification

__device__ float min(const float a, const float b)



Calculate the minimum value of the input `float` arguments.

Calculate the minimum value of the arguments `a` and `b`. Behavior is equivalent to fminf() function.

Note, this is different from `std:`: specification

__device__ float modff(float x, float *iptr)



Break down the input argument into fractional and integral parts.

Break down the argument `x` into fractional and integral parts. The integral part is stored in the argument `iptr`. Fractional and integral parts are given the same sign as the argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * modff( \\( \pm x \\) , `iptr`) returns a result with the same sign as `x`.

  * modff( \\( \pm \infty \\) , `iptr`) returns \\( \pm 0 \\) and stores \\( \pm \infty \\) in the object pointed to by `iptr`.

  * modff(NaN, `iptr`) stores a NaN in the object pointed to by `iptr` and returns a NaN.


__device__ float nanf(const char *tagp)



Returns “Not a Number” value.

Return a representation of a quiet NaN. Argument `tagp` selects one of the possible representations.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * nanf(`tagp`) returns NaN.


__device__ float nearbyintf(float x)



Round the input argument to the nearest integer.

Round argument `x` to an integer value in single precision floating-point format. Uses round to nearest rounding, with ties rounding to even.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * nearbyintf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * nearbyintf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * nearbyintf(NaN) returns NaN.


__device__ float nextafterf(float x, float y)



Return next representable single-precision floating-point value after argument `x` in the direction of `y`.

Calculate the next representable single-precision floating-point value following `x` in the direction of `y`. For example, if `y` is greater than `x`, nextafterf() returns the smallest representable number greater than `x`

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * nextafterf(`x`, `y`) = `y` if `x` equals `y`.

  * nextafterf(`x`, `y`) = `NaN` if either `x` or `y` are `NaN`.


__device__ float norm3df(float a, float b, float c)



Calculate the square root of the sum of squares of three coordinates of the argument.

Calculates the length of three dimensional vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the length of the 3D vector \\( \sqrt{a^2+b^2+c^2} \\).

  * In the presence of an exactly infinite coordinate \\( +\infty \\) is returned, even if there are NaNs.

  * returns +0, when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float norm4df(float a, float b, float c, float d)



Calculate the square root of the sum of squares of four coordinates of the argument.

Calculates the length of four dimensional vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the length of the 4D vector \\( \sqrt{a^2+b^2+c^2+d^2} \\).

  * In the presence of an exactly infinite coordinate \\( +\infty \\) is returned, even if there are NaNs.

  * returns +0, when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float normcdff(float x)



Calculate the standard normal cumulative distribution function.

Calculate the cumulative distribution function of the standard normal distribution for input argument `x`, \\( \Phi(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * normcdff( \\( +\infty \\) ) returns 1.

  * normcdff( \\( -\infty \\) ) returns +0

  * normcdff(NaN) returns NaN.


__device__ float normcdfinvf(float x)



Calculate the inverse of the standard normal cumulative distribution function.

Calculate the inverse of the standard normal cumulative distribution function for input argument `x`, \\( \Phi^{-1}(x) \\). The function is defined for input values in the interval \\( (0, 1) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * normcdfinvf( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * normcdfinvf(1) returns \\( +\infty \\).

  * normcdfinvf(`x`) returns NaN if `x` is not in the interval [0,1].

  * normcdfinvf(NaN) returns NaN.


__device__ float normf(int dim, float const *p)



Calculate the square root of the sum of squares of any number of coordinates.

Calculates the length of a vector `p`, dimension of which is passed as an argument without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the length of the dim-D vector \\( \sqrt{\sum_{i=0}^{dim-1} p_i^2} \\).

  * In the presence of an exactly infinite coordinate \\( +\infty \\) is returned, even if there are NaNs.

  * returns +0, when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float powf(float x, float y)



Calculate the value of first argument to the power of second argument.

Calculate the value of `x` to the power of `y`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


  * powf( \\( \pm 0 \\) , `y`) returns \\( \pm \infty \\) for `y` an odd integer less than 0.

  * powf( \\( \pm 0 \\) , `y`) returns \\( +\infty \\) for `y` less than 0 and not an odd integer.

  * powf( \\( \pm 0 \\) , `y`) returns \\( \pm 0 \\) for `y` an odd integer greater than 0.

  * powf( \\( \pm 0 \\) , `y`) returns +0 for `y` > 0 and not an odd integer.

  * powf(-1, \\( \pm \infty \\) ) returns 1.

  * powf(+1, `y`) returns 1 for any `y`, even a NaN.

  * powf(`x`, \\( \pm 0 \\) ) returns 1 for any `x`, even a NaN.

  * powf(`x`, `y`) returns a NaN for finite `x` < 0 and finite non-integer `y`.

  * powf(`x`, \\( -\infty \\) ) returns \\( +\infty \\) for \\( | x | < 1 \\).

  * powf(`x`, \\( -\infty \\) ) returns +0 for \\( | x | > 1 \\).

  * powf(`x`, \\( +\infty \\) ) returns +0 for \\( | x | < 1 \\).

  * powf(`x`, \\( +\infty \\) ) returns \\( +\infty \\) for \\( | x | > 1 \\).

  * powf( \\( -\infty \\) , `y`) returns -0 for `y` an odd integer less than 0.

  * powf( \\( -\infty \\) , `y`) returns +0 for `y` < 0 and not an odd integer.

  * powf( \\( -\infty \\) , `y`) returns \\( -\infty \\) for `y` an odd integer greater than 0.

  * powf( \\( -\infty \\) , `y`) returns \\( +\infty \\) for `y` > 0 and not an odd integer.

  * powf( \\( +\infty \\) , `y`) returns +0 for `y` < 0.

  * powf( \\( +\infty \\) , `y`) returns \\( +\infty \\) for `y` > 0.

  * powf(`x`, `y`) returns NaN if either `x` or `y` or both are NaN and `x` \\( \neq \\) +1 and `y` \\( \neq\pm 0 \\).


__device__ float rcbrtf(float x)



Calculate reciprocal cube root function.

Calculate reciprocal cube root function of `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * rcbrtf( \\( \pm 0 \\) ) returns \\( \pm \infty \\).

  * rcbrtf( \\( \pm \infty \\) ) returns \\( \pm 0 \\).

  * rcbrtf(NaN) returns NaN.


__device__ float remainderf(float x, float y)



Compute single-precision floating-point remainder.

Compute single-precision floating-point remainder `r` of dividing `x` by `y` for nonzero `y`. Thus \\( r = x - n y \\). The value `n` is the integer value nearest \\( \frac{x}{y} \\). In the case when \\( | n -\frac{x}{y} | = \frac{1}{2} \\) , the even `n` value is chosen.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * remainderf(`x`, \\( \pm 0 \\) ) returns NaN.

  * remainderf( \\( \pm \infty \\) , `y`) returns NaN.

  * remainderf(`x`, \\( \pm \infty \\) ) returns `x` for finite `x`.

  * If either argument is NaN, NaN is returned.


__device__ float remquof(float x, float y, int *quo)



Compute single-precision floating-point remainder and part of quotient.

Compute a single-precision floating-point remainder in the same way as the remainderf() function. Argument `quo` returns part of quotient upon division of `x` by `y`. Value `quo` has the same sign as \\( \frac{x}{y} \\) and may not be the exact quotient but agrees with the exact quotient in the low order 3 bits.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the remainder.

  * remquof(`x`, \\( \pm 0 \\) , `quo`) returns NaN and stores an unspecified value in the location to which `quo` points.

  * remquof( \\( \pm \infty \\) , `y`, `quo`) returns NaN and stores an unspecified value in the location to which `quo` points.

  * remquof(`x`, `y`, `quo`) returns NaN and stores an unspecified value in the location to which `quo` points if either of `x` or `y` is NaN.

  * remquof(`x`, \\( \pm \infty \\) , `quo`) returns `x` and stores zero in the location to which `quo` points for finite `x`.


__device__ float rhypotf(float x, float y)



Calculate one over the square root of the sum of squares of two arguments.

Calculates one over the length of the hypotenuse of a right triangle whose two sides have lengths `x` and `y` without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns one over the length of the hypotenuse \\( \frac{1}{\sqrt{x^2+y^2}} \\).

  * rhypotf(`x`,`y`), rhypotf(`y`,`x`), and rhypotf(`x`, `-y`) are equivalent.

  * rhypotf( \\( \pm \infty \\) ,`y`) returns +0, even if `y` is a NaN.

  * rhypotf( \\( \pm 0, \pm 0 \\)) returns \\( +\infty \\).

  * rhypotf(NaN, `y`) returns NaN, when `y` is not \\( \pm\infty \\).


__device__ float rintf(float x)



Round input to nearest integer value in floating-point.

Round `x` to the nearest integer value in floating-point format, with halfway cases rounded to the nearest even integer value.

Returns


Returns rounded integer value.

  * rintf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * rintf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * rintf(NaN) returns NaN.


__device__ float rnorm3df(float a, float b, float c)



Calculate one over the square root of the sum of squares of three coordinates.

Calculates one over the length of three dimension vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns one over the length of the 3D vector \\( \frac{1}{\sqrt{a^2+b^2+c^2}} \\).

  * In the presence of an exactly infinite coordinate \\( +0 \\) is returned, even if there are NaNs.

  * returns \\( +\infty \\), when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float rnorm4df(float a, float b, float c, float d)



Calculate one over the square root of the sum of squares of four coordinates.

Calculates one over the length of four dimension vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns one over the length of the 3D vector \\( \frac{1}{\sqrt{a^2+b^2+c^2+d^2}} \\).

  * In the presence of an exactly infinite coordinate \\( +0 \\) is returned, even if there are NaNs.

  * returns \\( +\infty \\), when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float rnormf(int dim, float const *p)



Calculate the reciprocal of square root of the sum of squares of any number of coordinates.

Calculates one over the length of vector `p`, dimension of which is passed as an argument, in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns one over the length of the vector \\( \frac{1}{\sqrt{\sum_{i=0}^{dim-1} p_i^2}} \\).

  * In the presence of an exactly infinite coordinate \\( +0 \\) is returned, even if there are NaNs.

  * returns \\( +\infty \\), when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float roundf(float x)



Round to nearest integer value in floating-point.

Round `x` to the nearest integer value in floating-point format, with halfway cases rounded away from zero.

Note

This function may be slower than alternate rounding methods. See rintf().

Returns


Returns rounded integer value.

  * roundf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * roundf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * roundf(NaN) returns NaN.


__device__ float rsqrtf(float x)



Calculate the reciprocal of the square root of the input argument.

Calculate the reciprocal of the nonnegative square root of `x`, \\( 1/\sqrt{x} \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns \\( 1/\sqrt{x} \\).

  * rsqrtf( \\( +\infty \\) ) returns +0.

  * rsqrtf( \\( \pm 0 \\) ) returns \\( \pm \infty \\).

  * rsqrtf(`x`) returns NaN if `x` is less than 0.

  * rsqrtf(NaN) returns NaN.


__device__ float scalblnf(float x, long int n)



Scale floating-point input by integer power of two.

Scale `x` by \\( 2^n \\) by efficient manipulation of the floating-point exponent.

Returns


Returns `x` * \\( 2^n \\).

  * scalblnf( \\( \pm 0 \\) , `n`) returns \\( \pm 0 \\).

  * scalblnf(`x`, 0) returns `x`.

  * scalblnf( \\( \pm \infty \\) , `n`) returns \\( \pm \infty \\).

  * scalblnf(NaN, `n`) returns NaN.


__device__ float scalbnf(float x, int n)



Scale floating-point input by integer power of two.

Scale `x` by \\( 2^n \\) by efficient manipulation of the floating-point exponent.

Returns


Returns `x` * \\( 2^n \\).

  * scalbnf( \\( \pm 0 \\) , `n`) returns \\( \pm 0 \\).

  * scalbnf(`x`, 0) returns `x`.

  * scalbnf( \\( \pm \infty \\) , `n`) returns \\( \pm \infty \\).

  * scalbnf(NaN, `n`) returns NaN.


__device__ __RETURN_TYPE signbit(float a)



Return the sign bit of the input.

Determine whether the floating-point value `a` is negative.

Returns


Reports the sign bit of all values including infinities, zeros, and NaNs.

  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is negative.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is negative.


__device__ void sincosf(float x, float *sptr, float *cptr)



Calculate the sine and cosine of the first input argument.

Calculate the sine and cosine of the first input argument `x` (measured in radians). The results for sine and cosine are written into the second argument, `sptr`, and, respectively, third argument, `cptr`.

See also

sinf() and cosf().

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

__device__ void sincospif(float x, float *sptr, float *cptr)



Calculate the sine and cosine of the first input argument \\( \times \pi \\).

Calculate the sine and cosine of the first input argument, `x` (measured in radians), \\( \times \pi \\). The results for sine and cosine are written into the second argument, `sptr`, and, respectively, third argument, `cptr`.

See also

sinpif() and cospif().

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

__device__ float sinf(float x)



Calculate the sine of the input argument.

Calculate the sine of the input argument `x` (measured in radians).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


  * sinf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sinf( \\( \pm \infty \\) ) returns NaN.

  * sinf(NaN) returns NaN.


__device__ float sinhf(float x)



Calculate the hyperbolic sine of the input argument.

Calculate the hyperbolic sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * sinhf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sinhf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * sinhf(NaN) returns NaN.


__device__ float sinpif(float x)



Calculate the sine of the input argument \\( \times \pi \\).

Calculate the sine of `x` \\( \times \pi \\) (measured in radians), where `x` is the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * sinpif( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sinpif( \\( \pm \infty \\) ) returns NaN.

  * sinpif(NaN) returns NaN.


__device__ float sqrtf(float x)



Calculate the square root of the input argument.

Calculate the nonnegative square root of `x`, \\( \sqrt{x} \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns \\( \sqrt{x} \\).

  * sqrtf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sqrtf( \\( +\infty \\) ) returns \\( +\infty \\).

  * sqrtf(`x`) returns NaN if `x` is less than 0.

  * sqrtf(NaN) returns NaN.


__device__ float tanf(float x)



Calculate the tangent of the input argument.

Calculate the tangent of the input argument `x` (measured in radians).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected.

Returns


  * tanf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * tanf( \\( \pm \infty \\) ) returns NaN.

  * tanf(NaN) returns NaN.


__device__ float tanhf(float x)



Calculate the hyperbolic tangent of the input argument.

Calculate the hyperbolic tangent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * tanhf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * tanhf( \\( \pm \infty \\) ) returns \\( \pm 1 \\).

  * tanhf(NaN) returns NaN.


__device__ float tgammaf(float x)



Calculate the gamma function of the input argument.

Calculate the gamma function of the input argument `x`, namely the value of \\( \Gamma(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


  * tgammaf( \\( \pm 0 \\) ) returns \\( \pm \infty \\).

  * tgammaf(`x`) returns NaN if `x` < 0 and `x` is an integer.

  * tgammaf( \\( -\infty \\) ) returns NaN.

  * tgammaf( \\( +\infty \\) ) returns \\( +\infty \\).

  * tgammaf(NaN) returns NaN.


__device__ float truncf(float x)



Truncate input argument to the integral part.

Round `x` to the nearest integer value that does not exceed `x` in magnitude.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns truncated integer value.

  * truncf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * truncf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * truncf(NaN) returns NaN.


__device__ float y0f(float x)



Calculate the value of the Bessel function of the second kind of order 0 for the input argument.

Calculate the value of the Bessel function of the second kind of order 0 for the input argument `x`, \\( Y_0(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the second kind of order 0.

  * y0f( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * y0f(`x`) returns NaN for `x` < 0.

  * y0f( \\( +\infty \\) ) returns +0.

  * y0f(NaN) returns NaN.


__device__ float y1f(float x)



Calculate the value of the Bessel function of the second kind of order 1 for the input argument.

Calculate the value of the Bessel function of the second kind of order 1 for the input argument `x`, \\( Y_1(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the second kind of order 1.

  * y1f( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * y1f(`x`) returns NaN for `x` < 0.

  * y1f( \\( +\infty \\) ) returns +0.

  * y1f(NaN) returns NaN.


__device__ float ynf(int n, float x)



Calculate the value of the Bessel function of the second kind of order n for the input argument.

Calculate the value of the Bessel function of the second kind of order `n` for the input argument `x`, \\( Y_n(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the second kind of order `n`.

  * ynf(`n`, `x`) returns NaN for `n` < 0.

  * ynf(`n`, \\( \pm 0 \\) ) returns \\( -\infty \\).

  * ynf(`n`, `x`) returns NaN for `x` < 0.

  * ynf(`n`, \\( +\infty \\) ) returns +0.

  * ynf(`n`, NaN) returns NaN.