# 8. Double Precision Mathematical Functions

**Source:** group__CUDA__MATH__DOUBLE.html


#  8\. Double Precision Mathematical Functions

This section describes double precision mathematical functions.

To use these functions, you do not need to include any additional header file in your program.

Functions

__device__ double acos(double x)


Calculate the arc cosine of the input argument.

__device__ double acosh(double x)


Calculate the nonnegative inverse hyperbolic cosine of the input argument.

__device__ double asin(double x)


Calculate the arc sine of the input argument.

__device__ double asinh(double x)


Calculate the inverse hyperbolic sine of the input argument.

__device__ double atan(double x)


Calculate the arc tangent of the input argument.

__device__ double atan2(double y, double x)


Calculate the arc tangent of the ratio of first and second input arguments.

__device__ double atanh(double x)


Calculate the inverse hyperbolic tangent of the input argument.

__device__ double cbrt(double x)


Calculate the cube root of the input argument.

__device__ double ceil(double x)


Calculate ceiling of the input argument.

__device__ double copysign(double x, double y)


Create value with given magnitude, copying sign of second value.

__device__ double cos(double x)


Calculate the cosine of the input argument.

__device__ double cosh(double x)


Calculate the hyperbolic cosine of the input argument.

__device__ double cospi(double x)


Calculate the cosine of the input argument \\(\times \pi\\) .

__device__ double cyl_bessel_i0(double x)


Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.

__device__ double cyl_bessel_i1(double x)


Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.

__device__ double erf(double x)


Calculate the error function of the input argument.

__device__ double erfc(double x)


Calculate the complementary error function of the input argument.

__device__ double erfcinv(double x)


Calculate the inverse complementary error function of the input argument.

__device__ double erfcx(double x)


Calculate the scaled complementary error function of the input argument.

__device__ double erfinv(double x)


Calculate the inverse error function of the input argument.

__device__ double exp(double x)


Calculate the base \\(e\\) exponential of the input argument.

__device__ double exp10(double x)


Calculate the base 10 exponential of the input argument.

__device__ double exp2(double x)


Calculate the base 2 exponential of the input argument.

__device__ double expm1(double x)


Calculate the base \\(e\\) exponential of the input argument, minus 1.

__device__ double fabs(double x)


Calculate the absolute value of the input argument.

__device__ double fdim(double x, double y)


Compute the positive difference between `x` and `y` .

__device__ double floor(double x)


Calculate the largest integer less than or equal to `x` .

__device__ double fma(double x, double y, double z)


Compute \\(x \times y + z\\) as a single operation.

__device__ double fmax(double, double)


Determine the maximum numeric value of the arguments.

__device__ double fmin(double x, double y)


Determine the minimum numeric value of the arguments.

__device__ double fmod(double x, double y)


Calculate the double-precision floating-point remainder of `x` / `y` .

__device__ double frexp(double x, int *nptr)


Extract mantissa and exponent of a floating-point value.

__device__ double hypot(double x, double y)


Calculate the square root of the sum of squares of two arguments.

__device__ int ilogb(double x)


Compute the unbiased integer exponent of the argument.

__device__ __RETURN_TYPE isfinite(double a)


Determine whether argument is finite.

__device__ __RETURN_TYPE isinf(double a)


Determine whether argument is infinite.

__device__ __RETURN_TYPE isnan(double a)


Determine whether argument is a NaN.

__device__ double j0(double x)


Calculate the value of the Bessel function of the first kind of order 0 for the input argument.

__device__ double j1(double x)


Calculate the value of the Bessel function of the first kind of order 1 for the input argument.

__device__ double jn(int n, double x)


Calculate the value of the Bessel function of the first kind of order n for the input argument.

__device__ double ldexp(double x, int exp)


Calculate the value of \\(x\cdot 2^{exp}\\) .

__device__ double lgamma(double x)


Calculate the natural logarithm of the absolute value of the gamma function of the input argument.

__device__ long long int llrint(double x)


Round input to nearest integer value.

__device__ long long int llround(double x)


Round to nearest integer value.

__device__ double log(double x)


Calculate the base \\(e\\) logarithm of the input argument.

__device__ double log10(double x)


Calculate the base 10 logarithm of the input argument.

__device__ double log1p(double x)


Calculate the value of \\(\log_{e}(1+x)\\) .

__device__ double log2(double x)


Calculate the base 2 logarithm of the input argument.

__device__ double logb(double x)


Calculate the floating-point representation of the exponent of the input argument.

__device__ long int lrint(double x)


Round input to nearest integer value.

__device__ long int lround(double x)


Round to nearest integer value.

__device__ double max(const float a, const double b)


Calculate the maximum value of the input `float` and `double` arguments.

__device__ double max(const double a, const float b)


Calculate the maximum value of the input `double` and `float` arguments.

__device__ double max(const double a, const double b)


Calculate the maximum value of the input `float` arguments.

__device__ double min(const float a, const double b)


Calculate the minimum value of the input `float` and `double` arguments.

__device__ double min(const double a, const double b)


Calculate the minimum value of the input `float` arguments.

__device__ double min(const double a, const float b)


Calculate the minimum value of the input `double` and `float` arguments.

__device__ double modf(double x, double *iptr)


Break down the input argument into fractional and integral parts.

__device__ double nan(const char *tagp)


Returns "Not a Number" value.

__device__ double nearbyint(double x)


Round the input argument to the nearest integer.

__device__ double nextafter(double x, double y)


Return next representable double-precision floating-point value after argument `x` in the direction of `y` .

__device__ double norm(int dim, double const *p)


Calculate the square root of the sum of squares of any number of coordinates.

__device__ double norm3d(double a, double b, double c)


Calculate the square root of the sum of squares of three coordinates of the argument.

__device__ double norm4d(double a, double b, double c, double d)


Calculate the square root of the sum of squares of four coordinates of the argument.

__device__ double normcdf(double x)


Calculate the standard normal cumulative distribution function.

__device__ double normcdfinv(double x)


Calculate the inverse of the standard normal cumulative distribution function.

__device__ double pow(double x, double y)


Calculate the value of first argument to the power of second argument.

__device__ double rcbrt(double x)


Calculate reciprocal cube root function.

__device__ double remainder(double x, double y)


Compute double-precision floating-point remainder.

__device__ double remquo(double x, double y, int *quo)


Compute double-precision floating-point remainder and part of quotient.

__device__ double rhypot(double x, double y)


Calculate one over the square root of the sum of squares of two arguments.

__device__ double rint(double x)


Round to nearest integer value in floating-point.

__device__ double rnorm(int dim, double const *p)


Calculate the reciprocal of square root of the sum of squares of any number of coordinates.

__device__ double rnorm3d(double a, double b, double c)


Calculate one over the square root of the sum of squares of three coordinates.

__device__ double rnorm4d(double a, double b, double c, double d)


Calculate one over the square root of the sum of squares of four coordinates.

__device__ double round(double x)


Round to nearest integer value in floating-point.

__device__ double rsqrt(double x)


Calculate the reciprocal of the square root of the input argument.

__device__ double scalbln(double x, long int n)


Scale floating-point input by integer power of two.

__device__ double scalbn(double x, int n)


Scale floating-point input by integer power of two.

__device__ __RETURN_TYPE signbit(double a)


Return the sign bit of the input.

__device__ double sin(double x)


Calculate the sine of the input argument.

__device__ void sincos(double x, double *sptr, double *cptr)


Calculate the sine and cosine of the first input argument.

__device__ void sincospi(double x, double *sptr, double *cptr)


Calculate the sine and cosine of the first input argument \\(\times \pi\\) .

__device__ double sinh(double x)


Calculate the hyperbolic sine of the input argument.

__device__ double sinpi(double x)


Calculate the sine of the input argument \\(\times \pi\\) .

__device__ double sqrt(double x)


Calculate the square root of the input argument.

__device__ double tan(double x)


Calculate the tangent of the input argument.

__device__ double tanh(double x)


Calculate the hyperbolic tangent of the input argument.

__device__ double tgamma(double x)


Calculate the gamma function of the input argument.

__device__ double trunc(double x)


Truncate input argument to the integral part.

__device__ double y0(double x)


Calculate the value of the Bessel function of the second kind of order 0 for the input argument.

__device__ double y1(double x)


Calculate the value of the Bessel function of the second kind of order 1 for the input argument.

__device__ double yn(int n, double x)


Calculate the value of the Bessel function of the second kind of order n for the input argument.

##  8.1. Functions

__device__ double acos(double x)



Calculate the arc cosine of the input argument.

Calculate the principal value of the arc cosine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Result will be in radians, in the interval [0, \\( \pi \\) ] for `x` inside [-1, +1].

  * acos(1) returns +0.

  * acos(`x`) returns NaN for `x` outside [-1, +1].

  * acos(NaN) returns NaN.


__device__ double acosh(double x)



Calculate the nonnegative inverse hyperbolic cosine of the input argument.

Calculate the nonnegative inverse hyperbolic cosine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Result will be in the interval [0, \\( +\infty \\) ].

  * acosh(1) returns 0.

  * acosh(`x`) returns NaN for `x` in the interval  \\( -\infty \\) , 1).

  * acosh( \\( +\infty \\) ) returns \\( +\infty \\).

  * acosh(NaN) returns NaN.


__device__ double asin(double x)[



Calculate the arc sine of the input argument.

Calculate the principal value of the arc sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Result will be in radians, in the interval [- \\( \pi \\) /2, + \\( \pi \\) /2] for `x` inside [-1, +1].

  * asin( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * asin(`x`) returns NaN for `x` outside [-1, +1].

  * asin(NaN) returns NaN.


__device__ double asinh(double x)



Calculate the inverse hyperbolic sine of the input argument.

Calculate the inverse hyperbolic sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * asinh( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * asinh( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * asinh(NaN) returns NaN.


__device__ double atan(double x)



Calculate the arc tangent of the input argument.

Calculate the principal value of the arc tangent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Result will be in radians, in the interval [- \\( \pi \\) /2, + \\( \pi \\) /2].

  * atan( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * atan( \\( \pm \infty \\) ) returns \\( \pm \pi \\) /2.

  * atan(NaN) returns NaN.


__device__ double atan2(double y, double x)



Calculate the arc tangent of the ratio of first and second input arguments.

Calculate the principal value of the arc tangent of the ratio of first and second input arguments `y` / `x`. The quadrant of the result is determined by the signs of inputs `y` and `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Result will be in radians, in the interval [- \\( \pi \\) , + \\( \pi \\) ].

  * atan2( \\( \pm 0 \\) , -0) returns \\( \pm \pi \\).

  * atan2( \\( \pm 0 \\) , +0) returns \\( \pm 0 \\).

  * atan2( \\( \pm 0 \\) , `x`) returns \\( \pm \pi \\) for `x` < 0.

  * atan2( \\( \pm 0 \\) , `x`) returns \\( \pm 0 \\) for `x` > 0.

  * atan2(`y`, \\( \pm 0 \\) ) returns \\( -\pi \\) /2 for `y` < 0.

  * atan2(`y`, \\( \pm 0 \\) ) returns \\( \pi \\) /2 for `y` > 0.

  * atan2( \\( \pm y \\) , \\( -\infty \\) ) returns \\( \pm \pi \\) for finite `y` > 0.

  * atan2( \\( \pm y \\) , \\( +\infty \\) ) returns \\( \pm 0 \\) for finite `y` > 0.

  * atan2( \\( \pm \infty \\) , `x`) returns \\( \pm \pi \\) /2 for finite `x`.

  * atan2( \\( \pm \infty \\) , \\( -\infty \\) ) returns \\( \pm 3\pi \\) /4.

  * atan2( \\( \pm \infty \\) , \\( +\infty \\) ) returns \\( \pm \pi \\) /4.

  * If either argument is NaN, NaN is returned.


__device__ double atanh(double x)



Calculate the inverse hyperbolic tangent of the input argument.

Calculate the inverse hyperbolic tangent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * atanh( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * atanh( \\( \pm 1 \\) ) returns \\( \pm \infty \\).

  * atanh(`x`) returns NaN for `x` outside interval [-1, 1].

  * atanh(NaN) returns NaN.


__device__ double cbrt(double x)



Calculate the cube root of the input argument.

Calculate the cube root of `x`, \\( x^{1/3} \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns \\( x^{1/3} \\).

  * cbrt( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * cbrt( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * cbrt(NaN) returns NaN.


__device__ double ceil(double x)



Calculate ceiling of the input argument.

Compute the smallest integer value not less than `x`.

Returns


Returns \\( \lceil x \rceil \\) expressed as a floating-point number.

  * ceil( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * ceil( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * ceil(NaN) returns NaN.


__device__ double copysign(double x, double y)



Create value with given magnitude, copying sign of second value.

Create a floating-point value with the magnitude `x` and the sign of `y`.

Returns


  * a value with the magnitude of `x` and the sign of `y`.

  * copysign(`NaN`, `y`) returns a `NaN` with the sign of `y`.


__device__ double cos(double x)



Calculate the cosine of the input argument.

Calculate the cosine of the input argument `x` (measured in radians).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * cos( \\( \pm 0 \\) ) returns 1.

  * cos( \\( \pm \infty \\) ) returns NaN.

  * cos(NaN) returns NaN.


__device__ double cosh(double x)



Calculate the hyperbolic cosine of the input argument.

Calculate the hyperbolic cosine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * cosh( \\( \pm 0 \\) ) returns 1.

  * cosh( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * cosh(NaN) returns NaN.


__device__ double cospi(double x)



Calculate the cosine of the input argument \\( \times \pi \\).

Calculate the cosine of `x` \\( \times \pi \\) (measured in radians), where `x` is the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * cospi( \\( \pm 0 \\) ) returns 1.

  * cospi( \\( \pm \infty \\) ) returns NaN.

  * cospi(NaN) returns NaN.


__device__ double cyl_bessel_i0(double x)



Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.

Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument `x`, \\( I_0(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the value of the regular modified cylindrical Bessel function of order 0.

  * cyl_bessel_i0( \\( \pm 0 \\)) returns +1.

  * cyl_bessel_i0( \\( \pm\infty \\)) returns \\( +\infty \\).

  * cyl_bessel_i0(NaN) returns NaN.


__device__ double cyl_bessel_i1(double x)



Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.

Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument `x`, \\( I_1(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the value of the regular modified cylindrical Bessel function of order 1.

  * cyl_bessel_i1( \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * cyl_bessel_i1( \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * cyl_bessel_i1(NaN) returns NaN.


__device__ double erf(double x)



Calculate the error function of the input argument.

Calculate the value of the error function for the input argument `x`, \\( \frac{2}{\sqrt \pi} \int_0^x e^{-t^2} dt \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * erf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * erf( \\( \pm \infty \\) ) returns \\( \pm 1 \\).

  * erf(NaN) returns NaN.


__device__ double erfc(double x)



Calculate the complementary error function of the input argument.

Calculate the complementary error function of the input argument `x`, 1 - erf(`x`).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * erfc( \\( -\infty \\) ) returns 2.

  * erfc( \\( +\infty \\) ) returns +0.

  * erfc(NaN) returns NaN.


__device__ double erfcinv(double x)



Calculate the inverse complementary error function of the input argument.

Calculate the inverse complementary error function \\( \operatorname{erfc}^{-1} \\) (`x`), of the input argument `x` in the interval [0, 2].

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * erfcinv( \\( \pm 0 \\) ) returns \\( +\infty \\).

  * erfcinv(2) returns \\( -\infty \\).

  * erfcinv(`x`) returns NaN for `x` outside [0, 2].

  * erfcinv(NaN) returns NaN.


__device__ double erfcx(double x)



Calculate the scaled complementary error function of the input argument.

Calculate the scaled complementary error function of the input argument `x`, \\( e^{x^2}\cdot \operatorname{erfc}(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * erfcx( \\( -\infty \\) ) returns \\( +\infty \\).

  * erfcx( \\( +\infty \\) ) returns +0.

  * erfcx(NaN) returns NaN.


__device__ double erfinv(double x)



Calculate the inverse error function of the input argument.

Calculate the inverse error function \\( \operatorname{erf}^{-1} \\) (`x`), of the input argument `x` in the interval [-1, 1].

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * erfinv( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * erfinv(1) returns \\( +\infty \\).

  * erfinv(-1) returns \\( -\infty \\).

  * erfinv(`x`) returns NaN for `x` outside [-1, +1].

  * erfinv(NaN) returns NaN.


__device__ double exp(double x)



Calculate the base \\( e \\) exponential of the input argument.

Calculate \\( e^x \\) , the base \\( e \\) exponential of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * exp( \\( \pm 0 \\) ) returns 1.

  * exp( \\( -\infty \\) ) returns +0.

  * exp( \\( +\infty \\) ) returns \\( +\infty \\).

  * exp(NaN) returns NaN.


__device__ double exp10(double x)



Calculate the base 10 exponential of the input argument.

Calculate \\( 10^x \\) , the base 10 exponential of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * exp10( \\( \pm 0 \\) ) returns 1.

  * exp10( \\( -\infty \\) ) returns +0.

  * exp10( \\( +\infty \\) ) returns \\( +\infty \\).

  * exp10(NaN) returns NaN.


__device__ double exp2(double x)



Calculate the base 2 exponential of the input argument.

Calculate \\( 2^x \\) , the base 2 exponential of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * exp2( \\( \pm 0 \\) ) returns 1.

  * exp2( \\( -\infty \\) ) returns +0.

  * exp2( \\( +\infty \\) ) returns \\( +\infty \\).

  * exp2(NaN) returns NaN.


__device__ double expm1(double x)



Calculate the base \\( e \\) exponential of the input argument, minus 1.

Calculate \\( e^x \\) -1, the base \\( e \\) exponential of the input argument `x`, minus 1.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * expm1( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * expm1( \\( -\infty \\) ) returns -1.

  * expm1( \\( +\infty \\) ) returns \\( +\infty \\).

  * expm1(NaN) returns NaN.


__device__ double fabs(double x)



Calculate the absolute value of the input argument.

Calculate the absolute value of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the absolute value of the input argument.

  * fabs( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * fabs( \\( \pm 0 \\) ) returns +0.

  * fabs(NaN) returns an unspecified NaN.


__device__ double fdim(double x, double y)



Compute the positive difference between `x` and `y`.

Compute the positive difference between `x` and `y`. The positive difference is `x` \- `y` when `x` > `y` and +0 otherwise.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the positive difference between `x` and `y`.

  * fdim(`x`, `y`) returns `x` \- `y` if `x` > `y`.

  * fdim(`x`, `y`) returns +0 if `x` \\( \leq \\) `y`.

  * If either argument is NaN, NaN is returned.


__device__ double floor(double x)



Calculate the largest integer less than or equal to `x`.

Calculates the largest integer value which is less than or equal to `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns \\( \lfloor x \rfloor \\) expressed as a floating-point number.

  * floor( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * floor( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * floor(NaN) returns NaN.


__device__ double fma(double x, double y, double z)



Compute \\( x \times y + z \\) as a single operation.

Compute the value of \\( x \times y + z \\) as a single ternary operation. After computing the value to infinite precision, the value is rounded once using round-to-nearest, ties-to-even rounding mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * fma( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * fma( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * fma(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * fma(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * fma(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * fma(`x`, `y`, \\( \mp 0 \\)) returns \\( +0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * fma(`x`, `y`, `z`) returns \\( +0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double fmax(double, double)



Determine the maximum numeric value of the arguments.

Determines the maximum numeric value of the arguments `x` and `y`. Treats NaN arguments as missing data. If one argument is a NaN and the other is legitimate numeric value, the numeric value is chosen.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the maximum numeric values of the arguments `x` and `y`.

  * If both arguments are NaN, returns NaN.

  * If one argument is NaN, returns the numeric argument.


__device__ double fmin(double x, double y)



Determine the minimum numeric value of the arguments.

Determines the minimum numeric value of the arguments `x` and `y`. Treats NaN arguments as missing data. If one argument is a NaN and the other is legitimate numeric value, the numeric value is chosen.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the minimum numeric value of the arguments `x` and `y`.

  * If both arguments are NaN, returns NaN.

  * If one argument is NaN, returns the numeric argument.


__device__ double fmod(double x, double y)



Calculate the double-precision floating-point remainder of `x` / `y`.

Calculate the double-precision floating-point remainder of `x` / `y`. The floating-point remainder of the division operation `x` / `y` calculated by this function is exactly the value `x - n*y`, where `n` is `x` / `y` with its fractional part truncated. The computed value will have the same sign as `x`, and its magnitude will be less than the magnitude of `y`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * Returns the floating-point remainder of `x` / `y`.

  * fmod( \\( \pm 0 \\) , `y`) returns \\( \pm 0 \\) if `y` is not zero.

  * fmod(`x`, \\( \pm \infty \\) ) returns `x` if `x` is finite.

  * fmod(`x`, `y`) returns NaN if `x` is \\( \pm\infty \\) or `y` is zero.

  * If either argument is NaN, NaN is returned.


__device__ double frexp(double x, int *nptr)



Extract mantissa and exponent of a floating-point value.

Decompose the floating-point value `x` into a component `m` for the normalized fraction element and another term `n` for the exponent. The absolute value of `m` will be greater than or equal to 0.5 and less than 1.0 or it will be equal to 0; \\( x = m\cdot 2^n \\). The integer exponent `n` will be stored in the location to which `nptr` points.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the fractional component `m`.

  * frexp( \\( \pm 0 \\) , `nptr`) returns \\( \pm 0 \\) and stores zero in the location pointed to by `nptr`.

  * frexp( \\( \pm \infty \\) , `nptr`) returns \\( \pm \infty \\) and stores an unspecified value in the location to which `nptr` points.

  * frexp(NaN, `y`) returns a NaN and stores an unspecified value in the location to which `nptr` points.


__device__ double hypot(double x, double y)



Calculate the square root of the sum of squares of two arguments.

Calculate the length of the hypotenuse of a right triangle whose two sides have lengths `x` and `y` without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the length of the hypotenuse \\( \sqrt{x^2+y^2} \\).

  * hypot(`x`,`y`), hypot(`y`,`x`), and hypot(`x`, `-y`) are equivalent.

  * hypot(`x`, \\( \pm 0 \\) ) is equivalent to fabs(`x`).

  * hypot( \\( \pm \infty \\) ,`y`) returns \\( +\infty \\) , even if `y` is a NaN.

  * hypot(NaN, `y`) returns NaN, when `y` is not \\( \pm\infty \\).


__device__ int ilogb(double x)



Compute the unbiased integer exponent of the argument.

Calculates the unbiased integer exponent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * If successful, returns the unbiased exponent of the argument.

  * ilogb( \\( \pm 0 \\) ) returns `INT_MIN`.

  * ilogb(NaN) returns `INT_MIN`.

  * ilogb( \\( \pm \infty \\) ) returns `INT_MAX`.

  * Note: above behavior does not take into account `FP_ILOGB0` nor `FP_ILOGBNAN`.


__device__ __RETURN_TYPE isfinite(double a)



Determine whether argument is finite.

Determine whether the floating-point value `a` is a finite value (zero, subnormal, or normal and not infinity or NaN).

Returns


  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is a finite value.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is a finite value.


__device__ __RETURN_TYPE isinf(double a)



Determine whether argument is infinite.

Determine whether the floating-point value `a` is an infinite value (positive or negative).

Returns


  * With Visual Studio 2013 host compiler: Returns true if and only if `a` is an infinite value.

  * With other host compilers: Returns a nonzero value if and only if `a` is an infinite value.


__device__ __RETURN_TYPE isnan(double a)



Determine whether argument is a NaN.

Determine whether the floating-point value `a` is a NaN.

Returns


  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is a NaN value.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is a NaN value.


__device__ double j0(double x)



Calculate the value of the Bessel function of the first kind of order 0 for the input argument.

Calculate the value of the Bessel function of the first kind of order 0 for the input argument `x`, \\( J_0(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the first kind of order 0.

  * j0( \\( \pm \infty \\) ) returns +0.

  * j0(NaN) returns NaN.


__device__ double j1(double x)



Calculate the value of the Bessel function of the first kind of order 1 for the input argument.

Calculate the value of the Bessel function of the first kind of order 1 for the input argument `x`, \\( J_1(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the first kind of order 1.

  * j1( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * j1( \\( \pm \infty \\) ) returns \\( \pm 0 \\).

  * j1(NaN) returns NaN.


__device__ double jn(int n, double x)



Calculate the value of the Bessel function of the first kind of order n for the input argument.

Calculate the value of the Bessel function of the first kind of order `n` for the input argument `x`, \\( J_n(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the first kind of order `n`.

  * jn(`n`, NaN) returns NaN.

  * jn(`n`, `x`) returns NaN for `n` < 0.

  * jn(`n`, \\( +\infty \\) ) returns +0.


__device__ double ldexp(double x, int exp)



Calculate the value of \\( x\cdot 2^{exp} \\).

Calculate the value of \\( x\cdot 2^{exp} \\) of the input arguments `x` and `exp`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * ldexp(`x`, `exp`) is equivalent to scalbn(`x`, `exp`).


__device__ double lgamma(double x)



Calculate the natural logarithm of the absolute value of the gamma function of the input argument.

Calculate the natural logarithm of the absolute value of the gamma function of the input argument `x`, namely the value of \\( \log_{e}\left|\Gamma(x)\right| \\)

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * lgamma(1) returns +0.

  * lgamma(2) returns +0.

  * lgamma(`x`) returns \\( +\infty \\) if `x` \\( \leq \\) 0 and `x` is an integer.

  * lgamma( \\( -\infty \\) ) returns \\( +\infty \\).

  * lgamma( \\( +\infty \\) ) returns \\( +\infty \\).

  * lgamma(NaN) returns NaN.


__device__ long long int llrint(double x)



Round input to nearest integer value.

Round `x` to the nearest integer value, with halfway cases rounded to the nearest even integer value. If the result is outside the range of the return type, the behavior is undefined.

Returns


Returns rounded integer value.

__device__ long long int llround(double x)



Round to nearest integer value.

Round `x` to the nearest integer value, with halfway cases rounded away from zero. If the result is outside the range of the return type, the behavior is undefined.

Note

This function may be slower than alternate rounding methods. See llrint().

Returns


Returns rounded integer value.

__device__ double log(double x)



Calculate the base \\( e \\) logarithm of the input argument.

Calculate the base \\( e \\) logarithm of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * log( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * log(1) returns +0.

  * log(`x`) returns NaN for `x` < 0.

  * log( \\( +\infty \\) ) returns \\( +\infty \\).

  * log(NaN) returns NaN.


__device__ double log10(double x)



Calculate the base 10 logarithm of the input argument.

Calculate the base 10 logarithm of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * log10( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * log10(1) returns +0.

  * log10(`x`) returns NaN for `x` < 0.

  * log10( \\( +\infty \\) ) returns \\( +\infty \\).

  * log10(NaN) returns NaN.


__device__ double log1p(double x)



Calculate the value of \\( \log_{e}(1+x) \\).

Calculate the value of \\( \log_{e}(1+x) \\) of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * log1p( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * log1p(-1) returns \\( -\infty \\).

  * log1p(`x`) returns NaN for `x` < -1.

  * log1p( \\( +\infty \\) ) returns \\( +\infty \\).

  * log1p(NaN) returns NaN.


__device__ double log2(double x)



Calculate the base 2 logarithm of the input argument.

Calculate the base 2 logarithm of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * log2( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * log2(1) returns +0.

  * log2(`x`) returns NaN for `x` < 0.

  * log2( \\( +\infty \\) ) returns \\( +\infty \\).

  * log2(NaN) returns NaN.


__device__ double logb(double x)



Calculate the floating-point representation of the exponent of the input argument.

Calculate the floating-point representation of the exponent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * logb( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * logb( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * logb(NaN) returns NaN.


__device__ long int lrint(double x)



Round input to nearest integer value.

Round `x` to the nearest integer value, with halfway cases rounded to the nearest even integer value. If the result is outside the range of the return type, the behavior is undefined.

Returns


Returns rounded integer value.

__device__ long int lround(double x)



Round to nearest integer value.

Round `x` to the nearest integer value, with halfway cases rounded away from zero. If the result is outside the range of the return type, the behavior is undefined.

Note

This function may be slower than alternate rounding methods. See lrint().

Returns


Returns rounded integer value.

__device__ double max(const float a, const double b)



Calculate the maximum value of the input `float` and `double` arguments.

Convert `float` argument `a` to `double`, followed by fmax().

Note, this is different from `std:`: specification

__device__ double max(const double a, const float b)



Calculate the maximum value of the input `double` and `float` arguments.

Convert `float` argument `b` to `double`, followed by fmax().

Note, this is different from `std:`: specification

__device__ double max(const double a, const double b)



Calculate the maximum value of the input `float` arguments.

Calculate the maximum value of the arguments `a` and `b`. Behavior is equivalent to fmax() function.

Note, this is different from `std:`: specification

__device__ double min(const float a, const double b)



Calculate the minimum value of the input `float` and `double` arguments.

Convert `float` argument `a` to `double`, followed by fmin().

Note, this is different from `std:`: specification

__device__ double min(const double a, const double b)



Calculate the minimum value of the input `float` arguments.

Calculate the minimum value of the arguments `a` and `b`. Behavior is equivalent to fmin() function.

Note, this is different from `std:`: specification

__device__ double min(const double a, const float b)



Calculate the minimum value of the input `double` and `float` arguments.

Convert `float` argument `b` to `double`, followed by fmin().

Note, this is different from `std:`: specification

__device__ double modf(double x, double *iptr)



Break down the input argument into fractional and integral parts.

Break down the argument `x` into fractional and integral parts. The integral part is stored in the argument `iptr`. Fractional and integral parts are given the same sign as the argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * modf( \\( \pm x \\) , `iptr`) returns a result with the same sign as `x`.

  * modf( \\( \pm \infty \\) , `iptr`) returns \\( \pm 0 \\) and stores \\( \pm \infty \\) in the object pointed to by `iptr`.

  * modf(NaN, `iptr`) stores a NaN in the object pointed to by `iptr` and returns a NaN.


__device__ double nan(const char *tagp)



Returns “Not a Number” value.

Return a representation of a quiet NaN. Argument `tagp` selects one of the possible representations.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * nan(`tagp`) returns NaN.


__device__ double nearbyint(double x)



Round the input argument to the nearest integer.

Round argument `x` to an integer value in double precision floating-point format. Uses round to nearest rounding, with ties rounding to even.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * nearbyint( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * nearbyint( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * nearbyint(NaN) returns NaN.


__device__ double nextafter(double x, double y)



Return next representable double-precision floating-point value after argument `x` in the direction of `y`.

Calculate the next representable double-precision floating-point value following `x` in the direction of `y`. For example, if `y` is greater than `x`, nextafter() returns the smallest representable number greater than `x`

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * nextafter(`x`, `y`) = `y` if `x` equals `y`.

  * nextafter(`x`, `y`) = `NaN` if either `x` or `y` are `NaN`.


__device__ double norm(int dim, double const *p)



Calculate the square root of the sum of squares of any number of coordinates.

Calculate the length of a vector p, dimension of which is passed as an argument `without` undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the length of the dim-D vector \\( \sqrt{\sum_{i=0}^{dim-1} p_i^2} \\).

  * In the presence of an exactly infinite coordinate \\( +\infty \\) is returned, even if there are NaNs.

  * returns +0, when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ double norm3d(double a, double b, double c)



Calculate the square root of the sum of squares of three coordinates of the argument.

Calculate the length of three dimensional vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the length of 3D vector \\( \sqrt{a^2+b^2+c^2} \\).

  * In the presence of an exactly infinite coordinate \\( +\infty \\) is returned, even if there are NaNs.

  * returns +0, when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ double norm4d(double a, double b, double c, double d)



Calculate the square root of the sum of squares of four coordinates of the argument.

Calculate the length of four dimensional vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the length of 4D vector \\( \sqrt{a^2+b^2+c^2+d^2} \\).

  * In the presence of an exactly infinite coordinate \\( +\infty \\) is returned, even if there are NaNs.

  * returns +0, when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ double normcdf(double x)



Calculate the standard normal cumulative distribution function.

Calculate the cumulative distribution function of the standard normal distribution for input argument `x`, \\( \Phi(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * normcdf( \\( +\infty \\) ) returns 1.

  * normcdf( \\( -\infty \\) ) returns +0.

  * normcdf(NaN) returns NaN.


__device__ double normcdfinv(double x)



Calculate the inverse of the standard normal cumulative distribution function.

Calculate the inverse of the standard normal cumulative distribution function for input argument `x`, \\( \Phi^{-1}(x) \\). The function is defined for input values in the interval \\( (0, 1) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * normcdfinv( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * normcdfinv(1) returns \\( +\infty \\).

  * normcdfinv(`x`) returns NaN if `x` is not in the interval [0,1].

  * normcdfinv(NaN) returns NaN.


__device__ double pow(double x, double y)



Calculate the value of first argument to the power of second argument.

Calculate the value of `x` to the power of `y`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * pow( \\( \pm 0 \\) , `y`) returns \\( \pm \infty \\) for `y` an odd integer less than 0.

  * pow( \\( \pm 0 \\) , `y`) returns \\( +\infty \\) for `y` less than 0 and not an odd integer.

  * pow( \\( \pm 0 \\) , `y`) returns \\( \pm 0 \\) for `y` an odd integer greater than 0.

  * pow( \\( \pm 0 \\) , `y`) returns +0 for `y` > 0 and not an odd integer.

  * pow(-1, \\( \pm \infty \\) ) returns 1.

  * pow(+1, `y`) returns 1 for any `y`, even a NaN.

  * pow(`x`, \\( \pm 0 \\) ) returns 1 for any `x`, even a NaN.

  * pow(`x`, `y`) returns a NaN for finite `x` < 0 and finite non-integer `y`.

  * pow(`x`, \\( -\infty \\) ) returns \\( +\infty \\) for \\( | x | < 1 \\).

  * pow(`x`, \\( -\infty \\) ) returns +0 for \\( | x | > 1 \\).

  * pow(`x`, \\( +\infty \\) ) returns +0 for \\( | x | < 1 \\).

  * pow(`x`, \\( +\infty \\) ) returns \\( +\infty \\) for \\( | x | > 1 \\).

  * pow( \\( -\infty \\) , `y`) returns -0 for `y` an odd integer less than 0.

  * pow( \\( -\infty \\) , `y`) returns +0 for `y` < 0 and not an odd integer.

  * pow( \\( -\infty \\) , `y`) returns \\( -\infty \\) for `y` an odd integer greater than 0.

  * pow( \\( -\infty \\) , `y`) returns \\( +\infty \\) for `y` > 0 and not an odd integer.

  * pow( \\( +\infty \\) , `y`) returns +0 for `y` < 0.

  * pow( \\( +\infty \\) , `y`) returns \\( +\infty \\) for `y` > 0.

  * pow(`x`, `y`) returns NaN if either `x` or `y` or both are NaN and `x` \\( \neq \\) +1 and `y` \\( \neq\pm 0 \\).


__device__ double rcbrt(double x)



Calculate reciprocal cube root function.

Calculate reciprocal cube root function of `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * rcbrt( \\( \pm 0 \\) ) returns \\( \pm \infty \\).

  * rcbrt( \\( \pm \infty \\) ) returns \\( \pm 0 \\).

  * rcbrt(NaN) returns NaN.


__device__ double remainder(double x, double y)



Compute double-precision floating-point remainder.

Compute double-precision floating-point remainder `r` of dividing `x` by `y` for nonzero `y`. Thus \\( r = x - n y \\). The value `n` is the integer value nearest \\( \frac{x}{y} \\). In the case when \\( | n -\frac{x}{y} | = \frac{1}{2} \\) , the even `n` value is chosen.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * remainder(`x`, \\( \pm 0 \\) ) returns NaN.

  * remainder( \\( \pm \infty \\) , `y`) returns NaN.

  * remainder(`x`, \\( \pm \infty \\) ) returns `x` for finite `x`.

  * If either argument is NaN, NaN is returned.


__device__ double remquo(double x, double y, int *quo)



Compute double-precision floating-point remainder and part of quotient.

Compute a double-precision floating-point remainder in the same way as the remainder() function. Argument `quo` returns part of quotient upon division of `x` by `y`. Value `quo` has the same sign as \\( \frac{x}{y} \\) and may not be the exact quotient but agrees with the exact quotient in the low order 3 bits.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the remainder.

  * remquo(`x`, \\( \pm 0 \\) , `quo`) returns NaN and stores an unspecified value in the location to which `quo` points.

  * remquo( \\( \pm \infty \\) , `y`, `quo`) returns NaN and stores an unspecified value in the location to which `quo` points.

  * remquo(`x`, `y`, `quo`) returns NaN and stores an unspecified value in the location to which `quo` points if either of `x` or `y` is NaN.

  * remquo(`x`, \\( \pm \infty \\) , `quo`) returns `x` and stores zero in the location to which `quo` points for finite `x`.


__device__ double rhypot(double x, double y)



Calculate one over the square root of the sum of squares of two arguments.

Calculate one over the length of the hypotenuse of a right triangle whose two sides have lengths `x` and `y` without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns one over the length of the hypotenuse \\( \frac{1}{\sqrt{x^2+y^2}} \\).

  * rhypot(`x`,`y`), rhypot(`y`,`x`), and rhypot(`x`, `-y`) are equivalent.

  * rhypot( \\( \pm \infty \\) ,`y`) returns +0, even if `y` is a NaN.

  * rhypot( \\( \pm 0, \pm 0 \\)) returns \\( +\infty \\).

  * rhypot(NaN, `y`) returns NaN, when `y` is not \\( \pm\infty \\).


__device__ double rint(double x)



Round to nearest integer value in floating-point.

Round `x` to the nearest integer value in floating-point format, with halfway cases rounded to the nearest even integer value.

Returns


Returns rounded integer value.

  * rint( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * rint( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * rint(NaN) returns NaN.


__device__ double rnorm(int dim, double const *p)



Calculate the reciprocal of square root of the sum of squares of any number of coordinates.

Calculates one over the length of vector `p`, dimension of which is passed as an argument, in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns one over the length of the vector \\( \frac{1}{\sqrt{\sum_{i=0}^{dim-1} p_i^2}} \\).

  * In the presence of an exactly infinite coordinate \\( +0 \\) is returned, even if there are NaNs.

  * returns \\( +\infty \\), when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ double rnorm3d(double a, double b, double c)



Calculate one over the square root of the sum of squares of three coordinates.

Calculate one over the length of three dimensional vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns one over the length of the 3D vector \\( \frac{1}{\sqrt{a^2+b^2+c^2}} \\).

  * In the presence of an exactly infinite coordinate \\( +0 \\) is returned, even if there are NaNs.

  * returns \\( +\infty \\), when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ double rnorm4d(double a, double b, double c, double d)



Calculate one over the square root of the sum of squares of four coordinates.

Calculate one over the length of four dimensional vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns one over the length of the 3D vector \\( \frac{1}{\sqrt{a^2+b^2+c^2+d^2}} \\).

  * In the presence of an exactly infinite coordinate \\( +0 \\) is returned, even if there are NaNs.

  * returns \\( +\infty \\), when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ double round(double x)



Round to nearest integer value in floating-point.

Round `x` to the nearest integer value in floating-point format, with halfway cases rounded away from zero.

Note

This function may be slower than alternate rounding methods. See rint().

Returns


Returns rounded integer value.

  * round( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * round( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * round(NaN) returns NaN.


__device__ double rsqrt(double x)



Calculate the reciprocal of the square root of the input argument.

Calculate the reciprocal of the nonnegative square root of `x`, \\( 1/\sqrt{x} \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns \\( 1/\sqrt{x} \\).

  * rsqrt( \\( +\infty \\) ) returns +0.

  * rsqrt( \\( \pm 0 \\) ) returns \\( \pm \infty \\).

  * rsqrt(`x`) returns NaN if `x` is less than 0.

  * rsqrt(NaN) returns NaN.


__device__ double scalbln(double x, long int n)



Scale floating-point input by integer power of two.

Scale `x` by \\( 2^n \\) by efficient manipulation of the floating-point exponent.

Returns


Returns `x` * \\( 2^n \\).

  * scalbln( \\( \pm 0 \\) , `n`) returns \\( \pm 0 \\).

  * scalbln(`x`, 0) returns `x`.

  * scalbln( \\( \pm \infty \\) , `n`) returns \\( \pm \infty \\).

  * scalbln(NaN, `n`) returns NaN.


__device__ double scalbn(double x, int n)



Scale floating-point input by integer power of two.

Scale `x` by \\( 2^n \\) by efficient manipulation of the floating-point exponent.

Returns


Returns `x` * \\( 2^n \\).

  * scalbn( \\( \pm 0 \\) , `n`) returns \\( \pm 0 \\).

  * scalbn(`x`, 0) returns `x`.

  * scalbn( \\( \pm \infty \\) , `n`) returns \\( \pm \infty \\).

  * scalbn(NaN, `n`) returns NaN.


__device__ __RETURN_TYPE signbit(double a)



Return the sign bit of the input.

Determine whether the floating-point value `a` is negative.

Returns


Reports the sign bit of all values including infinities, zeros, and NaNs.

  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is negative.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is negative.


__device__ double sin(double x)



Calculate the sine of the input argument.

Calculate the sine of the input argument `x` (measured in radians).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * sin( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sin( \\( \pm \infty \\) ) returns NaN.

  * sin(NaN) returns NaN.


__device__ void sincos(double x, double *sptr, double *cptr)



Calculate the sine and cosine of the first input argument.

Calculate the sine and cosine of the first input argument `x` (measured in radians). The results for sine and cosine are written into the second argument, `sptr`, and, respectively, third argument, `cptr`.

See also

sin() and cos().

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

__device__ void sincospi(double x, double *sptr, double *cptr)



Calculate the sine and cosine of the first input argument \\( \times \pi \\).

Calculate the sine and cosine of the first input argument, `x` (measured in radians), \\( \times \pi \\). The results for sine and cosine are written into the second argument, `sptr`, and, respectively, third argument, `cptr`.

See also

sinpi() and cospi().

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

__device__ double sinh(double x)



Calculate the hyperbolic sine of the input argument.

Calculate the hyperbolic sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * sinh( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sinh( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * sinh(NaN) returns NaN.


__device__ double sinpi(double x)



Calculate the sine of the input argument \\( \times \pi \\).

Calculate the sine of `x` \\( \times \pi \\) (measured in radians), where `x` is the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * sinpi( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sinpi( \\( \pm \infty \\) ) returns NaN.

  * sinpi(NaN) returns NaN.


__device__ double sqrt(double x)



Calculate the square root of the input argument.

Calculate the nonnegative square root of `x`, \\( \sqrt{x} \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns \\( \sqrt{x} \\).

  * sqrt( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sqrt( \\( +\infty \\) ) returns \\( +\infty \\).

  * sqrt(`x`) returns NaN if `x` is less than 0.

  * sqrt(NaN) returns NaN.


__device__ double tan(double x)



Calculate the tangent of the input argument.

Calculate the tangent of the input argument `x` (measured in radians).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * tan( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * tan( \\( \pm \infty \\) ) returns NaN.

  * tan(NaN) returns NaN.


__device__ double tanh(double x)



Calculate the hyperbolic tangent of the input argument.

Calculate the hyperbolic tangent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * tanh( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * tanh( \\( \pm \infty \\) ) returns \\( \pm 1 \\).

  * tanh(NaN) returns NaN.


__device__ double tgamma(double x)



Calculate the gamma function of the input argument.

Calculate the gamma function of the input argument `x`, namely the value of \\( \Gamma(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


  * tgamma( \\( \pm 0 \\) ) returns \\( \pm \infty \\).

  * tgamma(`x`) returns NaN if `x` < 0 and `x` is an integer.

  * tgamma( \\( -\infty \\) ) returns NaN.

  * tgamma( \\( +\infty \\) ) returns \\( +\infty \\).

  * tgamma(NaN) returns NaN.


__device__ double trunc(double x)



Truncate input argument to the integral part.

Round `x` to the nearest integer value that does not exceed `x` in magnitude.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns truncated integer value.

  * trunc( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * trunc( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * trunc(NaN) returns NaN.


__device__ double y0(double x)



Calculate the value of the Bessel function of the second kind of order 0 for the input argument.

Calculate the value of the Bessel function of the second kind of order 0 for the input argument `x`, \\( Y_0(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the second kind of order 0.

  * y0( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * y0(`x`) returns NaN for `x` < 0.

  * y0( \\( +\infty \\) ) returns +0.

  * y0(NaN) returns NaN.


__device__ double y1(double x)



Calculate the value of the Bessel function of the second kind of order 1 for the input argument.

Calculate the value of the Bessel function of the second kind of order 1 for the input argument `x`, \\( Y_1(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the second kind of order 1.

  * y1( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * y1(`x`) returns NaN for `x` < 0.

  * y1( \\( +\infty \\) ) returns +0.

  * y1(NaN) returns NaN.


__device__ double yn(int n, double x)



Calculate the value of the Bessel function of the second kind of order n for the input argument.

Calculate the value of the Bessel function of the second kind of order `n` for the input argument `x`, \\( Y_n(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Double-Precision Floating-Point Functions section.

Returns


Returns the value of the Bessel function of the second kind of order `n`.

  * yn(`n`, `x`) returns NaN for `n` < 0.

  * yn(`n`, \\( \pm 0 \\) ) returns \\( -\infty \\).

  * yn(`n`, `x`) returns NaN for `x` < 0.

  * yn(`n`, \\( +\infty \\) ) returns +0.

  * yn(`n`, NaN) returns NaN.