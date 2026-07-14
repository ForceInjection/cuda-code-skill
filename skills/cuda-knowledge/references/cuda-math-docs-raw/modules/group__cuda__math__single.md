# 6. Single Precision Mathematical Functions

**Source:** [group__CUDA__MATH__SINGLE.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html)

---

#  6\. Single Precision Mathematical Functions[](#single-precision-mathematical-functions "Permalink to this headline")

This section describes single precision mathematical functions. 

To use these functions, you do not need to include any additional header file in your program. 

Functions

__device__ float [acosf](#group__cuda__math__single_1ga63d1c22538561dc228fc230d10d85dca)(float x)
    

Calculate the arc cosine of the input argument.

__device__ float [acoshf](#group__cuda__math__single_1gab0f45cada398311319b50a00ff7e826e)(float x)
    

Calculate the nonnegative inverse hyperbolic cosine of the input argument.

__device__ float [asinf](#group__cuda__math__single_1ga82b2bb388724796ae8a30069abb3b386)(float x)
    

Calculate the arc sine of the input argument.

__device__ float [asinhf](#group__cuda__math__single_1ga74d4dabb94aa5c77ce31fd0ea987c083)(float x)
    

Calculate the inverse hyperbolic sine of the input argument.

__device__ float [atan2f](#group__cuda__math__single_1ga3f0bdfc73288f9dda45e5c9be7811c9d)(float y, float x)
    

Calculate the arc tangent of the ratio of first and second input arguments.

__device__ float [atanf](#group__cuda__math__single_1ga82629bb4eec2d8c9c95b9c69188beff9)(float x)
    

Calculate the arc tangent of the input argument.

__device__ float [atanhf](#group__cuda__math__single_1ga1b176d9d72adbf998b1960f830ad9dcc)(float x)
    

Calculate the inverse hyperbolic tangent of the input argument.

__device__ float [cbrtf](#group__cuda__math__single_1ga96d2384128af36ea9cb9b20d366900c7)(float x)
    

Calculate the cube root of the input argument.

__device__ float [ceilf](#group__cuda__math__single_1ga43a6f3aa4ccdb026b038a3fe9a80f65d)(float x)
    

Calculate ceiling of the input argument.

__device__ float [copysignf](#group__cuda__math__single_1gaf624240731f96c35e2bbf9aaa9217ad6)(float x, float y)
    

Create value with given magnitude, copying sign of second value.

__device__ float [cosf](#group__cuda__math__single_1ga20858ddd8f75a2c8332bdecd536057bf)(float x)
    

Calculate the cosine of the input argument.

__device__ float [coshf](#group__cuda__math__single_1ga34a53cc088d117bc7045caa111279799)(float x)
    

Calculate the hyperbolic cosine of the input argument.

__device__ float [cospif](#group__cuda__math__single_1ga6fc515121cf408a92ef611a3c6fdc5cc)(float x)
    

Calculate the cosine of the input argument \\(\times \pi\\) .

__device__ float [cyl_bessel_i0f](#group__cuda__math__single_1gaee787afb8a173c23b99d89239e245c59)(float x)
    

Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.

__device__ float [cyl_bessel_i1f](#group__cuda__math__single_1ga2505fc93886666a3ceec465ac5bfda1c)(float x)
    

Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.

__device__ float [erfcf](#group__cuda__math__single_1ga31faaaeab2a785191c3e0e66e030ceca)(float x)
    

Calculate the complementary error function of the input argument.

__device__ float [erfcinvf](#group__cuda__math__single_1ga47e42e1bbdda4a98b07fbca5e2a5c396)(float x)
    

Calculate the inverse complementary error function of the input argument.

__device__ float [erfcxf](#group__cuda__math__single_1gaec797649c94f21aecb8dc033a7b97353)(float x)
    

Calculate the scaled complementary error function of the input argument.

__device__ float [erff](#group__cuda__math__single_1ga3b8115ff34a107f4608152fd943dbf81)(float x)
    

Calculate the error function of the input argument.

__device__ float [erfinvf](#group__cuda__math__single_1ga5f5bf3cdea7cff0a0e4256ee4821b3df)(float x)
    

Calculate the inverse error function of the input argument.

__device__ float [exp10f](#group__cuda__math__single_1ga60f1de4fe78a907d915a52be29a799e7)(float x)
    

Calculate the base 10 exponential of the input argument.

__device__ float [exp2f](#group__cuda__math__single_1ga3e2984de99de67ca680c9bb4f4427f81)(float x)
    

Calculate the base 2 exponential of the input argument.

__device__ float [expf](#group__cuda__math__single_1gae2d7656fe00f9e750c6f3bde8cc0dca6)(float x)
    

Calculate the base \\(e\\) exponential of the input argument.

__device__ float [expm1f](#group__cuda__math__single_1ga832817212e7b0debe05d23ea37bdd748)(float x)
    

Calculate the base \\(e\\) exponential of the input argument, minus 1.

__device__ float [fabsf](#group__cuda__math__single_1gab00f8593e1bfb1985526020fbec4e0fc)(float x)
    

Calculate the absolute value of its argument.

__device__ float [fdimf](#group__cuda__math__single_1ga13959e5ca19c910e0d6f8e6ca5492149)(float x, float y)
    

Compute the positive difference between `x` and `y` .

__device__ float [fdividef](#group__cuda__math__single_1ga26852224a1347ee9b5edee12c869d206)(float x, float y)
    

Divide two floating-point values.

__device__ float [floorf](#group__cuda__math__single_1gadbff62f4c1647b9694f35d053eff5288)(float x)
    

Calculate the largest integer less than or equal to `x` .

__device__ float [fmaf](#group__cuda__math__single_1ga5910ee832dab4f5d37118e0a6811c195)(float x, float y, float z)
    

Compute \\(x \times y + z\\) as a single operation.

__device__ float [fmaxf](#group__cuda__math__single_1ga6e7516db46be25c33fb26e203287f2a3)(float x, float y)
    

Determine the maximum numeric value of the arguments.

__device__ float [fminf](#group__cuda__math__single_1gabf48322ad520d7b12542edf990dde8c0)(float x, float y)
    

Determine the minimum numeric value of the arguments.

__device__ float [fmodf](#group__cuda__math__single_1ga9255f64a2585463fea365c8273d23904)(float x, float y)
    

Calculate the floating-point remainder of `x` / `y` .

__device__ float [frexpf](#group__cuda__math__single_1ga56e8cba742e2f80647903dac9c93eb37)(float x, int *nptr)
    

Extract mantissa and exponent of a floating-point value.

__device__ float [hypotf](#group__cuda__math__single_1ga2880a4ebf5500aeb74fb01340ea91215)(float x, float y)
    

Calculate the square root of the sum of squares of two arguments.

__device__ int [ilogbf](#group__cuda__math__single_1ga4e9bcb254b97eb63abf3092233464131)(float x)
    

Compute the unbiased integer exponent of the argument.

__device__ __RETURN_TYPE [isfinite](#group__cuda__math__single_1ga57a3c8313f570282a1a7bcc78743b08e)(float a)
    

Determine whether argument is finite.

__device__ __RETURN_TYPE [isinf](#group__cuda__math__single_1ga0a62e45f335a23ee64ecad3fb87a72e3)(float a)
    

Determine whether argument is infinite.

__device__ __RETURN_TYPE [isnan](#group__cuda__math__single_1gaf8093cd7c372f91c9837a82fd368c711)(float a)
    

Determine whether argument is a NaN.

__device__ float [j0f](#group__cuda__math__single_1gaba3e4bad4109f5e8509dc1925fade7ce)(float x)
    

Calculate the value of the Bessel function of the first kind of order 0 for the input argument.

__device__ float [j1f](#group__cuda__math__single_1ga462954bfc6ada6132f28bd7fce41334e)(float x)
    

Calculate the value of the Bessel function of the first kind of order 1 for the input argument.

__device__ float [jnf](#group__cuda__math__single_1gadcd52a43c4f2d8d9148a022d6d6851dd)(int n, float x)
    

Calculate the value of the Bessel function of the first kind of order n for the input argument.

__device__ float [ldexpf](#group__cuda__math__single_1ga7d82accff3d8e3307d61e028c19c30cd)(float x, int exp)
    

Calculate the value of \\(x\cdot 2^{exp}\\) .

__device__ float [lgammaf](#group__cuda__math__single_1gaf7ffab2d685130195ba255e954e21130)(float x)
    

Calculate the natural logarithm of the absolute value of the gamma function of the input argument.

__device__ long long int [llrintf](#group__cuda__math__single_1ga7d4af230b5deee73fbfa9801f44f0616)(float x)
    

Round input to nearest integer value.

__device__ long long int [llroundf](#group__cuda__math__single_1gaf2a7fe8fb57e5b39886d776f75fdf5d6)(float x)
    

Round to nearest integer value.

__device__ float [log10f](#group__cuda__math__single_1gab49e218cf742a0eb08e5516dd5160585)(float x)
    

Calculate the base 10 logarithm of the input argument.

__device__ float [log1pf](#group__cuda__math__single_1ga9d53128ab5f7d6ebc4798f243481a6d7)(float x)
    

Calculate the value of \\(\log_{e}(1+x)\\) .

__device__ float [log2f](#group__cuda__math__single_1gafc9ae1bd4ebb4cd9533a50f1bf486f08)(float x)
    

Calculate the base 2 logarithm of the input argument.

__device__ float [logbf](#group__cuda__math__single_1ga9a86f57d529d7000b04cb30e859a21b7)(float x)
    

Calculate the floating-point representation of the exponent of the input argument.

__device__ float [logf](#group__cuda__math__single_1gacdaf041c4071f63cba0e51658b89ffa4)(float x)
    

Calculate the natural logarithm of the input argument.

__device__ long int [lrintf](#group__cuda__math__single_1ga639a876a55da8142dcd917ce6c12c27d)(float x)
    

Round input to nearest integer value.

__device__ long int [lroundf](#group__cuda__math__single_1ga4d10236b2afbafda2fd85825811b84e3)(float x)
    

Round to nearest integer value.

__device__ float [max](#group__cuda__math__single_1ga41e13ea2493aa9d89eb0745c9101c995)(const float a, const float b)
    

Calculate the maximum value of the input `float` arguments.

__device__ float [min](#group__cuda__math__single_1gac9f96a25ba62de9d44d4a7ca098aed5e)(const float a, const float b)
    

Calculate the minimum value of the input `float` arguments.

__device__ float [modff](#group__cuda__math__single_1ga7c49d2e467f6ca3cfc0362d84bb474ab)(float x, float *iptr)
    

Break down the input argument into fractional and integral parts.

__device__ float [nanf](#group__cuda__math__single_1ga372c640f910303dc4a7f17ce684322c5)(const char *tagp)
    

Returns "Not a Number" value.

__device__ float [nearbyintf](#group__cuda__math__single_1ga53c10d923def0d85af5a2b65b1a021f0)(float x)
    

Round the input argument to the nearest integer.

__device__ float [nextafterf](#group__cuda__math__single_1ga997fc003282f27b1c02c8a44fb4189f0)(float x, float y)
    

Return next representable single-precision floating-point value after argument `x` in the direction of `y` .

__device__ float [norm3df](#group__cuda__math__single_1ga921612f74ed8a71e62d40c547cab6dcf)(float a, float b, float c)
    

Calculate the square root of the sum of squares of three coordinates of the argument.

__device__ float [norm4df](#group__cuda__math__single_1ga2334d82818e94dcac4251cd045e1e281)(float a, float b, float c, float d)
    

Calculate the square root of the sum of squares of four coordinates of the argument.

__device__ float [normcdff](#group__cuda__math__single_1ga48e61ee0399db68ffbf01d4ce916153a)(float x)
    

Calculate the standard normal cumulative distribution function.

__device__ float [normcdfinvf](#group__cuda__math__single_1ga943a96334db31757c898d805df00b7b2)(float x)
    

Calculate the inverse of the standard normal cumulative distribution function.

__device__ float [normf](#group__cuda__math__single_1ga3384242f7010ba3d90f29ee0f3a67d9d)(int dim, float const *p)
    

Calculate the square root of the sum of squares of any number of coordinates.

__device__ float [powf](#group__cuda__math__single_1gab519b517c0036b3604d602f716a919dd)(float x, float y)
    

Calculate the value of first argument to the power of second argument.

__device__ float [rcbrtf](#group__cuda__math__single_1ga937164a0d40347821ad16b5cb5069c92)(float x)
    

Calculate reciprocal cube root function.

__device__ float [remainderf](#group__cuda__math__single_1ga36179ffa51305653b55c1e76f44154ff)(float x, float y)
    

Compute single-precision floating-point remainder.

__device__ float [remquof](#group__cuda__math__single_1gaa0d8ebba46ca705859d1c7462b53118d)(float x, float y, int *quo)
    

Compute single-precision floating-point remainder and part of quotient.

__device__ float [rhypotf](#group__cuda__math__single_1gaa53c41aebb09f501ea5e09a01145a932)(float x, float y)
    

Calculate one over the square root of the sum of squares of two arguments.

__device__ float [rintf](#group__cuda__math__single_1ga7791cd93108ffc6d24524f2e8635ccfd)(float x)
    

Round input to nearest integer value in floating-point.

__device__ float [rnorm3df](#group__cuda__math__single_1gaf97228e858bd11e2934c26cf54a1dff6)(float a, float b, float c)
    

Calculate one over the square root of the sum of squares of three coordinates.

__device__ float [rnorm4df](#group__cuda__math__single_1ga66a3b53292754ba1c455fb9b30b1e40a)(float a, float b, float c, float d)
    

Calculate one over the square root of the sum of squares of four coordinates.

__device__ float [rnormf](#group__cuda__math__single_1gac1a6c8305122ef2456ad9133e676138e)(int dim, float const *p)
    

Calculate the reciprocal of square root of the sum of squares of any number of coordinates.

__device__ float [roundf](#group__cuda__math__single_1gaa1c1521079e51b4f54771b16a7f8aeea)(float x)
    

Round to nearest integer value in floating-point.

__device__ float [rsqrtf](#group__cuda__math__single_1ga5a9bc318028131cfd13d10abfae1ae13)(float x)
    

Calculate the reciprocal of the square root of the input argument.

__device__ float [scalblnf](#group__cuda__math__single_1gac94fa1e3aea5f190b7ceb47917e722be)(float x, long int n)
    

Scale floating-point input by integer power of two.

__device__ float [scalbnf](#group__cuda__math__single_1gae5d0f588dbdbce27abe79ac3280a429f)(float x, int n)
    

Scale floating-point input by integer power of two.

__device__ __RETURN_TYPE [signbit](#group__cuda__math__single_1gaf105073ad5ef209e40942216f4ba6d8c)(float a)
    

Return the sign bit of the input.

__device__ void [sincosf](#group__cuda__math__single_1ga9456ff9df91a3874180d89a94b36fd46)(float x, float *sptr, float *cptr)
    

Calculate the sine and cosine of the first input argument.

__device__ void [sincospif](#group__cuda__math__single_1gaab8978300988c385e0aa4b6cba44225e)(float x, float *sptr, float *cptr)
    

Calculate the sine and cosine of the first input argument \\(\times \pi\\) .

__device__ float [sinf](#group__cuda__math__single_1ga4677d53159664972c54bb697b9c1bace)(float x)
    

Calculate the sine of the input argument.

__device__ float [sinhf](#group__cuda__math__single_1ga72c262cde9f805d08492c316fc0158d9)(float x)
    

Calculate the hyperbolic sine of the input argument.

__device__ float [sinpif](#group__cuda__math__single_1ga85a985e497f4199be19462387e062ae2)(float x)
    

Calculate the sine of the input argument \\(\times \pi\\) .

__device__ float [sqrtf](#group__cuda__math__single_1gacb80df3c252b3feb3cc88f992b955a14)(float x)
    

Calculate the square root of the input argument.

__device__ float [tanf](#group__cuda__math__single_1ga561a1e0eab1092d294d331caf9bb93c5)(float x)
    

Calculate the tangent of the input argument.

__device__ float [tanhf](#group__cuda__math__single_1ga7d925743801795775ca98ae83d4ba6e6)(float x)
    

Calculate the hyperbolic tangent of the input argument.

__device__ float [tgammaf](#group__cuda__math__single_1ga0e556a6b5d691277e3234f4548d9ae23)(float x)
    

Calculate the gamma function of the input argument.

__device__ float [truncf](#group__cuda__math__single_1ga86499f47865e04e1ca845927f41b3322)(float x)
    

Truncate input argument to the integral part.

__device__ float [y0f](#group__cuda__math__single_1ga87d0270856e29b6a34038c017513f811)(float x)
    

Calculate the value of the Bessel function of the second kind of order 0 for the input argument.

__device__ float [y1f](#group__cuda__math__single_1gabba94fdcb53f6a12f8bf5191697e8359)(float x)
    

Calculate the value of the Bessel function of the second kind of order 1 for the input argument.

__device__ float [ynf](#group__cuda__math__single_1ga383612b6d78a55003343521bca193ecd)(int n, float x)
    

Calculate the value of the Bessel function of the second kind of order n for the input argument.

##  6.1. Functions[](#functions "Permalink to this headline")

__device__ float acosf(float x)[](#_CPPv45acosff "Permalink to this definition")  

    

Calculate the arc cosine of the input argument. 

Calculate the principal value of the arc cosine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Result will be in radians, in the interval [0, \\( \pi \\) ] for `x` inside [-1, +1].

  * acosf(1) returns +0.

  * acosf(`x`) returns NaN for `x` outside [-1, +1].

  * acosf(NaN) returns NaN.


__device__ float acoshf(float x)[](#_CPPv46acoshff "Permalink to this definition")  

    

Calculate the nonnegative inverse hyperbolic cosine of the input argument. 

Calculate the nonnegative inverse hyperbolic cosine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Result will be in the interval [0, \\( +\infty \\) ].

  * acoshf(1) returns 0.

  * acoshf(`x`) returns NaN for `x` in the interval [ \\( -\infty \\) , 1).

  * acoshf( \\( +\infty \\) ) returns \\( +\infty \\).

  * acoshf(NaN) returns NaN.


__device__ float asinf(float x)[](#_CPPv45asinff "Permalink to this definition")  

    

Calculate the arc sine of the input argument. 

Calculate the principal value of the arc sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Result will be in radians, in the interval [- \\( \pi/2 \\) , + \\( \pi/2 \\) ] for `x` inside [-1, +1].

  * asinf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * asinf(`x`) returns NaN for `x` outside [-1, +1].

  * asinf(NaN) returns NaN.


__device__ float asinhf(float x)[](#_CPPv46asinhff "Permalink to this definition")  

    

Calculate the inverse hyperbolic sine of the input argument. 

Calculate the inverse hyperbolic sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * asinhf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * asinhf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * asinhf(NaN) returns NaN.


__device__ float atan2f(float y, float x)[](#_CPPv46atan2fff "Permalink to this definition")  

    

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


__device__ float atanf(float x)[](#_CPPv45atanff "Permalink to this definition")  

    

Calculate the arc tangent of the input argument. 

Calculate the principal value of the arc tangent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Result will be in radians, in the interval [- \\( \pi/2 \\) , + \\( \pi/2 \\) ].

  * atanf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * atanf( \\( \pm \infty \\) ) returns \\( \pm \pi \\) /2.

  * atanf(NaN) returns NaN.


__device__ float atanhf(float x)[](#_CPPv46atanhff "Permalink to this definition")  

    

Calculate the inverse hyperbolic tangent of the input argument. 

Calculate the inverse hyperbolic tangent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * atanhf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * atanhf( \\( \pm 1 \\) ) returns \\( \pm \infty \\).

  * atanhf(`x`) returns NaN for `x` outside interval [-1, 1].

  * atanhf(NaN) returns NaN.


__device__ float cbrtf(float x)[](#_CPPv45cbrtff "Permalink to this definition")  

    

Calculate the cube root of the input argument. 

Calculate the cube root of `x`, \\( x^{1/3} \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns \\( x^{1/3} \\).

  * cbrtf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * cbrtf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * cbrtf(NaN) returns NaN.


__device__ float ceilf(float x)[](#_CPPv45ceilff "Permalink to this definition")  

    

Calculate ceiling of the input argument. 

Compute the smallest integer value not less than `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns \\( \lceil x \rceil \\) expressed as a floating-point number.

  * ceilf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * ceilf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * ceilf(NaN) returns NaN.


__device__ float copysignf(float x, float y)[](#_CPPv49copysignfff "Permalink to this definition")  

    

Create value with given magnitude, copying sign of second value. 

Create a floating-point value with the magnitude `x` and the sign of `y`.

Returns
    

  * a value with the magnitude of `x` and the sign of `y`.

  * copysignf(`NaN`, `y`) returns a `NaN` with the sign of `y`. 


__device__ float cosf(float x)[](#_CPPv44cosff "Permalink to this definition")  

    

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


__device__ float coshf(float x)[](#_CPPv45coshff "Permalink to this definition")  

    

Calculate the hyperbolic cosine of the input argument. 

Calculate the hyperbolic cosine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * coshf( \\( \pm 0 \\) ) returns 1.

  * coshf( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * coshf(NaN) returns NaN.


__device__ float cospif(float x)[](#_CPPv46cospiff "Permalink to this definition")  

    

Calculate the cosine of the input argument \\( \times \pi \\). 

Calculate the cosine of `x` \\( \times \pi \\) (measured in radians), where `x` is the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * cospif( \\( \pm 0 \\) ) returns 1.

  * cospif( \\( \pm \infty \\) ) returns NaN.

  * cospif(NaN) returns NaN.


__device__ float cyl_bessel_i0f(float x)[](#_CPPv414cyl_bessel_i0ff "Permalink to this definition")  

    

Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument. 

Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument `x`, \\( I_0(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the value of the regular modified cylindrical Bessel function of order 0.

  * cyl_bessel_i0f( \\( \pm 0 \\)) returns +1.

  * cyl_bessel_i0f( \\( \pm\infty \\)) returns \\( +\infty \\).

  * cyl_bessel_i0f(NaN) returns NaN.


__device__ float cyl_bessel_i1f(float x)[](#_CPPv414cyl_bessel_i1ff "Permalink to this definition")  

    

Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument. 

Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument `x`, \\( I_1(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the value of the regular modified cylindrical Bessel function of order 1.

  * cyl_bessel_i1f( \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * cyl_bessel_i1f( \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * cyl_bessel_i1f(NaN) returns NaN.


__device__ float erfcf(float x)[](#_CPPv45erfcff "Permalink to this definition")  

    

Calculate the complementary error function of the input argument. 

Calculate the complementary error function of the input argument `x`, 1 - erf(`x`).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * erfcf( \\( -\infty \\) ) returns 2.

  * erfcf( \\( +\infty \\) ) returns +0.

  * erfcf(NaN) returns NaN.


__device__ float erfcinvf(float x)[](#_CPPv48erfcinvff "Permalink to this definition")  

    

Calculate the inverse complementary error function of the input argument. 

Calculate the inverse complementary error function \\( \operatorname{erfc}^{-1} \\) (`x`), of the input argument `x` in the interval [0, 2].

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * erfcinvf( \\( \pm 0 \\) ) returns \\( +\infty \\).

  * erfcinvf(2) returns \\( -\infty \\).

  * erfcinvf(`x`) returns NaN for `x` outside [0, 2].

  * erfcinvf(NaN) returns NaN.


__device__ float erfcxf(float x)[](#_CPPv46erfcxff "Permalink to this definition")  

    

Calculate the scaled complementary error function of the input argument. 

Calculate the scaled complementary error function of the input argument `x`, \\( e^{x^2}\cdot \operatorname{erfc}(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * erfcxf( \\( -\infty \\) ) returns \\( +\infty \\).

  * erfcxf( \\( +\infty \\) ) returns +0.

  * erfcxf(NaN) returns NaN.


__device__ float erff(float x)[](#_CPPv44erfff "Permalink to this definition")  

    

Calculate the error function of the input argument. 

Calculate the value of the error function for the input argument `x`, \\( \frac{2}{\sqrt \pi} \int_0^x e^{-t^2} dt \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * erff( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * erff( \\( \pm \infty \\) ) returns \\( \pm 1 \\).

  * erff(NaN) returns NaN.


__device__ float erfinvf(float x)[](#_CPPv47erfinvff "Permalink to this definition")  

    

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


__device__ float exp10f(float x)[](#_CPPv46exp10ff "Permalink to this definition")  

    

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


__device__ float exp2f(float x)[](#_CPPv45exp2ff "Permalink to this definition")  

    

Calculate the base 2 exponential of the input argument. 

Calculate \\( 2^x \\) , the base 2 exponential of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * exp2f( \\( \pm 0 \\) ) returns 1.

  * exp2f( \\( -\infty \\) ) returns +0.

  * exp2f( \\( +\infty \\) ) returns \\( +\infty \\).

  * exp2f(NaN) returns NaN.


__device__ float expf(float x)[](#_CPPv44expff "Permalink to this definition")  

    

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


__device__ float expm1f(float x)[](#_CPPv46expm1ff "Permalink to this definition")  

    

Calculate the base \\( e \\) exponential of the input argument, minus 1. 

Calculate \\( e^x \\) -1, the base \\( e \\) exponential of the input argument `x`, minus 1.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * expm1f( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * expm1f( \\( -\infty \\) ) returns -1.

  * expm1f( \\( +\infty \\) ) returns \\( +\infty \\).

  * expm1f(NaN) returns NaN.


__device__ float fabsf(float x)[](#_CPPv45fabsff "Permalink to this definition")  

    

Calculate the absolute value of its argument. 

Calculate the absolute value of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the absolute value of its argument.

  * fabsf( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * fabsf( \\( \pm 0 \\) ) returns +0.

  * fabsf(NaN) returns an unspecified NaN.


__device__ float fdimf(float x, float y)[](#_CPPv45fdimfff "Permalink to this definition")  

    

Compute the positive difference between `x` and `y`. 

Compute the positive difference between `x` and `y`. The positive difference is `x` \- `y` when `x` > `y` and +0 otherwise.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the positive difference between `x` and `y`.

  * fdimf(`x`, `y`) returns `x` \- `y` if `x` > `y`.

  * fdimf(`x`, `y`) returns +0 if `x` \\( \leq \\) `y`.

  * If either argument is NaN, NaN is returned. 


__device__ float fdividef(float x, float y)[](#_CPPv48fdividefff "Permalink to this definition")  

    

Divide two floating-point values. 

Compute `x` divided by `y`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected. 

Returns
    

Returns `x` / `y`.

  * Follows the regular division operation behavior by default.

  * If `-use_fast_math` is specified and is not amended by an explicit `-prec_div=true`, uses [__fdividef()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__cuda__math__intrinsic__single_1gac996beec34f94f6376d0674a6860e107) for higher performance


__device__ float floorf(float x)[](#_CPPv46floorff "Permalink to this definition")  

    

Calculate the largest integer less than or equal to `x`. 

Calculate the largest integer value which is less than or equal to `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns \\( \lfloor x \rfloor \\) expressed as a floating-point number.

  * floorf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * floorf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * floorf(NaN) returns NaN.


__device__ float fmaf(float x, float y, float z)[](#_CPPv44fmaffff "Permalink to this definition")  

    

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


__device__ float fmaxf(float x, float y)[](#_CPPv45fmaxfff "Permalink to this definition")  

    

Determine the maximum numeric value of the arguments. 

Determines the maximum numeric value of the arguments `x` and `y`. Treats NaN arguments as missing data. If one argument is a NaN and the other is legitimate numeric value, the numeric value is chosen.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the maximum numeric values of the arguments `x` and `y`.

  * If both arguments are NaN, returns NaN.

  * If one argument is NaN, returns the numeric argument.


__device__ float fminf(float x, float y)[](#_CPPv45fminfff "Permalink to this definition")  

    

Determine the minimum numeric value of the arguments. 

Determines the minimum numeric value of the arguments `x` and `y`. Treats NaN arguments as missing data. If one argument is a NaN and the other is legitimate numeric value, the numeric value is chosen.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the minimum numeric value of the arguments `x` and `y`.

  * If both arguments are NaN, returns NaN.

  * If one argument is NaN, returns the numeric argument.


__device__ float fmodf(float x, float y)[](#_CPPv45fmodfff "Permalink to this definition")  

    

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


__device__ float frexpf(float x, int *nptr)[](#_CPPv46frexpffPi "Permalink to this definition")  

    

Extract mantissa and exponent of a floating-point value. 

Decomposes the floating-point value `x` into a component `m` for the normalized fraction element and another term `n` for the exponent. The absolute value of `m` will be greater than or equal to 0.5 and less than 1.0 or it will be equal to 0; \\( x = m\cdot 2^n \\). The integer exponent `n` will be stored in the location to which `nptr` points.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the fractional component `m`.

  * frexpf( \\( \pm 0 \\) , `nptr`) returns \\( \pm 0 \\) and stores zero in the location pointed to by `nptr`.

  * frexpf( \\( \pm \infty \\) , `nptr`) returns \\( \pm \infty \\) and stores an unspecified value in the location to which `nptr` points.

  * frexpf(NaN, `y`) returns a NaN and stores an unspecified value in the location to which `nptr` points.


__device__ float hypotf(float x, float y)[](#_CPPv46hypotfff "Permalink to this definition")  

    

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


__device__ int ilogbf(float x)[](#_CPPv46ilogbff "Permalink to this definition")  

    

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


__device__ __RETURN_TYPE isfinite(float a)[](#_CPPv48isfinitef "Permalink to this definition")  

    

Determine whether argument is finite. 

Determine whether the floating-point value `a` is a finite value (zero, subnormal, or normal and not infinity or NaN).

Returns
    

  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is a finite value.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is a finite value. 


__device__ __RETURN_TYPE isinf(float a)[](#_CPPv45isinff "Permalink to this definition")  

    

Determine whether argument is infinite. 

Determine whether the floating-point value `a` is an infinite value (positive or negative).

Returns
    

  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is an infinite value.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is an infinite value. 


__device__ __RETURN_TYPE isnan(float a)[](#_CPPv45isnanf "Permalink to this definition")  

    

Determine whether argument is a NaN. 

Determine whether the floating-point value `a` is a NaN. 

Returns
    

  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is a NaN value.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is a NaN value. 


__device__ float j0f(float x)[](#_CPPv43j0ff "Permalink to this definition")  

    

Calculate the value of the Bessel function of the first kind of order 0 for the input argument. 

Calculate the value of the Bessel function of the first kind of order 0 for the input argument `x`, \\( J_0(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the value of the Bessel function of the first kind of order 0.

  * j0f( \\( \pm \infty \\) ) returns +0.

  * j0f(NaN) returns NaN.


__device__ float j1f(float x)[](#_CPPv43j1ff "Permalink to this definition")  

    

Calculate the value of the Bessel function of the first kind of order 1 for the input argument. 

Calculate the value of the Bessel function of the first kind of order 1 for the input argument `x`, \\( J_1(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the value of the Bessel function of the first kind of order 1.

  * j1f( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * j1f( \\( \pm \infty \\) ) returns \\( \pm 0 \\).

  * j1f(NaN) returns NaN.


__device__ float jnf(int n, float x)[](#_CPPv43jnfif "Permalink to this definition")  

    

Calculate the value of the Bessel function of the first kind of order n for the input argument. 

Calculate the value of the Bessel function of the first kind of order `n` for the input argument `x`, \\( J_n(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the value of the Bessel function of the first kind of order `n`.

  * jnf(`n`, NaN) returns NaN.

  * jnf(`n`, `x`) returns NaN for `n` < 0.

  * jnf(`n`, \\( +\infty \\) ) returns +0.


__device__ float ldexpf(float x, int exp)[](#_CPPv46ldexpffi "Permalink to this definition")  

    

Calculate the value of \\( x\cdot 2^{exp} \\). 

Calculate the value of \\( x\cdot 2^{exp} \\) of the input arguments `x` and `exp`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * ldexpf(`x`, `exp`) is equivalent to scalbnf(`x`, `exp`).


__device__ float lgammaf(float x)[](#_CPPv47lgammaff "Permalink to this definition")  

    

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


__device__ long long int llrintf(float x)[](#_CPPv47llrintff "Permalink to this definition")  

    

Round input to nearest integer value. 

Round `x` to the nearest integer value, with halfway cases rounded to the nearest even integer value. If the result is outside the range of the return type, the behavior is undefined.

Returns
    

Returns rounded integer value. 

__device__ long long int llroundf(float x)[](#_CPPv48llroundff "Permalink to this definition")  

    

Round to nearest integer value. 

Round `x` to the nearest integer value, with halfway cases rounded away from zero. If the result is outside the range of the return type, the behavior is undefined.

Note

This function may be slower than alternate rounding methods. See [llrintf()](#group__cuda__math__single_1ga7d4af230b5deee73fbfa9801f44f0616). 

Returns
    

Returns rounded integer value.

__device__ float log10f(float x)[](#_CPPv46log10ff "Permalink to this definition")  

    

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


__device__ float log1pf(float x)[](#_CPPv46log1pff "Permalink to this definition")  

    

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


__device__ float log2f(float x)[](#_CPPv45log2ff "Permalink to this definition")  

    

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


__device__ float logbf(float x)[](#_CPPv45logbff "Permalink to this definition")  

    

Calculate the floating-point representation of the exponent of the input argument. 

Calculate the floating-point representation of the exponent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * logbf( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * logbf( \\( \pm \infty \\) ) returns \\( +\infty \\).

  * logbf(NaN) returns NaN.


__device__ float logf(float x)[](#_CPPv44logff "Permalink to this definition")  

    

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


__device__ long int lrintf(float x)[](#_CPPv46lrintff "Permalink to this definition")  

    

Round input to nearest integer value. 

Round `x` to the nearest integer value, with halfway cases rounded to the nearest even integer value. If the result is outside the range of the return type, the behavior is undefined.

Returns
    

Returns rounded integer value. 

__device__ long int lroundf(float x)[](#_CPPv47lroundff "Permalink to this definition")  

    

Round to nearest integer value. 

Round `x` to the nearest integer value, with halfway cases rounded away from zero. If the result is outside the range of the return type, the behavior is undefined.

Note

This function may be slower than alternate rounding methods. See [lrintf()](#group__cuda__math__single_1ga639a876a55da8142dcd917ce6c12c27d). 

Returns
    

Returns rounded integer value.

__device__ float max(const float a, const float b)[](#_CPPv43maxKfKf "Permalink to this definition")  

    

Calculate the maximum value of the input `float` arguments. 

Calculate the maximum value of the arguments `a` and `b`. Behavior is equivalent to [fmaxf()](#group__cuda__math__single_1ga6e7516db46be25c33fb26e203287f2a3) function.

Note, this is different from `std:`: specification 

__device__ float min(const float a, const float b)[](#_CPPv43minKfKf "Permalink to this definition")  

    

Calculate the minimum value of the input `float` arguments. 

Calculate the minimum value of the arguments `a` and `b`. Behavior is equivalent to [fminf()](#group__cuda__math__single_1gabf48322ad520d7b12542edf990dde8c0) function.

Note, this is different from `std:`: specification 

__device__ float modff(float x, float *iptr)[](#_CPPv45modfffPf "Permalink to this definition")  

    

Break down the input argument into fractional and integral parts. 

Break down the argument `x` into fractional and integral parts. The integral part is stored in the argument `iptr`. Fractional and integral parts are given the same sign as the argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * modff( \\( \pm x \\) , `iptr`) returns a result with the same sign as `x`.

  * modff( \\( \pm \infty \\) , `iptr`) returns \\( \pm 0 \\) and stores \\( \pm \infty \\) in the object pointed to by `iptr`.

  * modff(NaN, `iptr`) stores a NaN in the object pointed to by `iptr` and returns a NaN.


__device__ float nanf(const char *tagp)[](#_CPPv44nanfPKc "Permalink to this definition")  

    

Returns “Not a Number” value. 

Return a representation of a quiet NaN. Argument `tagp` selects one of the possible representations.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * nanf(`tagp`) returns NaN.


__device__ float nearbyintf(float x)[](#_CPPv410nearbyintff "Permalink to this definition")  

    

Round the input argument to the nearest integer. 

Round argument `x` to an integer value in single precision floating-point format. Uses round to nearest rounding, with ties rounding to even.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * nearbyintf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * nearbyintf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * nearbyintf(NaN) returns NaN.


__device__ float nextafterf(float x, float y)[](#_CPPv410nextafterfff "Permalink to this definition")  

    

Return next representable single-precision floating-point value after argument `x` in the direction of `y`. 

Calculate the next representable single-precision floating-point value following `x` in the direction of `y`. For example, if `y` is greater than `x`, [nextafterf()](#group__cuda__math__single_1ga997fc003282f27b1c02c8a44fb4189f0) returns the smallest representable number greater than `x`

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * nextafterf(`x`, `y`) = `y` if `x` equals `y`.

  * nextafterf(`x`, `y`) = `NaN` if either `x` or `y` are `NaN`.


__device__ float norm3df(float a, float b, float c)[](#_CPPv47norm3dffff "Permalink to this definition")  

    

Calculate the square root of the sum of squares of three coordinates of the argument. 

Calculates the length of three dimensional vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the length of the 3D vector \\( \sqrt{a^2+b^2+c^2} \\).

  * In the presence of an exactly infinite coordinate \\( +\infty \\) is returned, even if there are NaNs.

  * returns +0, when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float norm4df(float a, float b, float c, float d)[](#_CPPv47norm4dfffff "Permalink to this definition")  

    

Calculate the square root of the sum of squares of four coordinates of the argument. 

Calculates the length of four dimensional vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the length of the 4D vector \\( \sqrt{a^2+b^2+c^2+d^2} \\).

  * In the presence of an exactly infinite coordinate \\( +\infty \\) is returned, even if there are NaNs.

  * returns +0, when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float normcdff(float x)[](#_CPPv48normcdfff "Permalink to this definition")  

    

Calculate the standard normal cumulative distribution function. 

Calculate the cumulative distribution function of the standard normal distribution for input argument `x`, \\( \Phi(x) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * normcdff( \\( +\infty \\) ) returns 1.

  * normcdff( \\( -\infty \\) ) returns +0

  * normcdff(NaN) returns NaN.


__device__ float normcdfinvf(float x)[](#_CPPv411normcdfinvff "Permalink to this definition")  

    

Calculate the inverse of the standard normal cumulative distribution function. 

Calculate the inverse of the standard normal cumulative distribution function for input argument `x`, \\( \Phi^{-1}(x) \\). The function is defined for input values in the interval \\( (0, 1) \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * normcdfinvf( \\( \pm 0 \\) ) returns \\( -\infty \\).

  * normcdfinvf(1) returns \\( +\infty \\).

  * normcdfinvf(`x`) returns NaN if `x` is not in the interval [0,1].

  * normcdfinvf(NaN) returns NaN.


__device__ float normf(int dim, float const *p)[](#_CPPv45normfiPKf "Permalink to this definition")  

    

Calculate the square root of the sum of squares of any number of coordinates. 

Calculates the length of a vector `p`, dimension of which is passed as an argument without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the length of the dim-D vector \\( \sqrt{\sum_{i=0}^{dim-1} p_i^2} \\).

  * In the presence of an exactly infinite coordinate \\( +\infty \\) is returned, even if there are NaNs.

  * returns +0, when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float powf(float x, float y)[](#_CPPv44powfff "Permalink to this definition")  

    

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


__device__ float rcbrtf(float x)[](#_CPPv46rcbrtff "Permalink to this definition")  

    

Calculate reciprocal cube root function. 

Calculate reciprocal cube root function of `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * rcbrtf( \\( \pm 0 \\) ) returns \\( \pm \infty \\).

  * rcbrtf( \\( \pm \infty \\) ) returns \\( \pm 0 \\).

  * rcbrtf(NaN) returns NaN.


__device__ float remainderf(float x, float y)[](#_CPPv410remainderfff "Permalink to this definition")  

    

Compute single-precision floating-point remainder. 

Compute single-precision floating-point remainder `r` of dividing `x` by `y` for nonzero `y`. Thus \\( r = x - n y \\). The value `n` is the integer value nearest \\( \frac{x}{y} \\). In the case when \\( | n -\frac{x}{y} | = \frac{1}{2} \\) , the even `n` value is chosen.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * remainderf(`x`, \\( \pm 0 \\) ) returns NaN.

  * remainderf( \\( \pm \infty \\) , `y`) returns NaN.

  * remainderf(`x`, \\( \pm \infty \\) ) returns `x` for finite `x`.

  * If either argument is NaN, NaN is returned.


__device__ float remquof(float x, float y, int *quo)[](#_CPPv47remquofffPi "Permalink to this definition")  

    

Compute single-precision floating-point remainder and part of quotient. 

Compute a single-precision floating-point remainder in the same way as the [remainderf()](#group__cuda__math__single_1ga36179ffa51305653b55c1e76f44154ff) function. Argument `quo` returns part of quotient upon division of `x` by `y`. Value `quo` has the same sign as \\( \frac{x}{y} \\) and may not be the exact quotient but agrees with the exact quotient in the low order 3 bits.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns the remainder.

  * remquof(`x`, \\( \pm 0 \\) , `quo`) returns NaN and stores an unspecified value in the location to which `quo` points.

  * remquof( \\( \pm \infty \\) , `y`, `quo`) returns NaN and stores an unspecified value in the location to which `quo` points.

  * remquof(`x`, `y`, `quo`) returns NaN and stores an unspecified value in the location to which `quo` points if either of `x` or `y` is NaN.

  * remquof(`x`, \\( \pm \infty \\) , `quo`) returns `x` and stores zero in the location to which `quo` points for finite `x`.


__device__ float rhypotf(float x, float y)[](#_CPPv47rhypotfff "Permalink to this definition")  

    

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


__device__ float rintf(float x)[](#_CPPv45rintff "Permalink to this definition")  

    

Round input to nearest integer value in floating-point. 

Round `x` to the nearest integer value in floating-point format, with halfway cases rounded to the nearest even integer value.

Returns
    

Returns rounded integer value.

  * rintf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * rintf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * rintf(NaN) returns NaN. 


__device__ float rnorm3df(float a, float b, float c)[](#_CPPv48rnorm3dffff "Permalink to this definition")  

    

Calculate one over the square root of the sum of squares of three coordinates. 

Calculates one over the length of three dimension vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns one over the length of the 3D vector \\( \frac{1}{\sqrt{a^2+b^2+c^2}} \\).

  * In the presence of an exactly infinite coordinate \\( +0 \\) is returned, even if there are NaNs.

  * returns \\( +\infty \\), when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float rnorm4df(float a, float b, float c, float d)[](#_CPPv48rnorm4dfffff "Permalink to this definition")  

    

Calculate one over the square root of the sum of squares of four coordinates. 

Calculates one over the length of four dimension vector in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns one over the length of the 3D vector \\( \frac{1}{\sqrt{a^2+b^2+c^2+d^2}} \\).

  * In the presence of an exactly infinite coordinate \\( +0 \\) is returned, even if there are NaNs.

  * returns \\( +\infty \\), when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float rnormf(int dim, float const *p)[](#_CPPv46rnormfiPKf "Permalink to this definition")  

    

Calculate the reciprocal of square root of the sum of squares of any number of coordinates. 

Calculates one over the length of vector `p`, dimension of which is passed as an argument, in Euclidean space without undue overflow or underflow.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns one over the length of the vector \\( \frac{1}{\sqrt{\sum_{i=0}^{dim-1} p_i^2}} \\).

  * In the presence of an exactly infinite coordinate \\( +0 \\) is returned, even if there are NaNs.

  * returns \\( +\infty \\), when all coordinates are \\( \pm 0 \\).

  * returns NaN, when at least one of the coordinates is NaN and none are infinite.


__device__ float roundf(float x)[](#_CPPv46roundff "Permalink to this definition")  

    

Round to nearest integer value in floating-point. 

Round `x` to the nearest integer value in floating-point format, with halfway cases rounded away from zero.

Note

This function may be slower than alternate rounding methods. See [rintf()](#group__cuda__math__single_1ga7791cd93108ffc6d24524f2e8635ccfd). 

Returns
    

Returns rounded integer value.

  * roundf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * roundf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * roundf(NaN) returns NaN.


__device__ float rsqrtf(float x)[](#_CPPv46rsqrtff "Permalink to this definition")  

    

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


__device__ float scalblnf(float x, long int n)[](#_CPPv48scalblnffl "Permalink to this definition")  

    

Scale floating-point input by integer power of two. 

Scale `x` by \\( 2^n \\) by efficient manipulation of the floating-point exponent.

Returns
    

Returns `x` * \\( 2^n \\).

  * scalblnf( \\( \pm 0 \\) , `n`) returns \\( \pm 0 \\).

  * scalblnf(`x`, 0) returns `x`.

  * scalblnf( \\( \pm \infty \\) , `n`) returns \\( \pm \infty \\).

  * scalblnf(NaN, `n`) returns NaN. 


__device__ float scalbnf(float x, int n)[](#_CPPv47scalbnffi "Permalink to this definition")  

    

Scale floating-point input by integer power of two. 

Scale `x` by \\( 2^n \\) by efficient manipulation of the floating-point exponent.

Returns
    

Returns `x` * \\( 2^n \\).

  * scalbnf( \\( \pm 0 \\) , `n`) returns \\( \pm 0 \\).

  * scalbnf(`x`, 0) returns `x`.

  * scalbnf( \\( \pm \infty \\) , `n`) returns \\( \pm \infty \\).

  * scalbnf(NaN, `n`) returns NaN. 


__device__ __RETURN_TYPE signbit(float a)[](#_CPPv47signbitf "Permalink to this definition")  

    

Return the sign bit of the input. 

Determine whether the floating-point value `a` is negative.

Returns
    

Reports the sign bit of all values including infinities, zeros, and NaNs.

  * With Visual Studio 2013 host compiler: __RETURN_TYPE is ‘bool’. Returns true if and only if `a` is negative.

  * With other host compilers: __RETURN_TYPE is ‘int’. Returns a nonzero value if and only if `a` is negative. 


__device__ void sincosf(float x, float *sptr, float *cptr)[](#_CPPv47sincosffPfPf "Permalink to this definition")  

    

Calculate the sine and cosine of the first input argument. 

Calculate the sine and cosine of the first input argument `x` (measured in radians). The results for sine and cosine are written into the second argument, `sptr`, and, respectively, third argument, `cptr`.

See also

[sinf()](#group__cuda__math__single_1ga4677d53159664972c54bb697b9c1bace) and [cosf()](#group__cuda__math__single_1ga20858ddd8f75a2c8332bdecd536057bf). 

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Note

This function is affected by the `use_fast_math` compiler flag. See the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section for a complete list of functions affected. 

__device__ void sincospif(float x, float *sptr, float *cptr)[](#_CPPv49sincospiffPfPf "Permalink to this definition")  

    

Calculate the sine and cosine of the first input argument \\( \times \pi \\). 

Calculate the sine and cosine of the first input argument, `x` (measured in radians), \\( \times \pi \\). The results for sine and cosine are written into the second argument, `sptr`, and, respectively, third argument, `cptr`.

See also

[sinpif()](#group__cuda__math__single_1ga85a985e497f4199be19462387e062ae2) and [cospif()](#group__cuda__math__single_1ga6fc515121cf408a92ef611a3c6fdc5cc). 

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

__device__ float sinf(float x)[](#_CPPv44sinff "Permalink to this definition")  

    

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


__device__ float sinhf(float x)[](#_CPPv45sinhff "Permalink to this definition")  

    

Calculate the hyperbolic sine of the input argument. 

Calculate the hyperbolic sine of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * sinhf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sinhf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * sinhf(NaN) returns NaN.


__device__ float sinpif(float x)[](#_CPPv46sinpiff "Permalink to this definition")  

    

Calculate the sine of the input argument \\( \times \pi \\). 

Calculate the sine of `x` \\( \times \pi \\) (measured in radians), where `x` is the input argument.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * sinpif( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * sinpif( \\( \pm \infty \\) ) returns NaN.

  * sinpif(NaN) returns NaN.


__device__ float sqrtf(float x)[](#_CPPv45sqrtff "Permalink to this definition")  

    

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


__device__ float tanf(float x)[](#_CPPv44tanff "Permalink to this definition")  

    

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


__device__ float tanhf(float x)[](#_CPPv45tanhff "Permalink to this definition")  

    

Calculate the hyperbolic tangent of the input argument. 

Calculate the hyperbolic tangent of the input argument `x`.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

  * tanhf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * tanhf( \\( \pm \infty \\) ) returns \\( \pm 1 \\).

  * tanhf(NaN) returns NaN.


__device__ float tgammaf(float x)[](#_CPPv47tgammaff "Permalink to this definition")  

    

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


__device__ float truncf(float x)[](#_CPPv46truncff "Permalink to this definition")  

    

Truncate input argument to the integral part. 

Round `x` to the nearest integer value that does not exceed `x` in magnitude.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Single-Precision Floating-Point Functions section. 

Returns
    

Returns truncated integer value.

  * truncf( \\( \pm 0 \\) ) returns \\( \pm 0 \\).

  * truncf( \\( \pm \infty \\) ) returns \\( \pm \infty \\).

  * truncf(NaN) returns NaN.


__device__ float y0f(float x)[](#_CPPv43y0ff "Permalink to this definition")  

    

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


__device__ float y1f(float x)[](#_CPPv43y1ff "Permalink to this definition")  

    

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


__device__ float ynf(int n, float x)[](#_CPPv43ynfif "Permalink to this definition")  

    

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