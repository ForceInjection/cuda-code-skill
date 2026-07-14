# 9. Double Precision Intrinsics

**Source:** [group__CUDA__MATH__INTRINSIC__DOUBLE.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)

---

#  9\. Double Precision Intrinsics[](#double-precision-intrinsics "Permalink to this headline")

This section describes double precision intrinsic functions that are only supported in device code. 

To use these functions, you do not need to include any additional header file in your program. 

Functions

__device__ double [__dadd_rd](#group__cuda__math__intrinsic__double_1gad39098ab6fd26648cffe5db8e2a45158)(double x, double y)
    

Add two floating-point values in round-down mode.

__device__ double [__dadd_rn](#group__cuda__math__intrinsic__double_1ga74f3cdb085ac807f35d429ab67aac193)(double x, double y)
    

Add two floating-point values in round-to-nearest-even mode.

__device__ double [__dadd_ru](#group__cuda__math__intrinsic__double_1ga8ef4f73ad7b62ff2e288087e8eb47c35)(double x, double y)
    

Add two floating-point values in round-up mode.

__device__ double [__dadd_rz](#group__cuda__math__intrinsic__double_1gaf520dfdc3b41b44681740296308e1ed1)(double x, double y)
    

Add two floating-point values in round-towards-zero mode.

__device__ double [__ddiv_rd](#group__cuda__math__intrinsic__double_1ga8a7184dfc723a76a91e7b087a2a0c494)(double x, double y)
    

Divide two floating-point values in round-down mode.

__device__ double [__ddiv_rn](#group__cuda__math__intrinsic__double_1gaf22df51b72de1f50fdaa316a47ef2f63)(double x, double y)
    

Divide two floating-point values in round-to-nearest-even mode.

__device__ double [__ddiv_ru](#group__cuda__math__intrinsic__double_1gaa8b59c6371a7df3735434e313a765ae0)(double x, double y)
    

Divide two floating-point values in round-up mode.

__device__ double [__ddiv_rz](#group__cuda__math__intrinsic__double_1gabd0f67c5396419f266027fa5be66d5a1)(double x, double y)
    

Divide two floating-point values in round-towards-zero mode.

__device__ double [__dmul_rd](#group__cuda__math__intrinsic__double_1ga0721a748fb3f7a07b75e3ca80c6447e6)(double x, double y)
    

Multiply two floating-point values in round-down mode.

__device__ double [__dmul_rn](#group__cuda__math__intrinsic__double_1gae77590b36e65f63b157db453ddb37e29)(double x, double y)
    

Multiply two floating-point values in round-to-nearest-even mode.

__device__ double [__dmul_ru](#group__cuda__math__intrinsic__double_1ga339ea2648f77714c87123a9198c899d3)(double x, double y)
    

Multiply two floating-point values in round-up mode.

__device__ double [__dmul_rz](#group__cuda__math__intrinsic__double_1gadc7aed39944bdacd98df5f30f4918e00)(double x, double y)
    

Multiply two floating-point values in round-towards-zero mode.

__device__ double [__drcp_rd](#group__cuda__math__intrinsic__double_1ga76b42ce1d523e8655454a061d0ed3728)(double x)
    

Compute \\(\frac{1}{x}\\) in round-down mode.

__device__ double [__drcp_rn](#group__cuda__math__intrinsic__double_1ga36354399debeb6772d61a430dc2ba65e)(double x)
    

Compute \\(\frac{1}{x}\\) in round-to-nearest-even mode.

__device__ double [__drcp_ru](#group__cuda__math__intrinsic__double_1ga4c05b47c04839e6ee616c84abe3264e1)(double x)
    

Compute \\(\frac{1}{x}\\) in round-up mode.

__device__ double [__drcp_rz](#group__cuda__math__intrinsic__double_1ga036d4c6182e8d98a127e686b46af0d2a)(double x)
    

Compute \\(\frac{1}{x}\\) in round-towards-zero mode.

__device__ double [__dsqrt_rd](#group__cuda__math__intrinsic__double_1ga30f36726737fda2d626c0c22119a9e46)(double x)
    

Compute \\(\sqrt{x}\\) in round-down mode.

__device__ double [__dsqrt_rn](#group__cuda__math__intrinsic__double_1gafc059c9318903bb71b8af4c6c5c11cca)(double x)
    

Compute \\(\sqrt{x}\\) in round-to-nearest-even mode.

__device__ double [__dsqrt_ru](#group__cuda__math__intrinsic__double_1gafc3365498bd0f2e3c92cca193b7bd636)(double x)
    

Compute \\(\sqrt{x}\\) in round-up mode.

__device__ double [__dsqrt_rz](#group__cuda__math__intrinsic__double_1gaff77be863e7bde0ce875ba0514a45f7c)(double x)
    

Compute \\(\sqrt{x}\\) in round-towards-zero mode.

__device__ double [__dsub_rd](#group__cuda__math__intrinsic__double_1ga33557b4e6079b7af6479f00bc967ff66)(double x, double y)
    

Subtract two floating-point values in round-down mode.

__device__ double [__dsub_rn](#group__cuda__math__intrinsic__double_1ga169a4b29f0db2ca1eae3f94ab5953d91)(double x, double y)
    

Subtract two floating-point values in round-to-nearest-even mode.

__device__ double [__dsub_ru](#group__cuda__math__intrinsic__double_1gab07fbe4d4228acc3fe9366fb41cd7ea4)(double x, double y)
    

Subtract two floating-point values in round-up mode.

__device__ double [__dsub_rz](#group__cuda__math__intrinsic__double_1gae3db86effaab85115908bfb7f43bce10)(double x, double y)
    

Subtract two floating-point values in round-towards-zero mode.

__device__ double [__fma_rd](#group__cuda__math__intrinsic__double_1gacaa00a783aab97d687005b464bcc4b02)(double x, double y, double z)
    

Compute \\(x \times y + z\\) as a single operation in round-down mode.

__device__ double [__fma_rn](#group__cuda__math__intrinsic__double_1ga57cb4940e3202c0e1302aa71703b7761)(double x, double y, double z)
    

Compute \\(x \times y + z\\) as a single operation in round-to-nearest-even mode.

__device__ double [__fma_ru](#group__cuda__math__intrinsic__double_1gaf4ac27af0c60df25d27c25bcab855a69)(double x, double y, double z)
    

Compute \\(x \times y + z\\) as a single operation in round-up mode.

__device__ double [__fma_rz](#group__cuda__math__intrinsic__double_1ga152174d74c12f3d5af74bbb18dd546bd)(double x, double y, double z)
    

Compute \\(x \times y + z\\) as a single operation in round-towards-zero mode.

##  9.1. Functions[](#functions "Permalink to this headline")

__device__ double __dadd_rd(double x, double y)[](#_CPPv49__dadd_rddd "Permalink to this definition")  

    

Add two floating-point values in round-down mode. 

Adds two floating-point values `x` and `y` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` \+ `y`.

  * __dadd_rd(`x`, `y`) is equivalent to __dadd_rd(`y`, `x`).

  * __dadd_rd(`x`, \\( \pm\infty \\)) returns \\( \pm\infty \\) for finite `x`.

  * __dadd_rd( \\( \pm\infty \\), \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * __dadd_rd( \\( \pm\infty \\), \\( \mp\infty \\)) returns NaN.

  * __dadd_rd( \\( \pm 0 \\), \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __dadd_rd(`x`, `-x`) returns \\( -0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __dadd_rn(double x, double y)[](#_CPPv49__dadd_rndd "Permalink to this definition")  

    

Add two floating-point values in round-to-nearest-even mode. 

Adds two floating-point values `x` and `y` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` \+ `y`.

  * __dadd_rn(`x`, `y`) is equivalent to __dadd_rn(`y`, `x`).

  * __dadd_rn(`x`, \\( \pm\infty \\)) returns \\( \pm\infty \\) for finite `x`.

  * __dadd_rn( \\( \pm\infty \\), \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * __dadd_rn( \\( \pm\infty \\), \\( \mp\infty \\)) returns NaN.

  * __dadd_rn( \\( \pm 0 \\), \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __dadd_rn(`x`, `-x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __dadd_ru(double x, double y)[](#_CPPv49__dadd_rudd "Permalink to this definition")  

    

Add two floating-point values in round-up mode. 

Adds two floating-point values `x` and `y` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` \+ `y`.

  * __dadd_ru(`x`, `y`) is equivalent to __dadd_ru(`y`, `x`).

  * __dadd_ru(`x`, \\( \pm\infty \\)) returns \\( \pm\infty \\) for finite `x`.

  * __dadd_ru( \\( \pm\infty \\), \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * __dadd_ru( \\( \pm\infty \\), \\( \mp\infty \\)) returns NaN.

  * __dadd_ru( \\( \pm 0 \\), \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __dadd_ru(`x`, `-x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __dadd_rz(double x, double y)[](#_CPPv49__dadd_rzdd "Permalink to this definition")  

    

Add two floating-point values in round-towards-zero mode. 

Adds two floating-point values `x` and `y` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` \+ `y`.

  * __dadd_rz(`x`, `y`) is equivalent to __dadd_rz(`y`, `x`).

  * __dadd_rz(`x`, \\( \pm\infty \\)) returns \\( \pm\infty \\) for finite `x`.

  * __dadd_rz( \\( \pm\infty \\), \\( \pm\infty \\)) returns \\( \pm\infty \\).

  * __dadd_rz( \\( \pm\infty \\), \\( \mp\infty \\)) returns NaN.

  * __dadd_rz( \\( \pm 0 \\), \\( \pm 0 \\)) returns \\( \pm 0 \\).

  * __dadd_rz(`x`, `-x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __ddiv_rd(double x, double y)[](#_CPPv49__ddiv_rddd "Permalink to this definition")  

    

Divide two floating-point values in round-down mode. 

Divides two floating-point values `x` by `y` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns `x` / `y`.

  * sign of the quotient `x` / `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __ddiv_rd( \\( \pm 0 \\), \\( \pm 0 \\)) returns NaN.

  * __ddiv_rd( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __ddiv_rd(`x`, \\( \pm\infty \\)) returns \\( 0 \\) of appropriate sign for finite `x`.

  * __ddiv_rd( \\( \pm\infty \\), `y`) returns \\( \infty \\) of appropriate sign for finite `y`.

  * __ddiv_rd(`x`, \\( \pm 0 \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __ddiv_rd( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for `y` \\( \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __ddiv_rn(double x, double y)[](#_CPPv49__ddiv_rndd "Permalink to this definition")  

    

Divide two floating-point values in round-to-nearest-even mode. 

Divides two floating-point values `x` by `y` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns `x` / `y`.

  * sign of the quotient `x` / `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __ddiv_rn( \\( \pm 0 \\), \\( \pm 0 \\)) returns NaN.

  * __ddiv_rn( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __ddiv_rn(`x`, \\( \pm\infty \\)) returns \\( 0 \\) of appropriate sign for finite `x`.

  * __ddiv_rn( \\( \pm\infty \\), `y`) returns \\( \infty \\) of appropriate sign for finite `y`.

  * __ddiv_rn(`x`, \\( \pm 0 \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __ddiv_rn( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for `y` \\( \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __ddiv_ru(double x, double y)[](#_CPPv49__ddiv_rudd "Permalink to this definition")  

    

Divide two floating-point values in round-up mode. 

Divides two floating-point values `x` by `y` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns `x` / `y`.

  * sign of the quotient `x` / `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __ddiv_ru( \\( \pm 0 \\), \\( \pm 0 \\)) returns NaN.

  * __ddiv_ru( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __ddiv_ru(`x`, \\( \pm\infty \\)) returns \\( 0 \\) of appropriate sign for finite `x`.

  * __ddiv_ru( \\( \pm\infty \\), `y`) returns \\( \infty \\) of appropriate sign for finite `y`.

  * __ddiv_ru(`x`, \\( \pm 0 \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __ddiv_ru( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for `y` \\( \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __ddiv_rz(double x, double y)[](#_CPPv49__ddiv_rzdd "Permalink to this definition")  

    

Divide two floating-point values in round-towards-zero mode. 

Divides two floating-point values `x` by `y` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns `x` / `y`.

  * sign of the quotient `x` / `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __ddiv_rz( \\( \pm 0 \\), \\( \pm 0 \\)) returns NaN.

  * __ddiv_rz( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __ddiv_rz(`x`, \\( \pm\infty \\)) returns \\( 0 \\) of appropriate sign for finite `x`.

  * __ddiv_rz( \\( \pm\infty \\), `y`) returns \\( \infty \\) of appropriate sign for finite `y`.

  * __ddiv_rz(`x`, \\( \pm 0 \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __ddiv_rz( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for `y` \\( \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __dmul_rd(double x, double y)[](#_CPPv49__dmul_rddd "Permalink to this definition")  

    

Multiply two floating-point values in round-down mode. 

Multiplies two floating-point values `x` and `y` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` * `y`.

  * sign of the product `x` * `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __dmul_rd(`x`, `y`) is equivalent to __dmul_rd(`y`, `x`).

  * __dmul_rd(`x`, \\( \pm\infty \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __dmul_rd( \\( \pm 0 \\), \\( \pm\infty \\)) returns NaN.

  * __dmul_rd( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for finite `y`.

  * If either argument is NaN, NaN is returned.


__device__ double __dmul_rn(double x, double y)[](#_CPPv49__dmul_rndd "Permalink to this definition")  

    

Multiply two floating-point values in round-to-nearest-even mode. 

Multiplies two floating-point values `x` and `y` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` * `y`.

  * sign of the product `x` * `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __dmul_rn(`x`, `y`) is equivalent to __dmul_rn(`y`, `x`).

  * __dmul_rn(`x`, \\( \pm\infty \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __dmul_rn( \\( \pm 0 \\), \\( \pm\infty \\)) returns NaN.

  * __dmul_rn( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for finite `y`.

  * If either argument is NaN, NaN is returned.


__device__ double __dmul_ru(double x, double y)[](#_CPPv49__dmul_rudd "Permalink to this definition")  

    

Multiply two floating-point values in round-up mode. 

Multiplies two floating-point values `x` and `y` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` * `y`.

  * sign of the product `x` * `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __dmul_ru(`x`, `y`) is equivalent to __dmul_ru(`y`, `x`).

  * __dmul_ru(`x`, \\( \pm\infty \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __dmul_ru( \\( \pm 0 \\), \\( \pm\infty \\)) returns NaN.

  * __dmul_ru( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for finite `y`.

  * If either argument is NaN, NaN is returned.


__device__ double __dmul_rz(double x, double y)[](#_CPPv49__dmul_rzdd "Permalink to this definition")  

    

Multiply two floating-point values in round-towards-zero mode. 

Multiplies two floating-point values `x` and `y` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` * `y`.

  * sign of the product `x` * `y` is XOR of the signs of `x` and `y` when neither inputs nor result are NaN.

  * __dmul_rz(`x`, `y`) is equivalent to __dmul_rz(`y`, `x`).

  * __dmul_rz(`x`, \\( \pm\infty \\)) returns \\( \infty \\) of appropriate sign for `x` \\( \neq 0 \\).

  * __dmul_rz( \\( \pm 0 \\), \\( \pm\infty \\)) returns NaN.

  * __dmul_rz( \\( \pm 0 \\), `y`) returns \\( 0 \\) of appropriate sign for finite `y`.

  * If either argument is NaN, NaN is returned.


__device__ double __drcp_rd(double x)[](#_CPPv49__drcp_rdd "Permalink to this definition")  

    

Compute \\( \frac{1}{x} \\) in round-down mode. 

Compute the reciprocal of `x` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns \\( \frac{1}{x} \\).

__device__ double __drcp_rn(double x)[](#_CPPv49__drcp_rnd "Permalink to this definition")  

    

Compute \\( \frac{1}{x} \\) in round-to-nearest-even mode. 

Compute the reciprocal of `x` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns \\( \frac{1}{x} \\).

__device__ double __drcp_ru(double x)[](#_CPPv49__drcp_rud "Permalink to this definition")  

    

Compute \\( \frac{1}{x} \\) in round-up mode. 

Compute the reciprocal of `x` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns \\( \frac{1}{x} \\).

__device__ double __drcp_rz(double x)[](#_CPPv49__drcp_rzd "Permalink to this definition")  

    

Compute \\( \frac{1}{x} \\) in round-towards-zero mode. 

Compute the reciprocal of `x` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns \\( \frac{1}{x} \\).

__device__ double __dsqrt_rd(double x)[](#_CPPv410__dsqrt_rdd "Permalink to this definition")  

    

Compute \\( \sqrt{x} \\) in round-down mode. 

Compute the square root of `x` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns \\( \sqrt{x} \\).

__device__ double __dsqrt_rn(double x)[](#_CPPv410__dsqrt_rnd "Permalink to this definition")  

    

Compute \\( \sqrt{x} \\) in round-to-nearest-even mode. 

Compute the square root of `x` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns \\( \sqrt{x} \\).

__device__ double __dsqrt_ru(double x)[](#_CPPv410__dsqrt_rud "Permalink to this definition")  

    

Compute \\( \sqrt{x} \\) in round-up mode. 

Compute the square root of `x` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns \\( \sqrt{x} \\).

__device__ double __dsqrt_rz(double x)[](#_CPPv410__dsqrt_rzd "Permalink to this definition")  

    

Compute \\( \sqrt{x} \\) in round-towards-zero mode. 

Compute the square root of `x` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Requires compute capability >= 2.0. 

Returns
    

Returns \\( \sqrt{x} \\).

__device__ double __dsub_rd(double x, double y)[](#_CPPv49__dsub_rddd "Permalink to this definition")  

    

Subtract two floating-point values in round-down mode. 

Subtracts two floating-point values `x` and `y` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` \- `y`.

  * __dsub_rd( \\( \pm\infty \\), `y`) returns \\( \pm\infty \\) for finite `y`.

  * __dsub_rd(`x`, \\( \pm\infty \\)) returns \\( \mp\infty \\) for finite `x`.

  * __dsub_rd( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __dsub_rd( \\( \pm\infty \\), \\( \mp\infty \\)) returns \\( \pm\infty \\).

  * __dsub_rd( \\( \pm 0 \\), \\( \mp 0 \\)) returns \\( \pm 0 \\).

  * __dsub_rd(`x`, `x`) returns \\( -0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __dsub_rn(double x, double y)[](#_CPPv49__dsub_rndd "Permalink to this definition")  

    

Subtract two floating-point values in round-to-nearest-even mode. 

Subtracts two floating-point values `x` and `y` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` \- `y`.

  * __dsub_rn( \\( \pm\infty \\), `y`) returns \\( \pm\infty \\) for finite `y`.

  * __dsub_rn(`x`, \\( \pm\infty \\)) returns \\( \mp\infty \\) for finite `x`.

  * __dsub_rn( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __dsub_rn( \\( \pm\infty \\), \\( \mp\infty \\)) returns \\( \pm\infty \\).

  * __dsub_rn( \\( \pm 0 \\), \\( \mp 0 \\)) returns \\( \pm 0 \\).

  * __dsub_rn(`x`, `x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __dsub_ru(double x, double y)[](#_CPPv49__dsub_rudd "Permalink to this definition")  

    

Subtract two floating-point values in round-up mode. 

Subtracts two floating-point values `x` and `y` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` \- `y`.

  * __dsub_ru( \\( \pm\infty \\), `y`) returns \\( \pm\infty \\) for finite `y`.

  * __dsub_ru(`x`, \\( \pm\infty \\)) returns \\( \mp\infty \\) for finite `x`.

  * __dsub_ru( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __dsub_ru( \\( \pm\infty \\), \\( \mp\infty \\)) returns \\( \pm\infty \\).

  * __dsub_ru( \\( \pm 0 \\), \\( \mp 0 \\)) returns \\( \pm 0 \\).

  * __dsub_ru(`x`, `x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __dsub_rz(double x, double y)[](#_CPPv49__dsub_rzdd "Permalink to this definition")  

    

Subtract two floating-point values in round-towards-zero mode. 

Subtracts two floating-point values `x` and `y` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

This operation will never be merged into a single multiply-add instruction. 

Returns
    

Returns `x` \- `y`.

  * __dsub_rz( \\( \pm\infty \\), `y`) returns \\( \pm\infty \\) for finite `y`.

  * __dsub_rz(`x`, \\( \pm\infty \\)) returns \\( \mp\infty \\) for finite `x`.

  * __dsub_rz( \\( \pm\infty \\), \\( \pm\infty \\)) returns NaN.

  * __dsub_rz( \\( \pm\infty \\), \\( \mp\infty \\)) returns \\( \pm\infty \\).

  * __dsub_rz( \\( \pm 0 \\), \\( \mp 0 \\)) returns \\( \pm 0 \\).

  * __dsub_rz(`x`, `x`) returns \\( +0 \\) for finite `x`, including \\( \pm 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __fma_rd(double x, double y, double z)[](#_CPPv48__fma_rdddd "Permalink to this definition")  

    

Compute \\( x \times y + z \\) as a single operation in round-down mode. 

Computes the value of \\( x \times y + z \\) as a single ternary operation, rounding the result once in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * __fma_rd( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * __fma_rd( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * __fma_rd(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * __fma_rd(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * __fma_rd(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fma_rd(`x`, `y`, \\( \mp 0 \\)) returns \\( -0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fma_rd(`x`, `y`, `z`) returns \\( -0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __fma_rn(double x, double y, double z)[](#_CPPv48__fma_rnddd "Permalink to this definition")  

    

Compute \\( x \times y + z \\) as a single operation in round-to-nearest-even mode. 

Computes the value of \\( x \times y + z \\) as a single ternary operation, rounding the result once in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * __fma_rn( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * __fma_rn( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * __fma_rn(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * __fma_rn(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * __fma_rn(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fma_rn(`x`, `y`, \\( \mp 0 \\)) returns \\( +0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fma_rn(`x`, `y`, `z`) returns \\( +0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __fma_ru(double x, double y, double z)[](#_CPPv48__fma_ruddd "Permalink to this definition")  

    

Compute \\( x \times y + z \\) as a single operation in round-up mode. 

Computes the value of \\( x \times y + z \\) as a single ternary operation, rounding the result once in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * __fma_ru( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * __fma_ru( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * __fma_ru(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * __fma_ru(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * __fma_ru(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fma_ru(`x`, `y`, \\( \mp 0 \\)) returns \\( +0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fma_ru(`x`, `y`, `z`) returns \\( +0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.


__device__ double __fma_rz(double x, double y, double z)[](#_CPPv48__fma_rzddd "Permalink to this definition")  

    

Compute \\( x \times y + z \\) as a single operation in round-towards-zero mode. 

Computes the value of \\( x \times y + z \\) as a single ternary operation, rounding the result once in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns the rounded value of \\( x \times y + z \\) as a single operation.

  * __fma_rz( \\( \pm \infty \\) , \\( \pm 0 \\) , `z`) returns NaN.

  * __fma_rz( \\( \pm 0 \\) , \\( \pm \infty \\) , `z`) returns NaN.

  * __fma_rz(`x`, `y`, \\( -\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( +\infty \\).

  * __fma_rz(`x`, `y`, \\( +\infty \\) ) returns NaN if \\( x \times y \\) is an exact \\( -\infty \\).

  * __fma_rz(`x`, `y`, \\( \pm 0 \\)) returns \\( \pm 0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fma_rz(`x`, `y`, \\( \mp 0 \\)) returns \\( +0 \\) if \\( x \times y \\) is exact \\( \pm 0 \\).

  * __fma_rz(`x`, `y`, `z`) returns \\( +0 \\) if \\( x \times y + z \\) is exactly zero and \\( z \neq 0 \\).

  * If either argument is NaN, NaN is returned.