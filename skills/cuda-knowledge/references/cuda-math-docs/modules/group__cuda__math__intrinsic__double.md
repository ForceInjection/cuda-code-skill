# 9. Double Precision Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__DOUBLE.html


#  9\. Double Precision Intrinsics

This section describes double precision intrinsic functions that are only supported in device code.

To use these functions, you do not need to include any additional header file in your program.

Functions

__device__ double __dadd_rd(double x, double y)


Add two floating-point values in round-down mode.

__device__ double __dadd_rn(double x, double y)


Add two floating-point values in round-to-nearest-even mode.

__device__ double __dadd_ru(double x, double y)


Add two floating-point values in round-up mode.

__device__ double __dadd_rz(double x, double y)


Add two floating-point values in round-towards-zero mode.

__device__ double __ddiv_rd(double x, double y)


Divide two floating-point values in round-down mode.

__device__ double __ddiv_rn(double x, double y)


Divide two floating-point values in round-to-nearest-even mode.

__device__ double __ddiv_ru(double x, double y)


Divide two floating-point values in round-up mode.

__device__ double __ddiv_rz(double x, double y)


Divide two floating-point values in round-towards-zero mode.

__device__ double __dmul_rd(double x, double y)


Multiply two floating-point values in round-down mode.

__device__ double __dmul_rn(double x, double y)


Multiply two floating-point values in round-to-nearest-even mode.

__device__ double __dmul_ru(double x, double y)


Multiply two floating-point values in round-up mode.

__device__ double __dmul_rz(double x, double y)


Multiply two floating-point values in round-towards-zero mode.

__device__ double __drcp_rd(double x)


Compute \\(\frac{1}{x}\\) in round-down mode.

__device__ double __drcp_rn(double x)


Compute \\(\frac{1}{x}\\) in round-to-nearest-even mode.

__device__ double __drcp_ru(double x)


Compute \\(\frac{1}{x}\\) in round-up mode.

__device__ double __drcp_rz(double x)


Compute \\(\frac{1}{x}\\) in round-towards-zero mode.

__device__ double __dsqrt_rd(double x)


Compute \\(\sqrt{x}\\) in round-down mode.

__device__ double __dsqrt_rn(double x)


Compute \\(\sqrt{x}\\) in round-to-nearest-even mode.

__device__ double __dsqrt_ru(double x)


Compute \\(\sqrt{x}\\) in round-up mode.

__device__ double __dsqrt_rz(double x)


Compute \\(\sqrt{x}\\) in round-towards-zero mode.

__device__ double __dsub_rd(double x, double y)


Subtract two floating-point values in round-down mode.

__device__ double __dsub_rn(double x, double y)


Subtract two floating-point values in round-to-nearest-even mode.

__device__ double __dsub_ru(double x, double y)


Subtract two floating-point values in round-up mode.

__device__ double __dsub_rz(double x, double y)


Subtract two floating-point values in round-towards-zero mode.

__device__ double __fma_rd(double x, double y, double z)


Compute \\(x \times y + z\\) as a single operation in round-down mode.

__device__ double __fma_rn(double x, double y, double z)


Compute \\(x \times y + z\\) as a single operation in round-to-nearest-even mode.

__device__ double __fma_ru(double x, double y, double z)


Compute \\(x \times y + z\\) as a single operation in round-up mode.

__device__ double __fma_rz(double x, double y, double z)


Compute \\(x \times y + z\\) as a single operation in round-towards-zero mode.

##  9.1. Functions

__device__ double __dadd_rd(double x, double y)



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


__device__ double __dadd_rn(double x, double y)



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


__device__ double __dadd_ru(double x, double y)



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


__device__ double __dadd_rz(double x, double y)



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


__device__ double __ddiv_rd(double x, double y)



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


__device__ double __ddiv_rn(double x, double y)



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


__device__ double __ddiv_ru(double x, double y)



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


__device__ double __ddiv_rz(double x, double y)



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


__device__ double __dmul_rd(double x, double y)



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


__device__ double __dmul_rn(double x, double y)



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


__device__ double __dmul_ru(double x, double y)



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


__device__ double __dmul_rz(double x, double y)



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


__device__ double __drcp_rd(double x)



Compute \\( \frac{1}{x} \\) in round-down mode.

Compute the reciprocal of `x` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Requires compute capability >= 2.0.

Returns


Returns \\( \frac{1}{x} \\).

__device__ double __drcp_rn(double x)



Compute \\( \frac{1}{x} \\) in round-to-nearest-even mode.

Compute the reciprocal of `x` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Requires compute capability >= 2.0.

Returns


Returns \\( \frac{1}{x} \\).

__device__ double __drcp_ru(double x)



Compute \\( \frac{1}{x} \\) in round-up mode.

Compute the reciprocal of `x` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Requires compute capability >= 2.0.

Returns


Returns \\( \frac{1}{x} \\).

__device__ double __drcp_rz(double x)



Compute \\( \frac{1}{x} \\) in round-towards-zero mode.

Compute the reciprocal of `x` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Requires compute capability >= 2.0.

Returns


Returns \\( \frac{1}{x} \\).

__device__ double __dsqrt_rd(double x)



Compute \\( \sqrt{x} \\) in round-down mode.

Compute the square root of `x` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Requires compute capability >= 2.0.

Returns


Returns \\( \sqrt{x} \\).

__device__ double __dsqrt_rn(double x)



Compute \\( \sqrt{x} \\) in round-to-nearest-even mode.

Compute the square root of `x` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Requires compute capability >= 2.0.

Returns


Returns \\( \sqrt{x} \\).

__device__ double __dsqrt_ru(double x)



Compute \\( \sqrt{x} \\) in round-up mode.

Compute the square root of `x` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Requires compute capability >= 2.0.

Returns


Returns \\( \sqrt{x} \\).

__device__ double __dsqrt_rz(double x)



Compute \\( \sqrt{x} \\) in round-towards-zero mode.

Compute the square root of `x` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.

Note

Requires compute capability >= 2.0.

Returns


Returns \\( \sqrt{x} \\).

__device__ double __dsub_rd(double x, double y)



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


__device__ double __dsub_rn(double x, double y)



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


__device__ double __dsub_ru(double x, double y)



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


__device__ double __dsub_rz(double x, double y)



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


__device__ double __fma_rd(double x, double y, double z)



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


__device__ double __fma_rn(double x, double y, double z)



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


__device__ double __fma_ru(double x, double y, double z)



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


__device__ double __fma_rz(double x, double y, double z)



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