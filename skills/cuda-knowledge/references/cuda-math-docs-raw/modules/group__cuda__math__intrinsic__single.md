# 7. Single Precision Intrinsics

**Source:** [group__CUDA__MATH__INTRINSIC__SINGLE.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)

---

#  7\. Single Precision Intrinsics[](#single-precision-intrinsics "Permalink to this headline")

This section describes single precision intrinsic functions that are only supported in device code. 

To use these functions, you do not need to include any additional header file in your program. 

Functions

__device__ float [__cosf](#group__cuda__math__intrinsic__single_1ga129ff4afc615da9a5886c77713094c32)(float x)
    

Calculate the fast approximate cosine of the input argument.

__device__ float [__exp10f](#group__cuda__math__intrinsic__single_1ga35e0d360ac27e6eb9cf1065b847d46af)(float x)
    

Calculate the fast approximate base 10 exponential of the input argument.

__device__ float [__expf](#group__cuda__math__intrinsic__single_1ga1beeb3ae544cfdde4a0a724ace025aed)(float x)
    

Calculate the fast approximate base \\(e\\) exponential of the input argument.

__device__ float2 [__fadd2_rd](#group__cuda__math__intrinsic__single_1ga4bb3dfb835319ced1298a9c4e3b12fd2)(float2 x, float2 y)
    

Compute vector add operation \\(x + y\\) in round-down mode.

__device__ float2 [__fadd2_rn](#group__cuda__math__intrinsic__single_1ga4e99cb7261ebe2f6a2bbfb61a4a275ad)(float2 x, float2 y)
    

Compute vector add operation \\(x + y\\) in round-to-nearest-even mode.

__device__ float2 [__fadd2_ru](#group__cuda__math__intrinsic__single_1ga773d1a5803e57459fc2005403cbe1f1a)(float2 x, float2 y)
    

Compute vector add operation \\(x + y\\) in round-up mode.

__device__ float2 [__fadd2_rz](#group__cuda__math__intrinsic__single_1gaeb684f4aed8aaaeec4a9b0152d01032b)(float2 x, float2 y)
    

Compute vector add operation \\(x + y\\) in round-towards-zero mode.

__device__ float [__fadd_rd](#group__cuda__math__intrinsic__single_1gaa8255ea2b671a8488813d9d3527e661a)(float x, float y)
    

Add two floating-point values in round-down mode.

__device__ float [__fadd_rn](#group__cuda__math__intrinsic__single_1ga34f8d82135041b06099d7709ff0bbe4e)(float x, float y)
    

Add two floating-point values in round-to-nearest-even mode.

__device__ float [__fadd_ru](#group__cuda__math__intrinsic__single_1ga3ba9e088a1f798c2542fe7d69736ac7e)(float x, float y)
    

Add two floating-point values in round-up mode.

__device__ float [__fadd_rz](#group__cuda__math__intrinsic__single_1ga8d108dee04b1aa0681fcdfb8f661dc9e)(float x, float y)
    

Add two floating-point values in round-towards-zero mode.

__device__ float [__fdiv_rd](#group__cuda__math__intrinsic__single_1gaa548a4045aee3b2dce24bccb26f90f82)(float x, float y)
    

Divide two floating-point values in round-down mode.

__device__ float [__fdiv_rn](#group__cuda__math__intrinsic__single_1ga1bd038de4c10eec819e00991fb5acf7e)(float x, float y)
    

Divide two floating-point values in round-to-nearest-even mode.

__device__ float [__fdiv_ru](#group__cuda__math__intrinsic__single_1ga8089bb061fd076bdb5b08566b4d0e0c2)(float x, float y)
    

Divide two floating-point values in round-up mode.

__device__ float [__fdiv_rz](#group__cuda__math__intrinsic__single_1gab1465a09b2a88d1c2048d0b0f612fb5c)(float x, float y)
    

Divide two floating-point values in round-towards-zero mode.

__device__ float [__fdividef](#group__cuda__math__intrinsic__single_1gac996beec34f94f6376d0674a6860e107)(float x, float y)
    

Calculate the fast approximate division of the input arguments.

__device__ float2 [__ffma2_rd](#group__cuda__math__intrinsic__single_1gaa8ce278c8b8a38261e47110261dce74f)(float2 x, float2 y, float2 z)
    

Compute vector fused multiply-add operation \\(x \times y + z\\) in round-down mode.

__device__ float2 [__ffma2_rn](#group__cuda__math__intrinsic__single_1ga42bf1176894d6e32a135b0f03450635e)(float2 x, float2 y, float2 z)
    

Compute vector fused multiply-add operation \\(x \times y + z\\) in round-to-nearest-even mode.

__device__ float2 [__ffma2_ru](#group__cuda__math__intrinsic__single_1ga4aab27fd257769c18a6393dd49617272)(float2 x, float2 y, float2 z)
    

Compute vector fused multiply-add operation \\(x \times y + z\\) in round-up mode.

__device__ float2 [__ffma2_rz](#group__cuda__math__intrinsic__single_1ga89996aff31e027c46a8ca0e98e751cd3)(float2 x, float2 y, float2 z)
    

Compute vector fused multiply-add operation \\(x \times y + z\\) in round-towards-zero mode.

__device__ float [__fmaf_ieee_rd](#group__cuda__math__intrinsic__single_1ga5ec4876d85bf5331ac63fa98f3fb57b4)(float x, float y, float z)
    

Compute fused multiply-add operation in round-down mode, ignore `-ftz=true` compiler flag.

__device__ float [__fmaf_ieee_rn](#group__cuda__math__intrinsic__single_1ga6535eebe86aba96a47c3ff4632640d99)(float x, float y, float z)
    

Compute fused multiply-add operation in round-to-nearest-even mode, ignore `-ftz=true` compiler flag.

__device__ float [__fmaf_ieee_ru](#group__cuda__math__intrinsic__single_1ga6c50feff625ea8d50ccf7fcfa30383eb)(float x, float y, float z)
    

Compute fused multiply-add operation in round-up mode, ignore `-ftz=true` compiler flag.

__device__ float [__fmaf_ieee_rz](#group__cuda__math__intrinsic__single_1gac39a4b9e57049f913213edcb0d855138)(float x, float y, float z)
    

Compute fused multiply-add operation in round-towards-zero mode, ignore `-ftz=true` compiler flag.

__device__ float [__fmaf_rd](#group__cuda__math__intrinsic__single_1ga5b12d5103d17eed423f1db706a9c80be)(float x, float y, float z)
    

Compute \\(x \times y + z\\) as a single operation, in round-down mode.

__device__ float [__fmaf_rn](#group__cuda__math__intrinsic__single_1ga77b36635a8fbdc96a7e08e201d589316)(float x, float y, float z)
    

Compute \\(x \times y + z\\) as a single operation, in round-to-nearest-even mode.

__device__ float [__fmaf_ru](#group__cuda__math__intrinsic__single_1gafe855a453ea92a580b79ad8a7e72bc49)(float x, float y, float z)
    

Compute \\(x \times y + z\\) as a single operation, in round-up mode.

__device__ float [__fmaf_rz](#group__cuda__math__intrinsic__single_1ga42a5f6c99064834ad50b2bfa7aa77731)(float x, float y, float z)
    

Compute \\(x \times y + z\\) as a single operation, in round-towards-zero mode.

__device__ float2 [__fmul2_rd](#group__cuda__math__intrinsic__single_1gaa7746418cf19c6e2672f37c35fd72f96)(float2 x, float2 y)
    

Compute vector multiply operation \\(x \times y\\) in round-down mode.

__device__ float2 [__fmul2_rn](#group__cuda__math__intrinsic__single_1gad4626f5b6503314f1f954ccf3714ab8b)(float2 x, float2 y)
    

Compute vector multiply operation \\(x \times y\\) in round-to-nearest-even mode.

__device__ float2 [__fmul2_ru](#group__cuda__math__intrinsic__single_1ga7629d4315bd4849529576dd1b018eaa3)(float2 x, float2 y)
    

Compute vector multiply operation \\(x \times y\\) in round-up mode.

__device__ float2 [__fmul2_rz](#group__cuda__math__intrinsic__single_1ga078c450ff284a1f36da2f21cdbbcc679)(float2 x, float2 y)
    

Compute vector multiply operation \\(x \times y\\) in round-towards-zero mode.

__device__ float [__fmul_rd](#group__cuda__math__intrinsic__single_1ga28d56d8747ca7960860cd9c67cd3fed6)(float x, float y)
    

Multiply two floating-point values in round-down mode.

__device__ float [__fmul_rn](#group__cuda__math__intrinsic__single_1ga4b9d2d5cb295c1442b00e6eff5248b97)(float x, float y)
    

Multiply two floating-point values in round-to-nearest-even mode.

__device__ float [__fmul_ru](#group__cuda__math__intrinsic__single_1gacd2f8b720306266f6e814345d4cf1b93)(float x, float y)
    

Multiply two floating-point values in round-up mode.

__device__ float [__fmul_rz](#group__cuda__math__intrinsic__single_1gaff448e40e1e71eb620159a40e5e62705)(float x, float y)
    

Multiply two floating-point values in round-towards-zero mode.

__device__ float [__frcp_rd](#group__cuda__math__intrinsic__single_1ga4f51ca3e41e5f9369bed0bf0c5f42971)(float x)
    

Compute \\(\frac{1}{x}\\) in round-down mode.

__device__ float [__frcp_rn](#group__cuda__math__intrinsic__single_1gaba455801af8ac9af405a5d37ef2f077b)(float x)
    

Compute \\(\frac{1}{x}\\) in round-to-nearest-even mode.

__device__ float [__frcp_ru](#group__cuda__math__intrinsic__single_1gad5b3289e40e510c2067c8f7a426aa884)(float x)
    

Compute \\(\frac{1}{x}\\) in round-up mode.

__device__ float [__frcp_rz](#group__cuda__math__intrinsic__single_1ga64032913321c328ae72111161af32268)(float x)
    

Compute \\(\frac{1}{x}\\) in round-towards-zero mode.

__device__ float [__frsqrt_rn](#group__cuda__math__intrinsic__single_1ga71ee45580cbeeea206297f0112aff42c)(float x)
    

Compute \\(1/\sqrt{x}\\) in round-to-nearest-even mode.

__device__ float [__fsqrt_rd](#group__cuda__math__intrinsic__single_1ga59566bdd0638a5b249dbda757f2bb06b)(float x)
    

Compute \\(\sqrt{x}\\) in round-down mode.

__device__ float [__fsqrt_rn](#group__cuda__math__intrinsic__single_1gaf021e85b5e9de141a0fc2ff6fbe85875)(float x)
    

Compute \\(\sqrt{x}\\) in round-to-nearest-even mode.

__device__ float [__fsqrt_ru](#group__cuda__math__intrinsic__single_1gab429e39b1790b4dfae0d0c4926f53fe2)(float x)
    

Compute \\(\sqrt{x}\\) in round-up mode.

__device__ float [__fsqrt_rz](#group__cuda__math__intrinsic__single_1ga094bf489bf492287424b1080569189f1)(float x)
    

Compute \\(\sqrt{x}\\) in round-towards-zero mode.

__device__ float [__fsub_rd](#group__cuda__math__intrinsic__single_1ga78502e666a0e6e7690230e118403df54)(float x, float y)
    

Subtract two floating-point values in round-down mode.

__device__ float [__fsub_rn](#group__cuda__math__intrinsic__single_1gae75e42e6637c178cd3c86a6e3774f7cb)(float x, float y)
    

Subtract two floating-point values in round-to-nearest-even mode.

__device__ float [__fsub_ru](#group__cuda__math__intrinsic__single_1ga4adc718465695ee57990318d5a650b1c)(float x, float y)
    

Subtract two floating-point values in round-up mode.

__device__ float [__fsub_rz](#group__cuda__math__intrinsic__single_1gac900725c2921068c2ad4f53039b6bacf)(float x, float y)
    

Subtract two floating-point values in round-towards-zero mode.

__device__ float [__log10f](#group__cuda__math__intrinsic__single_1ga2b7358a27b8231b592da81ff3143b9a8)(float x)
    

Calculate the fast approximate base 10 logarithm of the input argument.

__device__ float [__log2f](#group__cuda__math__intrinsic__single_1gafcc053f9040d50975aab00e44e7c6093)(float x)
    

Calculate the fast approximate base 2 logarithm of the input argument.

__device__ float [__logf](#group__cuda__math__intrinsic__single_1gaed5cef656578096892f104a27d5287c4)(float x)
    

Calculate the fast approximate base \\(e\\) logarithm of the input argument.

__device__ float [__powf](#group__cuda__math__intrinsic__single_1ga2c2b295816185f6ce2423471df529974)(float x, float y)
    

Calculate the fast approximate of \\(x^y\\) .

__device__ float [__saturatef](#group__cuda__math__intrinsic__single_1ga2c84f08e0db7117a14509d21c3aec04e)(float x)
    

Clamp the input argument to [+0.0, 1.0].

__device__ void [__sincosf](#group__cuda__math__intrinsic__single_1ga96089d6195f9befde96e14afafd931fb)(float x, float *sptr, float *cptr)
    

Calculate the fast approximate of sine and cosine of the first input argument.

__device__ float [__sinf](#group__cuda__math__intrinsic__single_1gafa0ea4b2cee94521792ead0deb03addb)(float x)
    

Calculate the fast approximate sine of the input argument.

__device__ float [__tanf](#group__cuda__math__intrinsic__single_1ga4fc8b7c67526a0195b9cb47287b5c121)(float x)
    

Calculate the fast approximate tangent of the input argument.

__device__ float [__tanhf](#group__cuda__math__intrinsic__single_1ga8d2d47cf3422dc1680659de61f8693a7)(float x)
    

Calculate the fast approximate hyperbolic tangent of the input argument.

##  7.1. Functions[](#functions "Permalink to this headline")

__device__ float __cosf(float x)[](#_CPPv46__cosff "Permalink to this definition")  

    

Calculate the fast approximate cosine of the input argument. 

Calculate the fast approximate cosine of the input argument `x`, measured in radians.

See also

[cosf()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1ga20858ddd8f75a2c8332bdecd536057bf) for further special case behavior specification. 

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns the approximate cosine of `x`.

__device__ float __exp10f(float x)[](#_CPPv48__exp10ff "Permalink to this definition")  

    

Calculate the fast approximate base 10 exponential of the input argument. 

Calculate the fast approximate base 10 exponential of the input argument `x`, \\( 10^x \\).

See also

[exp10f()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1ga60f1de4fe78a907d915a52be29a799e7) for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns an approximation to \\( 10^x \\). 

__device__ float __expf(float x)[](#_CPPv46__expff "Permalink to this definition")  

    

Calculate the fast approximate base \\( e \\) exponential of the input argument. 

Calculate the fast approximate base \\( e \\) exponential of the input argument `x`, \\( e^x \\).

See also

[expf()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1gae2d7656fe00f9e750c6f3bde8cc0dca6) for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns an approximation to \\( e^x \\). 

__device__ float2 __fadd2_rd(float2 x, float2 y)[](#_CPPv410__fadd2_rd6float26float2 "Permalink to this definition")  

    

Compute vector add operation \\( x + y \\) in round-down mode. 

Numeric behavior per component is the same as [__fadd_rd()](#group__cuda__math__intrinsic__single_1gaa8255ea2b671a8488813d9d3527e661a).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float2 __fadd2_rn(float2 x, float2 y)[](#_CPPv410__fadd2_rn6float26float2 "Permalink to this definition")  

    

Compute vector add operation \\( x + y \\) in round-to-nearest-even mode. 

Numeric behavior per component is the same as [__fadd_rn()](#group__cuda__math__intrinsic__single_1ga34f8d82135041b06099d7709ff0bbe4e).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float2 __fadd2_ru(float2 x, float2 y)[](#_CPPv410__fadd2_ru6float26float2 "Permalink to this definition")  

    

Compute vector add operation \\( x + y \\) in round-up mode. 

Numeric behavior per component is the same as [__fadd_ru()](#group__cuda__math__intrinsic__single_1ga3ba9e088a1f798c2542fe7d69736ac7e).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float2 __fadd2_rz(float2 x, float2 y)[](#_CPPv410__fadd2_rz6float26float2 "Permalink to this definition")  

    

Compute vector add operation \\( x + y \\) in round-towards-zero mode. 

Numeric behavior per component is the same as [__fadd_rz()](#group__cuda__math__intrinsic__single_1ga8d108dee04b1aa0681fcdfb8f661dc9e).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float __fadd_rd(float x, float y)[](#_CPPv49__fadd_rdff "Permalink to this definition")  

    

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


__device__ float __fadd_rn(float x, float y)[](#_CPPv49__fadd_rnff "Permalink to this definition")  

    

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


__device__ float __fadd_ru(float x, float y)[](#_CPPv49__fadd_ruff "Permalink to this definition")  

    

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


__device__ float __fadd_rz(float x, float y)[](#_CPPv49__fadd_rzff "Permalink to this definition")  

    

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


__device__ float __fdiv_rd(float x, float y)[](#_CPPv49__fdiv_rdff "Permalink to this definition")  

    

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


__device__ float __fdiv_rn(float x, float y)[](#_CPPv49__fdiv_rnff "Permalink to this definition")  

    

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


__device__ float __fdiv_ru(float x, float y)[](#_CPPv49__fdiv_ruff "Permalink to this definition")  

    

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


__device__ float __fdiv_rz(float x, float y)[](#_CPPv49__fdiv_rzff "Permalink to this definition")  

    

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


__device__ float __fdividef(float x, float y)[](#_CPPv410__fdividefff "Permalink to this definition")  

    

Calculate the fast approximate division of the input arguments. 

Calculate the fast approximate division of `x` by `y`.

See also

[__fdiv_rn()](#group__cuda__math__intrinsic__single_1ga1bd038de4c10eec819e00991fb5acf7e) for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns `x` / `y`.

  * __fdividef( \\( \infty \\) , `y`) returns NaN for \\( 2^{126} < |y| < 2^{128} \\).

  * __fdividef(`x`, `y`) returns 0 for \\( 2^{126} < |y| < 2^{128} \\) and finite \\( x \\). 


__device__ float2 __ffma2_rd(float2 x, float2 y, float2 z)[](#_CPPv410__ffma2_rd6float26float26float2 "Permalink to this definition")  

    

Compute vector fused multiply-add operation \\( x \times y + z \\) in round-down mode. 

Numeric behavior per component is the same as [__fmaf_rd()](#group__cuda__math__intrinsic__single_1ga5b12d5103d17eed423f1db706a9c80be).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float2 __ffma2_rn(float2 x, float2 y, float2 z)[](#_CPPv410__ffma2_rn6float26float26float2 "Permalink to this definition")  

    

Compute vector fused multiply-add operation \\( x \times y + z \\) in round-to-nearest-even mode. 

Numeric behavior per component is the same as [__fmaf_rn()](#group__cuda__math__intrinsic__single_1ga77b36635a8fbdc96a7e08e201d589316).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float2 __ffma2_ru(float2 x, float2 y, float2 z)[](#_CPPv410__ffma2_ru6float26float26float2 "Permalink to this definition")  

    

Compute vector fused multiply-add operation \\( x \times y + z \\) in round-up mode. 

Numeric behavior per component is the same as [__fmaf_ru()](#group__cuda__math__intrinsic__single_1gafe855a453ea92a580b79ad8a7e72bc49).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float2 __ffma2_rz(float2 x, float2 y, float2 z)[](#_CPPv410__ffma2_rz6float26float26float2 "Permalink to this definition")  

    

Compute vector fused multiply-add operation \\( x \times y + z \\) in round-towards-zero mode. 

Numeric behavior per component is the same as [__fmaf_rz()](#group__cuda__math__intrinsic__single_1ga42a5f6c99064834ad50b2bfa7aa77731).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float __fmaf_ieee_rd(float x, float y, float z)[](#_CPPv414__fmaf_ieee_rdfff "Permalink to this definition")  

    

Compute fused multiply-add operation in round-down mode, ignore `-ftz=true` compiler flag. 

Behavior is the same as [__fmaf_rd](#group__cuda__math__intrinsic__single_1ga5b12d5103d17eed423f1db706a9c80be)(`x`, `y`, `z`), the difference is in handling denormalized inputs and outputs: `-ftz` compiler flag has no effect. 

__device__ float __fmaf_ieee_rn(float x, float y, float z)[](#_CPPv414__fmaf_ieee_rnfff "Permalink to this definition")  

    

Compute fused multiply-add operation in round-to-nearest-even mode, ignore `-ftz=true` compiler flag. 

Behavior is the same as [__fmaf_rn](#group__cuda__math__intrinsic__single_1ga77b36635a8fbdc96a7e08e201d589316)(`x`, `y`, `z`), the difference is in handling denormalized inputs and outputs: `-ftz` compiler flag has no effect. 

__device__ float __fmaf_ieee_ru(float x, float y, float z)[](#_CPPv414__fmaf_ieee_rufff "Permalink to this definition")  

    

Compute fused multiply-add operation in round-up mode, ignore `-ftz=true` compiler flag. 

Behavior is the same as [__fmaf_ru](#group__cuda__math__intrinsic__single_1gafe855a453ea92a580b79ad8a7e72bc49)(`x`, `y`, `z`), the difference is in handling denormalized inputs and outputs: `-ftz` compiler flag has no effect. 

__device__ float __fmaf_ieee_rz(float x, float y, float z)[](#_CPPv414__fmaf_ieee_rzfff "Permalink to this definition")  

    

Compute fused multiply-add operation in round-towards-zero mode, ignore `-ftz=true` compiler flag. 

Behavior is the same as [__fmaf_rz](#group__cuda__math__intrinsic__single_1ga42a5f6c99064834ad50b2bfa7aa77731)(`x`, `y`, `z`), the difference is in handling denormalized inputs and outputs: `-ftz` compiler flag has no effect. 

__device__ float __fmaf_rd(float x, float y, float z)[](#_CPPv49__fmaf_rdfff "Permalink to this definition")  

    

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


__device__ float __fmaf_rn(float x, float y, float z)[](#_CPPv49__fmaf_rnfff "Permalink to this definition")  

    

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


__device__ float __fmaf_ru(float x, float y, float z)[](#_CPPv49__fmaf_rufff "Permalink to this definition")  

    

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


__device__ float __fmaf_rz(float x, float y, float z)[](#_CPPv49__fmaf_rzfff "Permalink to this definition")  

    

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


__device__ float2 __fmul2_rd(float2 x, float2 y)[](#_CPPv410__fmul2_rd6float26float2 "Permalink to this definition")  

    

Compute vector multiply operation \\( x \times y \\) in round-down mode. 

Numeric behavior per component is the same as [__fmul_rd()](#group__cuda__math__intrinsic__single_1ga28d56d8747ca7960860cd9c67cd3fed6).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float2 __fmul2_rn(float2 x, float2 y)[](#_CPPv410__fmul2_rn6float26float2 "Permalink to this definition")  

    

Compute vector multiply operation \\( x \times y \\) in round-to-nearest-even mode. 

Numeric behavior per component is the same as [__fmul_rn()](#group__cuda__math__intrinsic__single_1ga4b9d2d5cb295c1442b00e6eff5248b97).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float2 __fmul2_ru(float2 x, float2 y)[](#_CPPv410__fmul2_ru6float26float2 "Permalink to this definition")  

    

Compute vector multiply operation \\( x \times y \\) in round-up mode. 

Numeric behavior per component is the same as [__fmul_ru()](#group__cuda__math__intrinsic__single_1gacd2f8b720306266f6e814345d4cf1b93).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float2 __fmul2_rz(float2 x, float2 y)[](#_CPPv410__fmul2_rz6float26float2 "Permalink to this definition")  

    

Compute vector multiply operation \\( x \times y \\) in round-towards-zero mode. 

Numeric behavior per component is the same as [__fmul_rz()](#group__cuda__math__intrinsic__single_1gaff448e40e1e71eb620159a40e5e62705).

Note

This intrinsic requires compute capability >= 10.0. 

Note

The vector variants may not always provide better performance. 

__device__ float __fmul_rd(float x, float y)[](#_CPPv49__fmul_rdff "Permalink to this definition")  

    

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


__device__ float __fmul_rn(float x, float y)[](#_CPPv49__fmul_rnff "Permalink to this definition")  

    

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


__device__ float __fmul_ru(float x, float y)[](#_CPPv49__fmul_ruff "Permalink to this definition")  

    

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


__device__ float __fmul_rz(float x, float y)[](#_CPPv49__fmul_rzff "Permalink to this definition")  

    

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


__device__ float __frcp_rd(float x)[](#_CPPv49__frcp_rdf "Permalink to this definition")  

    

Compute \\( \frac{1}{x} \\) in round-down mode. 

Compute the reciprocal of `x` in round-down (to negative infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns \\( \frac{1}{x} \\).

  * __frcp_rd( \\( \pm 0 \\)) returns \\( \pm\infty \\).

  * __frcp_rd( \\( \pm\infty \\)) returns \\( \pm 0 \\).

  * __frcp_rd(NaN) returns NaN.


__device__ float __frcp_rn(float x)[](#_CPPv49__frcp_rnf "Permalink to this definition")  

    

Compute \\( \frac{1}{x} \\) in round-to-nearest-even mode. 

Compute the reciprocal of `x` in round-to-nearest-even mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns \\( \frac{1}{x} \\).

  * __frcp_rn( \\( \pm 0 \\)) returns \\( \pm\infty \\).

  * __frcp_rn( \\( \pm\infty \\)) returns \\( \pm 0 \\).

  * __frcp_rn(NaN) returns NaN.


__device__ float __frcp_ru(float x)[](#_CPPv49__frcp_ruf "Permalink to this definition")  

    

Compute \\( \frac{1}{x} \\) in round-up mode. 

Compute the reciprocal of `x` in round-up (to positive infinity) mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns \\( \frac{1}{x} \\).

  * __frcp_ru( \\( \pm 0 \\)) returns \\( \pm\infty \\).

  * __frcp_ru( \\( \pm\infty \\)) returns \\( \pm 0 \\).

  * __frcp_ru(NaN) returns NaN.


__device__ float __frcp_rz(float x)[](#_CPPv49__frcp_rzf "Permalink to this definition")  

    

Compute \\( \frac{1}{x} \\) in round-towards-zero mode. 

Compute the reciprocal of `x` in round-towards-zero mode.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns \\( \frac{1}{x} \\).

  * __frcp_rz( \\( \pm 0 \\)) returns \\( \pm\infty \\).

  * __frcp_rz( \\( \pm\infty \\)) returns \\( \pm 0 \\).

  * __frcp_rz(NaN) returns NaN.


__device__ float __frsqrt_rn(float x)[](#_CPPv411__frsqrt_rnf "Permalink to this definition")  

    

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


__device__ float __fsqrt_rd(float x)[](#_CPPv410__fsqrt_rdf "Permalink to this definition")  

    

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


__device__ float __fsqrt_rn(float x)[](#_CPPv410__fsqrt_rnf "Permalink to this definition")  

    

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


__device__ float __fsqrt_ru(float x)[](#_CPPv410__fsqrt_ruf "Permalink to this definition")  

    

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


__device__ float __fsqrt_rz(float x)[](#_CPPv410__fsqrt_rzf "Permalink to this definition")  

    

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


__device__ float __fsub_rd(float x, float y)[](#_CPPv49__fsub_rdff "Permalink to this definition")  

    

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


__device__ float __fsub_rn(float x, float y)[](#_CPPv49__fsub_rnff "Permalink to this definition")  

    

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


__device__ float __fsub_ru(float x, float y)[](#_CPPv49__fsub_ruff "Permalink to this definition")  

    

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


__device__ float __fsub_rz(float x, float y)[](#_CPPv49__fsub_rzff "Permalink to this definition")  

    

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


__device__ float __log10f(float x)[](#_CPPv48__log10ff "Permalink to this definition")  

    

Calculate the fast approximate base 10 logarithm of the input argument. 

Calculate the fast approximate base 10 logarithm of the input argument `x`.

See also

[log10f()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1gab49e218cf742a0eb08e5516dd5160585) for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns an approximation to \\( \log_{10}(x) \\). 

__device__ float __log2f(float x)[](#_CPPv47__log2ff "Permalink to this definition")  

    

Calculate the fast approximate base 2 logarithm of the input argument. 

Calculate the fast approximate base 2 logarithm of the input argument `x`.

See also

[log2f()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1gafc9ae1bd4ebb4cd9533a50f1bf486f08) for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns an approximation to \\( \log_2(x) \\). 

__device__ float __logf(float x)[](#_CPPv46__logff "Permalink to this definition")  

    

Calculate the fast approximate base \\( e \\) logarithm of the input argument. 

Calculate the fast approximate base \\( e \\) logarithm of the input argument `x`.

See also

[logf()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1gacdaf041c4071f63cba0e51658b89ffa4) for further special case behavior specification.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns an approximation to \\( \log_e(x) \\). 

__device__ float __powf(float x, float y)[](#_CPPv46__powfff "Permalink to this definition")  

    

Calculate the fast approximate of \\( x^y \\). 

Calculate the fast approximate of `x`, the first input argument, raised to the power of `y`, the second input argument, \\( x^y \\).

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns an approximation to \\( x^y \\).

__device__ float __saturatef(float x)[](#_CPPv411__saturateff "Permalink to this definition")  

    

Clamp the input argument to [+0.0, 1.0]. 

Clamp the input argument `x` to be within the interval [+0.0, 1.0]. 

Returns
    

  * __saturatef(`x`) returns +0 if \\( x \le 0 \\).

  * __saturatef(`x`) returns 1 if \\( x \ge 1 \\).

  * __saturatef(`x`) returns `x` if \\( 0 < x < 1 \\).

  * __saturatef(NaN) returns +0. 


__device__ void __sincosf(float x, float *sptr, float *cptr)[](#_CPPv49__sincosffPfPf "Permalink to this definition")  

    

Calculate the fast approximate of sine and cosine of the first input argument. 

Calculate the fast approximate of sine and cosine of the first input argument `x` (measured in radians). The results for sine and cosine are written into the second argument, `sptr`, and, respectively, third argument, `cptr`.

See also

[__sinf()](#group__cuda__math__intrinsic__single_1gafa0ea4b2cee94521792ead0deb03addb) and [__cosf()](#group__cuda__math__intrinsic__single_1ga129ff4afc615da9a5886c77713094c32). 

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Denorm input/output is flushed to sign preserving 0.0. 

__device__ float __sinf(float x)[](#_CPPv46__sinff "Permalink to this definition")  

    

Calculate the fast approximate sine of the input argument. 

Calculate the fast approximate sine of the input argument `x`, measured in radians.

See also

[sinf()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1ga4677d53159664972c54bb697b9c1bace) for further special case behavior specification. 

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

Output in the denormal range is flushed to sign preserving 0.0. 

Returns
    

Returns the approximate sine of `x`.

__device__ float __tanf(float x)[](#_CPPv46__tanff "Permalink to this definition")  

    

Calculate the fast approximate tangent of the input argument. 

Calculate the fast approximate tangent of the input argument `x`, measured in radians.

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Note

The result is computed as the fast divide of [__sinf()](#group__cuda__math__intrinsic__single_1gafa0ea4b2cee94521792ead0deb03addb) by [__cosf()](#group__cuda__math__intrinsic__single_1ga129ff4afc615da9a5886c77713094c32). Denormal output is flushed to sign-preserving 0.0. 

Returns
    

Returns the approximate tangent of `x`.

__device__ float __tanhf(float x)[](#_CPPv47__tanhff "Permalink to this definition")  

    

Calculate the fast approximate hyperbolic tangent of the input argument. 

Calculate the fast approximate hyperbolic tangent of the input argument `x`, measured in radians.

See also

[tanhf()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1ga7d925743801795775ca98ae83d4ba6e6) for further special case behavior specification. 

Note

For accuracy information, see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section. 

Returns
    

Returns the approximate hyperbolic tangent of `x`.