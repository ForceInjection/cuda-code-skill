# 11. Type Casting Intrinsics

**Source:** [group__CUDA__MATH__INTRINSIC__CAST.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html)

---

#  11\. Type Casting Intrinsics[](#type-casting-intrinsics "Permalink to this headline")

This section describes type casting intrinsic functions that are only supported in device code. 

To use these functions, you do not need to include any additional header file in your program. 

Functions

__device__ float [__double2float_rd](#group__cuda__math__intrinsic__cast_1ga64f62a06692dc6f683a2da3a25bd39de)(double x)
    

Convert a double to a float in round-down mode.

__device__ float [__double2float_rn](#group__cuda__math__intrinsic__cast_1ga44785801ae1807a2bc0fde3f4f9d6aae)(double x)
    

Convert a double to a float in round-to-nearest-even mode.

__device__ float [__double2float_ru](#group__cuda__math__intrinsic__cast_1ga4a5f6ab65413bdfb6080003392a2954b)(double x)
    

Convert a double to a float in round-up mode.

__device__ float [__double2float_rz](#group__cuda__math__intrinsic__cast_1gaa2bfeb44d4150a05d590a3c330511990)(double x)
    

Convert a double to a float in round-towards-zero mode.

__device__ int [__double2hiint](#group__cuda__math__intrinsic__cast_1gae8bbbe2efda5aa6f087c94e1e56463e8)(double x)
    

Reinterpret high 32 bits in a double as a signed integer.

__device__ int [__double2int_rd](#group__cuda__math__intrinsic__cast_1gaf91bc76a892f92d470595c3d95703531)(double x)
    

Convert a double to a signed int in round-down mode.

__device__ int [__double2int_rn](#group__cuda__math__intrinsic__cast_1ga57eccdd6189ac5527b52eeff42daaae5)(double x)
    

Convert a double to a signed int in round-to-nearest-even mode.

__device__ int [__double2int_ru](#group__cuda__math__intrinsic__cast_1gae3ad8b7aba740b05097cb47c70f2387a)(double x)
    

Convert a double to a signed int in round-up mode.

__device__ int [__double2int_rz](#group__cuda__math__intrinsic__cast_1gaf5ba8d1316a96fab5f34f0cfffd4fb07)(double x)
    

Convert a double to a signed int in round-towards-zero mode.

__device__ long long int [__double2ll_rd](#group__cuda__math__intrinsic__cast_1ga1b9a396549605e99ad6f0639d8efd0cc)(double x)
    

Convert a double to a signed 64-bit int in round-down mode.

__device__ long long int [__double2ll_rn](#group__cuda__math__intrinsic__cast_1gaff7010f874f73a459f0055dfa31d3531)(double x)
    

Convert a double to a signed 64-bit int in round-to-nearest-even mode.

__device__ long long int [__double2ll_ru](#group__cuda__math__intrinsic__cast_1ga410bc39567ec1105a1ef4658bdfa4c4f)(double x)
    

Convert a double to a signed 64-bit int in round-up mode.

__device__ long long int [__double2ll_rz](#group__cuda__math__intrinsic__cast_1ga1e10c3df99035e81bf902589410a8798)(double x)
    

Convert a double to a signed 64-bit int in round-towards-zero mode.

__device__ int [__double2loint](#group__cuda__math__intrinsic__cast_1ga8f685cb937cc96c273d197da7b1633f1)(double x)
    

Reinterpret low 32 bits in a double as a signed integer.

__device__ unsigned int [__double2uint_rd](#group__cuda__math__intrinsic__cast_1ga38ca3e31db652a84d318fe966231e60d)(double x)
    

Convert a double to an unsigned int in round-down mode.

__device__ unsigned int [__double2uint_rn](#group__cuda__math__intrinsic__cast_1ga551e5828292865856f306fb5b3eeb9e7)(double x)
    

Convert a double to an unsigned int in round-to-nearest-even mode.

__device__ unsigned int [__double2uint_ru](#group__cuda__math__intrinsic__cast_1ga872081e638057748cc55697cfbde0de7)(double x)
    

Convert a double to an unsigned int in round-up mode.

__device__ unsigned int [__double2uint_rz](#group__cuda__math__intrinsic__cast_1ga9b6b4afa2ad047c3d10b1a6ec4b540f2)(double x)
    

Convert a double to an unsigned int in round-towards-zero mode.

__device__ unsigned long long int [__double2ull_rd](#group__cuda__math__intrinsic__cast_1ga4041c7521153868536f5081b43aa6e94)(double x)
    

Convert a double to an unsigned 64-bit int in round-down mode.

__device__ unsigned long long int [__double2ull_rn](#group__cuda__math__intrinsic__cast_1ga33d309bee17d6e98fad0a3db0077e1d8)(double x)
    

Convert a double to an unsigned 64-bit int in round-to-nearest-even mode.

__device__ unsigned long long int [__double2ull_ru](#group__cuda__math__intrinsic__cast_1ga0b673695ff1ea93070b8795747d0bb0a)(double x)
    

Convert a double to an unsigned 64-bit int in round-up mode.

__device__ unsigned long long int [__double2ull_rz](#group__cuda__math__intrinsic__cast_1gaa228e8924a18963346001d7818665b3b)(double x)
    

Convert a double to an unsigned 64-bit int in round-towards-zero mode.

__device__ long long int [__double_as_longlong](#group__cuda__math__intrinsic__cast_1gaeb72123ab9c35d88842baf34140a8ca6)(double x)
    

Reinterpret bits in a double as a 64-bit signed integer.

__device__ int [__float2int_rd](#group__cuda__math__intrinsic__cast_1ga46c024bc92e746ef5c6a48ef0ce74b54)(float x)
    

Convert a float to a signed integer in round-down mode.

__device__ int [__float2int_rn](#group__cuda__math__intrinsic__cast_1gaa0223a729c7bda6096fc7fc212df32cd)(float x)
    

Convert a float to a signed integer in round-to-nearest-even mode.

__device__ int [__float2int_ru](#group__cuda__math__intrinsic__cast_1gaefb35242e2f0c38d448b7123782aa15f)(float)
    

Convert a float to a signed integer in round-up mode.

__device__ int [__float2int_rz](#group__cuda__math__intrinsic__cast_1ga0f1b36aae52d3051068da702e4e6e3df)(float x)
    

Convert a float to a signed integer in round-towards-zero mode.

__device__ long long int [__float2ll_rd](#group__cuda__math__intrinsic__cast_1ga46a5c844ad314deca27609a775732386)(float x)
    

Convert a float to a signed 64-bit integer in round-down mode.

__device__ long long int [__float2ll_rn](#group__cuda__math__intrinsic__cast_1ga8491bb2ed031de169206e591e5779f69)(float x)
    

Convert a float to a signed 64-bit integer in round-to-nearest-even mode.

__device__ long long int [__float2ll_ru](#group__cuda__math__intrinsic__cast_1ga9f3eb6391aaeed6d18da2615d51205eb)(float x)
    

Convert a float to a signed 64-bit integer in round-up mode.

__device__ long long int [__float2ll_rz](#group__cuda__math__intrinsic__cast_1gab3d41edef9134fc0aa91ecd6d9dcf0c6)(float x)
    

Convert a float to a signed 64-bit integer in round-towards-zero mode.

__device__ unsigned int [__float2uint_rd](#group__cuda__math__intrinsic__cast_1gadaebb858df76995537091414f4b54970)(float x)
    

Convert a float to an unsigned integer in round-down mode.

__device__ unsigned int [__float2uint_rn](#group__cuda__math__intrinsic__cast_1gaa32cbc1af788e704c26b48c1a4a87494)(float x)
    

Convert a float to an unsigned integer in round-to-nearest-even mode.

__device__ unsigned int [__float2uint_ru](#group__cuda__math__intrinsic__cast_1ga233419ac96d01a429517c0fdfd4be157)(float x)
    

Convert a float to an unsigned integer in round-up mode.

__device__ unsigned int [__float2uint_rz](#group__cuda__math__intrinsic__cast_1gad833e48202936c6c70402fdf7549ef23)(float x)
    

Convert a float to an unsigned integer in round-towards-zero mode.

__device__ unsigned long long int [__float2ull_rd](#group__cuda__math__intrinsic__cast_1ga321ba664bb669aec511eb3be642a2be1)(float x)
    

Convert a float to an unsigned 64-bit integer in round-down mode.

__device__ unsigned long long int [__float2ull_rn](#group__cuda__math__intrinsic__cast_1ga6a8ce4db49da2a1eb288722ac108636e)(float x)
    

Convert a float to an unsigned 64-bit integer in round-to-nearest-even mode.

__device__ unsigned long long int [__float2ull_ru](#group__cuda__math__intrinsic__cast_1gabf910da5dce52e6e4a1df307a8aff874)(float x)
    

Convert a float to an unsigned 64-bit integer in round-up mode.

__device__ unsigned long long int [__float2ull_rz](#group__cuda__math__intrinsic__cast_1ga116a44de48af36f1a4d19ca97c9e58cd)(float x)
    

Convert a float to an unsigned 64-bit integer in round-towards-zero mode.

__device__ int [__float_as_int](#group__cuda__math__intrinsic__cast_1gadcf392b5fe3086d1ae06a1a3dcb752b1)(float x)
    

Reinterpret bits in a float as a signed integer.

__device__ unsigned int [__float_as_uint](#group__cuda__math__intrinsic__cast_1gaaf4c4e5365416bb96b5937df9ffd9497)(float x)
    

Reinterpret bits in a float as a unsigned integer.

__device__ double [__hiloint2double](#group__cuda__math__intrinsic__cast_1ga5853355db4780e81fb09ac911a8f2def)(int hi, int lo)
    

Reinterpret high and low 32-bit integer values as a double.

__device__ double [__int2double_rn](#group__cuda__math__intrinsic__cast_1ga802fcfefe56a2b03839f3135ac5ffda3)(int x)
    

Convert a signed int to a double.

__device__ float [__int2float_rd](#group__cuda__math__intrinsic__cast_1ga5247baba1fb70321b7569ba35000f5f9)(int x)
    

Convert a signed integer to a float in round-down mode.

__device__ float [__int2float_rn](#group__cuda__math__intrinsic__cast_1ga08cd4822773d557bd092ab986dffa8f1)(int x)
    

Convert a signed integer to a float in round-to-nearest-even mode.

__device__ float [__int2float_ru](#group__cuda__math__intrinsic__cast_1ga0f3748c434f936ee459feff8e6f0de7d)(int x)
    

Convert a signed integer to a float in round-up mode.

__device__ float [__int2float_rz](#group__cuda__math__intrinsic__cast_1gaa3080c83713f5be18ac3488af0c1d8a9)(int x)
    

Convert a signed integer to a float in round-towards-zero mode.

__device__ float [__int_as_float](#group__cuda__math__intrinsic__cast_1ga99207ac41a048168ff039e8e6f1084c9)(int x)
    

Reinterpret bits in an integer as a float.

__device__ double [__ll2double_rd](#group__cuda__math__intrinsic__cast_1ga7fcf675dbfa5fcb854a07c4b005cca44)(long long int x)
    

Convert a signed 64-bit int to a double in round-down mode.

__device__ double [__ll2double_rn](#group__cuda__math__intrinsic__cast_1ga70e811280daf5925d1b377f143d76201)(long long int x)
    

Convert a signed 64-bit int to a double in round-to-nearest-even mode.

__device__ double [__ll2double_ru](#group__cuda__math__intrinsic__cast_1ga8a8f6881e01a8d0de4231157606a17d2)(long long int x)
    

Convert a signed 64-bit int to a double in round-up mode.

__device__ double [__ll2double_rz](#group__cuda__math__intrinsic__cast_1gafe3afeae8f48bb3d598e2c7c9417feb0)(long long int x)
    

Convert a signed 64-bit int to a double in round-towards-zero mode.

__device__ float [__ll2float_rd](#group__cuda__math__intrinsic__cast_1ga125d06ca0d57a1d856714af3dbfa68a2)(long long int x)
    

Convert a signed integer to a float in round-down mode.

__device__ float [__ll2float_rn](#group__cuda__math__intrinsic__cast_1ga5012f7d034ffd3affadc0c1ff4fb4c0b)(long long int x)
    

Convert a signed 64-bit integer to a float in round-to-nearest-even mode.

__device__ float [__ll2float_ru](#group__cuda__math__intrinsic__cast_1ga30000a07d935a414c23eb79a4f4fd353)(long long int x)
    

Convert a signed integer to a float in round-up mode.

__device__ float [__ll2float_rz](#group__cuda__math__intrinsic__cast_1ga2dc649ecea6a9f04e8a85f7cfd8444c1)(long long int x)
    

Convert a signed integer to a float in round-towards-zero mode.

__device__ double [__longlong_as_double](#group__cuda__math__intrinsic__cast_1ga354a51915a0c6465ac5e5c844724d987)(long long int x)
    

Reinterpret bits in a 64-bit signed integer as a double.

__device__ double [__uint2double_rn](#group__cuda__math__intrinsic__cast_1gaf214f35ac7732dc82db1a2fe8cf1760b)(unsigned int x)
    

Convert an unsigned int to a double.

__device__ float [__uint2float_rd](#group__cuda__math__intrinsic__cast_1gaa46b5ec834fc7a973b0f90a605a82cad)(unsigned int x)
    

Convert an unsigned integer to a float in round-down mode.

__device__ float [__uint2float_rn](#group__cuda__math__intrinsic__cast_1gaeef617486f7a4937b87933b54d41b03a)(unsigned int x)
    

Convert an unsigned integer to a float in round-to-nearest-even mode.

__device__ float [__uint2float_ru](#group__cuda__math__intrinsic__cast_1ga4c31d267e495c32d9718d21bb9b3470b)(unsigned int x)
    

Convert an unsigned integer to a float in round-up mode.

__device__ float [__uint2float_rz](#group__cuda__math__intrinsic__cast_1gae1aa4eb377adbff2d94875131f10bfc1)(unsigned int x)
    

Convert an unsigned integer to a float in round-towards-zero mode.

__device__ float [__uint_as_float](#group__cuda__math__intrinsic__cast_1ga156767b22e3d00f8fa6625804a1cff63)(unsigned int x)
    

Reinterpret bits in an unsigned integer as a float.

__device__ double [__ull2double_rd](#group__cuda__math__intrinsic__cast_1ga7e2b60cf5ff4113e569a64b9a2db0a74)(unsigned long long int x)
    

Convert an unsigned 64-bit int to a double in round-down mode.

__device__ double [__ull2double_rn](#group__cuda__math__intrinsic__cast_1ga9de34b6ace45642b96873f1415d42d24)(unsigned long long int x)
    

Convert an unsigned 64-bit int to a double in round-to-nearest-even mode.

__device__ double [__ull2double_ru](#group__cuda__math__intrinsic__cast_1ga5d582e599bf8d35e17ff9b90695b0ebf)(unsigned long long int x)
    

Convert an unsigned 64-bit int to a double in round-up mode.

__device__ double [__ull2double_rz](#group__cuda__math__intrinsic__cast_1ga3b9d77551c668db0930cc75e1121528a)(unsigned long long int x)
    

Convert an unsigned 64-bit int to a double in round-towards-zero mode.

__device__ float [__ull2float_rd](#group__cuda__math__intrinsic__cast_1ga79735bf3dfb053279868c89b3e1111dc)(unsigned long long int x)
    

Convert an unsigned integer to a float in round-down mode.

__device__ float [__ull2float_rn](#group__cuda__math__intrinsic__cast_1gaa655fbb8e0ce7082496658b9708c655a)(unsigned long long int x)
    

Convert an unsigned integer to a float in round-to-nearest-even mode.

__device__ float [__ull2float_ru](#group__cuda__math__intrinsic__cast_1ga8ab7e72ff20c099263ad64bc7d02f9ed)(unsigned long long int x)
    

Convert an unsigned integer to a float in round-up mode.

__device__ float [__ull2float_rz](#group__cuda__math__intrinsic__cast_1ga3e719eed3287f966e05121745cff0a41)(unsigned long long int x)
    

Convert an unsigned integer to a float in round-towards-zero mode.

##  11.1. Functions[](#functions "Permalink to this headline")

__device__ float __double2float_rd(double x)[](#_CPPv417__double2float_rdd "Permalink to this definition")  

    

Convert a double to a float in round-down mode. 

Convert the double-precision floating-point value `x` to a single-precision floating-point value in round-down (to negative infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __double2float_rn(double x)[](#_CPPv417__double2float_rnd "Permalink to this definition")  

    

Convert a double to a float in round-to-nearest-even mode. 

Convert the double-precision floating-point value `x` to a single-precision floating-point value in round-to-nearest-even mode. 

Returns
    

Returns converted value. 

__device__ float __double2float_ru(double x)[](#_CPPv417__double2float_rud "Permalink to this definition")  

    

Convert a double to a float in round-up mode. 

Convert the double-precision floating-point value `x` to a single-precision floating-point value in round-up (to positive infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __double2float_rz(double x)[](#_CPPv417__double2float_rzd "Permalink to this definition")  

    

Convert a double to a float in round-towards-zero mode. 

Convert the double-precision floating-point value `x` to a single-precision floating-point value in round-towards-zero mode. 

Returns
    

Returns converted value. 

__device__ int __double2hiint(double x)[](#_CPPv414__double2hiintd "Permalink to this definition")  

    

Reinterpret high 32 bits in a double as a signed integer. 

Reinterpret the high 32 bits in the double-precision floating-point value `x` as a signed integer. 

Returns
    

Returns reinterpreted value. 

__device__ int __double2int_rd(double x)[](#_CPPv415__double2int_rdd "Permalink to this definition")  

    

Convert a double to a signed int in round-down mode. 

Convert the double-precision floating-point value `x` to a signed integer value in round-down (to negative infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ int __double2int_rn(double x)[](#_CPPv415__double2int_rnd "Permalink to this definition")  

    

Convert a double to a signed int in round-to-nearest-even mode. 

Convert the double-precision floating-point value `x` to a signed integer value in round-to-nearest-even mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ int __double2int_ru(double x)[](#_CPPv415__double2int_rud "Permalink to this definition")  

    

Convert a double to a signed int in round-up mode. 

Convert the double-precision floating-point value `x` to a signed integer value in round-up (to positive infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ int __double2int_rz(double x)[](#_CPPv415__double2int_rzd "Permalink to this definition")  

    

Convert a double to a signed int in round-towards-zero mode. 

Convert the double-precision floating-point value `x` to a signed integer value in round-towards-zero mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ long long int __double2ll_rd(double x)[](#_CPPv414__double2ll_rdd "Permalink to this definition")  

    

Convert a double to a signed 64-bit int in round-down mode. 

Convert the double-precision floating-point value `x` to a signed 64-bit integer value in round-down (to negative infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ long long int __double2ll_rn(double x)[](#_CPPv414__double2ll_rnd "Permalink to this definition")  

    

Convert a double to a signed 64-bit int in round-to-nearest-even mode. 

Convert the double-precision floating-point value `x` to a signed 64-bit integer value in round-to-nearest-even mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ long long int __double2ll_ru(double x)[](#_CPPv414__double2ll_rud "Permalink to this definition")  

    

Convert a double to a signed 64-bit int in round-up mode. 

Convert the double-precision floating-point value `x` to a signed 64-bit integer value in round-up (to positive infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ long long int __double2ll_rz(double x)[](#_CPPv414__double2ll_rzd "Permalink to this definition")  

    

Convert a double to a signed 64-bit int in round-towards-zero mode. 

Convert the double-precision floating-point value `x` to a signed 64-bit integer value in round-towards-zero mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ int __double2loint(double x)[](#_CPPv414__double2lointd "Permalink to this definition")  

    

Reinterpret low 32 bits in a double as a signed integer. 

Reinterpret the low 32 bits in the double-precision floating-point value `x` as a signed integer. 

Returns
    

Returns reinterpreted value. 

__device__ unsigned int __double2uint_rd(double x)[](#_CPPv416__double2uint_rdd "Permalink to this definition")  

    

Convert a double to an unsigned int in round-down mode. 

Convert the double-precision floating-point value `x` to an unsigned integer value in round-down (to negative infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned int __double2uint_rn(double x)[](#_CPPv416__double2uint_rnd "Permalink to this definition")  

    

Convert a double to an unsigned int in round-to-nearest-even mode. 

Convert the double-precision floating-point value `x` to an unsigned integer value in round-to-nearest-even mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned int __double2uint_ru(double x)[](#_CPPv416__double2uint_rud "Permalink to this definition")  

    

Convert a double to an unsigned int in round-up mode. 

Convert the double-precision floating-point value `x` to an unsigned integer value in round-up (to positive infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned int __double2uint_rz(double x)[](#_CPPv416__double2uint_rzd "Permalink to this definition")  

    

Convert a double to an unsigned int in round-towards-zero mode. 

Convert the double-precision floating-point value `x` to an unsigned integer value in round-towards-zero mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned long long int __double2ull_rd(double x)[](#_CPPv415__double2ull_rdd "Permalink to this definition")  

    

Convert a double to an unsigned 64-bit int in round-down mode. 

Convert the double-precision floating-point value `x` to an unsigned 64-bit integer value in round-down (to negative infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned long long int __double2ull_rn(double x)[](#_CPPv415__double2ull_rnd "Permalink to this definition")  

    

Convert a double to an unsigned 64-bit int in round-to-nearest-even mode. 

Convert the double-precision floating-point value `x` to an unsigned 64-bit integer value in round-to-nearest-even mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned long long int __double2ull_ru(double x)[](#_CPPv415__double2ull_rud "Permalink to this definition")  

    

Convert a double to an unsigned 64-bit int in round-up mode. 

Convert the double-precision floating-point value `x` to an unsigned 64-bit integer value in round-up (to positive infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned long long int __double2ull_rz(double x)[](#_CPPv415__double2ull_rzd "Permalink to this definition")  

    

Convert a double to an unsigned 64-bit int in round-towards-zero mode. 

Convert the double-precision floating-point value `x` to an unsigned 64-bit integer value in round-towards-zero mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ long long int __double_as_longlong(double x)[](#_CPPv420__double_as_longlongd "Permalink to this definition")  

    

Reinterpret bits in a double as a 64-bit signed integer. 

Reinterpret the bits in the double-precision floating-point value `x` as a signed 64-bit integer. 

Returns
    

Returns reinterpreted value. 

__device__ int __float2int_rd(float x)[](#_CPPv414__float2int_rdf "Permalink to this definition")  

    

Convert a float to a signed integer in round-down mode. 

Convert the single-precision floating-point value `x` to a signed integer in round-down (to negative infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ int __float2int_rn(float x)[](#_CPPv414__float2int_rnf "Permalink to this definition")  

    

Convert a float to a signed integer in round-to-nearest-even mode. 

Convert the single-precision floating-point value `x` to a signed integer in round-to-nearest-even mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ int __float2int_ru(float)[](#_CPPv414__float2int_ruf "Permalink to this definition")  

    

Convert a float to a signed integer in round-up mode. 

Convert the single-precision floating-point value `x` to a signed integer in round-up (to positive infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ int __float2int_rz(float x)[](#_CPPv414__float2int_rzf "Permalink to this definition")  

    

Convert a float to a signed integer in round-towards-zero mode. 

Convert the single-precision floating-point value `x` to a signed integer in round-towards-zero mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ long long int __float2ll_rd(float x)[](#_CPPv413__float2ll_rdf "Permalink to this definition")  

    

Convert a float to a signed 64-bit integer in round-down mode. 

Convert the single-precision floating-point value `x` to a signed 64-bit integer in round-down (to negative infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ long long int __float2ll_rn(float x)[](#_CPPv413__float2ll_rnf "Permalink to this definition")  

    

Convert a float to a signed 64-bit integer in round-to-nearest-even mode. 

Convert the single-precision floating-point value `x` to a signed 64-bit integer in round-to-nearest-even mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ long long int __float2ll_ru(float x)[](#_CPPv413__float2ll_ruf "Permalink to this definition")  

    

Convert a float to a signed 64-bit integer in round-up mode. 

Convert the single-precision floating-point value `x` to a signed 64-bit integer in round-up (to positive infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ long long int __float2ll_rz(float x)[](#_CPPv413__float2ll_rzf "Permalink to this definition")  

    

Convert a float to a signed 64-bit integer in round-towards-zero mode. 

Convert the single-precision floating-point value `x` to a signed 64-bit integer in round-towards-zero mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned int __float2uint_rd(float x)[](#_CPPv415__float2uint_rdf "Permalink to this definition")  

    

Convert a float to an unsigned integer in round-down mode. 

Convert the single-precision floating-point value `x` to an unsigned integer in round-down (to negative infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned int __float2uint_rn(float x)[](#_CPPv415__float2uint_rnf "Permalink to this definition")  

    

Convert a float to an unsigned integer in round-to-nearest-even mode. 

Convert the single-precision floating-point value `x` to an unsigned integer in round-to-nearest-even mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned int __float2uint_ru(float x)[](#_CPPv415__float2uint_ruf "Permalink to this definition")  

    

Convert a float to an unsigned integer in round-up mode. 

Convert the single-precision floating-point value `x` to an unsigned integer in round-up (to positive infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned int __float2uint_rz(float x)[](#_CPPv415__float2uint_rzf "Permalink to this definition")  

    

Convert a float to an unsigned integer in round-towards-zero mode. 

Convert the single-precision floating-point value `x` to an unsigned integer in round-towards-zero mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned long long int __float2ull_rd(float x)[](#_CPPv414__float2ull_rdf "Permalink to this definition")  

    

Convert a float to an unsigned 64-bit integer in round-down mode. 

Convert the single-precision floating-point value `x` to an unsigned 64-bit integer in round-down (to negative infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned long long int __float2ull_rn(float x)[](#_CPPv414__float2ull_rnf "Permalink to this definition")  

    

Convert a float to an unsigned 64-bit integer in round-to-nearest-even mode. 

Convert the single-precision floating-point value `x` to an unsigned 64-bit integer in round-to-nearest-even mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned long long int __float2ull_ru(float x)[](#_CPPv414__float2ull_ruf "Permalink to this definition")  

    

Convert a float to an unsigned 64-bit integer in round-up mode. 

Convert the single-precision floating-point value `x` to an unsigned 64-bit integer in round-up (to positive infinity) mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ unsigned long long int __float2ull_rz(float x)[](#_CPPv414__float2ull_rzf "Permalink to this definition")  

    

Convert a float to an unsigned 64-bit integer in round-towards-zero mode. 

Convert the single-precision floating-point value `x` to an unsigned 64-bit integer in round-towards-zero mode. 

Note

When the floating-point input rounded to integral is outside the range of the return type, the behavior is undefined. 

Returns
    

Returns converted value. 

__device__ int __float_as_int(float x)[](#_CPPv414__float_as_intf "Permalink to this definition")  

    

Reinterpret bits in a float as a signed integer. 

Reinterpret the bits in the single-precision floating-point value `x` as a signed integer. 

Returns
    

Returns reinterpreted value. 

__device__ unsigned int __float_as_uint(float x)[](#_CPPv415__float_as_uintf "Permalink to this definition")  

    

Reinterpret bits in a float as a unsigned integer. 

Reinterpret the bits in the single-precision floating-point value `x` as a unsigned integer. 

Returns
    

Returns reinterpreted value. 

__device__ double __hiloint2double(int hi, int lo)[](#_CPPv416__hiloint2doubleii "Permalink to this definition")  

    

Reinterpret high and low 32-bit integer values as a double. 

Reinterpret the integer value of `hi` as the high 32 bits of a double-precision floating-point value and the integer value of `lo` as the low 32 bits of the same double-precision floating-point value. 

Returns
    

Returns reinterpreted value. 

__device__ double __int2double_rn(int x)[](#_CPPv415__int2double_rni "Permalink to this definition")  

    

Convert a signed int to a double. 

Convert the signed integer value `x` to a double-precision floating-point value. 

Returns
    

Returns converted value. 

__device__ float __int2float_rd(int x)[](#_CPPv414__int2float_rdi "Permalink to this definition")  

    

Convert a signed integer to a float in round-down mode. 

Convert the signed integer value `x` to a single-precision floating-point value in round-down (to negative infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __int2float_rn(int x)[](#_CPPv414__int2float_rni "Permalink to this definition")  

    

Convert a signed integer to a float in round-to-nearest-even mode. 

Convert the signed integer value `x` to a single-precision floating-point value in round-to-nearest-even mode. 

Returns
    

Returns converted value. 

__device__ float __int2float_ru(int x)[](#_CPPv414__int2float_rui "Permalink to this definition")  

    

Convert a signed integer to a float in round-up mode. 

Convert the signed integer value `x` to a single-precision floating-point value in round-up (to positive infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __int2float_rz(int x)[](#_CPPv414__int2float_rzi "Permalink to this definition")  

    

Convert a signed integer to a float in round-towards-zero mode. 

Convert the signed integer value `x` to a single-precision floating-point value in round-towards-zero mode. 

Returns
    

Returns converted value. 

__device__ float __int_as_float(int x)[](#_CPPv414__int_as_floati "Permalink to this definition")  

    

Reinterpret bits in an integer as a float. 

Reinterpret the bits in the signed integer value `x` as a single-precision floating-point value. 

Returns
    

Returns reinterpreted value. 

__device__ double __ll2double_rd(long long int x)[](#_CPPv414__ll2double_rdx "Permalink to this definition")  

    

Convert a signed 64-bit int to a double in round-down mode. 

Convert the signed 64-bit integer value `x` to a double-precision floating-point value in round-down (to negative infinity) mode. 

Returns
    

Returns converted value. 

__device__ double __ll2double_rn(long long int x)[](#_CPPv414__ll2double_rnx "Permalink to this definition")  

    

Convert a signed 64-bit int to a double in round-to-nearest-even mode. 

Convert the signed 64-bit integer value `x` to a double-precision floating-point value in round-to-nearest-even mode. 

Returns
    

Returns converted value. 

__device__ double __ll2double_ru(long long int x)[](#_CPPv414__ll2double_rux "Permalink to this definition")  

    

Convert a signed 64-bit int to a double in round-up mode. 

Convert the signed 64-bit integer value `x` to a double-precision floating-point value in round-up (to positive infinity) mode. 

Returns
    

Returns converted value. 

__device__ double __ll2double_rz(long long int x)[](#_CPPv414__ll2double_rzx "Permalink to this definition")  

    

Convert a signed 64-bit int to a double in round-towards-zero mode. 

Convert the signed 64-bit integer value `x` to a double-precision floating-point value in round-towards-zero mode. 

Returns
    

Returns converted value. 

__device__ float __ll2float_rd(long long int x)[](#_CPPv413__ll2float_rdx "Permalink to this definition")  

    

Convert a signed integer to a float in round-down mode. 

Convert the signed integer value `x` to a single-precision floating-point value in round-down (to negative infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __ll2float_rn(long long int x)[](#_CPPv413__ll2float_rnx "Permalink to this definition")  

    

Convert a signed 64-bit integer to a float in round-to-nearest-even mode. 

Convert the signed 64-bit integer value `x` to a single-precision floating-point value in round-to-nearest-even mode. 

Returns
    

Returns converted value. 

__device__ float __ll2float_ru(long long int x)[](#_CPPv413__ll2float_rux "Permalink to this definition")  

    

Convert a signed integer to a float in round-up mode. 

Convert the signed integer value `x` to a single-precision floating-point value in round-up (to positive infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __ll2float_rz(long long int x)[](#_CPPv413__ll2float_rzx "Permalink to this definition")  

    

Convert a signed integer to a float in round-towards-zero mode. 

Convert the signed integer value `x` to a single-precision floating-point value in round-towards-zero mode. 

Returns
    

Returns converted value. 

__device__ double __longlong_as_double(long long int x)[](#_CPPv420__longlong_as_doublex "Permalink to this definition")  

    

Reinterpret bits in a 64-bit signed integer as a double. 

Reinterpret the bits in the 64-bit signed integer value `x` as a double-precision floating-point value. 

Returns
    

Returns reinterpreted value. 

__device__ double __uint2double_rn(unsigned int x)[](#_CPPv416__uint2double_rnj "Permalink to this definition")  

    

Convert an unsigned int to a double. 

Convert the unsigned integer value `x` to a double-precision floating-point value. 

Returns
    

Returns converted value. 

__device__ float __uint2float_rd(unsigned int x)[](#_CPPv415__uint2float_rdj "Permalink to this definition")  

    

Convert an unsigned integer to a float in round-down mode. 

Convert the unsigned integer value `x` to a single-precision floating-point value in round-down (to negative infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __uint2float_rn(unsigned int x)[](#_CPPv415__uint2float_rnj "Permalink to this definition")  

    

Convert an unsigned integer to a float in round-to-nearest-even mode. 

Convert the unsigned integer value `x` to a single-precision floating-point value in round-to-nearest-even mode. 

Returns
    

Returns converted value. 

__device__ float __uint2float_ru(unsigned int x)[](#_CPPv415__uint2float_ruj "Permalink to this definition")  

    

Convert an unsigned integer to a float in round-up mode. 

Convert the unsigned integer value `x` to a single-precision floating-point value in round-up (to positive infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __uint2float_rz(unsigned int x)[](#_CPPv415__uint2float_rzj "Permalink to this definition")  

    

Convert an unsigned integer to a float in round-towards-zero mode. 

Convert the unsigned integer value `x` to a single-precision floating-point value in round-towards-zero mode. 

Returns
    

Returns converted value. 

__device__ float __uint_as_float(unsigned int x)[](#_CPPv415__uint_as_floatj "Permalink to this definition")  

    

Reinterpret bits in an unsigned integer as a float. 

Reinterpret the bits in the unsigned integer value `x` as a single-precision floating-point value. 

Returns
    

Returns reinterpreted value. 

__device__ double __ull2double_rd(unsigned long long int x)[](#_CPPv415__ull2double_rdy "Permalink to this definition")  

    

Convert an unsigned 64-bit int to a double in round-down mode. 

Convert the unsigned 64-bit integer value `x` to a double-precision floating-point value in round-down (to negative infinity) mode. 

Returns
    

Returns converted value. 

__device__ double __ull2double_rn(unsigned long long int x)[](#_CPPv415__ull2double_rny "Permalink to this definition")  

    

Convert an unsigned 64-bit int to a double in round-to-nearest-even mode. 

Convert the unsigned 64-bit integer value `x` to a double-precision floating-point value in round-to-nearest-even mode. 

Returns
    

Returns converted value. 

__device__ double __ull2double_ru(unsigned long long int x)[](#_CPPv415__ull2double_ruy "Permalink to this definition")  

    

Convert an unsigned 64-bit int to a double in round-up mode. 

Convert the unsigned 64-bit integer value `x` to a double-precision floating-point value in round-up (to positive infinity) mode. 

Returns
    

Returns converted value. 

__device__ double __ull2double_rz(unsigned long long int x)[](#_CPPv415__ull2double_rzy "Permalink to this definition")  

    

Convert an unsigned 64-bit int to a double in round-towards-zero mode. 

Convert the unsigned 64-bit integer value `x` to a double-precision floating-point value in round-towards-zero mode. 

Returns
    

Returns converted value. 

__device__ float __ull2float_rd(unsigned long long int x)[](#_CPPv414__ull2float_rdy "Permalink to this definition")  

    

Convert an unsigned integer to a float in round-down mode. 

Convert the unsigned integer value `x` to a single-precision floating-point value in round-down (to negative infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __ull2float_rn(unsigned long long int x)[](#_CPPv414__ull2float_rny "Permalink to this definition")  

    

Convert an unsigned integer to a float in round-to-nearest-even mode. 

Convert the unsigned integer value `x` to a single-precision floating-point value in round-to-nearest-even mode. 

Returns
    

Returns converted value. 

__device__ float __ull2float_ru(unsigned long long int x)[](#_CPPv414__ull2float_ruy "Permalink to this definition")  

    

Convert an unsigned integer to a float in round-up mode. 

Convert the unsigned integer value `x` to a single-precision floating-point value in round-up (to positive infinity) mode. 

Returns
    

Returns converted value. 

__device__ float __ull2float_rz(unsigned long long int x)[](#_CPPv414__ull2float_rzy "Permalink to this definition")  

    

Convert an unsigned integer to a float in round-towards-zero mode. 

Convert the unsigned integer value `x` to a single-precision floating-point value in round-towards-zero mode. 

Returns
    

Returns converted value.