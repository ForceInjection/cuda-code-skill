# 4. Half Precision Intrinsics

**Source:** [group__CUDA__MATH__INTRINSIC__HALF.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF.html)

---

#  4\. Half Precision Intrinsics[](#half-precision-intrinsics "Permalink to this headline")

This section describes half precision intrinsic functions. 

To use these functions, include the header file `cuda_fp16.h` in your program. All of the functions defined here are available in device code. Some of the functions are also available to host compilers, please refer to respective functions’ documentation for details.

NOTE: Aggressive floating-point optimizations performed by host or device compilers may affect numeric behavior of the functions implemented in this header.

The following macros are available to help users selectively enable/disable various definitions present in the header file:

  * `CUDA_NO_HALF` \- If defined, this macro will prevent the definition of additional type aliases in the global namespace, helping to avoid potential conflicts with symbols defined in the user program.

  * `__CUDA_NO_HALF_CONVERSIONS__` \- If defined, this macro will prevent the use of the C++ type conversions (converting constructors and conversion operators) that are common for built-in floating-point types, but may be undesirable for `half` which is essentially a user-defined type.

  * `__CUDA_NO_HALF_OPERATORS__` and `__CUDA_NO_HALF2_OPERATORS__` \- If defined, these macros will prevent the inadvertent use of usual arithmetic and comparison operators. This enforces the storage-only type semantics and prevents C++ style computations on `half` and `half2` types. 


Groups

[Half Arithmetic Constants](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF__CONSTANTS.html#group__cuda__math__intrinsic__half__constants)
    

To use these constants, include the header file `cuda_fp16.h` in your program.

[Half Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__cuda__math____half__arithmetic)
    

To use these functions, include the header file `cuda_fp16.h` in your program.

[Half Comparison Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#group__cuda__math____half__comparison)
    

To use these functions, include the header file `cuda_fp16.h` in your program.

[Half Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__cuda__math____half__functions)
    

To use these functions, include the header file `cuda_fp16.h` in your program.

[Half Precision Conversion and Data Movement](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc)
    

To use these functions, include the header file `cuda_fp16.h` in your program.

[Half2 Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__cuda__math____half2__arithmetic)
    

To use these functions, include the header file `cuda_fp16.h` in your program.

[Half2 Comparison Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__COMPARISON.html#group__cuda__math____half2__comparison)
    

To use these functions, include the header file `cuda_fp16.h` in your program.

[Half2 Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__cuda__math____half2__functions)
    

To use these functions, include the header file `cuda_fp16.h` in your program.

Structs

[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)
    

[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half) data type

[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2)
    

[__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#struct____half2) data type

[__half2_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2__raw.html#struct____half2__raw)
    

[__half2_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2__raw.html#struct____half2__raw) data type

[__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#struct____half__raw)
    

[__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#struct____half__raw) data type

Typedefs

[__nv_half](#group__cuda__math__intrinsic__half_1gae9346c2e791857fefc93141e8abbb973)
    

This datatype is an `__nv_` prefixed alias.

[__nv_half2](#group__cuda__math__intrinsic__half_1ga1b8823b035b23b1e85e62eea2bca7b94)
    

This datatype is an `__nv_` prefixed alias.

[__nv_half2_raw](#group__cuda__math__intrinsic__half_1ga0fc2581a3e84f7a243ac9621d9b72d33)
    

This datatype is an `__nv_` prefixed alias.

[__nv_half_raw](#group__cuda__math__intrinsic__half_1gaced553ec1d5e052a45d88d8a893d3765)
    

This datatype is an `__nv_` prefixed alias.

[half](#group__cuda__math__intrinsic__half_1ga3a4b5246a149b3134f45656c14b1f92b)
    

This datatype is meant to be the first-class or fundamental implementation of the half-precision numbers format.

[half2](#group__cuda__math__intrinsic__half_1ga1b22ac0f59c836bd0560a50f269de46c)
    

This datatype is meant to be the first-class or fundamental implementation of type for pairs of half-precision numbers.

[nv_half](#group__cuda__math__intrinsic__half_1ga5ecf930623ba3b275f39169975fdba2d)
    

This datatype is an `nv_` prefixed alias.

[nv_half2](#group__cuda__math__intrinsic__half_1ga87a356aba89538b1ad48e4ca356e43aa)
    

This datatype is an `nv_` prefixed alias.

##  4.9. Typedefs[](#typedefs "Permalink to this headline")

typedef [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") __nv_half[](#_CPPv49__nv_half "Permalink to this definition")  

    

This datatype is an `__nv_` prefixed alias. 

typedef [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") __nv_half2[](#_CPPv410__nv_half2 "Permalink to this definition")  

    

This datatype is an `__nv_` prefixed alias. 

typedef [__half2_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2__raw.html#_CPPv411__half2_raw "__half2_raw") __nv_half2_raw[](#_CPPv414__nv_half2_raw "Permalink to this definition")  

    

This datatype is an `__nv_` prefixed alias. 

typedef [__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#_CPPv410__half_raw "__half_raw") __nv_half_raw[](#_CPPv413__nv_half_raw "Permalink to this definition")  

    

This datatype is an `__nv_` prefixed alias. 

typedef [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") half[](#_CPPv44half "Permalink to this definition")  

    

This datatype is meant to be the first-class or fundamental implementation of the half-precision numbers format. 

Should be implemented in the compiler in the future. Current implementation is a simple typedef to a respective user-level type with underscores. 

typedef [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") half2[](#_CPPv45half2 "Permalink to this definition")  

    

This datatype is meant to be the first-class or fundamental implementation of type for pairs of half-precision numbers. 

Should be implemented in the compiler in the future. Current implementation is a simple typedef to a respective user-level type with underscores. 

typedef [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") nv_half[](#_CPPv47nv_half "Permalink to this definition")  

    

This datatype is an `nv_` prefixed alias. 

typedef [__half2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html#_CPPv47__half2 "__half2") nv_half2[](#_CPPv48nv_half2 "Permalink to this definition")  

    

This datatype is an `nv_` prefixed alias.