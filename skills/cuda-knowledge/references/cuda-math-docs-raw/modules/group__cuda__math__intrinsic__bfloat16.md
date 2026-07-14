# 5. Bfloat16 Precision Intrinsics

**Source:** [group__CUDA__MATH__INTRINSIC__BFLOAT16.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html)

---

#  5\. Bfloat16 Precision Intrinsics[](#bfloat16-precision-intrinsics "Permalink to this headline")

This section describes nv_bfloat16 precision intrinsic functions. 

To use these functions, include the header file `cuda_bf16.h` in your program. All of the functions defined here are available in device code. Some of the functions are also available to host compilers, please refer to respective functions’ documentation for details.

NOTE: Aggressive floating-point optimizations performed by host or device compilers may affect numeric behavior of the functions implemented in this header. Specific examples are:

  * [hsin(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#group__cuda__math____bfloat16__functions_1gaf31ae532d39ca19d22d3e32cba1cae64);

  * [hcos(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#group__cuda__math____bfloat16__functions_1ga094eb2de51c914219c4d0fbfd02fec05);

  * [h2sin(__nv_bfloat162)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT162__FUNCTIONS.html#group__cuda__math____bfloat162__functions_1ga2e77ae3fb1653dcefb47ba07d9587b5d);

  * [h2cos(__nv_bfloat162)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT162__FUNCTIONS.html#group__cuda__math____bfloat162__functions_1ga23ad268a57b40aaf009e264f932096e6);


The following macros are available to help users selectively enable/disable various definitions present in the header file:

  * `CUDA_NO_BFLOAT16` \- If defined, this macro will prevent the definition of additional type aliases in the global namespace, helping to avoid potential conflicts with symbols defined in the user program.

  * `__CUDA_NO_BFLOAT16_CONVERSIONS__` \- If defined, this macro will prevent the use of the C++ type conversions (converting constructors and conversion operators) that are common for built-in floating-point types, but may be undesirable for `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` which is essentially a user-defined type.

  * `__CUDA_NO_BFLOAT16_OPERATORS__` and `__CUDA_NO_BFLOAT162_OPERATORS__` \- If defined, these macros will prevent the inadvertent use of usual arithmetic and comparison operators. This enforces the storage-only type semantics and prevents C++ style computations on `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` and `[__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)` types. 


Groups

[Bfloat16 Arithmetic Constants](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16__CONSTANTS.html#group__cuda__math__intrinsic__bfloat16__constants)
    

To use these constants, include the header file `cuda_bf16.h` in your program.

[Bfloat16 Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html#group__cuda__math____bfloat16__arithmetic)
    

To use these functions, include the header file `cuda_bf16.h` in your program.

[Bfloat16 Comparison Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#group__cuda__math____bfloat16__comparison)
    

To use these functions, include the header file `cuda_bf16.h` in your program.

[Bfloat16 Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#group__cuda__math____bfloat16__functions)
    

To use these functions, include the header file `cuda_bf16.h` in your program.

[Bfloat16 Precision Conversion and Data Movement](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html#group__cuda__math____bfloat16__misc)
    

To use these functions, include the header file `cuda_bf16.h` in your program.

[Bfloat162 Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT162__ARITHMETIC.html#group__cuda__math____bfloat162__arithmetic)
    

To use these functions, include the header file `cuda_bf16.h` in your program.

[Bfloat162 Comparison Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT162__COMPARISON.html#group__cuda__math____bfloat162__comparison)
    

To use these functions, include the header file `cuda_bf16.h` in your program.

[Bfloat162 Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT162__FUNCTIONS.html#group__cuda__math____bfloat162__functions)
    

To use these functions, include the header file `cuda_bf16.h` in your program.

Structs

[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)
    

nv_bfloat16 datatype

[__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)
    

nv_bfloat162 datatype

[__nv_bfloat162_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162__raw.html#struct____nv__bfloat162__raw)
    

[__nv_bfloat162_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162__raw.html#struct____nv__bfloat162__raw) data type

[__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#struct____nv__bfloat16__raw)
    

[__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#struct____nv__bfloat16__raw) data type

Typedefs

[nv_bfloat16](#group__cuda__math__intrinsic__bfloat16_1ga620602d6c2e3900cd4cd8d8e6664358d)
    

This datatype is meant to be the first-class or fundamental implementation of the bfloat16 numbers format.

[nv_bfloat162](#group__cuda__math__intrinsic__bfloat16_1gad9aaa8323a2be8d4a1e91c5d82a8eb16)
    

This datatype is meant to be the first-class or fundamental implementation of type for pairs of bfloat16 numbers.

##  5.9. Typedefs[](#typedefs "Permalink to this headline")

typedef [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") nv_bfloat16[](#_CPPv411nv_bfloat16 "Permalink to this definition")  

    

This datatype is meant to be the first-class or fundamental implementation of the bfloat16 numbers format. 

Should be implemented in the compiler in the future. Current implementation is a simple typedef to a respective user-level type with underscores. 

typedef [__nv_bfloat162](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#_CPPv414__nv_bfloat162 "__nv_bfloat162") nv_bfloat162[](#_CPPv412nv_bfloat162 "Permalink to this definition")  

    

This datatype is meant to be the first-class or fundamental implementation of type for pairs of bfloat16 numbers. 

Should be implemented in the compiler in the future. Current implementation is a simple typedef to a respective user-level type with underscores.