# 5. Bfloat16 Precision Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__BFLOAT16.html


#  5\. Bfloat16 Precision Intrinsics

This section describes nv_bfloat16 precision intrinsic functions.

To use these functions, include the header file `cuda_bf16.h` in your program. All of the functions defined here are available in device code. Some of the functions are also available to host compilers, please refer to respective functions’ documentation for details.

NOTE: Aggressive floating-point optimizations performed by host or device compilers may affect numeric behavior of the functions implemented in this header. Specific examples are:

  * hsin(__nv_bfloat16);

  * hcos(__nv_bfloat16);

  * h2sin(__nv_bfloat162);

  * h2cos(__nv_bfloat162);


The following macros are available to help users selectively enable/disable various definitions present in the header file:

  * `CUDA_NO_BFLOAT16` \- If defined, this macro will prevent the definition of additional type aliases in the global namespace, helping to avoid potential conflicts with symbols defined in the user program.

  * `__CUDA_NO_BFLOAT16_CONVERSIONS__` \- If defined, this macro will prevent the use of the C++ type conversions (converting constructors and conversion operators) that are common for built-in floating-point types, but may be undesirable for `__nv_bfloat16` which is essentially a user-defined type.

  * `__CUDA_NO_BFLOAT16_OPERATORS__` and `__CUDA_NO_BFLOAT162_OPERATORS__` \- If defined, these macros will prevent the inadvertent use of usual arithmetic and comparison operators. This enforces the storage-only type semantics and prevents C++ style computations on `__nv_bfloat16` and `__nv_bfloat162` types.


Groups

Bfloat16 Arithmetic Constants


To use these constants, include the header file `cuda_bf16.h` in your program.

Bfloat16 Arithmetic Functions


To use these functions, include the header file `cuda_bf16.h` in your program.

Bfloat16 Comparison Functions


To use these functions, include the header file `cuda_bf16.h` in your program.

Bfloat16 Math Functions


To use these functions, include the header file `cuda_bf16.h` in your program.

Bfloat16 Precision Conversion and Data Movement


To use these functions, include the header file `cuda_bf16.h` in your program.

Bfloat162 Arithmetic Functions


To use these functions, include the header file `cuda_bf16.h` in your program.

Bfloat162 Comparison Functions


To use these functions, include the header file `cuda_bf16.h` in your program.

Bfloat162 Math Functions


To use these functions, include the header file `cuda_bf16.h` in your program.

Structs

__nv_bfloat16


nv_bfloat16 datatype

__nv_bfloat162


nv_bfloat162 datatype

__nv_bfloat162_raw


__nv_bfloat162_raw data type

__nv_bfloat16_raw


__nv_bfloat16_raw data type

Typedefs

nv_bfloat16


This datatype is meant to be the first-class or fundamental implementation of the bfloat16 numbers format.

nv_bfloat162


This datatype is meant to be the first-class or fundamental implementation of type for pairs of bfloat16 numbers.

##  5.9. Typedefs

typedef __nv_bfloat16 nv_bfloat16



This datatype is meant to be the first-class or fundamental implementation of the bfloat16 numbers format.

Should be implemented in the compiler in the future. Current implementation is a simple typedef to a respective user-level type with underscores.

typedef __nv_bfloat162 nv_bfloat162



This datatype is meant to be the first-class or fundamental implementation of type for pairs of bfloat16 numbers.

Should be implemented in the compiler in the future. Current implementation is a simple typedef to a respective user-level type with underscores.