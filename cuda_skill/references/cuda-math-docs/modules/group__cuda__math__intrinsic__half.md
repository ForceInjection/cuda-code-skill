# 4. Half Precision Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__HALF.html


#  4\. Half Precision Intrinsics

This section describes half precision intrinsic functions.

To use these functions, include the header file `cuda_fp16.h` in your program. All of the functions defined here are available in device code. Some of the functions are also available to host compilers, please refer to respective functions’ documentation for details.

NOTE: Aggressive floating-point optimizations performed by host or device compilers may affect numeric behavior of the functions implemented in this header.

The following macros are available to help users selectively enable/disable various definitions present in the header file:

  * `CUDA_NO_HALF` \- If defined, this macro will prevent the definition of additional type aliases in the global namespace, helping to avoid potential conflicts with symbols defined in the user program.

  * `__CUDA_NO_HALF_CONVERSIONS__` \- If defined, this macro will prevent the use of the C++ type conversions (converting constructors and conversion operators) that are common for built-in floating-point types, but may be undesirable for `half` which is essentially a user-defined type.

  * `__CUDA_NO_HALF_OPERATORS__` and `__CUDA_NO_HALF2_OPERATORS__` \- If defined, these macros will prevent the inadvertent use of usual arithmetic and comparison operators. This enforces the storage-only type semantics and prevents C++ style computations on `half` and `half2` types.


Groups

Half Arithmetic Constants


To use these constants, include the header file `cuda_fp16.h` in your program.

Half Arithmetic Functions


To use these functions, include the header file `cuda_fp16.h` in your program.

Half Comparison Functions


To use these functions, include the header file `cuda_fp16.h` in your program.

Half Math Functions


To use these functions, include the header file `cuda_fp16.h` in your program.

Half Precision Conversion and Data Movement


To use these functions, include the header file `cuda_fp16.h` in your program.

Half2 Arithmetic Functions


To use these functions, include the header file `cuda_fp16.h` in your program.

Half2 Comparison Functions


To use these functions, include the header file `cuda_fp16.h` in your program.

Half2 Math Functions


To use these functions, include the header file `cuda_fp16.h` in your program.

Structs

__half


__half data type

__half2


__half2 data type

__half2_raw


__half2_raw data type

__half_raw


__half_raw data type

Typedefs

__nv_half


This datatype is an `__nv_` prefixed alias.

__nv_half2


This datatype is an `__nv_` prefixed alias.

__nv_half2_raw


This datatype is an `__nv_` prefixed alias.

__nv_half_raw


This datatype is an `__nv_` prefixed alias.

half


This datatype is meant to be the first-class or fundamental implementation of the half-precision numbers format.

half2


This datatype is meant to be the first-class or fundamental implementation of type for pairs of half-precision numbers.

nv_half


This datatype is an `nv_` prefixed alias.

nv_half2


This datatype is an `nv_` prefixed alias.

##  4.9. Typedefs

typedef __half __nv_half



This datatype is an `__nv_` prefixed alias.

typedef __half2 __nv_half2



This datatype is an `__nv_` prefixed alias.

typedef __half2_raw __nv_half2_raw



This datatype is an `__nv_` prefixed alias.

typedef __half_raw __nv_half_raw



This datatype is an `__nv_` prefixed alias.

typedef __half half



This datatype is meant to be the first-class or fundamental implementation of the half-precision numbers format.

Should be implemented in the compiler in the future. Current implementation is a simple typedef to a respective user-level type with underscores.

typedef __half2 half2



This datatype is meant to be the first-class or fundamental implementation of type for pairs of half-precision numbers.

Should be implemented in the compiler in the future. Current implementation is a simple typedef to a respective user-level type with underscores.

typedef __half nv_half



This datatype is an `nv_` prefixed alias.

typedef __half2 nv_half2



This datatype is an `nv_` prefixed alias.