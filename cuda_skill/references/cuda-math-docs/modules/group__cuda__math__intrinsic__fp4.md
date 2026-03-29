# 1. FP4 Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__FP4.html


#  1\. FP4 Intrinsics

This section describes fp4 intrinsic functions.

To use these functions, include the header file `cuda_fp4.h` in your program.

The following macros are available to help users selectively enable/disable various definitions present in the header file:

  * `__CUDA_NO_FP4_CONVERSIONS__` \- If defined, this macro will prevent any use of the C++ type conversions (converting constructors and conversion operators) defined in the header.

  * `__CUDA_NO_FP4_CONVERSION_OPERATORS__` \- If defined, this macro will prevent any use of the C++ conversion operators from `fp4` to other types.


Note

Most of the operations defined here benefit from native HW support when compiled for specific GPU targets (e.g. devices of compute capability 10.0a), other targets use emulation path.

Groups

C++ struct for handling fp4 data type of e2m1 kind.


C++ struct for handling vector type of four fp4 values of e2m1 kind.


C++ struct for handling vector type of two fp4 values of e2m1 kind.


FP4 Conversion and Data Movement


To use these functions, include the header file `cuda_fp4.h` in your program.