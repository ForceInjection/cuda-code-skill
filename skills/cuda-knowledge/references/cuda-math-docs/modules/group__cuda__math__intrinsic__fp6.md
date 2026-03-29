# 2. FP6 Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__FP6.html


#  2\. FP6 Intrinsics

This section describes fp6 intrinsic functions.

To use these functions, include the header file `cuda_fp6.h` in your program.

The following macros are available to help users selectively enable/disable various definitions present in the header file:

  * `__CUDA_NO_FP6_CONVERSIONS__` \- If defined, this macro will prevent any use of the C++ type conversions (converting constructors and conversion operators) defined in the header.

  * `__CUDA_NO_FP6_CONVERSION_OPERATORS__` \- If defined, this macro will prevent any use of the C++ conversion operators from `fp6` to other types.


Note

Most of the operations defined here benefit from native HW support when compiled for specific GPU targets (e.g. devices of compute capability 10.0a), other targets use emulation path.

Groups

C++ struct for handling fp6 data type of e2m3 kind.


C++ struct for handling fp6 data type of e3m2 kind.


C++ struct for handling vector type of four fp6 values of e2m3 kind.


C++ struct for handling vector type of four fp6 values of e3m2 kind.


C++ struct for handling vector type of two fp6 values of e2m3 kind.


C++ struct for handling vector type of two fp6 values of e3m2 kind.


FP6 Conversion and Data Movement


To use these functions, include the header file `cuda_fp6.h` in your program.