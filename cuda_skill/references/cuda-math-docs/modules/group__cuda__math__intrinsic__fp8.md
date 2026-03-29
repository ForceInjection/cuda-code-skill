# 3. FP8 Intrinsics

**Source:** group__CUDA__MATH__INTRINSIC__FP8.html


#  3\. FP8 Intrinsics

This section describes fp8 intrinsic functions.

To use these functions, include the header file `cuda_fp8.h` in your program. The following macros are available to help users selectively enable/disable various definitions present in the header file:

  * `__CUDA_NO_FP8_CONVERSIONS__` \- If defined, this macro will prevent any use of the C++ type conversions (converting constructors and conversion operators) defined in the header.

  * `__CUDA_NO_FP8_CONVERSION_OPERATORS__` \- If defined, this macro will prevent any use of the C++ conversion operators from `fp8` to other types.


Groups

C++ struct for handling fp8 data type of e4m3 kind.


C++ struct for handling fp8 data type of e5m2 kind.


C++ struct for handling vector type of four fp8 values of e4m3 kind.


C++ struct for handling vector type of four fp8 values of e5m2 kind.


C++ struct for handling vector type of four scale factors of e8m0 kind.


C++ struct for handling vector type of two fp8 values of e4m3 kind.


C++ struct for handling vector type of two fp8 values of e5m2 kind.


C++ struct for handling vector type of two scale factors of e8m0 kind.


FP8 Conversion and Data Movement


To use these functions, include the header file `cuda_fp8.h` in your program.