# 3. FP8 Intrinsics

**Source:** [group__CUDA__MATH__INTRINSIC__FP8.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP8.html)

---

#  3\. FP8 Intrinsics[](#fp8-intrinsics "Permalink to this headline")

This section describes fp8 intrinsic functions. 

To use these functions, include the header file `cuda_fp8.h` in your program. The following macros are available to help users selectively enable/disable various definitions present in the header file:

  * `__CUDA_NO_FP8_CONVERSIONS__` \- If defined, this macro will prevent any use of the C++ type conversions (converting constructors and conversion operators) defined in the header.

  * `__CUDA_NO_FP8_CONVERSION_OPERATORS__` \- If defined, this macro will prevent any use of the C++ conversion operators from `fp8` to other types. 


Groups

[C++ struct for handling fp8 data type of e4m3 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__E4M3__STRUCT.html#group__cuda__math__fp8__e4m3__struct)
    

[C++ struct for handling fp8 data type of e5m2 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__E5M2__STRUCT.html#group__cuda__math__fp8__e5m2__struct)
    

[C++ struct for handling vector type of four fp8 values of e4m3 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8X4__E4M3__STRUCT.html#group__cuda__math__fp8x4__e4m3__struct)
    

[C++ struct for handling vector type of four fp8 values of e5m2 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8X4__E5M2__STRUCT.html#group__cuda__math__fp8x4__e5m2__struct)
    

[C++ struct for handling vector type of four scale factors of e8m0 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8X4__E8M0__STRUCT.html#group__cuda__math__fp8x4__e8m0__struct)
    

[C++ struct for handling vector type of two fp8 values of e4m3 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8X2__E4M3__STRUCT.html#group__cuda__math__fp8x2__e4m3__struct)
    

[C++ struct for handling vector type of two fp8 values of e5m2 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8X2__E5M2__STRUCT.html#group__cuda__math__fp8x2__e5m2__struct)
    

[C++ struct for handling vector type of two scale factors of e8m0 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8X2__E8M0__STRUCT.html#group__cuda__math__fp8x2__e8m0__struct)
    

[FP8 Conversion and Data Movement](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#group__cuda__math__fp8__misc)
    

To use these functions, include the header file `cuda_fp8.h` in your program.