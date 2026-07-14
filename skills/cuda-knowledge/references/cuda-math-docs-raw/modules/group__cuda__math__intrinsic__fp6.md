# 2. FP6 Intrinsics

**Source:** [group__CUDA__MATH__INTRINSIC__FP6.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP6.html)

---

#  2\. FP6 Intrinsics[](#fp6-intrinsics "Permalink to this headline")

This section describes fp6 intrinsic functions. 

To use these functions, include the header file `cuda_fp6.h` in your program.

The following macros are available to help users selectively enable/disable various definitions present in the header file:

  * `__CUDA_NO_FP6_CONVERSIONS__` \- If defined, this macro will prevent any use of the C++ type conversions (converting constructors and conversion operators) defined in the header.

  * `__CUDA_NO_FP6_CONVERSION_OPERATORS__` \- If defined, this macro will prevent any use of the C++ conversion operators from `fp6` to other types. 


Note

Most of the operations defined here benefit from native HW support when compiled for specific GPU targets (e.g. devices of compute capability 10.0a), other targets use emulation path.

Groups

[C++ struct for handling fp6 data type of e2m3 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__E2M3__STRUCT.html#group__cuda__math__fp6__e2m3__struct)
    

[C++ struct for handling fp6 data type of e3m2 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__E3M2__STRUCT.html#group__cuda__math__fp6__e3m2__struct)
    

[C++ struct for handling vector type of four fp6 values of e2m3 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6X4__E2M3__STRUCT.html#group__cuda__math__fp6x4__e2m3__struct)
    

[C++ struct for handling vector type of four fp6 values of e3m2 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6X4__E3M2__STRUCT.html#group__cuda__math__fp6x4__e3m2__struct)
    

[C++ struct for handling vector type of two fp6 values of e2m3 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6X2__E2M3__STRUCT.html#group__cuda__math__fp6x2__e2m3__struct)
    

[C++ struct for handling vector type of two fp6 values of e3m2 kind.](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6X2__E3M2__STRUCT.html#group__cuda__math__fp6x2__e3m2__struct)
    

[FP6 Conversion and Data Movement](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html#group__cuda__math__fp6__misc)
    

To use these functions, include the header file `cuda_fp6.h` in your program.