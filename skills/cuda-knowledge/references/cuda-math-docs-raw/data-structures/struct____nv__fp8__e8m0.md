# 15.20. __nv_fp8_e8m0

**Source:** [struct____nv__fp8__e8m0.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e8m0.html)

---

#  15.20. __nv_fp8_e8m0[](#nv-fp8-e8m0 "Permalink to this headline")

struct __nv_fp8_e8m0[](#_CPPv413__nv_fp8_e8m0 "Permalink to this definition")  

    

[__nv_fp8_e8m0](#struct____nv__fp8__e8m0) datatype 

This structure implements the datatype for handling 8-bit scale factors of `e8m0` kind: interpreted as powers of two with biased exponent. Bias equals to 127, so numbers 0 through 254 represent 2^-127 through 2^127. Number `0xFF` = 255 is reserved for NaN.

The structure implements converting constructors and operators. 

Public Functions

__nv_fp8_e8m0() = default[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp8_e8m0(const [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") f)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EK6__half "Permalink to this definition")  

    

Constructor from `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` data type, relies on `__NV_SATFINITE` behavior for large input values and `cudaRoundPosInf` for rounding. 

See also

[__nv_cvt_float_to_e8m0](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#group__cuda__math__fp8__misc_1ga02b68d6b0f87fbf9eaa1e64fc4dd3dfc) for further details 

__host__ __device__ inline explicit __nv_fp8_e8m0(const [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") f)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EK13__nv_bfloat16 "Permalink to this definition")  

    

Constructor from `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` data type, relies on `__NV_SATFINITE` behavior for large input values and `cudaRoundPosInf` for rounding. 

See also

[__nv_cvt_bfloat16raw_to_e8m0](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#group__cuda__math__fp8__misc_1gaad8e4fbc1206c9ba42c6fb48e70f4c05) for further details 

__host__ __device__ inline explicit __nv_fp8_e8m0(const double f)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKd "Permalink to this definition")  

    

Constructor from `double` data type, relies on `__NV_SATFINITE` behavior for large input values and `cudaRoundPosInf` for rounding. 

See also

[__nv_cvt_double_to_e8m0](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#group__cuda__math__fp8__misc_1ga396678cf3f6d6b036f66c6ecb1f4b145) for further details 

__host__ __device__ inline explicit __nv_fp8_e8m0(const float f)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKf "Permalink to this definition")  

    

Constructor from `float` data type, relies on `__NV_SATFINITE` behavior behavior for large input values and `cudaRoundPosInf` for rounding. 

See also

[__nv_cvt_float_to_e8m0](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#group__cuda__math__fp8__misc_1ga02b68d6b0f87fbf9eaa1e64fc4dd3dfc) for further details 

__host__ __device__ inline explicit __nv_fp8_e8m0(const int val)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKi "Permalink to this definition")  

    

Constructor from `int` data type, relies on `cudaRoundPosInf` rounding. 

__host__ __device__ inline explicit __nv_fp8_e8m0(const long int val)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKl "Permalink to this definition")  

    

Constructor from `long` `int` data type, relies on `cudaRoundPosInf` rounding. 

__host__ __device__ inline explicit __nv_fp8_e8m0(const long long int val)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKx "Permalink to this definition")  

    

Constructor from `long` `long` `int` data type, relies on `cudaRoundPosInf` rounding. 

__host__ __device__ inline explicit __nv_fp8_e8m0(const short int val)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKs "Permalink to this definition")  

    

Constructor from `short` `int` data type, relies on `cudaRoundPosInf` rounding. 

__host__ __device__ inline explicit __nv_fp8_e8m0(const unsigned int val)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKj "Permalink to this definition")  

    

Constructor from `unsigned` `int` data type, relies on `cudaRoundPosInf` rounding. 

__host__ __device__ inline explicit __nv_fp8_e8m0(const unsigned long int val)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKm "Permalink to this definition")  

    

Constructor from `unsigned` `long` `int` data type, relies on `cudaRoundPosInf` rounding. 

__host__ __device__ inline explicit __nv_fp8_e8m0(const unsigned long long int val)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKy "Permalink to this definition")  

    

Constructor from `unsigned` `long` `long` `int` data type, relies on `cudaRoundPosInf` rounding. 

__host__ __device__ inline explicit __nv_fp8_e8m0(const unsigned short int val)[](#_CPPv4N13__nv_fp8_e8m013__nv_fp8_e8m0EKt "Permalink to this definition")  

    

Constructor from `unsigned` `short` `int` data type, relies on `cudaRoundPosInf` rounding. 

__host__ __device__ inline explicit operator [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half")() const[](#_CPPv4NK13__nv_fp8_e8m0cv6__halfEv "Permalink to this definition")  

    

Conversion operator to `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` data type. 

__host__ __device__ inline explicit operator [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16")() const[](#_CPPv4NK13__nv_fp8_e8m0cv13__nv_bfloat16Ev "Permalink to this definition")  

    

Conversion operator to `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` data type. 

__host__ __device__ inline explicit operator bool() const[](#_CPPv4NK13__nv_fp8_e8m0cvbEv "Permalink to this definition")  

    

Conversion operator to `bool` data type. 

All values in input range are non-zero, so result is always `true`. 

__host__ __device__ inline explicit operator char() const[](#_CPPv4NK13__nv_fp8_e8m0cvcEv "Permalink to this definition")  

    

Conversion operator to an implementation defined `char` data type. 

Detects signedness of the `char` type and proceeds accordingly, see further details in signed and unsigned char operators.

Clamps inputs to the output range. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator double() const[](#_CPPv4NK13__nv_fp8_e8m0cvdEv "Permalink to this definition")  

    

Conversion operator to `double` data type. 

__host__ __device__ inline explicit operator float() const[](#_CPPv4NK13__nv_fp8_e8m0cvfEv "Permalink to this definition")  

    

Conversion operator to `float` data type. 

__host__ __device__ inline explicit operator int() const[](#_CPPv4NK13__nv_fp8_e8m0cviEv "Permalink to this definition")  

    

Conversion operator to `int` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator long int() const[](#_CPPv4NK13__nv_fp8_e8m0cvlEv "Permalink to this definition")  

    

Conversion operator to `long` `int` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero` if output type is 32-bit. `NaN` inputs convert to `0x8000000000000000ULL` if output type is 64-bit. 

__host__ __device__ inline explicit operator long long int() const[](#_CPPv4NK13__nv_fp8_e8m0cvxEv "Permalink to this definition")  

    

Conversion operator to `long` `long` `int` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `0x8000000000000000LL`. 

__host__ __device__ inline explicit operator short int() const[](#_CPPv4NK13__nv_fp8_e8m0cvsEv "Permalink to this definition")  

    

Conversion operator to `short` `int` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator signed char() const[](#_CPPv4NK13__nv_fp8_e8m0cvaEv "Permalink to this definition")  

    

Conversion operator to `signed` `char` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator unsigned char() const[](#_CPPv4NK13__nv_fp8_e8m0cvhEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `char` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator unsigned int() const[](#_CPPv4NK13__nv_fp8_e8m0cvjEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `int` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator unsigned long int() const[](#_CPPv4NK13__nv_fp8_e8m0cvmEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `long` `int` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero` if output type is 32-bit. `NaN` inputs convert to `0x8000000000000000ULL` if output type is 64-bit. 

__host__ __device__ inline explicit operator unsigned long long int() const[](#_CPPv4NK13__nv_fp8_e8m0cvyEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `long` `long` `int` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `0x8000000000000000ULL`. 

__host__ __device__ inline explicit operator unsigned short int() const[](#_CPPv4NK13__nv_fp8_e8m0cvtEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `short` `int` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`. 

Public Members

[__nv_fp8_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#_CPPv418__nv_fp8_storage_t "__nv_fp8_storage_t") __x[](#_CPPv4N13__nv_fp8_e8m03__xE "Permalink to this definition")  

    

Storage variable contains the 8-bit scale data.