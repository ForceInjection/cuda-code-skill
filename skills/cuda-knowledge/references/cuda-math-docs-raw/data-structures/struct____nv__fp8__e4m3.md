# 15.18. __nv_fp8_e4m3

**Source:** [struct____nv__fp8__e4m3.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html)

---

#  15.18. __nv_fp8_e4m3[](#nv-fp8-e4m3 "Permalink to this headline")

struct __nv_fp8_e4m3[](#_CPPv413__nv_fp8_e4m3 "Permalink to this definition")  

    

[__nv_fp8_e4m3](#struct____nv__fp8__e4m3) datatype 

This structure implements the datatype for storing `fp8` floating-point numbers of `e4m3` kind: with 1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits. The encoding doesn’t support Infinity. NaNs are limited to 0x7F and 0xFF values.

The structure implements converting constructors and operators. 

Public Functions

__nv_fp8_e4m3() = default[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3Ev "Permalink to this definition")  

    

Constructor by default. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") f)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EK6__half "Permalink to this definition")  

    

Constructor from `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") f)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EK13__nv_bfloat16 "Permalink to this definition")  

    

Constructor from `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const double f)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKd "Permalink to this definition")  

    

Constructor from `double` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const float f)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKf "Permalink to this definition")  

    

Constructor from `float` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const int val)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKi "Permalink to this definition")  

    

Constructor from `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const long int val)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKl "Permalink to this definition")  

    

Constructor from `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const long long int val)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKx "Permalink to this definition")  

    

Constructor from `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const short int val)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKs "Permalink to this definition")  

    

Constructor from `short` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const unsigned int val)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKj "Permalink to this definition")  

    

Constructor from `unsigned` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const unsigned long int val)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKm "Permalink to this definition")  

    

Constructor from `unsigned` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const unsigned long long int val)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKy "Permalink to this definition")  

    

Constructor from `unsigned` `long` `long` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit __nv_fp8_e4m3(const unsigned short int val)[](#_CPPv4N13__nv_fp8_e4m313__nv_fp8_e4m3EKt "Permalink to this definition")  

    

Constructor from `unsigned` `short` `int` data type, relies on `__NV_SATFINITE` behavior for out-of-range values. 

__host__ __device__ inline explicit operator [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half")() const[](#_CPPv4NK13__nv_fp8_e4m3cv6__halfEv "Permalink to this definition")  

    

Conversion operator to `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` data type. 

__host__ __device__ inline explicit operator [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16")() const[](#_CPPv4NK13__nv_fp8_e4m3cv13__nv_bfloat16Ev "Permalink to this definition")  

    

Conversion operator to `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` data type. 

__host__ __device__ inline explicit operator bool() const[](#_CPPv4NK13__nv_fp8_e4m3cvbEv "Permalink to this definition")  

    

Conversion operator to `bool` data type. 

+0 and -0 inputs convert to `false`. Non-zero inputs convert to `true`. 

__host__ __device__ inline explicit operator char() const[](#_CPPv4NK13__nv_fp8_e4m3cvcEv "Permalink to this definition")  

    

Conversion operator to an implementation defined `char` data type. 

Detects signedness of the `char` type and proceeds accordingly, see further details in signed and unsigned char operators.

Clamps inputs to the output range. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator double() const[](#_CPPv4NK13__nv_fp8_e4m3cvdEv "Permalink to this definition")  

    

Conversion operator to `double` data type. 

__host__ __device__ inline explicit operator float() const[](#_CPPv4NK13__nv_fp8_e4m3cvfEv "Permalink to this definition")  

    

Conversion operator to `float` data type. 

__host__ __device__ inline explicit operator int() const[](#_CPPv4NK13__nv_fp8_e4m3cviEv "Permalink to this definition")  

    

Conversion operator to `int` data type. 

`NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator long int() const[](#_CPPv4NK13__nv_fp8_e4m3cvlEv "Permalink to this definition")  

    

Conversion operator to `long` `int` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero` if output type is 32-bit. `NaN` inputs convert to `0x8000000000000000ULL` if output type is 64-bit. 

__host__ __device__ inline explicit operator long long int() const[](#_CPPv4NK13__nv_fp8_e4m3cvxEv "Permalink to this definition")  

    

Conversion operator to `long` `long` `int` data type. 

`NaN` inputs convert to `0x8000000000000000LL`. 

__host__ __device__ inline explicit operator short int() const[](#_CPPv4NK13__nv_fp8_e4m3cvsEv "Permalink to this definition")  

    

Conversion operator to `short` `int` data type. 

`NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator signed char() const[](#_CPPv4NK13__nv_fp8_e4m3cvaEv "Permalink to this definition")  

    

Conversion operator to `signed` `char` data type. 

Clamps too large inputs to the output range. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator unsigned char() const[](#_CPPv4NK13__nv_fp8_e4m3cvhEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `char` data type. 

Clamps negative and too large inputs to the output range. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator unsigned int() const[](#_CPPv4NK13__nv_fp8_e4m3cvjEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `int` data type. 

Clamps negative inputs to zero. `NaN` inputs convert to `zero`. 

__host__ __device__ inline explicit operator unsigned long int() const[](#_CPPv4NK13__nv_fp8_e4m3cvmEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `long` `int` data type. 

Clamps negative and too large inputs to the output range. `NaN` inputs convert to `zero` if output type is 32-bit. `NaN` inputs convert to `0x8000000000000000ULL` if output type is 64-bit. 

__host__ __device__ inline explicit operator unsigned long long int() const[](#_CPPv4NK13__nv_fp8_e4m3cvyEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `long` `long` `int` data type. 

Clamps negative inputs to zero. `NaN` inputs convert to `0x8000000000000000ULL`. 

__host__ __device__ inline explicit operator unsigned short int() const[](#_CPPv4NK13__nv_fp8_e4m3cvtEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `short` `int` data type. 

Clamps negative inputs to zero. `NaN` inputs convert to `zero`. 

Public Members

[__nv_fp8_storage_t](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html#_CPPv418__nv_fp8_storage_t "__nv_fp8_storage_t") __x[](#_CPPv4N13__nv_fp8_e4m33__xE "Permalink to this definition")  

    

Storage variable contains the `fp8` floating-point data.