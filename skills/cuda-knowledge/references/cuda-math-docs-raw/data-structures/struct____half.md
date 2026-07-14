# 15.1. __half

**Source:** [struct____half.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html)

---

#  15.1. __half[](#half "Permalink to this headline")

struct __half[](#_CPPv46__half "Permalink to this definition")  

    

[__half](#struct____half) data type 

This structure implements the datatype for storing half-precision floating-point numbers. The structure implements assignment, arithmetic and comparison operators, and type conversions. 16 bits are being used in total: 1 sign bit, 5 bits for the exponent, and the significand is being stored in 10 bits. The total precision is 11 bits. There are 15361 representable numbers within the interval [0.0, 1.0], endpoints included. On average we have log10(2**11) ~ 3.311 decimal digits.

The objective here is to provide IEEE754-compliant implementation of `binary16` type and arithmetic with limitations due to device HW not supporting floating-point exceptions. 

Public Functions

__half() = default[](#_CPPv4N6__half6__halfEv "Permalink to this definition")  

    

Constructor by default. 

Emtpy default constructor, result is uninitialized. 

__host__ __device__ inline constexpr __half(const [__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#_CPPv410__half_raw "__half_raw") &hr)[](#_CPPv4N6__half6__halfERK10__half_raw "Permalink to this definition")  

    

Constructor from `[__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#struct____half__raw)`. 

__host__ __device__ __tile__ explicit __half(const [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") f)[](#_CPPv4N6__half6__halfEK13__nv_bfloat16 "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` input using default round-to-nearest-even rounding mode. 

Need to include the header file `cuda_bf16.h`

__host__ __device__ __tile__ inline __half(const double f)[](#_CPPv4N6__half6__halfEKd "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `double` input using default round-to-nearest-even rounding mode. 

See also

[__double2half(double)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gadd809022bb43d8d0c5eb2c91687c3d6f) for further details. 

__host__ __device__ __tile__ inline __half(const float f)[](#_CPPv4N6__half6__halfEKf "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `float` input using default round-to-nearest-even rounding mode. 

See also

[__float2half(float)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gac45cbae133365f41c7c0861c4428676e) for further details. 

__host__ __device__ __tile__ inline __half(const int val)[](#_CPPv4N6__half6__halfEKi "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `int` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __half(const long long val)[](#_CPPv4N6__half6__halfEKx "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `long` `long` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __half(const long val)[](#_CPPv4N6__half6__halfEKl "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `long` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __half(const short val)[](#_CPPv4N6__half6__halfEKs "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `short` integer input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __half(const unsigned int val)[](#_CPPv4N6__half6__halfEKj "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `unsigned` `int` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __half(const unsigned long long val)[](#_CPPv4N6__half6__halfEKy "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `unsigned` `long` `long` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __half(const unsigned long val)[](#_CPPv4N6__half6__halfEKm "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `unsigned` `long` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __half(const unsigned short val)[](#_CPPv4N6__half6__halfEKt "Permalink to this definition")  

    

Construct `[__half](#struct____half)` from `unsigned` `short` integer input using default round-to-nearest-even rounding mode. 

__host__ __device__ operator [__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#_CPPv410__half_raw "__half_raw")() const[](#_CPPv4NK6__halfcv10__half_rawEv "Permalink to this definition")  

    

Type cast to `[__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#struct____half__raw)` operator. 

__host__ __device__ operator [__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#_CPPv410__half_raw "__half_raw")() volatile const[](#_CPPv4NVK6__halfcv10__half_rawEv "Permalink to this definition")  

    

Type cast to `[__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#struct____half__raw)` operator with `volatile` input. 

__host__ __device__ __tile__ inline operator bool() const[](#_CPPv4NK6__halfcvbEv "Permalink to this definition")  

    

Conversion operator to `bool` data type. 

+0 and -0 inputs convert to `false`. Non-zero inputs convert to `true`. 

__host__ __device__ __tile__ inline operator char() const[](#_CPPv4NK6__halfcvcEv "Permalink to this definition")  

    

Conversion operator to an implementation defined `char` data type. 

Using round-toward-zero rounding mode.

Detects signedness of the `char` type and proceeds accordingly, see further details in [__half2char_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1ga0de2c5abcd8c633b6c450ffcee489aaf) and [__half2uchar_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gaac0433bc8d1ea1a38d42ec4ab0863ebe). 

__host__ __device__ __tile__ operator float() const[](#_CPPv4NK6__halfcvfEv "Permalink to this definition")  

    

Type cast to `float` operator. 

__host__ __device__ __tile__ operator int() const[](#_CPPv4NK6__halfcviEv "Permalink to this definition")  

    

Conversion operator to `int` data type. 

Using round-toward-zero rounding mode.

See also

[__half2int_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gaaa53dbf7c9e0b948bc6868a4cdcc7422) for further details. 

__host__ __device__ __tile__ inline operator long() const[](#_CPPv4NK6__halfcvlEv "Permalink to this definition")  

    

Conversion operator to `long` data type. 

Using round-toward-zero rounding mode.

Detects size of the `long` type and proceeds accordingly, see further details in [__half2int_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gaaa53dbf7c9e0b948bc6868a4cdcc7422) and [__half2ll_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1ga90f0dbbabc6a3f603f37098511807473). 

__host__ __device__ __tile__ operator long long() const[](#_CPPv4NK6__halfcvxEv "Permalink to this definition")  

    

Conversion operator to `long` `long` data type. 

Using round-toward-zero rounding mode.

See also

[__half2ll_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1ga90f0dbbabc6a3f603f37098511807473) for further details. 

__host__ __device__ __tile__ operator short() const[](#_CPPv4NK6__halfcvsEv "Permalink to this definition")  

    

Conversion operator to `short` data type. 

Using round-toward-zero rounding mode.

See also

[__half2short_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1ga6d297f3aa9ae496370017c9f20d3a11d) for further details. 

__host__ __device__ __tile__ operator signed char() const[](#_CPPv4NK6__halfcvaEv "Permalink to this definition")  

    

Conversion operator to `signed` `char` data type. 

Using round-toward-zero rounding mode.

See also

[__half2char_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1ga0de2c5abcd8c633b6c450ffcee489aaf) for further details. 

__host__ __device__ __tile__ operator unsigned char() const[](#_CPPv4NK6__halfcvhEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `char` data type. 

Using round-toward-zero rounding mode.

See also

[__half2uchar_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gaac0433bc8d1ea1a38d42ec4ab0863ebe) for further details. 

__host__ __device__ __tile__ operator unsigned int() const[](#_CPPv4NK6__halfcvjEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `int` data type. 

Using round-toward-zero rounding mode.

See also

[__half2uint_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1ga7e4a6e48f1f457c06e4ce954ab5df158) for further details. 

__host__ __device__ __tile__ inline operator unsigned long() const[](#_CPPv4NK6__halfcvmEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `long` data type. 

Using round-toward-zero rounding mode.

Detects size of the `unsigned` `long` type and proceeds accordingly, see further details in [__half2uint_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1ga7e4a6e48f1f457c06e4ce954ab5df158) and [__half2ull_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gaa082cf8bf8dde9b2d7fa61f530bf0279). 

__host__ __device__ __tile__ operator unsigned long long() const[](#_CPPv4NK6__halfcvyEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `long` `long` data type. 

Using round-toward-zero rounding mode.

See also

[__half2ull_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gaa082cf8bf8dde9b2d7fa61f530bf0279) for further details. 

__host__ __device__ __tile__ operator unsigned short() const[](#_CPPv4NK6__halfcvtEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `short` data type. 

Using round-toward-zero rounding mode.

See also

[__half2ushort_rz(__half)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1ga631200f88c895c4ec3f20c4b74116bb0) for further details. 

__host__ __device__ [__half](#_CPPv46__half "__half") &operator=(const [__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#_CPPv410__half_raw "__half_raw") &hr)[](#_CPPv4N6__halfaSERK10__half_raw "Permalink to this definition")  

    

Assignment operator from `[__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#struct____half__raw)`. 

__host__ __device__ volatile [__half](#_CPPv46__half "__half") &operator=(const [__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#_CPPv410__half_raw "__half_raw") &hr) volatile[](#_CPPv4NV6__halfaSERK10__half_raw "Permalink to this definition")  

    

Assignment operator from `[__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#struct____half__raw)` to `volatile` `[__half](#struct____half)`. 

__host__ __device__ __tile__ [__half](#_CPPv46__half "__half") &operator=(const double f)[](#_CPPv4N6__halfaSEKd "Permalink to this definition")  

    

Type cast to `[__half](#struct____half)` assignment operator from `double` input using default round-to-nearest-even rounding mode. 

See also

[__double2half(double)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gadd809022bb43d8d0c5eb2c91687c3d6f) for further details. 

__host__ __device__ __tile__ [__half](#_CPPv46__half "__half") &operator=(const float f)[](#_CPPv4N6__halfaSEKf "Permalink to this definition")  

    

Type cast to `[__half](#struct____half)` assignment operator from `float` input using default round-to-nearest-even rounding mode. 

See also

[__float2half(float)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#group__cuda__math____half__misc_1gac45cbae133365f41c7c0861c4428676e) for further details. 

__host__ __device__ __tile__ [__half](#_CPPv46__half "__half") &operator=(const int val)[](#_CPPv4N6__halfaSEKi "Permalink to this definition")  

    

Type cast from `int` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__half](#_CPPv46__half "__half") &operator=(const long long val)[](#_CPPv4N6__halfaSEKx "Permalink to this definition")  

    

Type cast from `long` `long` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__half](#_CPPv46__half "__half") &operator=(const short val)[](#_CPPv4N6__halfaSEKs "Permalink to this definition")  

    

Type cast from `short` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__half](#_CPPv46__half "__half") &operator=(const unsigned int val)[](#_CPPv4N6__halfaSEKj "Permalink to this definition")  

    

Type cast from `unsigned` `int` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__half](#_CPPv46__half "__half") &operator=(const unsigned long long val)[](#_CPPv4N6__halfaSEKy "Permalink to this definition")  

    

Type cast from `unsigned` `long` `long` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__half](#_CPPv46__half "__half") &operator=(const unsigned short val)[](#_CPPv4N6__halfaSEKt "Permalink to this definition")  

    

Type cast from `unsigned` `short` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ volatile [__half](#_CPPv46__half "__half") &operator=(volatile const [__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#_CPPv410__half_raw "__half_raw") &hr) volatile[](#_CPPv4NV6__halfaSERVK10__half_raw "Permalink to this definition")  

    

Assignment operator from `volatile` `[__half_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html#struct____half__raw)` to `volatile` `[__half](#struct____half)`.