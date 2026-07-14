# 15.5. __nv_bfloat16

**Source:** [struct____nv__bfloat16.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html)

---

#  15.5. __nv_bfloat16[](#nv-bfloat16 "Permalink to this headline")

struct __nv_bfloat16[](#_CPPv413__nv_bfloat16 "Permalink to this definition")  

    

nv_bfloat16 datatype 

This structure implements the datatype for storing nv_bfloat16 floating-point numbers. The structure implements assignment operators and type conversions. 16 bits are being used in total: 1 sign bit, 8 bits for the exponent, and the significand is being stored in 7 bits. The total precision is 8 bits. 

Public Functions

__nv_bfloat16() = default[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16Ev "Permalink to this definition")  

    

Constructor by default. 

Emtpy default constructor, result is uninitialized. 

__host__ __device__ __tile__ inline explicit __nv_bfloat16(const [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") f)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16EK6__half "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` input using default round-to-nearest-even rounding mode. 

__host__ __device__ inline constexpr __nv_bfloat16(const [__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#_CPPv417__nv_bfloat16_raw "__nv_bfloat16_raw") &hr)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16ERK17__nv_bfloat16_raw "Permalink to this definition")  

    

Constructor from `[__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#struct____nv__bfloat16__raw)`. 

__host__ __device__ __tile__ inline __nv_bfloat16(const double f)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16EKd "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `double` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __nv_bfloat16(const float f)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16EKf "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `float` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __nv_bfloat16(const long val)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16EKl "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `long` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __nv_bfloat16(const unsigned long val)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16EKm "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `unsigned` `long` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __nv_bfloat16(int val)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16Ei "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `int` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __nv_bfloat16(long long val)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16Ex "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `long` `long` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __nv_bfloat16(short val)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16Es "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `short` integer input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __nv_bfloat16(unsigned int val)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16Ej "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `unsigned` `int` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __nv_bfloat16(unsigned long long val)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16Ey "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `unsigned` `long` `long` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ inline __nv_bfloat16(unsigned short val)[](#_CPPv4N13__nv_bfloat1613__nv_bfloat16Et "Permalink to this definition")  

    

Construct `[__nv_bfloat16](#struct____nv__bfloat16)` from `unsigned` `short` integer input using default round-to-nearest-even rounding mode. 

__host__ __device__ operator [__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#_CPPv417__nv_bfloat16_raw "__nv_bfloat16_raw")() const[](#_CPPv4NK13__nv_bfloat16cv17__nv_bfloat16_rawEv "Permalink to this definition")  

    

Type cast to `[__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#struct____nv__bfloat16__raw)` operator. 

__host__ __device__ operator [__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#_CPPv417__nv_bfloat16_raw "__nv_bfloat16_raw")() volatile const[](#_CPPv4NVK13__nv_bfloat16cv17__nv_bfloat16_rawEv "Permalink to this definition")  

    

Type cast to `[__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#struct____nv__bfloat16__raw)` operator with `volatile` input. 

__host__ __device__ __tile__ inline operator bool() const[](#_CPPv4NK13__nv_bfloat16cvbEv "Permalink to this definition")  

    

Conversion operator to `bool` data type. 

+0 and -0 inputs convert to `false`. Non-zero inputs convert to `true`. 

__host__ __device__ __tile__ inline operator char() const[](#_CPPv4NK13__nv_bfloat16cvcEv "Permalink to this definition")  

    

Conversion operator to an implementation defined `char` data type. 

Using round-toward-zero rounding mode.

Detects signedness of the `char` type and proceeds accordingly, see further details in signed and unsigned char operators. 

__host__ __device__ __tile__ operator float() const[](#_CPPv4NK13__nv_bfloat16cvfEv "Permalink to this definition")  

    

Type cast to `float` operator. 

__host__ __device__ __tile__ operator int() const[](#_CPPv4NK13__nv_bfloat16cviEv "Permalink to this definition")  

    

Conversion operator to `int` data type. 

Using round-toward-zero rounding mode.

See [__bfloat162int_rz(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html#group__cuda__math____bfloat16__misc_1ga727657fa0da18973162c7624752fd0fc) for further details 

__host__ __device__ __tile__ inline operator long() const[](#_CPPv4NK13__nv_bfloat16cvlEv "Permalink to this definition")  

    

Conversion operator to `long` data type. 

Using round-toward-zero rounding mode. 

__host__ __device__ __tile__ operator long long() const[](#_CPPv4NK13__nv_bfloat16cvxEv "Permalink to this definition")  

    

Conversion operator to `long` `long` data type. 

Using round-toward-zero rounding mode.

See [__bfloat162ll_rz(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html#group__cuda__math____bfloat16__misc_1gad6f9f5254ae97f824db9c04bc972cd7b) for further details 

__host__ __device__ __tile__ operator short() const[](#_CPPv4NK13__nv_bfloat16cvsEv "Permalink to this definition")  

    

Conversion operator to `short` data type. 

Using round-toward-zero rounding mode.

See [__bfloat162short_rz(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html#group__cuda__math____bfloat16__misc_1ga22837c57bbb02450a57ec39b0cf90d3e) for further details 

__host__ __device__ __tile__ operator signed char() const[](#_CPPv4NK13__nv_bfloat16cvaEv "Permalink to this definition")  

    

Conversion operator to `signed` `char` data type. 

Using round-toward-zero rounding mode.

See [__bfloat162char_rz(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html#group__cuda__math____bfloat16__misc_1ga9156b2ac85be47ae2ca7f4eaf53b6742) for further details 

__host__ __device__ __tile__ operator unsigned char() const[](#_CPPv4NK13__nv_bfloat16cvhEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `char` data type. 

Using round-toward-zero rounding mode.

See [__bfloat162uchar_rz(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html#group__cuda__math____bfloat16__misc_1ga71529a008b005da9850c82405c19ec82) for further details 

__host__ __device__ __tile__ operator unsigned int() const[](#_CPPv4NK13__nv_bfloat16cvjEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `int` data type. 

Using round-toward-zero rounding mode.

See [__bfloat162uint_rz(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html#group__cuda__math____bfloat16__misc_1ga1c6fcb5fd65f71df2c29c892bca71fe1) for further details 

__host__ __device__ __tile__ inline operator unsigned long() const[](#_CPPv4NK13__nv_bfloat16cvmEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `long` data type. 

Using round-toward-zero rounding mode. 

__host__ __device__ __tile__ operator unsigned long long() const[](#_CPPv4NK13__nv_bfloat16cvyEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `long` `long` data type. 

Using round-toward-zero rounding mode.

See [__bfloat162ull_rz(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html#group__cuda__math____bfloat16__misc_1gaad8adf7550b9fa34d04acca4adbbe16e) for further details 

__host__ __device__ __tile__ operator unsigned short() const[](#_CPPv4NK13__nv_bfloat16cvtEv "Permalink to this definition")  

    

Conversion operator to `unsigned` `short` data type. 

Using round-toward-zero rounding mode.

See [__bfloat162ushort_rz(__nv_bfloat16)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html#group__cuda__math____bfloat16__misc_1ga7154e6a785efd245a8af979884f5f532) for further details 

__host__ __device__ [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(const [__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#_CPPv417__nv_bfloat16_raw "__nv_bfloat16_raw") &hr)[](#_CPPv4N13__nv_bfloat16aSERK17__nv_bfloat16_raw "Permalink to this definition")  

    

Assignment operator from `[__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#struct____nv__bfloat16__raw)`. 

__host__ __device__ volatile [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(const [__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#_CPPv417__nv_bfloat16_raw "__nv_bfloat16_raw") &hr) volatile[](#_CPPv4NV13__nv_bfloat16aSERK17__nv_bfloat16_raw "Permalink to this definition")  

    

Assignment operator from `[__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#struct____nv__bfloat16__raw)` to `volatile` `[__nv_bfloat16](#struct____nv__bfloat16)`. 

__host__ __device__ __tile__ [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(const double f)[](#_CPPv4N13__nv_bfloat16aSEKd "Permalink to this definition")  

    

Type cast to `[__nv_bfloat16](#struct____nv__bfloat16)` assignment operator from `double` input using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(const float f)[](#_CPPv4N13__nv_bfloat16aSEKf "Permalink to this definition")  

    

Type cast to `[__nv_bfloat16](#struct____nv__bfloat16)` assignment operator from `float` input using default round-to-nearest-even rounding mode. 

__host__ __device__ volatile [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(volatile const [__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#_CPPv417__nv_bfloat16_raw "__nv_bfloat16_raw") &hr) volatile[](#_CPPv4NV13__nv_bfloat16aSERVK17__nv_bfloat16_raw "Permalink to this definition")  

    

Assignment operator from `volatile` `[__nv_bfloat16_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html#struct____nv__bfloat16__raw)` to `volatile` `[__nv_bfloat16](#struct____nv__bfloat16)`. 

__host__ __device__ __tile__ [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(int val)[](#_CPPv4N13__nv_bfloat16aSEi "Permalink to this definition")  

    

Type cast from `int` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(long long val)[](#_CPPv4N13__nv_bfloat16aSEx "Permalink to this definition")  

    

Type cast from `long` `long` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(short val)[](#_CPPv4N13__nv_bfloat16aSEs "Permalink to this definition")  

    

Type cast from `short` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(unsigned int val)[](#_CPPv4N13__nv_bfloat16aSEj "Permalink to this definition")  

    

Type cast from `unsigned` `int` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(unsigned long long val)[](#_CPPv4N13__nv_bfloat16aSEy "Permalink to this definition")  

    

Type cast from `unsigned` `long` `long` assignment operator, using default round-to-nearest-even rounding mode. 

__host__ __device__ __tile__ [__nv_bfloat16](#_CPPv413__nv_bfloat16 "__nv_bfloat16") &operator=(unsigned short val)[](#_CPPv4N13__nv_bfloat16aSEt "Permalink to this definition")  

    

Type cast from `unsigned` `short` assignment operator, using default round-to-nearest-even rounding mode.