# 15.6. __nv_bfloat162

**Source:** struct____nv__bfloat162.html


#  15.6. __nv_bfloat162

struct __nv_bfloat162



nv_bfloat162 datatype

This structure implements the datatype for storing two nv_bfloat16 floating-point numbers. The structure implements assignment, arithmetic and comparison operators, and type conversions.

  * NOTE: __nv_bfloat162 is visible to non-nvcc host compilers


Public Functions

__nv_bfloat162() = default



Constructor by default.

Emtpy default constructor, result is uninitialized.

__host__ __device__ __nv_bfloat162(__nv_bfloat162 &&src)



Move constructor, available for `C++11` and later dialects.

__host__ __device__ inline constexpr __nv_bfloat162(const __nv_bfloat16 &a, const __nv_bfloat16 &b)



Constructor from two `__nv_bfloat16` variables.

__host__ __device__ __nv_bfloat162(const __nv_bfloat162 &src)



Copy constructor.

__host__ __device__ __nv_bfloat162(const __nv_bfloat162_raw &h2r)



Constructor from `__nv_bfloat162_raw`.

__host__ __device__ operator __nv_bfloat162_raw() const



Conversion operator to `__nv_bfloat162_raw`.

__host__ __device__ __nv_bfloat162 &operator=(__nv_bfloat162 &&src)



Move assignment operator, available for `C++11` and later dialects.

__host__ __device__ __nv_bfloat162 &operator=(const __nv_bfloat162 &src)



Copy assignment operator.

__host__ __device__ __nv_bfloat162 &operator=(const __nv_bfloat162_raw &h2r)



Assignment operator from `__nv_bfloat162_raw`.

Public Members

__nv_bfloat16 x



Storage field holding lower `__nv_bfloat16` part.

__nv_bfloat16 y



Storage field holding upper `__nv_bfloat16` part.