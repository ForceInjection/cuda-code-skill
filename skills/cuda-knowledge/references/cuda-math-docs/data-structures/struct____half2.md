# 15.2. __half2

**Source:** struct____half2.html


#  15.2. __half2

struct __half2



__half2 data type

This structure implements the datatype for storing two half-precision floating-point numbers. The structure implements assignment, arithmetic and comparison operators, and type conversions.

  * NOTE: __half2 is visible to non-nvcc host compilers


Public Functions

__half2() = default



Constructor by default.

Emtpy default constructor, result is uninitialized.

__host__ __device__ inline constexpr __half2(const __half &a, const __half &b)



Constructor from two `__half` variables.

__host__ __device__ inline __half2(const __half2 &&src)



Move constructor, available for `C++11` and later dialects.

__host__ __device__ inline __half2(const __half2 &src)



Copy constructor.

__host__ __device__ inline __half2(const __half2_raw &h2r)



Constructor from `__half2_raw`.

__host__ __device__ operator __half2_raw() const



Conversion operator to `__half2_raw`.

__host__ __device__ __half2 &operator=(const __half2 &&src)



Move assignment operator, available for `C++11` and later dialects.

__host__ __device__ __half2 &operator=(const __half2 &src)



Copy assignment operator.

__host__ __device__ __half2 &operator=(const __half2_raw &h2r)



Assignment operator from `__half2_raw`.

Public Members

__half x



Storage field holding lower `__half` part.

__half y



Storage field holding upper `__half` part.