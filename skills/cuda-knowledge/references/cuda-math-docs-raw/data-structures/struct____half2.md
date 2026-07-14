# 15.2. __half2

**Source:** [struct____half2.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2.html)

---

#  15.2. __half2[](#half2 "Permalink to this headline")

struct __half2[](#_CPPv47__half2 "Permalink to this definition")  

    

[__half2](#struct____half2) data type 

This structure implements the datatype for storing two half-precision floating-point numbers. The structure implements assignment, arithmetic and comparison operators, and type conversions.

  * NOTE: [__half2](#struct____half2) is visible to non-nvcc host compilers 


Public Functions

__half2() = default[](#_CPPv4N7__half27__half2Ev "Permalink to this definition")  

    

Constructor by default. 

Emtpy default constructor, result is uninitialized. 

__host__ __device__ inline constexpr __half2(const [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") &a, const [__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") &b)[](#_CPPv4N7__half27__half2ERK6__halfRK6__half "Permalink to this definition")  

    

Constructor from two `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` variables. 

__host__ __device__ inline __half2(const [__half2](#_CPPv4N7__half27__half2ERRK7__half2 "__half2::__half2") &&src)[](#_CPPv4N7__half27__half2ERRK7__half2 "Permalink to this definition")  

    

Move constructor, available for `C++11` and later dialects. 

__host__ __device__ inline __half2(const [__half2](#_CPPv4N7__half27__half2ERK7__half2 "__half2::__half2") &src)[](#_CPPv4N7__half27__half2ERK7__half2 "Permalink to this definition")  

    

Copy constructor. 

__host__ __device__ inline __half2(const [__half2_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2__raw.html#_CPPv411__half2_raw "__half2_raw") &h2r)[](#_CPPv4N7__half27__half2ERK11__half2_raw "Permalink to this definition")  

    

Constructor from `[__half2_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2__raw.html#struct____half2__raw)`. 

__host__ __device__ operator [__half2_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2__raw.html#_CPPv411__half2_raw "__half2_raw")() const[](#_CPPv4NK7__half2cv11__half2_rawEv "Permalink to this definition")  

    

Conversion operator to `[__half2_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2__raw.html#struct____half2__raw)`. 

__host__ __device__ [__half2](#_CPPv47__half2 "__half2") &operator=(const [__half2](#_CPPv47__half2 "__half2") &&src)[](#_CPPv4N7__half2aSERRK7__half2 "Permalink to this definition")  

    

Move assignment operator, available for `C++11` and later dialects. 

__host__ __device__ [__half2](#_CPPv47__half2 "__half2") &operator=(const [__half2](#_CPPv47__half2 "__half2") &src)[](#_CPPv4N7__half2aSERK7__half2 "Permalink to this definition")  

    

Copy assignment operator. 

__host__ __device__ [__half2](#_CPPv47__half2 "__half2") &operator=(const [__half2_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2__raw.html#_CPPv411__half2_raw "__half2_raw") &h2r)[](#_CPPv4N7__half2aSERK11__half2_raw "Permalink to this definition")  

    

Assignment operator from `[__half2_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half2__raw.html#struct____half2__raw)`. 

Public Members

[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") x[](#_CPPv4N7__half21xE "Permalink to this definition")  

    

Storage field holding lower `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` part. 

[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#_CPPv46__half "__half") y[](#_CPPv4N7__half21yE "Permalink to this definition")  

    

Storage field holding upper `[__half](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html#struct____half)` part.