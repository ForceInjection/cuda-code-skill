# 15.6. __nv_bfloat162

**Source:** [struct____nv__bfloat162.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html)

---

#  15.6. __nv_bfloat162[](#nv-bfloat162 "Permalink to this headline")

struct __nv_bfloat162[](#_CPPv414__nv_bfloat162 "Permalink to this definition")  

    

nv_bfloat162 datatype 

This structure implements the datatype for storing two nv_bfloat16 floating-point numbers. The structure implements assignment, arithmetic and comparison operators, and type conversions.

  * NOTE: [__nv_bfloat162](#struct____nv__bfloat162) is visible to non-nvcc host compilers 


Public Functions

__nv_bfloat162() = default[](#_CPPv4N14__nv_bfloat16214__nv_bfloat162Ev "Permalink to this definition")  

    

Constructor by default. 

Emtpy default constructor, result is uninitialized. 

__host__ __device__ __nv_bfloat162([__nv_bfloat162](#_CPPv4N14__nv_bfloat16214__nv_bfloat162ERR14__nv_bfloat162 "__nv_bfloat162::__nv_bfloat162") &&src)[](#_CPPv4N14__nv_bfloat16214__nv_bfloat162ERR14__nv_bfloat162 "Permalink to this definition")  

    

Move constructor, available for `C++11` and later dialects. 

__host__ __device__ inline constexpr __nv_bfloat162(const [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") &a, const [__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") &b)[](#_CPPv4N14__nv_bfloat16214__nv_bfloat162ERK13__nv_bfloat16RK13__nv_bfloat16 "Permalink to this definition")  

    

Constructor from two `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` variables. 

__host__ __device__ __nv_bfloat162(const [__nv_bfloat162](#_CPPv4N14__nv_bfloat16214__nv_bfloat162ERK14__nv_bfloat162 "__nv_bfloat162::__nv_bfloat162") &src)[](#_CPPv4N14__nv_bfloat16214__nv_bfloat162ERK14__nv_bfloat162 "Permalink to this definition")  

    

Copy constructor. 

__host__ __device__ __nv_bfloat162(const [__nv_bfloat162_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162__raw.html#_CPPv418__nv_bfloat162_raw "__nv_bfloat162_raw") &h2r)[](#_CPPv4N14__nv_bfloat16214__nv_bfloat162ERK18__nv_bfloat162_raw "Permalink to this definition")  

    

Constructor from `[__nv_bfloat162_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162__raw.html#struct____nv__bfloat162__raw)`. 

__host__ __device__ operator [__nv_bfloat162_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162__raw.html#_CPPv418__nv_bfloat162_raw "__nv_bfloat162_raw")() const[](#_CPPv4NK14__nv_bfloat162cv18__nv_bfloat162_rawEv "Permalink to this definition")  

    

Conversion operator to `[__nv_bfloat162_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162__raw.html#struct____nv__bfloat162__raw)`. 

__host__ __device__ [__nv_bfloat162](#_CPPv414__nv_bfloat162 "__nv_bfloat162") &operator=([__nv_bfloat162](#_CPPv414__nv_bfloat162 "__nv_bfloat162") &&src)[](#_CPPv4N14__nv_bfloat162aSERR14__nv_bfloat162 "Permalink to this definition")  

    

Move assignment operator, available for `C++11` and later dialects. 

__host__ __device__ [__nv_bfloat162](#_CPPv414__nv_bfloat162 "__nv_bfloat162") &operator=(const [__nv_bfloat162](#_CPPv414__nv_bfloat162 "__nv_bfloat162") &src)[](#_CPPv4N14__nv_bfloat162aSERK14__nv_bfloat162 "Permalink to this definition")  

    

Copy assignment operator. 

__host__ __device__ [__nv_bfloat162](#_CPPv414__nv_bfloat162 "__nv_bfloat162") &operator=(const [__nv_bfloat162_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162__raw.html#_CPPv418__nv_bfloat162_raw "__nv_bfloat162_raw") &h2r)[](#_CPPv4N14__nv_bfloat162aSERK18__nv_bfloat162_raw "Permalink to this definition")  

    

Assignment operator from `[__nv_bfloat162_raw](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162__raw.html#struct____nv__bfloat162__raw)`. 

Public Members

[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") x[](#_CPPv4N14__nv_bfloat1621xE "Permalink to this definition")  

    

Storage field holding lower `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` part. 

[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#_CPPv413__nv_bfloat16 "__nv_bfloat16") y[](#_CPPv4N14__nv_bfloat1621yE "Permalink to this definition")  

    

Storage field holding upper `[__nv_bfloat16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html#struct____nv__bfloat16)` part.