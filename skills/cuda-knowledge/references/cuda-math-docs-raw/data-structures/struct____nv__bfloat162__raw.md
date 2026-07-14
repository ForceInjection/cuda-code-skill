# 15.7. __nv_bfloat162_raw

**Source:** [struct____nv__bfloat162__raw.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162__raw.html)

---

#  15.7. __nv_bfloat162_raw[](#nv-bfloat162-raw "Permalink to this headline")

struct __nv_bfloat162_raw[](#_CPPv418__nv_bfloat162_raw "Permalink to this definition")  

    

[__nv_bfloat162_raw](#struct____nv__bfloat162__raw) data type 

Type allows static initialization of `nv_bfloat162` until it becomes a built-in type.

  * Note: this initialization is as a bit-field representation of `nv_bfloat162`, and not a conversion from `short2` to `nv_bfloat162`. Such representation will be deprecated in a future version of CUDA.

  * Note: this is visible to non-nvcc compilers, including C-only compilations 


Public Members

unsigned short x[](#_CPPv4N18__nv_bfloat162_raw1xE "Permalink to this definition")  

    

Storage field contains bits of the lower `nv_bfloat16` part. 

unsigned short y[](#_CPPv4N18__nv_bfloat162_raw1yE "Permalink to this definition")  

    

Storage field contains bits of the upper `nv_bfloat16` part.