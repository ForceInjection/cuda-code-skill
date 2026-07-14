# 15.8. __nv_bfloat16_raw

**Source:** [struct____nv__bfloat16__raw.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat16__raw.html)

---

#  15.8. __nv_bfloat16_raw[](#nv-bfloat16-raw "Permalink to this headline")

struct __nv_bfloat16_raw[](#_CPPv417__nv_bfloat16_raw "Permalink to this definition")  

    

[__nv_bfloat16_raw](#struct____nv__bfloat16__raw) data type 

Type allows static initialization of `nv_bfloat16` until it becomes a built-in type.

  * Note: this initialization is as a bit-field representation of `nv_bfloat16`, and not a conversion from `short` to `nv_bfloat16`. Such representation will be deprecated in a future version of CUDA.

  * Note: this is visible to non-nvcc compilers, including C-only compilations 


Public Members

unsigned short x[](#_CPPv4N17__nv_bfloat16_raw1xE "Permalink to this definition")  

    

Storage field contains bits representation of the `nv_bfloat16` floating-point number.