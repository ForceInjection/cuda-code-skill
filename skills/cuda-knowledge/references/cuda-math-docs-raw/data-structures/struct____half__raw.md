# 15.4. __half_raw

**Source:** [struct____half__raw.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half__raw.html)

---

#  15.4. __half_raw[](#half-raw "Permalink to this headline")

struct __half_raw[](#_CPPv410__half_raw "Permalink to this definition")  

    

[__half_raw](#struct____half__raw) data type 

Type allows static initialization of `half` until it becomes a built-in type.

  * Note: this initialization is as a bit-field representation of `half`, and not a conversion from `short` to `half`. Such representation will be deprecated in a future version of CUDA.

  * Note: this is visible to non-nvcc compilers, including C-only compilations 


Public Members

unsigned short x[](#_CPPv4N10__half_raw1xE "Permalink to this definition")  

    

Storage field contains bits representation of the `half` floating-point number.