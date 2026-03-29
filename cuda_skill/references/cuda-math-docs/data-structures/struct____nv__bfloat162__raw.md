# 15.7. __nv_bfloat162_raw

**Source:** struct____nv__bfloat162__raw.html


#  15.7. __nv_bfloat162_raw

struct __nv_bfloat162_raw



__nv_bfloat162_raw data type

Type allows static initialization of `nv_bfloat162` until it becomes a built-in type.

  * Note: this initialization is as a bit-field representation of `nv_bfloat162`, and not a conversion from `short2` to `nv_bfloat162`. Such representation will be deprecated in a future version of CUDA.

  * Note: this is visible to non-nvcc compilers, including C-only compilations


Public Members

unsigned short x



Storage field contains bits of the lower `nv_bfloat16` part.

unsigned short y



Storage field contains bits of the upper `nv_bfloat16` part.