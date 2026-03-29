# 15.3. __half2_raw

**Source:** struct____half2__raw.html


#  15.3. __half2_raw

struct __half2_raw



__half2_raw data type

Type allows static initialization of `half2` until it becomes a built-in type.

  * Note: this initialization is as a bit-field representation of `half2`, and not a conversion from `short2` to `half2`. Such representation will be deprecated in a future version of CUDA.

  * Note: this is visible to non-nvcc compilers, including C-only compilations


Public Members

unsigned short x



Storage field contains bits of the lower `half` part.

unsigned short y



Storage field contains bits of the upper `half` part.