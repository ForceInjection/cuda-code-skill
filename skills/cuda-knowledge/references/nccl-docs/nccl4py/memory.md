# Memory Management

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/memory.html

---

# Memory Management[](#memory-management "Permalink to this heading")

NCCL-backed device memory allocation; see [Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#mem-allocator) for usage details. For zero-copy registration of existing buffers, see [`Communicator.register_buffer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_buffer "nccl.core.Communicator.register_buffer") and [`Communicator.register_window()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_window "nccl.core.Communicator.register_window").

## mem_alloc[](#mem-alloc "Permalink to this heading")

nccl.core.mem_alloc(_size : int_, _device : [Device](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Device.html#cuda.core.Device "\(in cuda.core\)") | int | None = None_) → [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)")[](#nccl.core.mem_alloc "Permalink to this definition")
    

Allocates GPU buffer memory using NCCL’s memory allocator.

The actual allocated size may be larger than requested due to buffer granularity requirements from NCCL optimizations. The returned buffer can be explicitly freed with [`mem_free()`](#nccl.core.mem_free "nccl.core.mem_free") or automatically freed when garbage collected.

Parameters:
    

  * **size** – Number of bytes to allocate.

  * **device** – Target CUDA device. Defaults to the current device.


Returns:
    

A CUDA buffer object backed by NCCL-managed memory. The buffer is allocated on the specified device; the current device is restored after allocation.

## mem_free[](#mem-free "Permalink to this heading")

nccl.core.mem_free(_buf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)")_) → None[](#nccl.core.mem_free "Permalink to this definition")
    

Frees memory allocated by [`mem_alloc()`](#nccl.core.mem_alloc "nccl.core.mem_alloc").

Explicit deallocation is optional. Memory is automatically freed when the Buffer object is garbage collected.

Parameters:
    

**buf** – The buffer to free.