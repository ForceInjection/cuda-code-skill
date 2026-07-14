# Framework Interop

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/interop.html

---

# Framework Interop[](#framework-interop "Permalink to this heading")

Lazy-loaded helpers for allocating CuPy arrays and PyTorch tensors backed by NCCL-managed memory, plus resolvers that translate framework objects into the `(ptr, count, dtype, device_id)` tuple NCCL expects. The submodules are imported on first attribute access via `nccl.core.cupy` and `nccl.core.torch`.

## CuPy[](#cupy "Permalink to this heading")

nccl.core.interop.cupy.empty(_shape: int | tuple[int, ...], dtype: str | np.dtype | cupy.dtype | type = <class 'float'>, order: Literal['C', 'F'] = 'C'_) → cupy.ndarray[](#nccl.core.interop.cupy.empty "Permalink to this definition")
    

Creates an uninitialized CuPy array backed by NCCL-allocated memory.

Returns an array filled with uninitialized data using NCCL’s memory allocator. This provides a CuPy-compatible interface while using NCCL’s memory allocator for efficient GPU memory management in distributed scenarios. Unlike cupy.empty, the underlying memory is allocated through NCCL.

Memory is automatically freed when the array is garbage collected; no explicit free call is required. For zero-copy optimization, register the array using [`register_buffer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_buffer "nccl.core.Communicator.register_buffer") or [`register_window()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_window "nccl.core.Communicator.register_window").

Parameters:
    

  * **shape** – Shape of the array.

  * **dtype** – Data type specifier. Defaults to `float`.

  * **order** – Memory layout. ‘C’ for row-major (C-style), ‘F’ for column-major (Fortran-style). Defaults to ‘C’.


Returns:
    

An uninitialized CuPy array backed by NCCL-allocated memory.

Raises:
    

  * [**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If order is not ‘C’ or ‘F’.

  * **ModuleNotFoundError** – If CuPy is not installed.


nccl.core.interop.cupy.resolve_array(_array : cupy.ndarray_) → tuple[int, int, [NcclDataType](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclDataType "nccl.core.typing.NcclDataType"), int][](#nccl.core.interop.cupy.resolve_array "Permalink to this definition")
    

Resolves a CuPy array to its NCCL buffer descriptor.

Parameters:
    

**array** – CuPy array to resolve.

Returns:
    

_Tuple of (ptr, count, dtype, device_id)_ – device pointer, element count, NCCL data type, and CUDA device ID.

Raises:
    

  * **ModuleNotFoundError** – If CuPy is not installed.

  * [**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If array is not a CuPy ndarray or its dtype has no NCCL equivalent.


## PyTorch[](#pytorch "Permalink to this heading")

nccl.core.interop.torch.empty(_* size_, _dtype : torch.dtype | None = None_, _device : torch.device | int | str | None = None_, _morder : Literal['C', 'F'] = 'C'_) → torch.Tensor[](#nccl.core.interop.torch.empty "Permalink to this definition")
    

Creates an uninitialized PyTorch tensor backed by NCCL-allocated memory.

Returns a tensor filled with uninitialized data using NCCL’s memory allocator. This provides a PyTorch-compatible interface while using NCCL’s memory allocator for efficient GPU memory management in distributed scenarios. Unlike torch.empty, the underlying memory is allocated through NCCL.

Memory is automatically freed when the tensor is garbage collected; no explicit free call is required. For zero-copy optimization, register the tensor using [`register_buffer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_buffer "nccl.core.Communicator.register_buffer") or [`register_window()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_window "nccl.core.Communicator.register_window").

Parameters:
    

  * ***size** – A sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a single list/tuple.

  * **dtype** – Desired data type of the tensor. If `None`, uses torch.get_default_dtype(). Defaults to `None`.

  * **device** – Device of the tensor. If `None`, uses the current CUDA device. Defaults to `None`.

  * **morder** – Memory layout. ‘C’ for row-major (C-style), ‘F’ for column-major (Fortran-style). Defaults to ‘C’.


Returns:
    

An uninitialized PyTorch tensor backed by NCCL-allocated memory.

Raises:
    

  * [**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If morder is not ‘C’ or ‘F’, or device is not a CUDA device.

  * **ModuleNotFoundError** – If PyTorch is not installed.


nccl.core.interop.torch.resolve_tensor(_tensor : torch.Tensor_) → tuple[int, int, [NcclDataType](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclDataType "nccl.core.typing.NcclDataType"), int][](#nccl.core.interop.torch.resolve_tensor "Permalink to this definition")
    

Resolves a PyTorch tensor to its NCCL buffer descriptor.

Parameters:
    

**tensor** – PyTorch tensor to resolve.

Returns:
    

_Tuple of (ptr, count, dtype, device_id)_ – device pointer, element count, NCCL data type, and CUDA device ID.

Raises:
    

  * **ModuleNotFoundError** – If PyTorch is not installed.

  * [**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If tensor is not a PyTorch tensor or its dtype has no NCCL equivalent.