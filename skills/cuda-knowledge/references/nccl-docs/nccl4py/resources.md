# Communicator Resources

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html

---

# Communicator Resources[](#communicator-resources "Permalink to this heading")

Resource handles owned by a [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.Communicator"). They share a common lifecycle: each handle is tracked by its owning communicator and is released either explicitly via its `close()` method or automatically when the communicator is destroyed or aborted.

## CommResource[](#commresource "Permalink to this heading")

_class _nccl.core.resources.CommResource(_comm_ptr : int_)[](#nccl.core.resources.CommResource "Permalink to this definition")
    

Bases: `ABC`

Abstract base class for NCCL communicator-owned resources.

Resources are tied to a specific communicator. They can be released explicitly via [`close()`](#nccl.core.resources.CommResource.close "nccl.core.resources.CommResource.close"), and are released automatically when the owning communicator is destroyed or aborted.

close() → None[](#nccl.core.resources.CommResource.close "Permalink to this definition")
    

Explicitly deallocates the resource.

Idempotent: safe to call multiple times.

_property _is_valid _: bool_[](#nccl.core.resources.CommResource.is_valid "Permalink to this definition")
    

Whether the resource has been initialized and is still valid (not closed).

## RegisteredBufferHandle[](#registeredbufferhandle "Permalink to this heading")

_class _nccl.core.RegisteredBufferHandle(_comm_ptr : int_, _buffer_ptr : int_, _size : int_)[](#nccl.core.RegisteredBufferHandle "Permalink to this definition")
    

Bases: [`CommResource`](#nccl.core.resources.CommResource "nccl.core.resources.CommResource")

NCCL registered buffer handle for zero-copy optimized communication.

Registers a user buffer with the communicator to enable performance optimizations in NCCL operations. Created by [`Communicator.register_buffer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_buffer "nccl.core.Communicator.register_buffer"). The registration handle can be released explicitly via `close()`, or automatically when the owning communicator is destroyed or aborted.

_property _handle _: int_[](#nccl.core.RegisteredBufferHandle.handle "Permalink to this definition")
    

Registration handle for NCCL operations.

Raises:
    

**RuntimeError** – If the buffer has been deregistered or the handle is invalid.

_property _size _: int_[](#nccl.core.RegisteredBufferHandle.size "Permalink to this definition")
    

Size of the registered buffer in bytes.

## RegisteredWindowHandle[](#registeredwindowhandle "Permalink to this heading")

_class _nccl.core.RegisteredWindowHandle(_comm_ptr : int_, _buffer_ptr : int_, _size : int_, _flags : [WindowFlag](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.WindowFlag "nccl.core.constants.WindowFlag") | None = None_)[](#nccl.core.RegisteredWindowHandle "Permalink to this definition")
    

Bases: [`CommResource`](#nccl.core.resources.CommResource "nccl.core.resources.CommResource")

NCCL registered window handle for Remote Memory Access (RMA) operations.

Registers a memory window with the communicator for one-sided communication patterns. Created by [`Communicator.register_window()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_window "nccl.core.Communicator.register_window"). Registration is collective: all ranks must call [`Communicator.register_window()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_window "nccl.core.Communicator.register_window") with equal buffer sizes by default. Deregistration is local. The window handle can be released explicitly via `close()`, or automatically when the owning communicator is destroyed or aborted.

_property _is_valid _: bool_[](#nccl.core.RegisteredWindowHandle.is_valid "Permalink to this definition")
    

Whether the resource has been initialized and is still valid (not closed).

_property _handle _: int_[](#nccl.core.RegisteredWindowHandle.handle "Permalink to this definition")
    

Window handle for NCCL operations.

Raises:
    

**RuntimeError** – If the window has been deregistered or the handle is invalid.

_property _size _: int_[](#nccl.core.RegisteredWindowHandle.size "Permalink to this definition")
    

Size of the registered window in bytes.

_property _user_ptr _: int_[](#nccl.core.RegisteredWindowHandle.user_ptr "Permalink to this definition")
    

Original user buffer pointer registered with this window.

Raises:
    

**RuntimeError** – If the window has been deregistered.

get_lsa_multimem_device_pointer(_offset : int = 0_) → int | None[](#nccl.core.RegisteredWindowHandle.get_lsa_multimem_device_pointer "Permalink to this definition")
    

Returns the LSA multicast device pointer for this window.

Returns a device pointer suitable for multicast operations over the LSA (Load/Store Accessible) team. The pointer is valid as long as the window and communicator remain alive.

Parameters:
    

**offset** – Byte offset within the window buffer. Defaults to 0.

Returns:
    

Device pointer as int, or `None` if multimem is not supported.

Raises:
    

**RuntimeError** – If the window has been closed.

get_lsa_device_pointer(_lsa_rank : int_, _offset : int = 0_) → int[](#nccl.core.RegisteredWindowHandle.get_lsa_device_pointer "Permalink to this definition")
    

Returns the LSA device pointer for a peer within the LSA team.

Returns a device pointer to the peer’s window buffer addressable from the local GPU via LSA (Load/Store Accessible) mapping.

Parameters:
    

  * **lsa_rank** – Rank within the LSA team (0 to lsa_size - 1).

  * **offset** – Byte offset within the window buffer. Defaults to 0.


Returns:
    

Device pointer as int.

Raises:
    

**RuntimeError** – If the window has been closed.

get_peer_device_pointer(_peer : int_, _offset : int = 0_) → int | None[](#nccl.core.RegisteredWindowHandle.get_peer_device_pointer "Permalink to this definition")
    

Returns a device pointer to a peer’s window buffer by world rank.

If the peer is not reachable via LSA, returns `None`.

Parameters:
    

  * **peer** – World rank of the peer (0 to nranks - 1).

  * **offset** – Byte offset within the window buffer. Defaults to 0.


Returns:
    

Device pointer as int, or `None` if the peer is not reachable via LSA.

Raises:
    

**RuntimeError** – If the window has been closed.

## CustomRedOp[](#customredop "Permalink to this heading")

_class _nccl.core.CustomRedOp(_comm_ptr : int_, _scalar_ptr : int_, _datatype : [NcclDataType](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclDataType "nccl.core.typing.NcclDataType")_, _residence : nccl.bindings.nccl.ScalarResidence_)[](#nccl.core.CustomRedOp "Permalink to this definition")
    

Bases: [`CommResource`](#nccl.core.resources.CommResource "nccl.core.resources.CommResource")

NCCL user-defined custom reduction operator.

Created by [`Communicator.create_pre_mul_sum()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#nccl.core.Communicator.create_pre_mul_sum "nccl.core.Communicator.create_pre_mul_sum"). The PreMulSum operator performs `output = scalar * sum(inputs)`, useful for averaging or weighted reductions. The operator can be released explicitly via `close()`, or automatically when the owning communicator is destroyed or aborted.

_property _op _: int_[](#nccl.core.CustomRedOp.op "Permalink to this definition")
    

Operator handle for use in reduction operations.

Raises:
    

**RuntimeError** – If the operator has been destroyed or is invalid.

## DevCommResource[](#devcommresource "Permalink to this heading")

_class _nccl.core.DevCommResource(_comm_ptr : int_, _requirements_ptr : int_)[](#nccl.core.DevCommResource "Permalink to this definition")
    

Bases: [`CommResource`](#nccl.core.resources.CommResource "nccl.core.resources.CommResource")

NCCL device communicator resource for device-side operations.

Wraps `ncclDevComm_t` and manages its lifecycle. Created by [`Communicator.create_dev_comm()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html#nccl.core.Communicator.create_dev_comm "nccl.core.Communicator.create_dev_comm"). The device communicator is automatically destroyed when the parent communicator is destroyed or aborted.

_property _ptr _: int_[](#nccl.core.DevCommResource.ptr "Permalink to this definition")
    

Raw pointer to the underlying [`ncclDevComm_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevComm "ncclDevComm") structure.

Raises:
    

**RuntimeError** – If the device communicator has been destroyed.