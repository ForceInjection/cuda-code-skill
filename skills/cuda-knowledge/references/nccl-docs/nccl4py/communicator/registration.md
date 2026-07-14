# Memory Registration Methods

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html

---

# Memory Registration Methods[](#memory-registration-methods "Permalink to this heading")

Methods on [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.Communicator") for registering buffers and windows for zero-copy and RMA operations. The returned handle classes are documented under [Memory Management](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/memory.html).

## register_buffer[](#register-buffer "Permalink to this heading")

Communicator.register_buffer(_buffer : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_) → [RegisteredBufferHandle](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.RegisteredBufferHandle "nccl.core.resources.RegisteredBufferHandle")[](#nccl.core.Communicator.register_buffer "Permalink to this definition")
    

Registers a buffer with this communicator for zero-copy communication.

Registered buffers can enable performance optimizations in NCCL operations. Buffer size is automatically derived from buffer count and dtype. The returned [`RegisteredBufferHandle`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.RegisteredBufferHandle "nccl.core.RegisteredBufferHandle") is tracked by the communicator and may be released explicitly via its `close()` method, or automatically when the communicator is destroyed or aborted.

Parameters:
    

**buffer** – Buffer to register (array, Buffer, or buffer-like object).

Returns:
    

[`RegisteredBufferHandle`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.RegisteredBufferHandle "nccl.core.RegisteredBufferHandle") for the registered buffer.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the buffer is on the wrong device or the communicator is not initialized.

See also

[`ncclCommRegister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommRegister "ncclCommRegister")

## register_window[](#register-window "Permalink to this heading")

Communicator.register_window(_buffer : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _flags : [WindowFlag](#nccl.core.WindowFlag "nccl.core.constants.WindowFlag") | None = None_) → [RegisteredWindowHandle](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.RegisteredWindowHandle "nccl.core.resources.RegisteredWindowHandle") | None[](#nccl.core.Communicator.register_window "Permalink to this definition")
    

Collectively registers a local buffer into an NCCL window.

This is a collective call: every rank in the communicator must participate, and buffer size must be equal among ranks by default. Buffer size is automatically derived from buffer count and dtype. If called within a group, the handle value may not be filled until `ncclGroupEnd` completes. For non-blocking communicators, the handle may remain `0` until [`get_async_error()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html#nccl.core.Communicator.get_async_error "nccl.core.Communicator.get_async_error") reports success.

The returned [`RegisteredWindowHandle`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.RegisteredWindowHandle "nccl.core.RegisteredWindowHandle") is tracked by the communicator and may be released explicitly via its `close()` method, or automatically when the communicator is destroyed or aborted.

Parameters:
    

  * **buffer** – Local buffer to register as a window.

  * **flags** – Window registration flags. Defaults to `None` ([`DEFAULT`](#nccl.core.WindowFlag.DEFAULT "nccl.core.WindowFlag.DEFAULT")).


Returns:
    

[`RegisteredWindowHandle`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.RegisteredWindowHandle "nccl.core.RegisteredWindowHandle") for the registered window, or `None` if NCCL returns a NULL handle (e.g. windows are unsupported on this platform).

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the buffer is on the wrong device or the communicator is not initialized.

See also

[`ncclCommWindowRegister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister")

## WindowFlag[](#windowflag "Permalink to this heading")

_class _nccl.core.WindowFlag(_value_ , _names= <not given>_, _*values_ , _module=None_ , _qualname=None_ , _type=None_ , _start=1_ , _boundary=None_)[](#nccl.core.WindowFlag "Permalink to this definition")
    

Bases: `IntFlag`

Window registration behavior flags for [`Communicator.register_window()`](#nccl.core.Communicator.register_window "nccl.core.Communicator.register_window").

DEFAULT _ = 0_[](#nccl.core.WindowFlag.DEFAULT "Permalink to this definition")
    

Default window registration.

COLL_SYMMETRIC _ = 1_[](#nccl.core.WindowFlag.COLL_SYMMETRIC "Permalink to this definition")
    

Collective symmetric window registration.

STRICT_ORDERING _ = 2_[](#nccl.core.WindowFlag.STRICT_ORDERING "Permalink to this definition")
    

Strict ordering for window operations.