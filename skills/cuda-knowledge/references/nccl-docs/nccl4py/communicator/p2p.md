# Point-to-Point and Signal Methods

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html

---

# Point-to-Point and Signal Methods[](#point-to-point-and-signal-methods "Permalink to this heading")

Methods on [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.Communicator") for point-to-point and signal/wait operations. See [Point To Point Communication Functions](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html) for the corresponding C API.

## send[](#send "Permalink to this heading")

Communicator.send(_sendbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _peer : int_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.send "Permalink to this definition")
    

Sends a buffer to a peer rank.

Parameters:
    

  * **sendbuf** – Source buffer to send.

  * **peer** – Destination rank ID.

  * **stream** – CUDA stream for the operation. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the buffer specification is invalid, the buffer is on the wrong device, or the communicator is not initialized.

See also

[`ncclSend()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclSend "ncclSend")

## recv[](#recv "Permalink to this heading")

Communicator.recv(_recvbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _peer : int_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.recv "Permalink to this definition")
    

Receives data into a buffer from a peer rank.

Parameters:
    

  * **recvbuf** – Destination buffer to receive into.

  * **peer** – Source rank ID.

  * **stream** – CUDA stream for the operation. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the buffer specification is invalid, the buffer is on the wrong device, or the communicator is not initialized.

See also

[`ncclRecv()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclRecv "ncclRecv")

## signal[](#signal "Permalink to this heading")

Communicator.signal(_peer : int_, _signal_index : int = 0_, _context : int = 0_, _flags : int = 0_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.signal "Permalink to this definition")
    

Sends a signal to a peer rank.

Enqueues a signal operation on the specified CUDA stream that notifies the target peer rank. The peer can wait for this signal using [`wait_signal()`](#nccl.core.Communicator.wait_signal "nccl.core.Communicator.wait_signal").

Parameters:
    

  * **peer** – Target rank to send the signal to.

  * **signal_index** – Signal index identifier. Currently must be 0.

  * **context** – Context identifier. Currently must be 0.

  * **flags** – Reserved for future use. Currently must be 0.

  * **stream** – CUDA stream to enqueue the signal operation on. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

See also

[`ncclSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclSignal "ncclSignal")

## wait_signal[](#wait-signal "Permalink to this heading")

Communicator.wait_signal(_descs : [WaitSignalDesc](#nccl.core.WaitSignalDesc "nccl.core.communicator.WaitSignalDesc") | Sequence[[WaitSignalDesc](#nccl.core.WaitSignalDesc "nccl.core.communicator.WaitSignalDesc")]_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.wait_signal "Permalink to this definition")
    

Waits for signals as described by the signal descriptor(s).

Enqueues a wait operation on the specified CUDA stream that blocks until the required signals from peer ranks are received. Each descriptor specifies a peer rank and the number of signal operations to wait for from that peer.

Parameters:
    

  * **descs** – One or more [`WaitSignalDesc`](#nccl.core.WaitSignalDesc "nccl.core.WaitSignalDesc") descriptors specifying which peers to wait for and how many signals to expect from each.

  * **stream** – CUDA stream to enqueue the wait operation on. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

See also

[`ncclWaitSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclWaitSignal "ncclWaitSignal")

## put_signal[](#put-signal "Permalink to this heading")

Communicator.put_signal(_local_buffer : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _peer : int_, _peer_window : [RegisteredWindowHandle](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.RegisteredWindowHandle "nccl.core.resources.RegisteredWindowHandle")_, _peer_window_offset : int = 0_, _signal_index : int = 0_, _context : int = 0_, _flags : int = 0_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.put_signal "Permalink to this definition")
    

Puts data from a local buffer to a peer’s window and sends a signal.

Enqueues a put-with-signal operation on the specified CUDA stream that transfers the local buffer contents to the target peer’s registered window and notifies that peer. The peer can wait for this signal (and thus for the put to complete) using [`wait_signal()`](#nccl.core.Communicator.wait_signal "nccl.core.Communicator.wait_signal"). The peer’s memory must be registered with [`register_window()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_window "nccl.core.Communicator.register_window"); pass the peer’s window handle as `peer_window` (e.g. obtained via an allgather of window handles).

Parameters:
    

  * **local_buffer** – Source buffer whose contents are put to the peer.

  * **peer** – Target rank to put the data to and send the signal to.

  * **peer_window** – Peer’s [`RegisteredWindowHandle`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.RegisteredWindowHandle "nccl.core.RegisteredWindowHandle") (from [`register_window()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#nccl.core.Communicator.register_window "nccl.core.Communicator.register_window")).

  * **peer_window_offset** – Offset in the peer’s window in elements. Defaults to 0.

  * **signal_index** – Signal index identifier. Currently must be 0.

  * **context** – Context identifier. Currently must be 0.

  * **flags** – Reserved for future use. Currently must be 0.

  * **stream** – CUDA stream to enqueue the put_signal operation on. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized, or if the buffer specification is invalid or the buffer is on a different device than the communicator.

See also

[`ncclPutSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclPutSignal "ncclPutSignal")

## WaitSignalDesc[](#waitsignaldesc "Permalink to this heading")

_class _nccl.core.WaitSignalDesc(_peer : int_, _op_count : int = 1_, _signal_index : int = 0_, _context : int = 0_)[](#nccl.core.WaitSignalDesc "Permalink to this definition")
    

Bases: `object`

Descriptor for a wait-signal operation.

Describes a single signal-wait operation for use with [`Communicator.wait_signal()`](#nccl.core.Communicator.wait_signal "nccl.core.Communicator.wait_signal"). Each descriptor specifies which peer to wait for, how many signal operations to wait for, and additional context for the wait operation.

peer _: int_[](#nccl.core.WaitSignalDesc.peer "Permalink to this definition")
    

Target peer rank to wait for signals from.

op_count _: int_ _ = 1_[](#nccl.core.WaitSignalDesc.op_count "Permalink to this definition")
    

Number of signal operations to wait for from the peer. Defaults to 1.

signal_index _: int_ _ = 0_[](#nccl.core.WaitSignalDesc.signal_index "Permalink to this definition")
    

Signal index identifier. Currently must be 0. Defaults to 0.

context _: int_ _ = 0_[](#nccl.core.WaitSignalDesc.context "Permalink to this definition")
    

Context identifier. Currently must be 0. Defaults to 0.