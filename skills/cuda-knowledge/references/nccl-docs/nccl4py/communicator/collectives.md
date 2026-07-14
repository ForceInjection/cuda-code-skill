# Collective Communication Methods

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html

---

# Collective Communication Methods[](#collective-communication-methods "Permalink to this heading")

Methods on [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.Communicator") for collective communication. See [Collective Communication Functions](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html) for the corresponding C API.

## allreduce[](#allreduce "Permalink to this heading")

Communicator.allreduce(_sendbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _recvbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _op : [NcclRedOp](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclRedOp "nccl.core.typing.NcclRedOp") | [CustomRedOp](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.CustomRedOp "nccl.core.resources.CustomRedOp")_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.allreduce "Permalink to this definition")
    

All-reduce variant of [`reduce()`](#nccl.core.Communicator.reduce "nccl.core.Communicator.reduce").

Equivalent to `reduce(sendbuf, recvbuf, op, root=None, stream=stream)`: reduces data across all ranks and stores identical copies in each rank’s recvbuf. See [`reduce()`](#nccl.core.Communicator.reduce "nccl.core.Communicator.reduce") for argument semantics.

See also

[`reduce()`](#nccl.core.Communicator.reduce "nccl.core.Communicator.reduce"), [`ncclAllReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllReduce "ncclAllReduce")

## broadcast[](#broadcast "Permalink to this heading")

Communicator.broadcast(_sendbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI | Any_, _recvbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _root : int_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.broadcast "Permalink to this definition")
    

Copies data from `sendbuf` on the root rank to all ranks’ `recvbuf`.

`sendbuf` is only used on the root rank and is ignored on other ranks.

On the root rank, both buffers must have matching data types and `sendcount == recvcount`. Element count is inferred from `recvbuf`: `count = recvcount`. In-place operation occurs when `sendbuf` and `recvbuf` resolve to the same device memory address.

Parameters:
    

  * **sendbuf** – Source buffer (only used on the root rank).

  * **recvbuf** – Destination buffer that will receive the broadcast data.

  * **root** – Root rank that broadcasts the data (0 to `nranks - 1`).

  * **stream** – CUDA stream for the operation. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If send and receive buffers have mismatched dtypes, mismatched counts, are on the wrong device, are invalid specifications, or the communicator is not initialized.

See also

[`ncclBroadcast()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclBroadcast "ncclBroadcast")

## reduce[](#reduce "Permalink to this heading")

Communicator.reduce(_sendbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _recvbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI | Any_, _op : [NcclRedOp](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclRedOp "nccl.core.typing.NcclRedOp") | [CustomRedOp](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.CustomRedOp "nccl.core.resources.CustomRedOp")_, _root : int | None = None_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.reduce "Permalink to this definition")
    

Reduces data from all ranks using the specified operation.

Supports two modes. In AllReduce mode (`root` is `None`) all ranks receive the reduced result in `recvbuf`. In Reduce mode (`root` specified) only the root rank receives the reduced result; `recvbuf` is ignored on other ranks.

Both buffers must have matching data types where used. Element count is inferred from `sendbuf`: `count = sendcount`. In AllReduce mode, all ranks must have `recvcount >= sendcount`; in Reduce mode, only the root rank requires `recvcount >= sendcount`. In-place operation occurs when `sendbuf` and `recvbuf` resolve to the same device memory address.

Parameters:
    

  * **sendbuf** – Source buffer containing data to be reduced.

  * **recvbuf** – Destination buffer for the reduced result. Only used on the root rank in Reduce mode.

  * **op** – Reduction operator (e.g. `SUM`, `MAX`, `MIN`, `AVG`, `PROD`, or a [`CustomRedOp`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.CustomRedOp "nccl.core.CustomRedOp")).

  * **root** – Root rank that receives the reduced result (0 to `nranks - 1`). If `None`, performs an all-reduce. Defaults to `None`.

  * **stream** – CUDA stream for the operation. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If send and receive buffers have mismatched dtypes, mismatched counts, are on the wrong device, are invalid specifications, or the communicator is not initialized.

See also

[`ncclAllReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllReduce "ncclAllReduce"), [`ncclReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduce "ncclReduce")

## allgather[](#allgather "Permalink to this heading")

Communicator.allgather(_sendbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _recvbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.allgather "Permalink to this definition")
    

All-gather variant of [`gather()`](#nccl.core.Communicator.gather "nccl.core.Communicator.gather").

Equivalent to `gather(sendbuf, recvbuf, root=None, stream=stream)`: gathers `sendcount` values from each rank and places identical copies of the concatenated result in every rank’s recvbuf. See [`gather()`](#nccl.core.Communicator.gather "nccl.core.Communicator.gather") for argument semantics.

See also

[`gather()`](#nccl.core.Communicator.gather "nccl.core.Communicator.gather"), [`ncclAllGather()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllGather "ncclAllGather")

## reduce_scatter[](#reduce-scatter "Permalink to this heading")

Communicator.reduce_scatter(_sendbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _recvbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _op : [NcclRedOp](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclRedOp "nccl.core.typing.NcclRedOp") | [CustomRedOp](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.CustomRedOp "nccl.core.resources.CustomRedOp")_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.reduce_scatter "Permalink to this definition")
    

Reduces data from all ranks and scatters the result across ranks.

Each rank receives a different portion of the reduced result: rank `i` receives the i-th block in its `recvbuf`.

Both buffers must have matching data types. Element count is inferred from `sendbuf`: `count = sendcount / nranks`. `sendcount` must be `>= nranks` and `recvcount` must be `>= count`. In-place operation occurs when `recvbuf` resolves to `sendbuf_address + rank * count`.

Parameters:
    

  * **sendbuf** – Source buffer (size `>= nranks * recvcount` elements).

  * **recvbuf** – Destination buffer with `recvcount` elements.

  * **op** – Reduction operator (e.g. `SUM`, `MAX`, `MIN`, `AVG`, `PROD`, or a [`CustomRedOp`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.CustomRedOp "nccl.core.CustomRedOp")).

  * **stream** – CUDA stream for the operation. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If send and receive buffers have mismatched dtypes, `sendbuf` is too small, are on the wrong device, are invalid specifications, or the communicator is not initialized.

See also

[`ncclReduceScatter()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduceScatter "ncclReduceScatter")

## alltoall[](#alltoall "Permalink to this heading")

Communicator.alltoall(_sendbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _recvbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.alltoall "Permalink to this definition")
    

Each rank sends and receives `count` values to and from every other rank.

Data sent to destination rank `j` is taken from `sendbuf + j * count` and data received from source rank `i` is placed at `recvbuf + i * count`.

Both buffers must have matching data types. Element count is inferred from `sendbuf`: `count = sendcount / nranks`. `sendcount` must be `>= nranks` and `recvcount` must be `>= sendcount`.

Parameters:
    

  * **sendbuf** – Source buffer (size `>= nranks * count` elements).

  * **recvbuf** – Destination buffer (size `>= nranks * count` elements).

  * **stream** – CUDA stream for the operation. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If send and receive buffers have mismatched dtypes, buffer sizes are incompatible with `nranks`, are on the wrong device, are invalid specifications, or the communicator is not initialized.

See also

[`ncclAlltoAll()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAlltoAll "ncclAlltoAll")

## gather[](#gather "Permalink to this heading")

Communicator.gather(_sendbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _recvbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI | Any_, _root : int | None = None_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.gather "Permalink to this definition")
    

Gathers `sendcount` values from all ranks.

Supports two modes. In AllGather mode (`root` is `None`) values are gathered from all ranks and identical copies of the result are placed in each `recvbuf`. In Gather mode (`root` specified) values are gathered to the specified root rank only; `recvbuf` is ignored on other ranks.

Both buffers must have matching data types where used. Element count is inferred from `sendbuf`: `count = sendcount`. Data from rank `i` is placed at `recvbuf + i * sendcount`. AllGather mode requires `recvcount >= nranks * sendcount` on every rank; Gather mode requires it only on the root rank.

In-place operation occurs when `sendbuf` resolves to `recvbuf_address + rank * sendcount` in AllGather mode, or to `recvbuf_address + root * sendcount` in Gather mode.

Parameters:
    

  * **sendbuf** – Source buffer containing `sendcount` elements.

  * **recvbuf** – Destination buffer (size `>= nranks * sendcount` elements). In Gather mode, only used on the root rank.

  * **root** – Root rank that receives the gathered data (0 to `nranks - 1`). If `None`, performs an all-gather. Defaults to `None`.

  * **stream** – CUDA stream for the operation. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If send and receive buffers have mismatched dtypes, `recvbuf` is too small, are on the wrong device, are invalid specifications, or the communicator is not initialized.

See also

[`ncclAllGather()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllGather "ncclAllGather"), [`ncclGather()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclGather "ncclGather")

## scatter[](#scatter "Permalink to this heading")

Communicator.scatter(_sendbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI | Any_, _recvbuf : [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _root : int_, _*_ , _stream : [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream "\(in cuda.core\)") | cuda.core.typing.IsStreamType | int | None = None_) → None[](#nccl.core.Communicator.scatter "Permalink to this definition")
    

Scatters data from the root rank to all ranks.

Each rank receives `count` elements from the root rank. On the root rank, `count` elements from `sendbuf + i * count` are sent to rank `i`. `sendbuf` is not used on non-root ranks.

On the root rank, both buffers must have matching data types. Element count is inferred from `recvbuf`: `count = recvcount`. The root rank requires `sendcount >= nranks` and `sendcount / nranks == recvcount`. In-place operation occurs when `recvbuf` resolves to `sendbuf_address + root * count`.

Parameters:
    

  * **sendbuf** – Source buffer (only used on the root rank, size `>= nranks * count` elements).

  * **recvbuf** – Destination buffer with `count` elements.

  * **root** – Root rank that scatters the data (0 to `nranks - 1`).

  * **stream** – CUDA stream for the operation. Defaults to `None` (the default stream).


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If send and receive buffers have mismatched dtypes, `sendbuf` is too small on the root rank, are on the wrong device, are invalid specifications, or the communicator is not initialized.

See also

[`ncclScatter()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclScatter "ncclScatter")

## create_pre_mul_sum[](#create-pre-mul-sum "Permalink to this heading")

Communicator.create_pre_mul_sum(_scalar : int | float | numpy.ndarray | [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer "\(in cuda.core\)") | SupportsDLPack | SupportsCAI_, _datatype : [NcclDataType](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclDataType "nccl.core.typing.NcclDataType") | None = None_) → [CustomRedOp](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.CustomRedOp "nccl.core.resources.CustomRedOp")[](#nccl.core.Communicator.create_pre_mul_sum "Permalink to this definition")
    

Creates a PreMulSum custom reduction operator.

Performs `output = scalar * sum(inputs)` and is useful for averaging (`scalar = 1/N`) or weighted reductions. The returned [`CustomRedOp`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.CustomRedOp "nccl.core.CustomRedOp") is tracked by the communicator and may be released explicitly via its `close()` method, or automatically when the communicator is destroyed or aborted.

Parameters:
    

  * **scalar** – Scalar multiplier value. A Python int or float is converted to a NumPy array using host memory. A NumPy array must contain exactly 1 element and uses host memory. An `NcclSupportedBuffer` is treated as a device buffer with exactly 1 element.

  * **datatype** – NCCL data type of the scalar and reduction. If `None`, it is inferred from `scalar`: Python `int` becomes `int64` and Python `float` becomes `float64` (NumPy’s natural dtypes); a NumPy array uses the array’s dtype; a device buffer uses the buffer’s dtype.


Returns:
    

[`CustomRedOp`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.CustomRedOp "nccl.core.CustomRedOp") for the PreMulSum operator.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized; the scalar type is unsupported; the NumPy array or device buffer does not contain exactly 1 element; or the requested datatype does not match a device buffer’s dtype.

See also

[`ncclRedOpCreatePreMulSum()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#c.ncclRedOpCreatePreMulSum "ncclRedOpCreatePreMulSum")