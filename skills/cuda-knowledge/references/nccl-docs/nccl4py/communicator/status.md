# Status and Utility Methods

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html

---

# Status and Utility Methods[](#status-and-utility-methods "Permalink to this heading")

Methods on [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.Communicator") for resource cleanup and error/status queries.

## close_all_resources[](#close-all-resources "Permalink to this heading")

Communicator.close_all_resources() → None[](#nccl.core.Communicator.close_all_resources "Permalink to this definition")
    

Closes all resources owned by this communicator.

Called automatically during [`destroy()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.destroy "nccl.core.Communicator.destroy") and [`abort()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.abort "nccl.core.Communicator.abort"), but can be called manually. Performs best-effort cleanup, ignoring any errors that occur during resource deallocation. Idempotent: safe to call multiple times.

## get_last_error[](#get-last-error "Permalink to this heading")

Communicator.get_last_error() → str[](#nccl.core.Communicator.get_last_error "Permalink to this definition")
    

Returns the last error string for this communicator.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

## get_async_error[](#get-async-error "Permalink to this heading")

Communicator.get_async_error() → nccl.bindings.nccl.Result[](#nccl.core.Communicator.get_async_error "Permalink to this definition")
    

Queries the progress and potential errors of asynchronous NCCL operations.

Operations without a stream argument (e.g. [`finalize()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.finalize "nccl.core.Communicator.finalize")) are complete when they return `ncclSuccess`. Operations with a stream argument (e.g. [`reduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#nccl.core.Communicator.reduce "nccl.core.Communicator.reduce")) return `ncclSuccess` when posted but may report errors through this method until completed. If any NCCL function returns `ncclInProgress`, users must query the communicator state until it becomes `ncclSuccess` before calling another NCCL function.

Before the state becomes `ncclSuccess`, do not issue CUDA kernels on streams used by NCCL. If an error occurs, destroy the communicator with [`abort()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.abort "nccl.core.Communicator.abort"); nothing can be assumed about the completion or correctness of enqueued operations after an error.

Returns:
    

Current state of the communicator (`ncclSuccess`, `ncclInProgress`, or an error code).

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

See also

[`ncclCommGetAsyncError()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommGetAsyncError "ncclCommGetAsyncError")

## get_mem_stat[](#get-mem-stat "Permalink to this heading")

Communicator.get_mem_stat(_stat : [NcclCommMemStat](#nccl.core.NcclCommMemStat "nccl.core.typing.NcclCommMemStat")_) → int[](#nccl.core.Communicator.get_mem_stat "Permalink to this definition")
    

Queries communicator memory statistics.

Parameters:
    

**stat** – The memory statistic to query.

Returns:
    

The memory statistic value (bytes, or 0/1 for GPU_MEM_SUSPENDED).

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

## NcclCommMemStat[](#ncclcommmemstat "Permalink to this heading")

_class _nccl.core.NcclCommMemStat(_value_ , _names= <not given>_, _*values_ , _module=None_ , _qualname=None_ , _type=None_ , _start=1_ , _boundary=None_)[](#nccl.core.NcclCommMemStat "Permalink to this definition")
    

Bases: `IntEnum`

Memory-statistic selector, mirroring [`ncclCommMemStat_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclCommMemStat_t "ncclCommMemStat_t").

Used as the `stat` argument of [`Communicator.get_mem_stat()`](#nccl.core.Communicator.get_mem_stat "nccl.core.Communicator.get_mem_stat") to identify which memory statistic to query. All values are returned in bytes except [`GPU_MEM_SUSPENDED`](#nccl.core.NcclCommMemStat.GPU_MEM_SUSPENDED "nccl.core.NcclCommMemStat.GPU_MEM_SUSPENDED"), which is a 0/1 flag.

GPU_MEM_SUSPEND _ = 0_[](#nccl.core.NcclCommMemStat.GPU_MEM_SUSPEND "Permalink to this definition")
    

Communicator-allocated GPU memory that can be released by [`Communicator.suspend()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.suspend "nccl.core.Communicator.suspend") (bytes).

GPU_MEM_SUSPENDED _ = 1_[](#nccl.core.NcclCommMemStat.GPU_MEM_SUSPENDED "Permalink to this definition")
    

Whether communicator-allocated GPU memory is currently suspended (`0` = active, `1` = suspended).

GPU_MEM_PERSIST _ = 2_[](#nccl.core.NcclCommMemStat.GPU_MEM_PERSIST "Permalink to this definition")
    

Communicator-allocated GPU memory that cannot be suspended (bytes).

GPU_MEM_TOTAL _ = 3_[](#nccl.core.NcclCommMemStat.GPU_MEM_TOTAL "Permalink to this definition")
    

Total communicator-allocated GPU memory tracked by NCCL (bytes).

## get_error_string[](#get-error-string "Permalink to this heading")

Module-level helper to render an NCCL result code as a human-readable string.

nccl.core.get_error_string(_nccl_result : _nccl_bindings.Result | int_) → str[](#nccl.core.get_error_string "Permalink to this definition")
    

Returns a human-readable error string for an NCCL result code.

Parameters:
    

**nccl_result** – NCCL result code.

Returns:
    

Human-readable error message corresponding to the result code.