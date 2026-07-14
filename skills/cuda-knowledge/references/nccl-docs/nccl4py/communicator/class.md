# Communicator Class

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html

---

# Communicator Class[](#communicator-class "Permalink to this heading")

_class _nccl.core.Communicator(_ptr : int = 0_)[](#nccl.core.Communicator "Permalink to this definition")
    

Bases: `object`

NCCL communicator for collective and point-to-point operations.

A communicator represents a group of ranks that can perform collective operations (e.g. allreduce, broadcast) and point-to-point operations (send/recv). Each rank has a unique ID in `[0, nranks)`.

Communicator instances expose a number of properties for inspection (`ptr`, `nranks`, `device`, `rank`, plus device-API related properties like `cuda_dev`, `nvml_dev`, `device_api_support`, `multimem_support`, `gin_type`, `n_lsa_teams`, `host_rma_support`, `railed_gin_type`); see the per-property documentation for details.

__init__(_ptr : int = 0_) → None[](#nccl.core.Communicator.__init__ "Permalink to this definition")
    

Initializes a communicator with a raw NCCL pointer.

Unlike the [`init()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.init "nccl.core.Communicator.init") classmethod, this constructor allows `ptr=0` for creating null communicators (e.g. when [`split()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.split "nccl.core.Communicator.split") excludes a rank). A null communicator can later be initialized via [`initialize()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.initialize "nccl.core.Communicator.initialize"), or used as the caller for [`grow()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.grow "nccl.core.Communicator.grow") to join an existing communicator.

Parameters:
    

**ptr** – Integer representing an NCCL communicator pointer (0 for a null communicator). Defaults to 0.

## Properties[](#properties "Permalink to this heading")

### Identity[](#identity "Permalink to this heading")

Communicator.ptr[](#nccl.core.Communicator.ptr "Permalink to this definition")
    

Raw pointer to the underlying [`ncclComm_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") structure (0 if destroyed or null).

Communicator.is_valid[](#nccl.core.Communicator.is_valid "Permalink to this definition")
    

Whether the communicator is valid (not destroyed or null).

Communicator.nranks[](#nccl.core.Communicator.nranks "Permalink to this definition")
    

Total number of ranks in the communicator.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.device[](#nccl.core.Communicator.device "Permalink to this definition")
    

CUDA device associated with this communicator.

Returns a [`cuda.core.Device`](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Device.html#cuda.core.Device "\(in cuda.core\)") providing additional functionality such as `to_system_device` for obtaining the NVML device, device properties, and synchronization.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.rank[](#nccl.core.Communicator.rank "Permalink to this definition")
    

This caller’s rank within the communicator (0 to nranks - 1).

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

### Device-API capability[](#device-api-capability "Permalink to this heading")

These properties reflect the underlying NCCL [`ncclCommProperties_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclCommProperties_t "ncclCommProperties_t") structure.

Communicator.cuda_dev[](#nccl.core.Communicator.cuda_dev "Permalink to this definition")
    

CUDA device ID associated with this communicator.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.nvml_dev[](#nccl.core.Communicator.nvml_dev "Permalink to this definition")
    

NVML device ID for the GPU associated with this communicator.

Uses the NVML indexing space, which may differ from CUDA indexing.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.device_api_support[](#nccl.core.Communicator.device_api_support "Permalink to this definition")
    

Whether device-side NCCL operations are supported on this platform.

If False, a device communicator cannot be created.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.multimem_support[](#nccl.core.Communicator.multimem_support "Permalink to this definition")
    

Whether ranks in the same LSA team can communicate using multimem.

If False, a device communicator cannot be created with multimem resources.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.gin_type[](#nccl.core.Communicator.gin_type "Permalink to this definition")
    

GPU Interconnect Network (GIN) type.

If equal to NcclGinType.NONE, a device communicator cannot be created with GIN resources.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.n_lsa_teams[](#nccl.core.Communicator.n_lsa_teams "Permalink to this definition")
    

Number of Local Shared Array (LSA) teams for this communicator.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.host_rma_support[](#nccl.core.Communicator.host_rma_support "Permalink to this definition")
    

Whether host RMA is supported on this communicator.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.railed_gin_type[](#nccl.core.Communicator.railed_gin_type "Permalink to this definition")
    

Railed GIN type supported by this communicator.

If equal to [`NcclGinType.NONE`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html#nccl.core.NcclGinType.NONE "nccl.core.NcclGinType.NONE"), a device communicator cannot be created with GIN connection type [`NcclGinConnectionType.RAIL`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html#nccl.core.NcclGinConnectionType.RAIL "nccl.core.NcclGinConnectionType.RAIL").

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.