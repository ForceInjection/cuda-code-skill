# Device Communicator Setup

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html

---

# Device Communicator Setup[](#device-communicator-setup "Permalink to this heading")

Host-side methods and resources for creating an NCCL device communicator. The device-side communication primitives themselves are available only from CUDA kernels and are documented under the C device API ([Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device.html)); this page covers what the Python (host) side exposes for bootstrapping them. The configuration object passed to [`Communicator.create_dev_comm()`](#nccl.core.Communicator.create_dev_comm "nccl.core.Communicator.create_dev_comm") is documented in [Configuration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html).

## create_dev_comm[](#create-dev-comm "Permalink to this heading")

Communicator.create_dev_comm(_requirements : [NCCLDevCommRequirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html#nccl.core.NCCLDevCommRequirements "nccl.core.communicator.NCCLDevCommRequirements") | None = None_) → [DevCommResource](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.DevCommResource "nccl.core.resources.DevCommResource")[](#nccl.core.Communicator.create_dev_comm "Permalink to this definition")
    

Creates a device communicator for device-side NCCL operations.

Device communicators enable direct GPU kernel access to NCCL communication primitives. Multiple device communicators can be created from one host communicator. The returned [`DevCommResource`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.DevCommResource "nccl.core.DevCommResource") is tracked by the communicator and may be released explicitly via its `close()` method, or automatically when the communicator is destroyed or aborted. Access the device communicator pointer via [`DevCommResource.ptr`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.DevCommResource.ptr "nccl.core.DevCommResource.ptr") or `resource.dev_comm.ptr`.

Parameters:
    

**requirements** – Configuration for device communicator resource allocation. If `None`, a default [`NCCLDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html#nccl.core.NCCLDevCommRequirements "nccl.core.NCCLDevCommRequirements") is used. Defaults to `None`.

Returns:
    

[`DevCommResource`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/resources.html#nccl.core.DevCommResource "nccl.core.DevCommResource") for the device communicator.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

See also

[`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate")

## GIN type enums[](#gin-type-enums "Permalink to this heading")

GPU Interconnect Network (GIN) enums describing what device-side network transport is available on a communicator and which connection topology the user requires.

### NcclGinType[](#ncclgintype "Permalink to this heading")

_class _nccl.core.NcclGinType(_value_ , _names= <not given>_, _*values_ , _module=None_ , _qualname=None_ , _type=None_ , _start=1_ , _boundary=None_)[](#nccl.core.NcclGinType "Permalink to this definition")
    

Bases: `IntEnum`

GIN transport type, mirroring [`ncclGinType_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGinType_t "ncclGinType_t").

Reported by [`Communicator.gin_type`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator.gin_type "nccl.core.Communicator.gin_type") and [`Communicator.railed_gin_type`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator.railed_gin_type "nccl.core.Communicator.railed_gin_type") to indicate which device-side network transport, if any, is available on the communicator.

NONE _ = 0_[](#nccl.core.NcclGinType.NONE "Permalink to this definition")
    

GIN not available on this communicator.

PROXY _ = 2_[](#nccl.core.NcclGinType.PROXY "Permalink to this definition")
    

Proxy-based GIN. Network operations issued from a device kernel are relayed through a CPU proxy thread.

GDAKI _ = 3_[](#nccl.core.NcclGinType.GDAKI "Permalink to this definition")
    

GPUDirect Async Kernel-Initiated (GDA-KI). The kernel directly issues network operations to the NIC, bypassing the CPU proxy.

GPI _ = 4_[](#nccl.core.NcclGinType.GPI "Permalink to this definition")
    

GPU-Push Interface. GPU threads push network descriptors directly to a NIC-visible MMIO queue, with no CPU involvement and no memory barriers.

### NcclGinConnectionType[](#ncclginconnectiontype "Permalink to this heading")

_class _nccl.core.NcclGinConnectionType(_value_ , _names= <not given>_, _*values_ , _module=None_ , _qualname=None_ , _type=None_ , _start=1_ , _boundary=None_)[](#nccl.core.NcclGinConnectionType "Permalink to this definition")
    

Bases: `IntEnum`

GIN connection topology, mirroring [`ncclGinConnectionType_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGinConnectionType_t "ncclGinConnectionType_t").

Set on the `gin_connection_type` field of [`NCCLDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html#nccl.core.NCCLDevCommRequirements "nccl.core.NCCLDevCommRequirements") before calling [`Communicator.create_dev_comm()`](#nccl.core.Communicator.create_dev_comm "nccl.core.Communicator.create_dev_comm") to declare which peers must be reachable via GIN from device code.

NONE _ = 0_[](#nccl.core.NcclGinConnectionType.NONE "Permalink to this definition")
    

No GIN connection requested.

FULL _ = 1_[](#nccl.core.NcclGinConnectionType.FULL "Permalink to this definition")
    

Fully connected. Every rank in the communicator must be reachable from every other rank via GIN.

RAIL _ = 2_[](#nccl.core.NcclGinConnectionType.RAIL "Permalink to this definition")
    

Rail-restricted. Ranks must be reachable via GIN only within the same rail (network plane).