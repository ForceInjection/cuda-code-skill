# Next >

**Source:** group__CUDA__TYPES.html


Search In: Entire Site Just This Document clear search search

 v13.3.1


  * 1\. Difference between the driver and runtime APIs

  * 2\. API synchronization behavior

  * 3\. Stream synchronization behavior

  * 4\. Graph object thread safety

  * 5\. Rules for version mixing

  * 6\. Modules

    * 6.1. Data types used by CUDA driver

    * 6.2. Error Handling

    * 6.3. Initialization

    * 6.4. Version Management

    * 6.5. Device Management

    * [6.6. Device Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED)

    * 6.7. Primary Context Management

    * 6.8. Context Management

    * [6.9. Context Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED)

    * 6.10. Module Management

    * [6.11. Module Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED)

    * 6.12. Library Management

    * 6.13. Memory Management

    * 6.14. Virtual Memory Management

    * 6.15. Stream Ordered Memory Allocator

    * 6.16. Multicast Object Management

    * 6.17. Logical Endpoint

    * 6.18. Unified Addressing

    * 6.19. Stream Management

    * 6.20. Event Management

    * 6.21. External Resource Interoperability

    * 6.22. Stream Memory Operations

    * 6.23. Execution Control

    * [6.24. Execution Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED)

    * 6.25. Graph Management

    * 6.26. Occupancy

    * [6.27. Texture Reference Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED)

    * [6.28. Surface Reference Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFREF__DEPRECATED.html#group__CUDA__SURFREF__DEPRECATED)

    * 6.29. Texture Object Management

    * 6.30. Surface Object Management

    * 6.31. Tensor Map Object Managment

    * 6.32. Peer Context Memory Access

    * 6.33. Graphics Interoperability

    * 6.34. Driver Entry Point Access

    * 6.35. Coredump Attributes Control API

    * 6.36. Green Contexts

    * 6.37. Error Log Management Functions

    * 6.38. CUDA Checkpointing

    * [6.39. Profiler Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER__DEPRECATED.html#group__CUDA__PROFILER__DEPRECATED)

    * 6.40. Profiler Control

    * 6.41. OpenGL Interoperability

      * [6.41.1. OpenGL Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED)

    * 6.42. Direct3D 9 Interoperability

      * [6.42.1. Direct3D 9 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED)

    * 6.43. Direct3D 10 Interoperability

      * [6.43.1. Direct3D 10 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED)

    * 6.44. Direct3D 11 Interoperability

      * [6.44.1. Direct3D 11 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D11__DEPRECATED.html#group__CUDA__D3D11__DEPRECATED)

    * 6.45. VDPAU Interoperability

    * 6.46. EGL Interoperability

  * 7\. Data Structures

    * 7.1. CU_DEV_SM_RESOURCE_GROUP_PARAMS

    * 7.2. CUaccessPolicyWindow_v1

    * 7.3. CUarrayMapInfo_v1

    * 7.4. CUasyncNotificationInfo

    * 7.5. CUcheckpointCheckpointArgs

    * 7.6. CUcheckpointGpuPair

    * 7.7. CUcheckpointLockArgs

    * 7.8. CUcheckpointRestoreArgs

    * 7.9. CUcheckpointUnlockArgs

    * 7.10. CUctxCigParam

    * 7.11. CUctxCreateParams

    * 7.12. CUDA_ARRAY3D_DESCRIPTOR_v2

    * 7.13. CUDA_ARRAY_DESCRIPTOR_v2

    * 7.14. CUDA_ARRAY_MEMORY_REQUIREMENTS_v1

    * 7.15. CUDA_ARRAY_SPARSE_PROPERTIES_v1

    * 7.16. CUDA_BATCH_MEM_OP_NODE_PARAMS_v1

    * 7.17. CUDA_BATCH_MEM_OP_NODE_PARAMS_v2

    * 7.18. CUDA_CHILD_GRAPH_NODE_PARAMS

    * 7.19. CUDA_CONDITIONAL_NODE_PARAMS

    * 7.20. CUDA_EVENT_RECORD_NODE_PARAMS

    * 7.21. CUDA_EVENT_WAIT_NODE_PARAMS

    * 7.22. CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1

    * 7.23. CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2

    * 7.24. CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1

    * 7.25. CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2

    * 7.26. CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1

    * 7.27. CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1

    * 7.28. CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1

    * 7.29. CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1

    * 7.30. CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1

    * 7.31. CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1

    * 7.32. CUDA_GRAPH_INSTANTIATE_PARAMS

    * 7.33. CUDA_HOST_NODE_PARAMS_v1

    * 7.34. CUDA_HOST_NODE_PARAMS_v2

    * 7.35. CUDA_KERNEL_NODE_PARAMS_v1

    * 7.36. CUDA_KERNEL_NODE_PARAMS_v2

    * 7.37. CUDA_KERNEL_NODE_PARAMS_v3

    * 7.38. CUDA_LAUNCH_PARAMS_v1

    * 7.39. CUDA_MEM_ALLOC_NODE_PARAMS_v1

    * 7.40. CUDA_MEM_ALLOC_NODE_PARAMS_v2

    * 7.41. CUDA_MEM_FREE_NODE_PARAMS

    * 7.42. CUDA_MEMCPY2D_v2

    * 7.43. CUDA_MEMCPY3D_PEER_v1

    * 7.44. CUDA_MEMCPY3D_v2

    * 7.45. CUDA_MEMCPY_NODE_PARAMS

    * 7.46. CUDA_MEMSET_NODE_PARAMS_v1

    * 7.47. CUDA_MEMSET_NODE_PARAMS_v2

    * 7.48. CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1

    * 7.49. CUDA_RESOURCE_DESC_v1

    * 7.50. CUDA_RESOURCE_VIEW_DESC_v1

    * 7.51. CUDA_TEXTURE_DESC_v1

    * 7.52. CUdevprop_v1

    * 7.53. CUdevResource

    * 7.54. CUdevSmResource

    * 7.55. CUdevWorkqueueConfigResource

    * 7.56. CUdevWorkqueueResource

    * 7.57. CUeglFrame_v1

    * 7.58. CUexecAffinityParam_v1

    * 7.59. CUexecAffinitySmCount_v1

    * 7.60. CUextent3D_v1

    * 7.61. CUgraphEdgeData

    * 7.62. CUgraphExecUpdateResultInfo_v1

    * 7.63. CUgraphNodeParams

    * 7.64. CUipcEventHandle_v1

    * 7.65. CUipcMemHandle_v1

    * 7.66. CUlaunchAttribute

    * 7.67. CUlaunchAttributeValue

    * 7.68. CUlaunchConfig

    * 7.69. CUlaunchMemSyncDomainMap

    * 7.70. CUlogicalEndpointFabricHandle

    * 7.71. CUlogicalEndpointProp

    * 7.72. CUmemAccessDesc_v1

    * 7.73. CUmemAllocationProp_v1

    * 7.74. CUmemcpy3DOperand_v1

    * 7.75. CUmemcpyAttributes_v1

    * 7.76. CUmemDecompressParams

    * 7.77. CUmemFabricHandle_v1

    * 7.78. CUmemLocation_v1

    * 7.79. CUmemPoolProps_v1

    * 7.80. CUmemPoolPtrExportData_v1

    * 7.81. CUmulticastObjectProp_v1

    * 7.82. CUoffset3D_v1

    * 7.83. CUstreamBatchMemOpParams_v1

    * 7.84. CUstreamCigCaptureParams

    * 7.85. CUstreamCigParam

    * 7.86. CUtensorMap

  * 8\. Data Fields

  * 9\. Deprecated List


## Search Results


< Previous | Next >

CUDA Driver API (PDF) \- v13.3.1 (older) \- Last updated June 29, 2026 \- [Send Feedback](mailto:CUDAIssues@nvidia.com?subject=CUDA%20Toolkit%20Documentation%20Feedback:%20CUDA%20Driver%20API)

## 6.1. Data types used by CUDA driver

### Classes

struct

CUDA_ARRAY3D_DESCRIPTOR_v2


struct

CUDA_ARRAY_DESCRIPTOR_v2


struct

CUDA_ARRAY_MEMORY_REQUIREMENTS_v1


struct

CUDA_ARRAY_SPARSE_PROPERTIES_v1


struct

CUDA_BATCH_MEM_OP_NODE_PARAMS_v1


struct

CUDA_BATCH_MEM_OP_NODE_PARAMS_v2


struct

CUDA_CHILD_GRAPH_NODE_PARAMS


struct

CUDA_CONDITIONAL_NODE_PARAMS


struct

CUDA_EVENT_RECORD_NODE_PARAMS


struct

CUDA_EVENT_WAIT_NODE_PARAMS


struct

CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1


struct

CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1


struct

CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1


struct

CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1


struct

CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1


struct

CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1


struct

CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1


struct

CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2


struct

CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1


struct

CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2


struct

CUDA_GRAPH_INSTANTIATE_PARAMS


struct

CUDA_HOST_NODE_PARAMS_v1


struct

CUDA_HOST_NODE_PARAMS_v2


struct

CUDA_KERNEL_NODE_PARAMS_v1


struct

CUDA_KERNEL_NODE_PARAMS_v2


struct

CUDA_KERNEL_NODE_PARAMS_v3


struct

CUDA_LAUNCH_PARAMS_v1


struct

CUDA_MEMCPY2D_v2


struct

CUDA_MEMCPY3D_PEER_v1


struct

CUDA_MEMCPY3D_v2


struct

CUDA_MEMCPY_NODE_PARAMS


struct

CUDA_MEMSET_NODE_PARAMS_v1


struct

CUDA_MEMSET_NODE_PARAMS_v2


struct

CUDA_MEM_ALLOC_NODE_PARAMS_v1


struct

CUDA_MEM_ALLOC_NODE_PARAMS_v2


struct

CUDA_MEM_FREE_NODE_PARAMS


struct

CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1


struct

CUDA_RESOURCE_DESC_v1


struct

CUDA_RESOURCE_VIEW_DESC_v1


struct

CUDA_TEXTURE_DESC_v1


struct

CUaccessPolicyWindow_v1


struct

CUarrayMapInfo_v1


struct

CUasyncNotificationInfo


struct

CUcheckpointCheckpointArgs


struct

CUcheckpointGpuPair


struct

CUcheckpointLockArgs


struct

CUcheckpointRestoreArgs


struct

CUcheckpointUnlockArgs


struct

CUctxCigParam


struct

CUctxCreateParams


struct

CUdevprop_v1


struct

CUeglFrame_v1


struct

CUexecAffinityParam_v1


struct

CUexecAffinitySmCount_v1


struct

CUextent3D_v1


struct

CUgraphEdgeData


struct

CUgraphExecUpdateResultInfo_v1


struct

CUgraphNodeParams


struct

CUipcEventHandle_v1


struct

CUipcMemHandle_v1


struct

CUlaunchAttribute


union

CUlaunchAttributeValue


struct

CUlaunchConfig


struct

CUlaunchMemSyncDomainMap


struct

CUmemAccessDesc_v1


struct

CUmemAllocationProp_v1


struct

CUmemFabricHandle_v1


struct

CUmemLocation_v1


struct

CUmemPoolProps_v1


struct

CUmemPoolPtrExportData_v1


struct

CUmemcpy3DOperand_v1


struct

CUmemcpyAttributes_v1


struct

CUmulticastObjectProp_v1


struct

CUoffset3D_v1


union

CUstreamBatchMemOpParams_v1


struct

CUstreamCigCaptureParams


struct

CUstreamCigParam


struct

CUtensorMap



### Defines

#define CUDA_ARRAY3D_2DARRAY 0x01

#define CUDA_ARRAY3D_COLOR_ATTACHMENT 0x20

#define CUDA_ARRAY3D_CUBEMAP 0x04

#define CUDA_ARRAY3D_DEFERRED_MAPPING 0x80

#define CUDA_ARRAY3D_DEPTH_TEXTURE 0x10

#define CUDA_ARRAY3D_LAYERED 0x01

#define CUDA_ARRAY3D_SPARSE 0x40

#define CUDA_ARRAY3D_SURFACE_LDST 0x02

#define CUDA_ARRAY3D_TEXTURE_GATHER 0x08

#define CUDA_ARRAY3D_VIDEO_ENCODE_DECODE 0x100

#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC 0x02

#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC 0x01

#define CUDA_EGL_INFINITE_TIMEOUT 0xFFFFFFFF

#define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1

#define CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC 0x01

#define CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC 0x02

#define CUDA_NVSCISYNC_ATTR_SIGNAL 0x1

#define CUDA_NVSCISYNC_ATTR_WAIT 0x2

#define CUDA_VERSION 13030

#define CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL 0x1

#define CU_DEVICE_CPU ((CUdevice)-1)

#define CU_DEVICE_INVALID ((CUdevice)-2)

#define CU_GRAPH_COND_ASSIGN_DEFAULT 0x1

#define CU_GRAPH_KERNEL_NODE_PORT_DEFAULT 0

#define CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER 2

#define CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC 1

#define CU_IPC_HANDLE_SIZE 64

#define CU_LAUNCH_KERNEL_REQUIRED_BLOCK_DIM 1

#define CU_LAUNCH_PARAM_BUFFER_POINTER

#define CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT 0x01

#define CU_LAUNCH_PARAM_BUFFER_SIZE

#define CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT 0x02

#define CU_LAUNCH_PARAM_END

#define CU_LAUNCH_PARAM_END_AS_INT 0x00

#define CU_MEMHOSTALLOC_DEVICEMAP 0x02

#define CU_MEMHOSTALLOC_PORTABLE 0x01

#define CU_MEMHOSTALLOC_WRITECOMBINED 0x04

#define CU_MEMHOSTREGISTER_DEVICEMAP 0x02

#define CU_MEMHOSTREGISTER_IOMEMORY 0x04

#define CU_MEMHOSTREGISTER_PORTABLE 0x01

#define CU_MEMHOSTREGISTER_READ_ONLY 0x08

#define CU_MEM_CREATE_USAGE_HW_DECOMPRESS 0x2

#define CU_MEM_CREATE_USAGE_TILE_POOL 0x1

#define CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS 0x2

#define CU_PARAM_TR_DEFAULT -1

#define CU_STREAM_LEGACY ((CUstream)0x1)

#define CU_STREAM_PER_THREAD ((CUstream)0x2)

#define CU_TENSOR_MAP_NUM_QWORDS 16

#define CU_TRSA_OVERRIDE_FORMAT 0x01

#define CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION 0x20

#define CU_TRSF_NORMALIZED_COORDINATES 0x02

#define CU_TRSF_READ_AS_INTEGER 0x01

#define CU_TRSF_SEAMLESS_CUBEMAP 0x40

#define CU_TRSF_SRGB 0x10

#define MAX_PLANES 3


### Typedefs
