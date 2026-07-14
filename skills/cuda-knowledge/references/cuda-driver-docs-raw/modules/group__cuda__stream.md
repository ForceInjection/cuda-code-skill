# Stream Management

**Source:** [group__CUDA__STREAM.html](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)

---

Search In: Entire Site Just This Document clear search search

[ v13.3.1](https://docs.nvidia.com/cuda/index.html "The root of the site.")

[](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)

  * [1\. Difference between the driver and runtime APIs ](https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html#driver-vs-runtime-api)

  * [2\. API synchronization behavior ](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior)

  * [3\. Stream synchronization behavior](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior)

  * [4\. Graph object thread safety](https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html#graphs-thread-safety)

  * [5\. Rules for version mixing ](https://docs.nvidia.com/cuda/cuda-driver-api/version-mixing-rules.html#version-mixing-rules)

  * [6\. Modules](https://docs.nvidia.com/cuda/cuda-driver-api/modules.html#modules)

    * [6.1. Data types used by CUDA driver](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES)

    * [6.2. Error Handling](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR)

    * [6.3. Initialization](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE)

    * [6.4. Version Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION)

    * [6.5. Device Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE)

    * [6.6. Device Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED)

    * [6.7. Primary Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX)

    * [6.8. Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX)

    * [6.9. Context Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED)

    * [6.10. Module Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE)

    * [6.11. Module Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED)

    * [6.12. Library Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY)

    * [6.13. Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM)

    * [6.14. Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA)

    * [6.15. Stream Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC)

    * [6.16. Multicast Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST)

    * [6.17. Logical Endpoint](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LOGICAL__ENDPOINT.html#group__CUDA__LOGICAL__ENDPOINT)

    * [6.18. Unified Addressing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED)

    * [6.19. Stream Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM)

    * [6.20. Event Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT)

    * [6.21. External Resource Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP)

    * [6.22. Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html#group__CUDA__MEMOP)

    * [6.23. Execution Control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC)

    * [6.24. Execution Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED)

    * [6.25. Graph Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH)

    * [6.26. Occupancy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY)

    * [6.27. Texture Reference Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED)

    * [6.28. Surface Reference Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFREF__DEPRECATED.html#group__CUDA__SURFREF__DEPRECATED)

    * [6.29. Texture Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT)

    * [6.30. Surface Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFOBJECT.html#group__CUDA__SURFOBJECT)

    * [6.31. Tensor Map Object Managment](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY)

    * [6.32. Peer Context Memory Access](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS)

    * [6.33. Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS)

    * [6.34. Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html#group__CUDA__DRIVER__ENTRY__POINT)

    * [6.35. Coredump Attributes Control API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP)

    * [6.36. Green Contexts](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS)

    * [6.37. Error Log Management Functions](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LOGS.html#group__CUDA__LOGS)

    * [6.38. CUDA Checkpointing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html#group__CUDA__CHECKPOINT)

    * [6.39. Profiler Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER__DEPRECATED.html#group__CUDA__PROFILER__DEPRECATED)

    * [6.40. Profiler Control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER.html#group__CUDA__PROFILER)

    * [6.41. OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GL.html#group__CUDA__GL)

      * [6.41.1. OpenGL Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED)

    * [6.42. Direct3D 9 Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D9.html#group__CUDA__D3D9)

      * [6.42.1. Direct3D 9 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED)

    * [6.43. Direct3D 10 Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D10.html#group__CUDA__D3D10)

      * [6.43.1. Direct3D 10 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED)

    * [6.44. Direct3D 11 Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D11.html#group__CUDA__D3D11)

      * [6.44.1. Direct3D 11 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D11__DEPRECATED.html#group__CUDA__D3D11__DEPRECATED)

    * [6.45. VDPAU Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VDPAU.html#group__CUDA__VDPAU)

    * [6.46. EGL Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EGL.html#group__CUDA__EGL)

  * [7\. Data Structures](https://docs.nvidia.com/cuda/cuda-driver-api/annotated.html#annotated)

    * [7.1. CU_DEV_SM_RESOURCE_GROUP_PARAMS](https://docs.nvidia.com/cuda/cuda-driver-api/structCU__DEV__SM__RESOURCE__GROUP__PARAMS.html#structCU__DEV__SM__RESOURCE__GROUP__PARAMS)

    * [7.2. CUaccessPolicyWindow_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUaccessPolicyWindow__v1.html#structCUaccessPolicyWindow__v1)

    * [7.3. CUarrayMapInfo_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1)

    * [7.4. CUasyncNotificationInfo](https://docs.nvidia.com/cuda/cuda-driver-api/structCUasyncNotificationInfo.html#structCUasyncNotificationInfo)

    * [7.5. CUcheckpointCheckpointArgs](https://docs.nvidia.com/cuda/cuda-driver-api/structCUcheckpointCheckpointArgs.html#structCUcheckpointCheckpointArgs)

    * [7.6. CUcheckpointGpuPair](https://docs.nvidia.com/cuda/cuda-driver-api/structCUcheckpointGpuPair.html#structCUcheckpointGpuPair)

    * [7.7. CUcheckpointLockArgs](https://docs.nvidia.com/cuda/cuda-driver-api/structCUcheckpointLockArgs.html#structCUcheckpointLockArgs)

    * [7.8. CUcheckpointRestoreArgs](https://docs.nvidia.com/cuda/cuda-driver-api/structCUcheckpointRestoreArgs.html#structCUcheckpointRestoreArgs)

    * [7.9. CUcheckpointUnlockArgs](https://docs.nvidia.com/cuda/cuda-driver-api/structCUcheckpointUnlockArgs.html#structCUcheckpointUnlockArgs)

    * [7.10. CUctxCigParam](https://docs.nvidia.com/cuda/cuda-driver-api/structCUctxCigParam.html#structCUctxCigParam)

    * [7.11. CUctxCreateParams](https://docs.nvidia.com/cuda/cuda-driver-api/structCUctxCreateParams.html#structCUctxCreateParams)

    * [7.12. CUDA_ARRAY3D_DESCRIPTOR_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2)

    * [7.13. CUDA_ARRAY_DESCRIPTOR_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2)

    * [7.14. CUDA_ARRAY_MEMORY_REQUIREMENTS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1.html#structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1)

    * [7.15. CUDA_ARRAY_SPARSE_PROPERTIES_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1)

    * [7.16. CUDA_BATCH_MEM_OP_NODE_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1)

    * [7.17. CUDA_BATCH_MEM_OP_NODE_PARAMS_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__BATCH__MEM__OP__NODE__PARAMS__v2.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v2)

    * [7.18. CUDA_CHILD_GRAPH_NODE_PARAMS](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__CHILD__GRAPH__NODE__PARAMS.html#structCUDA__CHILD__GRAPH__NODE__PARAMS)

    * [7.19. CUDA_CONDITIONAL_NODE_PARAMS](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__CONDITIONAL__NODE__PARAMS.html#structCUDA__CONDITIONAL__NODE__PARAMS)

    * [7.20. CUDA_EVENT_RECORD_NODE_PARAMS](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EVENT__RECORD__NODE__PARAMS.html#structCUDA__EVENT__RECORD__NODE__PARAMS)

    * [7.21. CUDA_EVENT_WAIT_NODE_PARAMS](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EVENT__WAIT__NODE__PARAMS.html#structCUDA__EVENT__WAIT__NODE__PARAMS)

    * [7.22. CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1)

    * [7.23. CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v2.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v2)

    * [7.24. CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1)

    * [7.25. CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v2.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v2)

    * [7.26. CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1)

    * [7.27. CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1)

    * [7.28. CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1)

    * [7.29. CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1)

    * [7.30. CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXTERNAL__SEMAPHORE__SIGNAL__PARAMS__v1.html#structCUDA__EXTERNAL__SEMAPHORE__SIGNAL__PARAMS__v1)

    * [7.31. CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__EXTERNAL__SEMAPHORE__WAIT__PARAMS__v1.html#structCUDA__EXTERNAL__SEMAPHORE__WAIT__PARAMS__v1)

    * [7.32. CUDA_GRAPH_INSTANTIATE_PARAMS](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__GRAPH__INSTANTIATE__PARAMS.html#structCUDA__GRAPH__INSTANTIATE__PARAMS)

    * [7.33. CUDA_HOST_NODE_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__HOST__NODE__PARAMS__v1.html#structCUDA__HOST__NODE__PARAMS__v1)

    * [7.34. CUDA_HOST_NODE_PARAMS_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__HOST__NODE__PARAMS__v2.html#structCUDA__HOST__NODE__PARAMS__v2)

    * [7.35. CUDA_KERNEL_NODE_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__KERNEL__NODE__PARAMS__v1.html#structCUDA__KERNEL__NODE__PARAMS__v1)

    * [7.36. CUDA_KERNEL_NODE_PARAMS_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2)

    * [7.37. CUDA_KERNEL_NODE_PARAMS_v3](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__KERNEL__NODE__PARAMS__v3.html#structCUDA__KERNEL__NODE__PARAMS__v3)

    * [7.38. CUDA_LAUNCH_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1)

    * [7.39. CUDA_MEM_ALLOC_NODE_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__MEM__ALLOC__NODE__PARAMS__v1.html#structCUDA__MEM__ALLOC__NODE__PARAMS__v1)

    * [7.40. CUDA_MEM_ALLOC_NODE_PARAMS_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__MEM__ALLOC__NODE__PARAMS__v2.html#structCUDA__MEM__ALLOC__NODE__PARAMS__v2)

    * [7.41. CUDA_MEM_FREE_NODE_PARAMS](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__MEM__FREE__NODE__PARAMS.html#structCUDA__MEM__FREE__NODE__PARAMS)

    * [7.42. CUDA_MEMCPY2D_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2)

    * [7.43. CUDA_MEMCPY3D_PEER_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__MEMCPY3D__PEER__v1.html#structCUDA__MEMCPY3D__PEER__v1)

    * [7.44. CUDA_MEMCPY3D_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2)

    * [7.45. CUDA_MEMCPY_NODE_PARAMS](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__MEMCPY__NODE__PARAMS.html#structCUDA__MEMCPY__NODE__PARAMS)

    * [7.46. CUDA_MEMSET_NODE_PARAMS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__MEMSET__NODE__PARAMS__v1.html#structCUDA__MEMSET__NODE__PARAMS__v1)

    * [7.47. CUDA_MEMSET_NODE_PARAMS_v2](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__MEMSET__NODE__PARAMS__v2.html#structCUDA__MEMSET__NODE__PARAMS__v2)

    * [7.48. CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__POINTER__ATTRIBUTE__P2P__TOKENS__v1.html#structCUDA__POINTER__ATTRIBUTE__P2P__TOKENS__v1)

    * [7.49. CUDA_RESOURCE_DESC_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1)

    * [7.50. CUDA_RESOURCE_VIEW_DESC_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1)

    * [7.51. CUDA_TEXTURE_DESC_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1)

    * [7.52. CUdevprop_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUdevprop__v1.html#structCUdevprop__v1)

    * [7.53. CUdevResource](https://docs.nvidia.com/cuda/cuda-driver-api/structCUdevResource.html#structCUdevResource)

    * [7.54. CUdevSmResource](https://docs.nvidia.com/cuda/cuda-driver-api/structCUdevSmResource.html#structCUdevSmResource)

    * [7.55. CUdevWorkqueueConfigResource](https://docs.nvidia.com/cuda/cuda-driver-api/structCUdevWorkqueueConfigResource.html#structCUdevWorkqueueConfigResource)

    * [7.56. CUdevWorkqueueResource](https://docs.nvidia.com/cuda/cuda-driver-api/structCUdevWorkqueueResource.html#structCUdevWorkqueueResource)

    * [7.57. CUeglFrame_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUeglFrame__v1.html#structCUeglFrame__v1)

    * [7.58. CUexecAffinityParam_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUexecAffinityParam__v1.html#structCUexecAffinityParam__v1)

    * [7.59. CUexecAffinitySmCount_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUexecAffinitySmCount__v1.html#structCUexecAffinitySmCount__v1)

    * [7.60. CUextent3D_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUextent3D__v1.html#structCUextent3D__v1)

    * [7.61. CUgraphEdgeData](https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphEdgeData.html#structCUgraphEdgeData)

    * [7.62. CUgraphExecUpdateResultInfo_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphExecUpdateResultInfo__v1.html#structCUgraphExecUpdateResultInfo__v1)

    * [7.63. CUgraphNodeParams](https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphNodeParams.html#structCUgraphNodeParams)

    * [7.64. CUipcEventHandle_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUipcEventHandle__v1.html#structCUipcEventHandle__v1)

    * [7.65. CUipcMemHandle_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUipcMemHandle__v1.html#structCUipcMemHandle__v1)

    * [7.66. CUlaunchAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/structCUlaunchAttribute.html#structCUlaunchAttribute)

    * [7.67. CUlaunchAttributeValue](https://docs.nvidia.com/cuda/cuda-driver-api/unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue)

    * [7.68. CUlaunchConfig](https://docs.nvidia.com/cuda/cuda-driver-api/structCUlaunchConfig.html#structCUlaunchConfig)

    * [7.69. CUlaunchMemSyncDomainMap](https://docs.nvidia.com/cuda/cuda-driver-api/structCUlaunchMemSyncDomainMap.html#structCUlaunchMemSyncDomainMap)

    * [7.70. CUlogicalEndpointFabricHandle](https://docs.nvidia.com/cuda/cuda-driver-api/structCUlogicalEndpointFabricHandle.html#structCUlogicalEndpointFabricHandle)

    * [7.71. CUlogicalEndpointProp](https://docs.nvidia.com/cuda/cuda-driver-api/structCUlogicalEndpointProp.html#structCUlogicalEndpointProp)

    * [7.72. CUmemAccessDesc_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemAccessDesc__v1.html#structCUmemAccessDesc__v1)

    * [7.73. CUmemAllocationProp_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1)

    * [7.74. CUmemcpy3DOperand_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemcpy3DOperand__v1.html#structCUmemcpy3DOperand__v1)

    * [7.75. CUmemcpyAttributes_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemcpyAttributes__v1.html#structCUmemcpyAttributes__v1)

    * [7.76. CUmemDecompressParams](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemDecompressParams.html#structCUmemDecompressParams)

    * [7.77. CUmemFabricHandle_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemFabricHandle__v1.html#structCUmemFabricHandle__v1)

    * [7.78. CUmemLocation_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemLocation__v1.html#structCUmemLocation__v1)

    * [7.79. CUmemPoolProps_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemPoolProps__v1.html#structCUmemPoolProps__v1)

    * [7.80. CUmemPoolPtrExportData_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemPoolPtrExportData__v1.html#structCUmemPoolPtrExportData__v1)

    * [7.81. CUmulticastObjectProp_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUmulticastObjectProp__v1.html#structCUmulticastObjectProp__v1)

    * [7.82. CUoffset3D_v1](https://docs.nvidia.com/cuda/cuda-driver-api/structCUoffset3D__v1.html#structCUoffset3D__v1)

    * [7.83. CUstreamBatchMemOpParams_v1](https://docs.nvidia.com/cuda/cuda-driver-api/unionCUstreamBatchMemOpParams__v1.html#unionCUstreamBatchMemOpParams__v1)

    * [7.84. CUstreamCigCaptureParams](https://docs.nvidia.com/cuda/cuda-driver-api/structCUstreamCigCaptureParams.html#structCUstreamCigCaptureParams)

    * [7.85. CUstreamCigParam](https://docs.nvidia.com/cuda/cuda-driver-api/structCUstreamCigParam.html#structCUstreamCigParam)

    * [7.86. CUtensorMap](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html#structCUtensorMap)

  * [8\. Data Fields](https://docs.nvidia.com/cuda/cuda-driver-api/functions.html#functions)

  * [9\. Deprecated List](https://docs.nvidia.com/cuda/cuda-driver-api/deprecated.html#deprecated)


## Search Results


[< Previous](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html) | [Next >](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)

CUDA Driver API ([PDF](https://docs.nvidia.com/cuda/pdf/CUDA_Driver_API.pdf)) \- v13.3.1 ([older](https://developer.nvidia.com/cuda-toolkit-archive)) \- Last updated June 29, 2026 \- [Send Feedback](mailto:CUDAIssues@nvidia.com?subject=CUDA%20Toolkit%20Documentation%20Feedback:%20CUDA%20Driver%20API)

## 6.19. Stream Management

This section describes the stream management functions of the low-level CUDA driver application programming interface. 

### Typedefs

typedef [CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9)* ( *[CUgraphRecaptureCallback](#group__CUDA__STREAM_1g17628a1a991a6b5c4fcf0e86283c23ac) )( void*  data,  CUgraphNode node, const CUgraphNodeParams*  originalParams, const CUgraphNodeParams*  recaptureParams,  CUgraphRecaptureStatus status ) 
    

### Functions

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamAddCallback](#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ge5743a8c48527f1040107a68205c5ba9) callback, void* userData, unsigned int  flags ) 
    Add a callback to a compute stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamAttachMemAsync](#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUdeviceptr](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e) dptr, size_t length, unsigned int  flags ) 
    Attach memory to a stream asynchronously. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamBeginCapture](#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669) mode ) 
    Begins graph capture on a stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamBeginCaptureToCig](#group__CUDA__STREAM_1g05756a51c341a98172c5993487b76c39) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCigCaptureParams](https://docs.nvidia.com/cuda/cuda-driver-api/structCUstreamCigCaptureParams.html#structCUstreamCigCaptureParams)* streamCigCaptureParams ) 
    Begins capture to CIG on a stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamBeginCaptureToGraph](#group__CUDA__STREAM_1gac495e0527d1dd6437f95ee482f61865) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUgraph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a) hGraph, const [CUgraphNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c)* dependencies, const [CUgraphEdgeData](https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphEdgeData.html#structCUgraphEdgeData)* dependencyData, size_t numDependencies, [CUstreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669) mode ) 
    Begins graph capture on a stream to an existing graph. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamBeginRecaptureToGraph](#group__CUDA__STREAM_1g34e9dd281a5f5744a484addabfebbb64) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669) mode, [CUgraph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a) hGraph, [CUgraphRecaptureCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g17628a1a991a6b5c4fcf0e86283c23ac) callbackFunc, void* userData ) 
    Begin graph capture on a stream to an existing graph. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamCopyAttributes](#group__CUDA__STREAM_1g680f5399f6126cc4a99bc5eee4c2eff0) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) dst, [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) src ) 
    Copies attributes from source stream to destination stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamCreate](#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a)* phStream, unsigned int  Flags ) 
    Create a stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamCreateWithPriority](#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a)* phStream, unsigned int  flags, int  priority ) 
    Create a stream with the given priority. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamDestroy](#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream ) 
    Destroys a stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamEndCapture](#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUgraph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a)* phGraph ) 
    Ends capture on a stream, returning the captured graph. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamEndCaptureToCig](#group__CUDA__STREAM_1gc21bd0a08a9c60f4ab2dd5594e829746) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream ) 
    Ends CIG capture on a stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamGetAttribute](#group__CUDA__STREAM_1g0598bb5295f3a62761b93c2d63d2266c) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamAttrID](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331) attr, [CUstreamAttrValue](https://docs.nvidia.com/cuda/cuda-driver-api/unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue)* value_out ) 
    Queries stream attribute. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamGetCaptureInfo](#group__CUDA__STREAM_1g85f03299332d6cf37578409d0e4b47ce) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCaptureStatus](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g7480c0f2bd19894e54fcd2c04d6efb91)* captureStatus_out, cuuint64_t* id_out, [CUgraph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a)* graph_out, const [CUgraphNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c)** dependencies_out, const [CUgraphEdgeData](https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphEdgeData.html#structCUgraphEdgeData)** edgeData_out, size_t* numDependencies_out ) 
    Query a stream's capture state. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamGetCtx](#group__CUDA__STREAM_1g1107907025eaa3387fdc590a9379a681) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUcontext](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9)* pctx ) 
    Query the context associated with a stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamGetCtx_v2](#group__CUDA__STREAM_1gd7eab81f618ec370a92c5e7d88ea11fa) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUcontext](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9)* pCtx, [CUgreenCtx](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268)* pGreenCtx ) 
    Query the contexts associated with a stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamGetDevice](#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUdevice](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a)* device ) 
    Returns the device handle of the stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamGetFlags](#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, unsigned int* flags ) 
    Query the flags of a given stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamGetId](#group__CUDA__STREAM_1g5dafd2b6f48caeb13d5110a7f21e60e3) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, unsigned long long* streamId ) 
    Returns the unique Id associated with the stream handle supplied. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamGetPriority](#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, int* priority ) 
    Query the priority of a given stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamIsCapturing](#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCaptureStatus](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g7480c0f2bd19894e54fcd2c04d6efb91)* captureStatus ) 
    Returns a stream's capture status. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamQuery](#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream ) 
    Determine status of a compute stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamSetAttribute](#group__CUDA__STREAM_1ga2c5fc0292861a42f264af6ca48be8c0) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamAttrID](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331) attr, const [CUstreamAttrValue](https://docs.nvidia.com/cuda/cuda-driver-api/unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue)* value ) 
    Sets stream attribute. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamSynchronize](#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream ) 
    Wait until a stream's tasks are completed. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamUpdateCaptureDependencies](#group__CUDA__STREAM_1g0cd3210434f3e0796c492cfa0d4b4bd1) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUgraphNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c)* dependencies, const [CUgraphEdgeData](https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphEdgeData.html#structCUgraphEdgeData)* dependencyData, size_t numDependencies, unsigned int  flags ) 
    Update the set of dependencies in a capturing stream. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuStreamWaitEvent](#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f) ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUevent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b) hEvent, unsigned int  Flags ) 
    Make a compute stream wait on an event. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuThreadExchangeStreamCaptureMode](#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d) ( [CUstreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669)* mode ) 
    Swaps the stream capture interaction mode for a thread. 

### Typedefs

CUresult* ( *CUgraphRecaptureCallback )( void*  data,  CUgraphNode node, const CUgraphNodeParams*  originalParams, const CUgraphNodeParams*  recaptureParams,  CUgraphRecaptureStatus status ) 
    

Callback function invoked when node parameter mismatches are detected while recapturing to an existing graph. Parameter struct pointers are only valid within the callback. 

######  Parameters 

`data`
    User parameter provided at beginning of recapture 
`CUgraphNode node`
    
`originalParams`
    The original node parameters from the graph 
`recaptureParams`
    The node parameters received during the recapture 
`CUgraphRecaptureStatus status`
    

###### Returns

Error code for the callback. Anything other than CUDA_SUCCESS will cause the recapture to fail immediately. 

### Functions

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamAddCallback ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ge5743a8c48527f1040107a68205c5ba9) callback, void* userData, unsigned int  flags ) 
    

Add a callback to a compute stream. 

######  Parameters 

`hStream`
    \- Stream to add callback to 
`callback`
    \- The function to call once preceding stream operations are complete 
`userData`
    \- User specified data to be passed to the callback function 
`flags`
    \- Reserved for future use, must be 0

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21), [CUDA_ERROR_NOT_SUPPORTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b)

###### Description

Note:

This function is slated for eventual deprecation and removal. If you do not require the callback to execute in case of a device error, consider using [cuLaunchHostFunc](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f "Enqueues a host function call in a stream."). Additionally, this function is not supported with [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream.") and [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph."), unlike [cuLaunchHostFunc](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f "Enqueues a host function call in a stream."). 

Adds a callback to be called on the host after all currently enqueued items in the stream have completed. For each cuStreamAddCallback call, the callback will be executed exactly once. The callback will block later work in the stream until it is finished. 

The callback may be passed [CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d) or an error code. In the event of a device error, all subsequently executed callbacks will receive an appropriate [CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9). 

Callbacks must not make any CUDA API calls. Attempting to use a CUDA API will result in [CUDA_ERROR_NOT_PERMITTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3). Callbacks must not perform any synchronization that may depend on outstanding device work or other callbacks that are not mandated to run earlier. Callbacks without a mandated order (in independent streams) execute in undefined order and may be serialized. 

For the purposes of Unified Memory, callback execution makes a number of guarantees: 

  * The callback stream is considered idle for the duration of the callback. Thus, for example, a callback may always use memory attached to the callback stream. 

  * The start of execution of a callback has the same effect as synchronizing an event recorded in the same stream immediately prior to the callback. It thus synchronizes streams which have been "joined" prior to the callback. 

  * Adding device work to any stream does not have the effect of making the stream active until all preceding host functions and stream callbacks have executed. Thus, for example, a callback might use global attached memory even if work has been added to another stream, if the work has been ordered behind the callback with an event. 

  * Completion of a callback does not cause a stream to become active except as described above. The callback stream will remain idle if no device work follows the callback, and will remain idle across consecutive callbacks without device work in between. Thus, for example, stream synchronization can be done by signaling from a callback at the end of the stream. 


Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuMemAllocManaged](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32 "Allocates memory that will be automatically managed by the Unified Memory system."), [cuStreamAttachMemAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533 "Attach memory to a stream asynchronously."), [cuLaunchHostFunc](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f "Enqueues a host function call in a stream."), [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamAttachMemAsync ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUdeviceptr](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e) dptr, size_t length, unsigned int  flags ) 
    

Attach memory to a stream asynchronously. 

######  Parameters 

`hStream`
    \- Stream in which to enqueue the attach operation 
`dptr`
    \- Pointer to memory (must be a pointer to managed memory or to a valid host-accessible region of system-allocated pageable memory) 
`length`
    \- Length of memory 
`flags`
    \- Must be one of [CUmemAttach_flags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g17c5d5f9b585aa2d6f121847d1a78f4c)

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21), [CUDA_ERROR_NOT_SUPPORTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b)

###### Description

Enqueues an operation in `hStream` to specify stream association of `length` bytes of memory starting from `dptr`. This function is a stream-ordered operation, meaning that it is dependent on, and will only take effect when, previous work in stream has completed. Any previous association is automatically replaced. 

`dptr` must point to one of the following types of memories: 

  * managed memory declared using the __managed__ keyword or allocated with [cuMemAllocManaged](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32 "Allocates memory that will be automatically managed by the Unified Memory system."). 

  * a valid host-accessible region of system-allocated pageable memory. This type of memory may only be specified if the device associated with the stream reports a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd). 


For managed allocations, `length` must be either zero or the entire allocation's size. Both indicate that the entire allocation's stream association is being changed. Currently, it is not possible to change stream association for a portion of a managed allocation. 

For pageable host allocations, `length` must be non-zero. 

The stream association is specified using `flags` which must be one of [CUmemAttach_flags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g17c5d5f9b585aa2d6f121847d1a78f4c). If the [CU_MEM_ATTACH_GLOBAL](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c0b42aae6a29b41b734d4c0dea6c33313) flag is specified, the memory can be accessed by any stream on any device. If the [CU_MEM_ATTACH_HOST](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c8b59c62cab9c7a762338e5fae92e2e9c) flag is specified, the program makes a guarantee that it won't access the memory on the device from any stream on a device that has a zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e). If the [CU_MEM_ATTACH_SINGLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c4b70b6a5e039f61eccc6b8db3dfac442) flag is specified and `hStream` is associated with a device that has a zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e), the program makes a guarantee that it will only access the memory on the device from `hStream`. It is illegal to attach singly to the NULL stream, because the NULL stream is a virtual global stream and not a specific stream. An error will be returned in this case. 

When memory is associated with a single stream, the Unified Memory system will allow CPU access to this memory region so long as all operations in `hStream` have completed, regardless of whether other streams are active. In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream activity instead of whole-GPU activity. 

Accessing memory on the device from streams that are not associated with it will produce undefined results. No error checking is performed by the Unified Memory system to ensure that kernels launched into other streams do not access this region. 

It is a program's responsibility to order calls to [cuStreamAttachMemAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533 "Attach memory to a stream asynchronously.") via events, synchronization or other means to ensure legal access to memory at all times. Data visibility and coherency will be changed appropriately for all kernels which follow a stream-association change. 

If `hStream` is destroyed while data is associated with it, the association is removed and the association reverts to the default visibility of the allocation as specified at [cuMemAllocManaged](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32 "Allocates memory that will be automatically managed by the Unified Memory system."). For __managed__ variables, the default association is always [CU_MEM_ATTACH_GLOBAL](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c0b42aae6a29b41b734d4c0dea6c33313). Note that destroying a stream is an asynchronous operation, and as a result, the change to default association won't happen until all work in the stream has completed. 

Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuMemAllocManaged](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32 "Allocates memory that will be automatically managed by the Unified Memory system."), [cudaStreamAttachMemAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g496353d630c29c44a2e33f531a3944d1)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamBeginCapture ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669) mode ) 
    

Begins graph capture on a stream. 

######  Parameters 

`hStream`
    \- Stream in which to initiate capture 
`mode`
    \- Controls the interaction of this capture sequence with other API calls that are potentially unsafe. For more details see [cuThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d "Swaps the stream capture interaction mode for a thread."). 

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb)

###### Description

Begin graph capture on `hStream`. When a stream is in capture mode, all operations pushed into the stream will not be executed, but will instead be captured into a graph, which will be returned via [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph."). Capture may not be initiated if `stream` is CU_STREAM_LEGACY. Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in capture mode. The capture mode may be queried via [cuStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca "Returns a stream's capture status."). A unique id representing the capture sequence may be queried via [cuStreamGetCaptureInfo](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g85f03299332d6cf37578409d0e4b47ce "Query a stream's capture state."). 

If `mode` is not CU_STREAM_CAPTURE_MODE_RELAXED, [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph.") must be called on this stream from the same thread. 

Note:

Kernels captured using this API must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca "Returns a stream's capture status."), [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph."), [cuThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d "Swaps the stream capture interaction mode for a thread.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamBeginCaptureToCig ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCigCaptureParams](https://docs.nvidia.com/cuda/cuda-driver-api/structCUstreamCigCaptureParams.html#structCUstreamCigCaptureParams)* streamCigCaptureParams ) 
    

Begins capture to CIG on a stream. 

######  Parameters 

`hStream`
    \- Stream in which to initiate capture to CIG 
`streamCigCaptureParams`
    \- CIG capture parameters

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_NOT_SUPPORTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), 

###### Description

Support for CIG streams with D3D12 can be determined using [cuDeviceGetAttribute()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266 "Returns information about the device.") with [CU_DEVICE_ATTRIBUTE_D3D12_CIG_STREAMS_SUPPORTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a31a7828183c94492a9fbc7867c684e5ec). 

Begin CIG (CUDA in Graphics) capture on `hStream` for the graphics API as provided in `streamCigCaptureParams`. When a stream is in CIG capture mode, all operations pushed into the stream will not be executed, but will instead be captured into a graphics API command list/command buffer. All kernel launches and memory copy/memory set operations on the CIG stream will be recorded. When the command list is executed by the graphics API, all the stream's operations will execute in order along with other graphics API commands in the command list. 

CIG stream capture may not be initiated if `stream` is CU_STREAM_LEGACY. Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in CIG capture mode. 

The context must be also created in CIG mode previously, otherwise this operation will fail and [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d) will be returned. 

Data from the graphics client can be shared with CUDA via the `streamSharedData` in `streamCigCaptureParams`. The format of `streamSharedData` is dependent on the type of the graphics client. For D3D12, `streamSharedData` is an ID3D12CommandList object pointer. The command list must be in ready state for recording commands whenever kernels are launched on the stream. The command list provided must belong to the graphics API device that the CIG context was created with, otherwise the behavior will be undefined. 

The stream object may not be destroyed until its associated command list has finished executing on the GPU. The command list/command buffer used for capture may not be submitted for execution before a call to [cuStreamEndCaptureToCig](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1gc21bd0a08a9c60f4ab2dd5594e829746 "Ends CIG capture on a stream.") is made on the associated stream. 

Graphics resources to be accessed by work recorded on the CIG stream must use UAV barriers on the command list prior to recording work that accesses them on the stream. 

Resubmission of the same recorded command list is not allowed. Further more, care must be taken for the order of execution of the recorded CUDA work with regards to other CUDA work submitted under the same CIG context. Out-of-order execution can lead to device hangs or exceptions. 

CIG capture mode operates similarly to `cuStreamBeginCapture` with the `CU_STREAM_CAPTURE_MODE_RELAXED` option. There are additional limitations to streams in CIG capture mode. The following functions are not allowed for CIG streams whether directly or indirectly via a recorded graph launch: [cuLaunchHostFunc](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f "Enqueues a host function call in a stream.")[cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream.")[cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed.")[cuStreamWaitValue32](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g629856339de7bc6606047385addbb398 "Wait on a memory location.")[cuStreamWaitValue64](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g6910c1258c5f15aa5d699f0fd60d6933 "Wait on a memory location.")[cuStreamBatchMemOp](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g764c442de9b671f9dec856e8ae531ed1 "Batch operations to synchronize the stream via memory operations.")[cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream.")[cuStreamBeginCaptureToGraph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1gac495e0527d1dd6437f95ee482f61865 "Begins graph capture on a stream to an existing graph.")[cuMemAllocAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f "Allocates memory with stream ordered semantics.")[cuMemFreeAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf "Frees memory with stream ordered semantics.")

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamEndCaptureToCig](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1gc21bd0a08a9c60f4ab2dd5594e829746 "Ends CIG capture on a stream."), [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."), [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamBeginCaptureToGraph ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUgraph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a) hGraph, const [CUgraphNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c)* dependencies, const [CUgraphEdgeData](https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphEdgeData.html#structCUgraphEdgeData)* dependencyData, size_t numDependencies, [CUstreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669) mode ) 
    

Begins graph capture on a stream to an existing graph. 

######  Parameters 

`hStream`
    \- Stream in which to initiate capture. 
`hGraph`
    \- Graph to capture into. 
`dependencies`
    \- Dependencies of the first node captured in the stream. Can be NULL if numDependencies is 0. 
`dependencyData`
    \- Optional array of data associated with each dependency. 
`numDependencies`
    \- Number of dependencies. 
`mode`
    \- Controls the interaction of this capture sequence with other API calls that are potentially unsafe. For more details see [cuThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d "Swaps the stream capture interaction mode for a thread."). 

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb)

###### Description

Begin graph capture on `hStream`, placing new nodes into an existing graph. When a stream is in capture mode, all operations pushed into the stream will not be executed, but will instead be captured into `hGraph`. The graph will not be instantiable until the user calls [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph."). 

Capture may not be initiated if `stream` is CU_STREAM_LEGACY. Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in capture mode. The capture mode may be queried via [cuStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca "Returns a stream's capture status."). A unique id representing the capture sequence may be queried via [cuStreamGetCaptureInfo](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g85f03299332d6cf37578409d0e4b47ce "Query a stream's capture state."). 

If `mode` is not CU_STREAM_CAPTURE_MODE_RELAXED, [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph.") must be called on this stream from the same thread. 

Note:

Kernels captured using this API must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."), [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca "Returns a stream's capture status."), [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph."), [cuThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d "Swaps the stream capture interaction mode for a thread."), [cuGraphAddNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23 "Adds a node of arbitrary type to a graph.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamBeginRecaptureToGraph ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669) mode, [CUgraph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a) hGraph, [CUgraphRecaptureCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g17628a1a991a6b5c4fcf0e86283c23ac) callbackFunc, void* userData ) 
    

Begin graph capture on a stream to an existing graph. 

######  Parameters 

`hStream`
    \- Stream in which to initiate capture 
`mode`
    \- Controls the interaction of this capture sequence with other API calls that are potentially unsafe. For more details see [cuThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d "Swaps the stream capture interaction mode for a thread."). 
`hGraph`
    \- Existing CUDA graph to be captured into 
`callbackFunc`
    \- Function that will be called for all parameter mismatches from the original graph 
`userData`
    \- A generic pointer to user data that is passed into the callback function

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), 

###### Description

Begin graph capture on `hStream` to the existing `hGraph`. The node creation order while recapturing the graph must be identical to the original graph. The recapture will fail immediately for: * Topology mismatches between the existing graph and the recaptured graph * Parameter mismatches for memory allocation or free nodes 

Any other node parameter mismatches during recapture can be configured to call the function provided in `callbackFunc`. The recapture will fail immediately if the callback returns anything other than CUDA_SUCCESS. 

If the recapture fails for any reason, the `graph` will be in an undefined state and should be destroyed. 

See cuStreamBeginCapture for additional detail on beginning the capture.

Note:

Any user objects associated with `graph` will be released prior to the recapture. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."), [cuStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca "Returns a stream's capture status."), [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph."), [cuThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d "Swaps the stream capture interaction mode for a thread.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamCopyAttributes ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) dst, [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) src ) 
    

Copies attributes from source stream to destination stream. 

######  Parameters 

`dst`
    Destination stream 
`src`
    Source stream For list of attributes see CUstreamAttrID

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb)

###### Description

Copies attributes from source stream `src` to destination stream `dst`. Both streams must have the same context. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamCreate ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a)* phStream, unsigned int  Flags ) 
    

Create a stream. 

######  Parameters 

`phStream`
    \- Returned newly created stream 
`Flags`
    \- Parameters for stream creation

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02)[CUDA_ERROR_EXTERNAL_DEVICE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9ceb154e2a824d8397aee018dea64d7ad)

###### Description

Creates a stream and returns a handle in `phStream`. The `Flags` argument determines behaviors of the stream. 

Valid values for `Flags` are: 

  * [CU_STREAM_DEFAULT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg775cb4ffbb7adf91e190067d9ad1752aaa5df0ec96f491f1be1124fdf265a066): Default stream creation flag. 

  * [CU_STREAM_NON_BLOCKING](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg775cb4ffbb7adf91e190067d9ad1752a89727d1d315214a6301abe98b419aff6): Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream), and that the created stream should perform no implicit synchronization with stream 0. 


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority."), [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context."), [cuStreamGetPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484 "Query the priority of a given stream."), [cuStreamGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7 "Query the flags of a given stream."), [cuStreamGetDevice](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b "Returns the device handle of the stream.")[cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream."), [cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamCreateWithPriority ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a)* phStream, unsigned int  flags, int  priority ) 
    

Create a stream with the given priority. 

######  Parameters 

`phStream`
    \- Returned newly created stream 
`flags`
    \- Flags for stream creation. See [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream.") for a list of valid flags 
`priority`
    \- Stream priority. Lower numbers represent higher priorities. See [cuCtxGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091 "Returns numerical values that correspond to the least and greatest stream priorities.") for more information about meaningful stream priorities that can be passed. 

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02)[CUDA_ERROR_EXTERNAL_DEVICE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9ceb154e2a824d8397aee018dea64d7ad)

###### Description

Creates a stream with the specified priority and returns a handle in `phStream`. This affects the scheduling priority of work in the stream. Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order. 

`priority` follows a convention where lower numbers represent higher priorities. '0' represents default priority. The range of meaningful numerical priorities can be queried using [cuCtxGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091 "Returns numerical values that correspond to the least and greatest stream priorities."). If the specified priority is outside the numerical range returned by [cuCtxGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091 "Returns numerical values that correspond to the least and greatest stream priorities."), it will automatically be clamped to the lowest or the highest number in the range. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * Stream priorities are supported only on GPUs with compute capability 3.5 or higher.

  * In the current implementation, only compute kernels launched in priority streams are affected by the stream's priority. Stream priorities have no effect on host-to-device and device-to-host memory operations. 


**See also:**

[cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context."), [cuStreamGetPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484 "Query the priority of a given stream."), [cuCtxGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091 "Returns numerical values that correspond to the least and greatest stream priorities."), [cuStreamGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7 "Query the flags of a given stream."), [cuStreamGetDevice](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b "Returns the device handle of the stream."), [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream."), [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamDestroy ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream ) 
    

Destroys a stream. 

######  Parameters 

`hStream`
    \- Stream to destroy

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21)[CUDA_ERROR_EXTERNAL_DEVICE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9ceb154e2a824d8397aee018dea64d7ad)

###### Description

Destroys the stream specified by `hStream`. 

In case the device is still doing work in the stream `hStream` when [cuStreamDestroy()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream.") is called, the function will return immediately and the resources associated with `hStream` will be released automatically once the device has completed all work in `hStream`. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream."), [cudaStreamDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamEndCapture ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUgraph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a)* phGraph ) 
    

Ends capture on a stream, returning the captured graph. 

######  Parameters 

`hStream`
    \- Stream to query 
`phGraph`
    \- The captured graph

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e979282fa9b0bd6a56167b5ddf44391440)

###### Description

End capture on `hStream`, returning the captured graph via `phGraph`. Capture must have been initiated on `hStream` via a call to [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."). If capture was invalidated, due to a violation of the rules of stream capture, then a NULL graph will be returned. 

If the `mode` argument to [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream.") was not CU_STREAM_CAPTURE_MODE_RELAXED, this call must be from the same thread as [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."). 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."), [cuStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca "Returns a stream's capture status."), [cuGraphDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g718cfd9681f078693d4be2426fd689c8 "Destroys a graph.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamEndCaptureToCig ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream ) 
    

Ends CIG capture on a stream. 

######  Parameters 

`hStream`
    \- Stream to end CIG capture

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e979282fa9b0bd6a56167b5ddf44391440)

###### Description

End CIG capture on `hStream`. Capture must have been initiated on `hStream` via a call to [cuStreamBeginCaptureToCig](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g05756a51c341a98172c5993487b76c39 "Begins capture to CIG on a stream."). Once this function is called, `hStream` will exit CIG capture mode and return to its original state, thus removing all CIG stream restrictions. Also, the command list/command buffer that was associated with `hStream` in the previous call to [cuStreamBeginCaptureToCig](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g05756a51c341a98172c5993487b76c39 "Begins capture to CIG on a stream.") is now allowed to be submitted for execution on the graphics API. However, the stream may not be destroyed until execution of the command list is fully done on the GPU. This requirements extends also to all streams dependant on the CIG stream (e.g. via event waits). 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamBeginCaptureToCig](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g05756a51c341a98172c5993487b76c39 "Begins capture to CIG on a stream.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamGetAttribute ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamAttrID](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331) attr, [CUstreamAttrValue](https://docs.nvidia.com/cuda/cuda-driver-api/unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue)* value_out ) 
    

Queries stream attribute. 

######  Parameters 

`hStream`
    
`attr`
    
`value_out`
    

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21)

###### Description

Queries attribute `attr` from `hStream` and stores it in corresponding member of `value_out`. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamGetCaptureInfo ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCaptureStatus](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g7480c0f2bd19894e54fcd2c04d6efb91)* captureStatus_out, cuuint64_t* id_out, [CUgraph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a)* graph_out, const [CUgraphNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c)** dependencies_out, const [CUgraphEdgeData](https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphEdgeData.html#structCUgraphEdgeData)** edgeData_out, size_t* numDependencies_out ) 
    

Query a stream's capture state. 

######  Parameters 

`hStream`
    \- The stream to query 
`captureStatus_out`
    \- Location to return the capture status of the stream; required 
`id_out`
    \- Optional location to return an id for the capture sequence, which is unique over the lifetime of the process 
`graph_out`
    \- Optional location to return the graph being captured into. All operations other than destroy and node removal are permitted on the graph while the capture sequence is in progress. This API does not transfer ownership of the graph, which is transferred or destroyed at [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph."). Note that the graph handle may be invalidated before end of capture for certain errors. Nodes that are or become unreachable from the original stream at [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph.") due to direct actions on the graph do not trigger [CUDA_ERROR_STREAM_CAPTURE_UNJOINED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9426e5dd5af746f6ee25aeb0f9fd32402). 
`dependencies_out`
    \- Optional location to store a pointer to an array of nodes. The next node to be captured in the stream will depend on this set of nodes, absent operations such as event wait which modify this set. The array pointer is valid until the next API call which operates on the stream or until the capture is terminated. The node handles may be copied out and are valid until they or the graph is destroyed. The driver-owned array may also be passed directly to APIs that operate on the graph (not the stream) without copying. 
`edgeData_out`
    \- Optional location to store a pointer to an array of graph edge data. This array parallels `dependencies_out`; the next node to be added has an edge to `dependencies_out`[i] with annotation `edgeData_out`[i] for each `i`. The array pointer is valid until the next API call which operates on the stream or until the capture is terminated. 
`numDependencies_out`
    \- Optional location to store the size of the array returned in dependencies_out.

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_STREAM_CAPTURE_IMPLICIT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9960a55453736ec87ca941f9bc2d80abe), [CUDA_ERROR_LOSSY_QUERY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7)

###### Description

Query stream state related to stream capture.

If called on [CU_STREAM_LEGACY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa) (the "null stream") while a stream not created with [CU_STREAM_NON_BLOCKING](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg775cb4ffbb7adf91e190067d9ad1752a89727d1d315214a6301abe98b419aff6) is capturing, returns [CUDA_ERROR_STREAM_CAPTURE_IMPLICIT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9960a55453736ec87ca941f9bc2d80abe). 

Valid data (other than capture status) is returned only if both of the following are true: 

  * the call returns CUDA_SUCCESS

  * the returned capture status is [CU_STREAM_CAPTURE_STATUS_ACTIVE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7480c0f2bd19894e54fcd2c04d6efb91c799fa3d867e2b300dfc45a6e90bc15d)


If `edgeData_out` is non-NULL then `dependencies_out` must be as well. If `dependencies_out` is non-NULL and `edgeData_out` is NULL, but there is non-zero edge data for one or more of the current stream dependencies, the call will return [CUDA_ERROR_LOSSY_QUERY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7). 

Note:

  * Graph objects are not threadsafe. [More here](https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html#graphs-thread-safety). 

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."), [cuStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca "Returns a stream's capture status."), [cuStreamUpdateCaptureDependencies](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g0cd3210434f3e0796c492cfa0d4b4bd1 "Update the set of dependencies in a capturing stream.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamGetCtx ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUcontext](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9)* pctx ) 
    

Query the context associated with a stream. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`pctx`
    \- Returned context associated with the stream

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21), [CUDA_ERROR_NOT_SUPPORTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b)

###### Description

Returns the CUDA context that the stream is associated with.

If the stream was created via the API [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context."), the returned context is equivalent to the one returned by [cuCtxFromGreenCtx()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb "Returns a CUcontext handle for a green context.") on the green context associated with the stream at creation time. 

The stream handle `hStream` can refer to any of the following: 

  * a stream created via any of the CUDA driver APIs such as [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream.") and [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority."), or their runtime API equivalents such as [cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3) and [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540). The returned context is the context that was active in the calling thread when the stream was created. Passing an invalid handle will result in undefined behavior. 

  * any of the special streams such as the NULL stream, [CU_STREAM_LEGACY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa) and [CU_STREAM_PER_THREAD](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c). The runtime API equivalents of these are also accepted, which are NULL, [cudaStreamLegacy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246) and [cudaStreamPerThread](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197) respectively. Specifying any of the special handles will return the context current to the calling thread. If no context is current to the calling thread, [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d) is returned. 


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority."), [cuStreamGetPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484 "Query the priority of a given stream."), [cuStreamGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7 "Query the flags of a given stream."), [cuStreamGetDevice](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b "Returns the device handle of the stream.")[cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream."), [cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamGetCtx_v2 ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUcontext](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9)* pCtx, [CUgreenCtx](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268)* pGreenCtx ) 
    

Query the contexts associated with a stream. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`pCtx`
    \- Returned regular context associated with the stream 
`pGreenCtx`
    \- Returned green context if the stream is associated with a green context or NULL if not

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21)

###### Description

Returns the contexts that the stream is associated with.

If the stream is associated with a green context, the API returns the green context in `pGreenCtx` and the primary context of the associated device in `pCtx`. 

If the stream is associated with a regular context, the API returns the regular context in `pCtx` and NULL in `pGreenCtx`. 

The stream handle `hStream` can refer to any of the following: 

  * a stream created via any of the CUDA driver APIs such as [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority.") and [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context."), or their runtime API equivalents such as [cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3) and [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540). Passing an invalid handle will result in undefined behavior. 

  * any of the special streams such as the NULL stream, [CU_STREAM_LEGACY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa) and [CU_STREAM_PER_THREAD](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c). The runtime API equivalents of these are also accepted, which are NULL, [cudaStreamLegacy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246) and [cudaStreamPerThread](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197) respectively. If any of the special handles are specified, the API will operate on the context current to the calling thread. If a green context (that was converted via [cuCtxFromGreenCtx()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb "Returns a CUcontext handle for a green context.") before setting it current) is current to the calling thread, the API will return the green context in `pGreenCtx` and the primary context of the associated device in `pCtx`. If a regular context is current, the API returns the regular context in `pCtx` and NULL in `pGreenCtx`. Note that specifying [CU_STREAM_PER_THREAD](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c) or [cudaStreamPerThread](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197) will return [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21) if a green context is current to the calling thread. If no context is current to the calling thread, [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d) is returned. 


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream.")[cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority."), [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context."), [cuStreamGetPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484 "Query the priority of a given stream."), [cuStreamGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7 "Query the flags of a given stream."), [cuStreamGetDevice](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b "Returns the device handle of the stream."), [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream."), [cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3), 

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamGetDevice ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUdevice](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a)* device ) 
    

Returns the device handle of the stream. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`device`
    \- Returns the device to which a stream belongs

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21), [CUDA_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02)

###### Description

Returns in `*device` the device handle of the stream 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context."), [cuStreamGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7 "Query the flags of a given stream.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamGetFlags ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, unsigned int* flags ) 
    

Query the flags of a given stream. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`flags`
    \- Pointer to an unsigned integer in which the stream's flags are returned The value returned in `flags` is a logical 'OR' of all flags that were used while creating this stream. See [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream.") for the list of valid flags 

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21), [CUDA_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02)

###### Description

Query the flags of a stream created using [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority.") or [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context.") and return the flags in `flags`. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context."), [cuStreamGetPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484 "Query the priority of a given stream."), [cudaStreamGetFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ga7f311f88126d751b9a7d3302ad6d0f8), [cuStreamGetDevice](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b "Returns the device handle of the stream.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamGetId ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, unsigned long long* streamId ) 
    

Returns the unique Id associated with the stream handle supplied. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`streamId`
    \- Pointer to store the Id of the stream

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21)

###### Description

Returns in `streamId` the unique Id which is associated with the given stream handle. The Id is unique for the life of the program. 

The stream handle `hStream` can refer to any of the following: 

  * a stream created via any of the CUDA driver APIs such as [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream.") and [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority."), or their runtime API equivalents such as [cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3) and [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540). Passing an invalid handle will result in undefined behavior. 

  * any of the special streams such as the NULL stream, [CU_STREAM_LEGACY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa) and [CU_STREAM_PER_THREAD](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c). The runtime API equivalents of these are also accepted, which are NULL, [cudaStreamLegacy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246) and [cudaStreamPerThread](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197) respectively. 


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamGetPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484 "Query the priority of a given stream."), [cudaStreamGetId](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g5799ae8dd744e561dfdeda02c53e82df)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamGetPriority ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, int* priority ) 
    

Query the priority of a given stream. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`priority`
    \- Pointer to a signed integer in which the stream's priority is returned 

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21), [CUDA_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02)

###### Description

Query the priority of a stream created using [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority.") or [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context.") and return the priority in `priority`. Note that if the stream was created with a priority outside the numerical range returned by [cuCtxGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091 "Returns numerical values that correspond to the least and greatest stream priorities."), this function returns the clamped priority. See [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority.") for details about priority clamping. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471 "Create a stream with the given priority."), [cuGreenCtxStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a "Create a stream for use in the green context."), [cuCtxGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091 "Returns numerical values that correspond to the least and greatest stream priorities."), [cuStreamGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7 "Query the flags of a given stream."), [cuStreamGetDevice](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b "Returns the device handle of the stream."), [cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamIsCapturing ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamCaptureStatus](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g7480c0f2bd19894e54fcd2c04d6efb91)* captureStatus ) 
    

Returns a stream's capture status. 

######  Parameters 

`hStream`
    \- Stream to query 
`captureStatus`
    \- Returns the stream's capture status

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_STREAM_CAPTURE_IMPLICIT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9960a55453736ec87ca941f9bc2d80abe)

###### Description

Return the capture status of `hStream` via `captureStatus`. After a successful call, `*captureStatus` will contain one of the following: 

  * [CU_STREAM_CAPTURE_STATUS_NONE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7480c0f2bd19894e54fcd2c04d6efb91e4023001f651dbdd3e3f55a1afc87fb3): The stream is not capturing. 

  * [CU_STREAM_CAPTURE_STATUS_ACTIVE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7480c0f2bd19894e54fcd2c04d6efb91c799fa3d867e2b300dfc45a6e90bc15d): The stream is capturing. 

  * [CU_STREAM_CAPTURE_STATUS_INVALIDATED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7480c0f2bd19894e54fcd2c04d6efb916b8a69837a782cd52243d481a2c6f51a): The stream was capturing but an error has invalidated the capture sequence. The capture sequence must be terminated with [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph.") on the stream where it was initiated in order to continue using `hStream`. 


Note that, if this is called on [CU_STREAM_LEGACY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa) (the "null stream") while a blocking stream in the same context is capturing, it will return [CUDA_ERROR_STREAM_CAPTURE_IMPLICIT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9960a55453736ec87ca941f9bc2d80abe) and `*captureStatus` is unspecified after the call. The blocking stream capture is not invalidated. 

When a blocking stream is capturing, the legacy stream is in an unusable state until the blocking stream capture is terminated. The legacy stream is not supported for stream capture, but attempted use would have an implicit dependency on the capturing stream(s). 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."), [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamQuery ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream ) 
    

Determine status of a compute stream. 

######  Parameters 

`hStream`
    \- Stream to query status of

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21), [CUDA_ERROR_NOT_READY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34)

###### Description

Returns [CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d) if all operations in the stream specified by `hStream` have completed, or [CUDA_ERROR_NOT_READY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34) if not. 

For the purposes of Unified Memory, a return value of [CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d) is equivalent to having called [cuStreamSynchronize()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."). 

Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream."), [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamSetAttribute ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUstreamAttrID](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331) attr, const [CUstreamAttrValue](https://docs.nvidia.com/cuda/cuda-driver-api/unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue)* value ) 
    

Sets stream attribute. 

######  Parameters 

`hStream`
    
`attr`
    
`value`
    

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21)

###### Description

Sets attribute `attr` on `hStream` from corresponding attribute of `value`. The updated attribute will be applied to subsequent work submitted to the stream. It will not affect previously submitted work. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamSynchronize ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream ) 
    

Wait until a stream's tasks are completed. 

######  Parameters 

`hStream`
    \- Stream to wait for

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21)

###### Description

Waits until the device has completed all operations in the stream specified by `hStream`. If the context was created with the [CU_CTX_SCHED_BLOCKING_SYNC](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852) flag, the CPU thread will block until the stream is finished with all of its tasks. 

Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f "Make a compute stream wait on an event."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream."), [cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamUpdateCaptureDependencies ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUgraphNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c)* dependencies, const [CUgraphEdgeData](https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphEdgeData.html#structCUgraphEdgeData)* dependencyData, size_t numDependencies, unsigned int  flags ) 
    

Update the set of dependencies in a capturing stream. 

######  Parameters 

`hStream`
    \- The stream to update 
`dependencies`
    \- The set of dependencies to add 
`dependencyData`
    \- Optional array of data associated with each dependency. 
`numDependencies`
    \- The size of the dependencies array 
`flags`
    \- See above

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_ILLEGAL_STATE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9f5fd9c15b97a258f341bab23b0b505a5)

###### Description

Modifies the dependency set of a capturing stream. The dependency set is the set of nodes that the next captured node in the stream will depend on along with the edge data for those dependencies. 

Valid flags are [CU_STREAM_ADD_CAPTURE_DEPENDENCIES](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggef58550e3d1f6d73c7e326455e744663bab808cd5e4e683f7000cb109973604e) and [CU_STREAM_SET_CAPTURE_DEPENDENCIES](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggef58550e3d1f6d73c7e326455e744663e3ada3eef9666e592a2d4c3301d08fca). These control whether the set passed to the API is added to the existing set or replaces it. A flags value of 0 defaults to [CU_STREAM_ADD_CAPTURE_DEPENDENCIES](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggef58550e3d1f6d73c7e326455e744663bab808cd5e4e683f7000cb109973604e). 

Nodes that are removed from the dependency set via this API do not result in [CUDA_ERROR_STREAM_CAPTURE_UNJOINED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9426e5dd5af746f6ee25aeb0f9fd32402) if they are unreachable from the stream at [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph."). 

Returns [CUDA_ERROR_ILLEGAL_STATE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9f5fd9c15b97a258f341bab23b0b505a5) if the stream is not capturing. 

**See also:**

[cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."), [cuStreamGetCaptureInfo](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g85f03299332d6cf37578409d0e4b47ce "Query a stream's capture state.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuStreamWaitEvent ( [CUstream](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a) hStream, [CUevent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b) hEvent, unsigned int  Flags ) 
    

Make a compute stream wait on an event. 

######  Parameters 

`hStream`
    \- Stream to wait 
`hEvent`
    \- Event to wait on (may not be NULL) 
`Flags`
    \- See CUevent_capture_flags

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_INVALID_HANDLE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21), 

###### Description

Makes all future work submitted to `hStream` wait for all work captured in `hEvent`. See [cuEventRecord()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1 "Records an event.") for details on what is captured by an event. The synchronization will be performed efficiently on the device when applicable. `hEvent` may be from a different context or device than `hStream`. 

flags include: 

  * [CU_EVENT_WAIT_DEFAULT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg0dbe4cec219cab20846e3f269a5440d4ab2546b7da3337d9dd2bdec73c032e18): Default event creation flag. 

  * [CU_EVENT_WAIT_EXTERNAL](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg0dbe4cec219cab20846e3f269a5440d42e696252699844df830094402b2a83d7): Event is captured in the graph as an external event node when performing stream capture. This flag is invalid outside of stream capture. 


Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4 "Create a stream."), [cuEventRecord](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1 "Records an event."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b "Determine status of a compute stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad "Wait until a stream's tasks are completed."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483 "Add a callback to a compute stream."), [cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758 "Destroys a stream."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9)

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuThreadExchangeStreamCaptureMode ( [CUstreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669)* mode ) 
    

Swaps the stream capture interaction mode for a thread. 

######  Parameters 

`mode`
    \- Pointer to mode value to swap with the current mode

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb)

###### Description

Sets the calling thread's stream capture interaction mode to the value contained in `*mode`, and overwrites `*mode` with the previous mode for the thread. To facilitate deterministic behavior across function or module boundaries, callers are encouraged to use this API in a push-pop fashion: 
    
    
    ‎     [CUstreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669) mode = desiredMode;
               [cuThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d "Swaps the stream capture interaction mode for a thread.")(&mode);
               ...
               [cuThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d "Swaps the stream capture interaction mode for a thread.")(&mode); // restore previous mode

During stream capture (see [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream.")), some actions, such as a call to [cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356), may be unsafe. In the case of [cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356), the operation is not enqueued asynchronously to a stream, and is not observed by stream capture. Therefore, if the sequence of operations captured via [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream.") depended on the allocation being replayed whenever the graph is launched, the captured graph would be invalid. 

Therefore, stream capture places restrictions on API calls that can be made within or concurrently to a [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream.")-[cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c "Ends capture on a stream, returning the captured graph.") sequence. This behavior can be controlled via this API and flags to [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream."). 

A thread's mode is one of the following: 

  * `CU_STREAM_CAPTURE_MODE_GLOBAL:` This is the default mode. If the local thread has an ongoing capture sequence that was not initiated with `CU_STREAM_CAPTURE_MODE_RELAXED` at `cuStreamBeginCapture`, or if any other thread has a concurrent capture sequence initiated with `CU_STREAM_CAPTURE_MODE_GLOBAL`, this thread is prohibited from potentially unsafe API calls. 

  * `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL:` If the local thread has an ongoing capture sequence not initiated with `CU_STREAM_CAPTURE_MODE_RELAXED`, it is prohibited from potentially unsafe API calls. Concurrent capture sequences in other threads are ignored. 

  * `CU_STREAM_CAPTURE_MODE_RELAXED:` The local thread is not prohibited from potentially unsafe API calls. Note that the thread is still prohibited from API calls which necessarily conflict with stream capture, for example, attempting [cuEventQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef "Queries an event's status.") on an event that was last recorded inside a capture sequence. 


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143 "Begins graph capture on a stream.")

* * *

![](https://docs.nvidia.com/cuda/common/formatting/NVIDIA-LogoBlack.svg)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Manage My Privacy](https://www.nvidia.com/en-us/privacy-center/) | [Do Not Sell or Share My Data](https://www.nvidia.com/en-us/preferences/email-preferences/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2026 NVIDIA Corporation

![](https://docs.nvidia.com/akam/13/pixel_6b113425?a=dD1jM2VhNTQ4NDliMmJjYjU4NWIwZmJkMWJkZmUzZmExZTBhYjdlNDkyJmpzPW9mZg==)