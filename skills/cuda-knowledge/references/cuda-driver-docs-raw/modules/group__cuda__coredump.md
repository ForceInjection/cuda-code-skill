# Coredump Attributes Control API

**Source:** [group__CUDA__COREDUMP.html](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html)

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


[< Previous](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html) | [Next >](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)

CUDA Driver API ([PDF](https://docs.nvidia.com/cuda/pdf/CUDA_Driver_API.pdf)) \- v13.3.1 ([older](https://developer.nvidia.com/cuda-toolkit-archive)) \- Last updated June 29, 2026 \- [Send Feedback](mailto:CUDAIssues@nvidia.com?subject=CUDA%20Toolkit%20Documentation%20Feedback:%20CUDA%20Driver%20API)

## 6.35. Coredump Attributes Control API

This section describes the coredump attribute control functions of the low-level CUDA driver application programming interface. 

### Typedefs

typedef CUcoredumpCallbackEntry_st * [CUcoredumpCallbackHandle](#group__CUDA__COREDUMP_1g9076700453e0d7c89ae1f74ca0eef555)
    Opaque handle representing a registered coredump status callback. 
typedef void(CUDA_CB* [CUcoredumpStatusCallback](#group__CUDA__COREDUMP_1g538185ddcc12f5eb9b7b7ecf8e9fd77c) )( void*  userData,  int pid,  CUdevice dev ) 
    Callback function prototype for GPU coredump status notifications. 

### Enumerations

enum [CUCoredumpGenerationFlags](#group__CUDA__COREDUMP_1g516d6bb94a388c0efa9f50efa6d215c9)
    
enum [CUcoredumpSettings](#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62)
    

### Functions

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuCoredumpDeregisterCompleteCallback](#group__CUDA__COREDUMP_1g0755f85ac8123062db6c4b2da0c654e7) ( [CUcoredumpCallbackHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9076700453e0d7c89ae1f74ca0eef555 "Opaque handle representing a registered coredump status callback. ") callback ) 
    Deregister a previously registered coredump complete callback. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuCoredumpDeregisterStartCallback](#group__CUDA__COREDUMP_1gd740dbaebff72cb7f155338104b6a675) ( [CUcoredumpCallbackHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9076700453e0d7c89ae1f74ca0eef555 "Opaque handle representing a registered coredump status callback. ") callback ) 
    Deregister a previously registered coredump start callback. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuCoredumpGetAttribute](#group__CUDA__COREDUMP_1g56d7eb4975c7eb8e2b4eb0713fd8cedd) ( [CUcoredumpSettings](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62) attrib, void* value, size_t* size ) 
    Allows caller to fetch a coredump attribute value for the current context. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuCoredumpGetAttributeGlobal](#group__CUDA__COREDUMP_1g5cb5b7ddf41a2c3631eed8d00c4ae819) ( [CUcoredumpSettings](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62) attrib, void* value, size_t* size ) 
    Allows caller to fetch a coredump attribute value for the entire application. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuCoredumpRegisterCompleteCallback](#group__CUDA__COREDUMP_1g2ccc1cc9d01135950fcc9c2a8ec6f9b5) ( [CUcoredumpStatusCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g538185ddcc12f5eb9b7b7ecf8e9fd77c "Callback function prototype for GPU coredump status notifications. ") callback, void* userData, [CUcoredumpCallbackHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9076700453e0d7c89ae1f74ca0eef555 "Opaque handle representing a registered coredump status callback. ")* callbackOut ) 
    Register a callback to be invoked when a GPU coredump completes. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuCoredumpRegisterStartCallback](#group__CUDA__COREDUMP_1gff8a2e3192675a2a5b15db50125d04fc) ( [CUcoredumpStatusCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g538185ddcc12f5eb9b7b7ecf8e9fd77c "Callback function prototype for GPU coredump status notifications. ") callback, void* userData, [CUcoredumpCallbackHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9076700453e0d7c89ae1f74ca0eef555 "Opaque handle representing a registered coredump status callback. ")* callbackOut ) 
    Register a callback to be invoked when a GPU coredump begins. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuCoredumpSetAttribute](#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb) ( [CUcoredumpSettings](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62) attrib, void* value, size_t* size ) 
    Allows caller to set a coredump attribute value for the current context. 
[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) [cuCoredumpSetAttributeGlobal](#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990) ( [CUcoredumpSettings](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62) attrib, void* value, size_t* size ) 
    Allows caller to set a coredump attribute value globally. 

### Typedefs

typedef CUcoredumpCallbackEntry_st * CUcoredumpCallbackHandle
    

Opaque handle representing a registered coredump status callback. This handle is returned when registering a callback and must be provided when deregistering the callback. 

void(CUDA_CB* CUcoredumpStatusCallback )( void*  userData,  int pid,  CUdevice dev ) 
    

Callback function prototype for GPU coredump status notifications. This callback will be invoked when a GPU coredump begins or completes, depending on which registration function was used. The callback executes synchronously during the coredump process. 

######  Parameters 

`userData`
    \- User-provided data pointer that was passed during registration 
`int pid`
    
`CUdevice dev`
    

### Enumerations

enum CUCoredumpGenerationFlags
    

Flags for controlling coredump contents 

######  Values 

CU_COREDUMP_DEFAULT_FLAGS = 0
    
CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = (1<<0)
    
CU_COREDUMP_SKIP_GLOBAL_MEMORY = (1<<1)
    
CU_COREDUMP_SKIP_SHARED_MEMORY = (1<<2)
    
CU_COREDUMP_SKIP_LOCAL_MEMORY = (1<<3)
    
CU_COREDUMP_SKIP_ABORT = (1<<4)
    
CU_COREDUMP_SKIP_CONSTBANK_MEMORY = (1<<5)
    
CU_COREDUMP_GZIP_COMPRESS = (1<<6)
    
CU_COREDUMP_FAULTED_CONTEXTS_ONLY = (1<<7)
    
CU_COREDUMP_NO_ERRBAR_AT_EXIT = (1<<30)
    
CU_COREDUMP_LOG_ONLY = (1<<31)
    
CU_COREDUMP_LIGHTWEIGHT_FLAGS = CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES |CU_COREDUMP_SKIP_GLOBAL_MEMORY |CU_COREDUMP_SKIP_SHARED_MEMORY |CU_COREDUMP_SKIP_LOCAL_MEMORY |CU_COREDUMP_SKIP_CONSTBANK_MEMORY
    

enum CUcoredumpSettings
    

Flags for choosing a coredump attribute to get/set 

######  Values 

CU_COREDUMP_ENABLE_ON_EXCEPTION = 1
    
CU_COREDUMP_TRIGGER_HOST
    
CU_COREDUMP_LIGHTWEIGHT
    
CU_COREDUMP_ENABLE_USER_TRIGGER
    
CU_COREDUMP_FILE
    
CU_COREDUMP_PIPE
    
CU_COREDUMP_GENERATION_FLAGS
    
CU_COREDUMP_MAX
    

### Functions

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuCoredumpDeregisterCompleteCallback ( [CUcoredumpCallbackHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9076700453e0d7c89ae1f74ca0eef555 "Opaque handle representing a registered coredump status callback. ") callback ) 
    

Deregister a previously registered coredump complete callback. 

######  Parameters 

`callback`
    \- The callback handle to deregister

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb)

###### Description

This function removes a callback that was registered with [cuCoredumpRegisterCompleteCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g2ccc1cc9d01135950fcc9c2a8ec6f9b5 "Register a callback to be invoked when a GPU coredump completes."). The callback handle becomes invalid after this call. 

Note:

It is the caller's responsibility to deregister callbacks before they go out of scope.

**See also:**

[cuCoredumpRegisterCompleteCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g2ccc1cc9d01135950fcc9c2a8ec6f9b5 "Register a callback to be invoked when a GPU coredump completes.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuCoredumpDeregisterStartCallback ( [CUcoredumpCallbackHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9076700453e0d7c89ae1f74ca0eef555 "Opaque handle representing a registered coredump status callback. ") callback ) 
    

Deregister a previously registered coredump start callback. 

######  Parameters 

`callback`
    \- The callback handle to deregister

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb)

###### Description

This function removes a callback that was registered with [cuCoredumpRegisterStartCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1gff8a2e3192675a2a5b15db50125d04fc "Register a callback to be invoked when a GPU coredump begins."). The callback handle becomes invalid after this call. 

Note:

It is the caller's responsibility to deregister callbacks before they go out of scope.

**See also:**

[cuCoredumpRegisterStartCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1gff8a2e3192675a2a5b15db50125d04fc "Register a callback to be invoked when a GPU coredump begins.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuCoredumpGetAttribute ( [CUcoredumpSettings](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62) attrib, void* value, size_t* size ) 
    

Allows caller to fetch a coredump attribute value for the current context. 

######  Parameters 

`attrib`
    \- The enum defining which value to fetch. 
`value`
    \- void* containing the requested data. 
`size`
    \- The size of the memory region `value` points to. 

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_NOT_PERMITTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_CONTEXT_IS_DESTROYED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139)

###### Description

Returns in `*value` the requested value specified by `attrib`. It is up to the caller to ensure that the data type and size of `*value` matches the request. 

If the caller calls this function with `*value` equal to NULL, the size of the memory region (in bytes) expected for `attrib` will be placed in `size`. 

The supported attributes are: 

  * CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where true means that GPU exceptions from this context will create a coredump at the location specified by CU_COREDUMP_FILE. The default value is false unless set to true globally or locally, or the CU_CTX_USER_COREDUMP_ENABLE flag was set during context creation. 

  * CU_COREDUMP_TRIGGER_HOST: Bool where true means that the host CPU will also create a coredump. The default value is true unless set to false globally or or locally. This value is deprecated as of CUDA 12.5 - raise the CU_COREDUMP_SKIP_ABORT flag to disable host device abort() if needed. 

  * CU_COREDUMP_LIGHTWEIGHT: Bool where true means that any resulting coredumps will not have a dump of GPU memory or non-reloc ELF images. The default value is false unless set to true globally or locally. This attribute is deprecated as of CUDA 12.5, please use CU_COREDUMP_GENERATION_FLAGS instead. 

  * CU_COREDUMP_ENABLE_USER_TRIGGER: Bool where true means that a coredump can be created by writing to the system pipe specified by CU_COREDUMP_PIPE. The default value is false unless set to true globally or locally. 

  * CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where any coredumps generated by this context will be written. The default value is core.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA applications and PID is the process ID of the CUDA application. 

  * CU_COREDUMP_PIPE: String of up to 1023 characters that defines the name of the pipe that will be monitored if user-triggered coredumps are enabled. The default value is corepipe.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA application and PID is the process ID of the CUDA application. 

  * CU_COREDUMP_GENERATION_FLAGS: An integer with values to allow granular control the data contained in a coredump specified as a bitwise OR combination of the following values: + CU_COREDUMP_DEFAULT_FLAGS - if set by itself, coredump generation returns to its default settings of including all memory regions that it is able to access + CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES \- Coredump will not include the data from CUDA source modules that are not relocated at runtime. + CU_COREDUMP_SKIP_GLOBAL_MEMORY \- Coredump will not include device-side global data that does not belong to any context. + CU_COREDUMP_SKIP_SHARED_MEMORY \- Coredump will not include grid-scale shared memory for the warp that the dumped kernel belonged to. + CU_COREDUMP_SKIP_LOCAL_MEMORY \- Coredump will not include local memory from the kernel. + CU_COREDUMP_LIGHTWEIGHT_FLAGS - Enables all of the above options. Equiavlent to setting the CU_COREDUMP_LIGHTWEIGHT attribute to true. + CU_COREDUMP_SKIP_ABORT - If set, GPU exceptions will not raise an abort() in the host CPU process. Same functional goal as CU_COREDUMP_TRIGGER_HOST but better reflects the default behavior. + CU_COREDUMP_SKIP_CONSTBANK_MEMORY - Coredump will not include constbank memory. + CU_COREDUMP_GZIP_COMPRESS - The generated coredump will be compressed with gzip, and .gz suffix will be appended to the filename, if it's not a part of it already. + CU_COREDUMP_FAULTED_CONTEXTS_ONLY - The coredump will only include contexts that have encountered an exception or a trap. + CU_COREDUMP_NO_ERRBAR_AT_EXIT - By default, when coredumps are requested, the GPU will ensure memory faults and other errors prevent warps from exiting, if possible. This can potentially affect the performance of the application. Setting this flag will disable this functionality, making it possible for faulted warps to exit, but also avoiding the potential performance hit. + CU_COREDUMP_LOG_ONLY - Setting this flag will disable actual generation of the coredump file, but exception details will still be logged. 


**See also:**

[cuCoredumpGetAttributeGlobal](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g5cb5b7ddf41a2c3631eed8d00c4ae819 "Allows caller to fetch a coredump attribute value for the entire application."), [cuCoredumpSetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb "Allows caller to set a coredump attribute value for the current context."), [cuCoredumpSetAttributeGlobal](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990 "Allows caller to set a coredump attribute value globally.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuCoredumpGetAttributeGlobal ( [CUcoredumpSettings](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62) attrib, void* value, size_t* size ) 
    

Allows caller to fetch a coredump attribute value for the entire application. 

######  Parameters 

`attrib`
    \- The enum defining which value to fetch. 
`value`
    \- void* containing the requested data. 
`size`
    \- The size of the memory region `value` points to. 

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb)

###### Description

Returns in `*value` the requested value specified by `attrib`. It is up to the caller to ensure that the data type and size of `*value` matches the request. 

If the caller calls this function with `*value` equal to NULL, the size of the memory region (in bytes) expected for `attrib` will be placed in `size`. 

The supported attributes are: 

  * CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where true means that GPU exceptions from this context will create a coredump at the location specified by CU_COREDUMP_FILE. The default value is false. 

  * CU_COREDUMP_TRIGGER_HOST: Bool where true means that the host CPU will also create a coredump. The default value is true unless set to false globally or or locally. This value is deprecated as of CUDA 12.5 - raise the CU_COREDUMP_SKIP_ABORT flag to disable host device abort() if needed. 

  * CU_COREDUMP_LIGHTWEIGHT: Bool where true means that any resulting coredumps will not have a dump of GPU memory or non-reloc ELF images. The default value is false. This attribute is deprecated as of CUDA 12.5, please use CU_COREDUMP_GENERATION_FLAGS instead. 

  * CU_COREDUMP_ENABLE_USER_TRIGGER: Bool where true means that a coredump can be created by writing to the system pipe specified by CU_COREDUMP_PIPE. The default value is false. 

  * CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where any coredumps generated by this context will be written. The default value is core.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA applications and PID is the process ID of the CUDA application. 

  * CU_COREDUMP_PIPE: String of up to 1023 characters that defines the name of the pipe that will be monitored if user-triggered coredumps are enabled. The default value is corepipe.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA application and PID is the process ID of the CUDA application. 

  * CU_COREDUMP_GENERATION_FLAGS: An integer with values to allow granular control the data contained in a coredump specified as a bitwise OR combination of the following values: + CU_COREDUMP_DEFAULT_FLAGS - if set by itself, coredump generation returns to its default settings of including all memory regions that it is able to access + CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES \- Coredump will not include the data from CUDA source modules that are not relocated at runtime. + CU_COREDUMP_SKIP_GLOBAL_MEMORY \- Coredump will not include device-side global data that does not belong to any context. + CU_COREDUMP_SKIP_SHARED_MEMORY \- Coredump will not include grid-scale shared memory for the warp that the dumped kernel belonged to. + CU_COREDUMP_SKIP_LOCAL_MEMORY \- Coredump will not include local memory from the kernel. + CU_COREDUMP_LIGHTWEIGHT_FLAGS - Enables all of the above options. Equiavlent to setting the CU_COREDUMP_LIGHTWEIGHT attribute to true. + CU_COREDUMP_SKIP_ABORT - If set, GPU exceptions will not raise an abort() in the host CPU process. Same functional goal as CU_COREDUMP_TRIGGER_HOST but better reflects the default behavior. + CU_COREDUMP_SKIP_CONSTBANK_MEMORY - Coredump will not include constbank memory. + CU_COREDUMP_GZIP_COMPRESS - The generated coredump will be compressed with gzip, and .gz suffix will be appended to the filename, if it's not a part of it already. + CU_COREDUMP_FAULTED_CONTEXTS_ONLY - The coredump will only include contexts that have encountered an exception or a trap. + CU_COREDUMP_NO_ERRBAR_AT_EXIT - By default, when coredumps are requested, the GPU will ensure memory faults and other errors prevent warps from exiting, if possible. This can potentially affect the performance of the application. Setting this flag will disable this functionality, making it possible for faulted warps to exit, but also avoiding the potential performance hit. + CU_COREDUMP_LOG_ONLY - Setting this flag will disable actual generation of the coredump file, but exception details will still be logged. 


**See also:**

[cuCoredumpGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g56d7eb4975c7eb8e2b4eb0713fd8cedd "Allows caller to fetch a coredump attribute value for the current context."), [cuCoredumpSetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb "Allows caller to set a coredump attribute value for the current context."), [cuCoredumpSetAttributeGlobal](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990 "Allows caller to set a coredump attribute value globally.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuCoredumpRegisterCompleteCallback ( [CUcoredumpStatusCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g538185ddcc12f5eb9b7b7ecf8e9fd77c "Callback function prototype for GPU coredump status notifications. ") callback, void* userData, [CUcoredumpCallbackHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9076700453e0d7c89ae1f74ca0eef555 "Opaque handle representing a registered coredump status callback. ")* callbackOut ) 
    

Register a callback to be invoked when a GPU coredump completes. 

######  Parameters 

`callback`
    \- The callback function to register 
`userData`
    \- User data pointer to pass to the callback 
`callbackOut`
    \- Location to store the callback handle (optional, may be NULL)

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02)

###### Description

This function registers a callback that will be called when a GPU coredump has been fully collected and written to disk. Callbacks are executed in the order they were registered. The same callback function can be registered multiple times with different userData, and each registration will receive a unique handle. 

Note:

Callbacks execute synchronously during the coredump process and will block coredump progress while running.

**See also:**

[cuCoredumpDeregisterCompleteCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g0755f85ac8123062db6c4b2da0c654e7 "Deregister a previously registered coredump complete callback."), [cuCoredumpRegisterStartCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1gff8a2e3192675a2a5b15db50125d04fc "Register a callback to be invoked when a GPU coredump begins.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuCoredumpRegisterStartCallback ( [CUcoredumpStatusCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g538185ddcc12f5eb9b7b7ecf8e9fd77c "Callback function prototype for GPU coredump status notifications. ") callback, void* userData, [CUcoredumpCallbackHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9076700453e0d7c89ae1f74ca0eef555 "Opaque handle representing a registered coredump status callback. ")* callbackOut ) 
    

Register a callback to be invoked when a GPU coredump begins. 

######  Parameters 

`callback`
    \- The callback function to register 
`userData`
    \- User data pointer to pass to the callback 
`callbackOut`
    \- Location to store the callback handle (optional, may be NULL)

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02)

###### Description

This function registers a callback that will be called when a GPU coredump is initiated, before any coredump data is collected. Callbacks are executed in the order they were registered. The same callback function can be registered multiple times with different userData, and each registration will receive a unique handle. 

Note:

Callbacks execute synchronously during the coredump process and will block coredump progress while running.

**See also:**

[cuCoredumpDeregisterStartCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1gd740dbaebff72cb7f155338104b6a675 "Deregister a previously registered coredump start callback."), [cuCoredumpRegisterCompleteCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g2ccc1cc9d01135950fcc9c2a8ec6f9b5 "Register a callback to be invoked when a GPU coredump completes.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuCoredumpSetAttribute ( [CUcoredumpSettings](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62) attrib, void* value, size_t* size ) 
    

Allows caller to set a coredump attribute value for the current context. 

######  Parameters 

`attrib`
    \- The enum defining which value to set. 
`value`
    \- void* containing the requested data. 
`size`
    \- The size of the memory region `value` points to. 

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_NOT_PERMITTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3), [CUDA_ERROR_DEINITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a), [CUDA_ERROR_NOT_INITIALIZED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8), [CUDA_ERROR_INVALID_CONTEXT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d), [CUDA_ERROR_CONTEXT_IS_DESTROYED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139), [CUDA_ERROR_NOT_SUPPORTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b)

###### Description

This function should be considered an alternate interface to the CUDA-GDB environment variables defined in this document: <https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-coredump>

An important design decision to note is that any coredump environment variable values set before CUDA initializes will take permanent precedence over any values set with this function. This decision was made to ensure no change in behavior for any users that may be currently using these variables to get coredumps. 

`*value` shall contain the requested value specified by `set`. It is up to the caller to ensure that the data type and size of `*value` matches the request. 

If the caller calls this function with `*value` equal to NULL, the size of the memory region (in bytes) expected for `set` will be placed in `size`. 

/note This function will return [CUDA_ERROR_NOT_SUPPORTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b) if the caller attempts to set CU_COREDUMP_ENABLE_ON_EXCEPTION on a GPU of with Compute Capability < 6.0. [cuCoredumpSetAttributeGlobal](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990 "Allows caller to set a coredump attribute value globally.") works on those platforms as an alternative. 

/note CU_COREDUMP_ENABLE_USER_TRIGGER and CU_COREDUMP_PIPE cannot be set on a per-context basis.

The supported attributes are: 

  * CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where true means that GPU exceptions from this context will create a coredump at the location specified by CU_COREDUMP_FILE. The default value is false. 

  * CU_COREDUMP_TRIGGER_HOST: Bool where true means that the host CPU will also create a coredump. The default value is true unless set to false globally or or locally. This value is deprecated as of CUDA 12.5 - raise the CU_COREDUMP_SKIP_ABORT flag to disable host device abort() if needed. 

  * CU_COREDUMP_LIGHTWEIGHT: Bool where true means that any resulting coredumps will not have a dump of GPU memory or non-reloc ELF images. The default value is false. This attribute is deprecated as of CUDA 12.5, please use CU_COREDUMP_GENERATION_FLAGS instead. 

  * CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where any coredumps generated by this context will be written. The default value is core.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA applications and PID is the process ID of the CUDA application. 

  * CU_COREDUMP_GENERATION_FLAGS: An integer with values to allow granular control the data contained in a coredump specified as a bitwise OR combination of the following values: + CU_COREDUMP_DEFAULT_FLAGS - if set by itself, coredump generation returns to its default settings of including all memory regions that it is able to access + CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES \- Coredump will not include the data from CUDA source modules that are not relocated at runtime. + CU_COREDUMP_SKIP_GLOBAL_MEMORY \- Coredump will not include device-side global data that does not belong to any context. + CU_COREDUMP_SKIP_SHARED_MEMORY \- Coredump will not include grid-scale shared memory for the warp that the dumped kernel belonged to. + CU_COREDUMP_SKIP_LOCAL_MEMORY \- Coredump will not include local memory from the kernel. + CU_COREDUMP_LIGHTWEIGHT_FLAGS - Enables all of the above options. Equiavlent to setting the CU_COREDUMP_LIGHTWEIGHT attribute to true. + CU_COREDUMP_SKIP_ABORT - If set, GPU exceptions will not raise an abort() in the host CPU process. Same functional goal as CU_COREDUMP_TRIGGER_HOST but better reflects the default behavior. + CU_COREDUMP_SKIP_CONSTBANK_MEMORY - Coredump will not include constbank memory. + CU_COREDUMP_GZIP_COMPRESS - The generated coredump will be compressed with gzip, and .gz suffix will be appended to the filename, if it's not a part of it already. + CU_COREDUMP_FAULTED_CONTEXTS_ONLY - The coredump will only include contexts that have encountered an exception or a trap. + CU_COREDUMP_NO_ERRBAR_AT_EXIT - By default, when coredumps are requested, the GPU will ensure memory faults and other errors prevent warps from exiting, if possible. This can potentially affect the performance of the application. Setting this flag will disable this functionality, making it possible for faulted warps to exit, but also avoiding the potential performance hit. + CU_COREDUMP_LOG_ONLY - Setting this flag will disable actual generation of the coredump file, but exception details will still be logged. 


Note:

CU_COREDUMP_GENERATION_FLAGS replaces all previously set coredump flags. Mixing CU_COREDUMP_GENERATION_FLAGS with the deprecated boolean attributes (CU_COREDUMP_TRIGGER_HOST, CU_COREDUMP_LIGHTWEIGHT) can result in undefined behavior. To avoid issues, either use only CU_COREDUMP_GENERATION_FLAGS or combine all desired flag bits (including CU_COREDUMP_SKIP_ABORT) in a single call. 

**See also:**

[cuCoredumpGetAttributeGlobal](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g5cb5b7ddf41a2c3631eed8d00c4ae819 "Allows caller to fetch a coredump attribute value for the entire application."), [cuCoredumpGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g56d7eb4975c7eb8e2b4eb0713fd8cedd "Allows caller to fetch a coredump attribute value for the current context."), [cuCoredumpSetAttributeGlobal](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990 "Allows caller to set a coredump attribute value globally.")

[CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9) cuCoredumpSetAttributeGlobal ( [CUcoredumpSettings](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62) attrib, void* value, size_t* size ) 
    

Allows caller to set a coredump attribute value globally. 

######  Parameters 

`attrib`
    \- The enum defining which value to set. 
`value`
    \- void* containing the requested data. 
`size`
    \- The size of the memory region `value` points to. 

###### Returns

[CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d), [CUDA_ERROR_INVALID_VALUE](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb), [CUDA_ERROR_NOT_PERMITTED](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3)

###### Description

This function should be considered an alternate interface to the CUDA-GDB environment variables defined in this document: <https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-coredump>

An important design decision to note is that any coredump environment variable values set before CUDA initializes will take permanent precedence over any values set with this function. This decision was made to ensure no change in behavior for any users that may be currently using these variables to get coredumps. 

`*value` shall contain the requested value specified by `set`. It is up to the caller to ensure that the data type and size of `*value` matches the request. 

If the caller calls this function with `*value` equal to NULL, the size of the memory region (in bytes) expected for `set` will be placed in `size`. 

The supported attributes are: 

  * CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where true means that GPU exceptions from this context will create a coredump at the location specified by CU_COREDUMP_FILE. The default value is false. 

  * CU_COREDUMP_TRIGGER_HOST: Bool where true means that the host CPU will also create a coredump. The default value is true unless set to false globally or or locally. This value is deprecated as of CUDA 12.5 - raise the CU_COREDUMP_SKIP_ABORT flag to disable host device abort() if needed. 

  * CU_COREDUMP_LIGHTWEIGHT: Bool where true means that any resulting coredumps will not have a dump of GPU memory or non-reloc ELF images. The default value is false. This attribute is deprecated as of CUDA 12.5, please use CU_COREDUMP_GENERATION_FLAGS instead. 

  * CU_COREDUMP_ENABLE_USER_TRIGGER: Bool where true means that a coredump can be created by writing to the system pipe specified by CU_COREDUMP_PIPE. The default value is false. 

  * CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where any coredumps generated by this context will be written. The default value is core.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA applications and PID is the process ID of the CUDA application. 

  * CU_COREDUMP_PIPE: String of up to 1023 characters that defines the name of the pipe that will be monitored if user-triggered coredumps are enabled. This value may not be changed after CU_COREDUMP_ENABLE_USER_TRIGGER is set to true. The default value is corepipe.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA application and PID is the process ID of the CUDA application. 

  * CU_COREDUMP_GENERATION_FLAGS: An integer with values to allow granular control the data contained in a coredump specified as a bitwise OR combination of the following values: + CU_COREDUMP_DEFAULT_FLAGS - if set by itself, coredump generation returns to its default settings of including all memory regions that it is able to access + CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES \- Coredump will not include the data from CUDA source modules that are not relocated at runtime. + CU_COREDUMP_SKIP_GLOBAL_MEMORY \- Coredump will not include device-side global data that does not belong to any context. + CU_COREDUMP_SKIP_SHARED_MEMORY \- Coredump will not include grid-scale shared memory for the warp that the dumped kernel belonged to. + CU_COREDUMP_SKIP_LOCAL_MEMORY \- Coredump will not include local memory from the kernel. + CU_COREDUMP_LIGHTWEIGHT_FLAGS - Enables all of the above options. Equiavlent to setting the CU_COREDUMP_LIGHTWEIGHT attribute to true. + CU_COREDUMP_SKIP_ABORT - If set, GPU exceptions will not raise an abort() in the host CPU process. Same functional goal as CU_COREDUMP_TRIGGER_HOST but better reflects the default behavior. + CU_COREDUMP_SKIP_CONSTBANK_MEMORY - Coredump will not include constbank memory. + CU_COREDUMP_GZIP_COMPRESS - The generated coredump will be compressed with gzip, and .gz suffix will be appended to the filename, if it's not a part of it already. + CU_COREDUMP_FAULTED_CONTEXTS_ONLY - The coredump will only include contexts that have encountered an exception or a trap. + CU_COREDUMP_NO_ERRBAR_AT_EXIT - By default, when coredumps are requested, the GPU will ensure memory faults and other errors prevent warps from exiting, if possible. This can potentially affect the performance of the application. Setting this flag will disable this functionality, making it possible for faulted warps to exit, but also avoiding the potential performance hit. + CU_COREDUMP_LOG_ONLY - Setting this flag will disable actual generation of the coredump file, but exception details will still be logged. 


Note:

CU_COREDUMP_GENERATION_FLAGS replaces all previously set coredump flags. Mixing CU_COREDUMP_GENERATION_FLAGS with the deprecated boolean attributes (CU_COREDUMP_TRIGGER_HOST, CU_COREDUMP_LIGHTWEIGHT) can result in undefined behavior. To avoid issues, either use only CU_COREDUMP_GENERATION_FLAGS or combine all desired flag bits (including CU_COREDUMP_SKIP_ABORT) in a single call. 

**See also:**

[cuCoredumpGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g56d7eb4975c7eb8e2b4eb0713fd8cedd "Allows caller to fetch a coredump attribute value for the current context."), [cuCoredumpGetAttributeGlobal](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g5cb5b7ddf41a2c3631eed8d00c4ae819 "Allows caller to fetch a coredump attribute value for the entire application."), [cuCoredumpSetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb "Allows caller to set a coredump attribute value for the current context.")

* * *

![](https://docs.nvidia.com/cuda/common/formatting/NVIDIA-LogoBlack.svg)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Manage My Privacy](https://www.nvidia.com/en-us/privacy-center/) | [Do Not Sell or Share My Data](https://www.nvidia.com/en-us/preferences/email-preferences/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2026 NVIDIA Corporation

![](https://docs.nvidia.com/akam/13/pixel_6b113425?a=dD1jM2VhNTQ4NDliMmJjYjU4NWIwZmJkMWJkZmUzZmExZTBhYjdlNDkyJmpzPW9mZg==)