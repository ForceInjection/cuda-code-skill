# Next >

**Source:** [group__CUDART__DEVICE.html](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)

---

### Functions

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaChooseDevice](#group__CUDART__DEVICE_1gf61f9ae0fe2d93b5b968756684a49460) ( int* device, const [cudaDeviceProp](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp)* prop ) 
     Select compute-device which best matches criteria. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceFlushGPUDirectRDMAWrites](#group__CUDART__DEVICE_1g3fc2853f4f30ce29019a8db822354d12) ( [cudaFlushGPUDirectRDMAWritesTarget](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g79b83831173a16801d2fb885b02e54a4) target, [cudaFlushGPUDirectRDMAWritesScope](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g1ed1a981bc94ef42e16f73c91faf8f1a) scope ) 
     Blocks until remote writes are visible to the specified scope. 
__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetAttribute](#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151) ( int* value, [cudaDeviceAttr](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd) attr, int  device ) 
     Returns information about the device. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetByPCIBusId](#group__CUDART__DEVICE_1g65f57fb8d0981ca03f6f9b20031c3e5d) ( int* device, const char* pciBusId ) 
     Returns a handle to a compute device. 
__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetCacheConfig](#group__CUDART__DEVICE_1gd9bf5eae6d464de05aa3840df9f5deeb) ( [cudaFuncCache *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gb980f35ed69ee7991704de29a13de49b)* pCacheConfig ) 
     Returns the preferred cache configuration for the current device. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetDefaultMemPool](#group__CUDART__DEVICE_1g69aee77b793f13fa79c5bc0b566845b7) ( [cudaMemPool_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gd7121b19b2347b777c07223c6ff4cdad)* memPool, int  device ) 
     Returns the default mempool of a device. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetHostAtomicCapabilities](#group__CUDART__DEVICE_1g2d9faed0ab26c97aabace7b177391c8d) ( unsigned int* capabilities, const [cudaAtomicOperation *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g097391b0c5ed60f698c6b5d729b901fc)* operations, unsigned int  count, int  device ) 
     Queries details about atomic operations supported between the device and host. 
__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetLimit](#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2) ( size_t* pValue, [cudaLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4c4b34c054d383b0e9a63ab0ffc93651) limit ) 
     Return resource limits. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetMemPool](#group__CUDART__DEVICE_1g9001cd1b2301101e81508f4b7714c97e) ( [cudaMemPool_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gd7121b19b2347b777c07223c6ff4cdad)* memPool, int  device ) 
     Gets the current mempool for a device. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetNvSciSyncAttributes](#group__CUDART__DEVICE_1gf70468b1bb3a7483d1cc393eb7301f2f) ( void* nvSciSyncAttrList, int  device, int  flags ) 
     Return NvSciSync attributes that this device can support. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetP2PAtomicCapabilities](#group__CUDART__DEVICE_1ga608cadee2598ca942b362db73267c2b) ( unsigned int* capabilities, const [cudaAtomicOperation *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g097391b0c5ed60f698c6b5d729b901fc)* operations, unsigned int  count, int  srcDevice, int  dstDevice ) 
     Queries details about atomic operations supported between two devices. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetP2PAttribute](#group__CUDART__DEVICE_1gc63e5bf168e53b2daf71904eab048fa9) ( int* value, [cudaDeviceP2PAttr](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g2f597e2acceab33f60bd61c41fea0c1b) attr, int  srcDevice, int  dstDevice ) 
     Queries attributes of the link between two devices. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetPCIBusId](#group__CUDART__DEVICE_1gea264dad3d8c4898e0b82213c0253def) ( char* pciBusId, int  len, int  device ) 
     Returns a PCI Bus Id string for the device. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetStreamPriorityRange](#group__CUDART__DEVICE_1gfdb79818f7c0ee7bc585648c91770275) ( int* leastPriority, int* greatestPriority ) 
     Returns numerical values that correspond to the least and greatest stream priorities. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceGetTexture1DLinearMaxWidth](#group__CUDART__DEVICE_1g45f0345fd7a3697d0766596593920f61) ( size_t* maxWidthInElements, const [cudaChannelFormatDesc](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaChannelFormatDesc.html#structcudaChannelFormatDesc)* fmtDesc, int  device ) 
     Returns the maximum number of elements allocatable in a 1D linear texture for a given element size. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceRegisterAsyncNotification](#group__CUDART__DEVICE_1gcff9794f21aa34d3b5ccc5b6b245da4a) ( int  device, cudaAsyncCallback callbackFunc, void* userData, [cudaAsyncCallbackHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf5d1b31c73cce10a9ae82d3cc11157d2)* callback ) 
     Registers a callback function to receive async notifications. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceReset](#group__CUDART__DEVICE_1gef69dd5c6d0206c2b8d099abac61f217) ( void ) 
     Destroy all allocations and reset all state on the current device in the current process. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceSetCacheConfig](#group__CUDART__DEVICE_1g6c9cc78ca80490386cf593b4baa35a15) ( [cudaFuncCache](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gb980f35ed69ee7991704de29a13de49b) cacheConfig ) 
     Sets the preferred cache configuration for the current device. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceSetLimit](#group__CUDART__DEVICE_1g05956f16eaa47ef3a4efee84563ccb7d) ( [cudaLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4c4b34c054d383b0e9a63ab0ffc93651) limit, size_t value ) 
     Set resource limits. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceSetMemPool](#group__CUDART__DEVICE_1g02bb50d2e2af83e5f24d0b2ab9dadc82) ( int  device, [cudaMemPool_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gd7121b19b2347b777c07223c6ff4cdad) memPool ) 
     Sets the current memory pool of a device. 
__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceSynchronize](#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d) ( void ) 
     Wait for compute device to finish. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaDeviceUnregisterAsyncNotification](#group__CUDART__DEVICE_1geeeae22bfbfc38915ed9b7b58376e64d) ( int  device, [cudaAsyncCallbackHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf5d1b31c73cce10a9ae82d3cc11157d2) callback ) 
     Unregisters an async notification callback. 
__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaGetDevice](#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78) ( int* device ) 
     Returns which device is currently being used. 
__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaGetDeviceCount](#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f) ( int* count ) 
     Returns the number of compute-capable devices. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaGetDeviceFlags](#group__CUDART__DEVICE_1gf830794caf068b71638c6182bba8f77a) ( unsigned int* flags ) 
     Gets the flags for the current device. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaGetDeviceProperties](#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0) ( [cudaDeviceProp](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp)* prop, int  device ) 
     Returns information about the compute-device. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaInitDevice](#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419) ( int  device, unsigned int  deviceFlags, unsigned int  flags ) 
     Initialize device to be used for GPU executions. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaIpcCloseMemHandle](#group__CUDART__DEVICE_1g02bb3632b5d223db6acae5f8744e2c91) ( void* devPtr ) 
     Attempts to close memory mapped with cudaIpcOpenMemHandle. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaIpcGetEventHandle](#group__CUDART__DEVICE_1g89a3abe1e9a11d08c665176669109784) ( [cudaIpcEventHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaIpcEventHandle__t.html#structcudaIpcEventHandle__t)* handle, [cudaEvent_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247) event ) 
     Gets an interprocess handle for a previously allocated event. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaIpcGetMemHandle](#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539) ( [cudaIpcMemHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaIpcMemHandle__t.html#structcudaIpcMemHandle__t)* handle, void* devPtr ) 
     Gets an interprocess memory handle for an existing device memory allocation. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaIpcOpenEventHandle](#group__CUDART__DEVICE_1g9691446ab0aec1d6e528357387ed87b2) ( [cudaEvent_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247)* event, [cudaIpcEventHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaIpcEventHandle__t.html#structcudaIpcEventHandle__t) handle ) 
     Opens an interprocess event handle for use in the current process. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaIpcOpenMemHandle](#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9) ( void** devPtr, [cudaIpcMemHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaIpcMemHandle__t.html#structcudaIpcMemHandle__t) handle, unsigned int  flags ) 
     Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaSetDevice](#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb) ( int  device ) 
     Set device to be used for GPU executions. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaSetDeviceFlags](#group__CUDART__DEVICE_1g69e73c7dda3fc05306ae7c811a690fac) ( unsigned int  flags ) 
     Sets flags to be used for device executions. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaSetValidDevices](#group__CUDART__DEVICE_1g1b9336c70f2299405f67a4f8496d7cfe) ( int* device_arr, int  len ) 
     Set a list of devices that can be used for CUDA. 

### Functions

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaChooseDevice ( int* device, const [cudaDeviceProp](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp)* prop ) 
    

Select compute-device which best matches criteria. 

######  Parameters 

`device`
    \- Device with best match 
`prop`
    \- Desired device properties

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Returns in `*device` the device which has properties that best match `*prop`. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f "Returns the number of compute-capable devices."), [cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78 "Returns which device is currently being used."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0 "Returns information about the compute-device."), [cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419 "Initialize device to be used for GPU executions.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceFlushGPUDirectRDMAWrites ( [cudaFlushGPUDirectRDMAWritesTarget](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g79b83831173a16801d2fb885b02e54a4) target, [cudaFlushGPUDirectRDMAWritesScope](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g1ed1a981bc94ef42e16f73c91faf8f1a) scope ) 
    

Blocks until remote writes are visible to the specified scope. 

######  Parameters 

`target`
    \- The target of the operation, see cudaFlushGPUDirectRDMAWritesTarget 
`scope`
    \- The scope of the operation, see cudaFlushGPUDirectRDMAWritesScope

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164), 

###### Description

Blocks until remote writes to the target context via mappings created through GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see <https://docs.nvidia.com/cuda/gpudirect-rdma> for more information), are visible to the specified scope. 

If the scope equals or lies within the scope indicated by [cudaDevAttrGPUDirectRDMAWritesOrdering](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cd881d5b0c5bf7777e382920af4d743bde), the call will be a no-op and can be safely omitted for performance. This can be determined by comparing the numerical values between the two enums, with smaller scopes having smaller values. 

Users may query support for this API via [cudaDevAttrGPUDirectRDMAFlushWritesOptions](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cd7b3462e13705248606f39fb5381446ee). 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cuFlushGPUDirectRDMAWrites](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g265e3c82ef0f0fe035f85c4c45a8fbdf)

__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetAttribute ( int* value, [cudaDeviceAttr](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd) attr, int  device ) 
    

Returns information about the device. 

######  Parameters 

`value`
    \- Returned device attribute value 
`attr`
    \- Device attribute to query 
`device`
    \- Device number to query

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Returns in `*value` the integer value of the attribute `attr` on device `device`. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f "Returns the number of compute-capable devices."), [cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78 "Returns which device is currently being used."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaChooseDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf61f9ae0fe2d93b5b968756684a49460 "Select compute-device which best matches criteria."), [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0 "Returns information about the compute-device."), [cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419 "Initialize device to be used for GPU executions."), [cuDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetByPCIBusId ( int* device, const char* pciBusId ) 
    

Returns a handle to a compute device. 

######  Parameters 

`device`
    \- Returned device ordinal
`pciBusId`
    \- String in one of the following forms: [domain]:[bus]:[device].[function] [domain]:[bus]:[device] [bus]:[device].[function] where `domain`, `bus`, `device`, and `function` are all hexadecimal values 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a)

###### Description

Returns in `*device` a device ordinal given a PCI bus ID string. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaDeviceGetPCIBusId](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gea264dad3d8c4898e0b82213c0253def "Returns a PCI Bus Id string for the device."), [cuDeviceGetByPCIBusId](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1ga89cd3fa06334ba7853ed1232c5ebe2a)

__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetCacheConfig ( [cudaFuncCache *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gb980f35ed69ee7991704de29a13de49b)* pCacheConfig ) 
    

Returns the preferred cache configuration for the current device. 

######  Parameters 

`pCacheConfig`
    \- Returned cache configuration

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591)

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this returns through `pCacheConfig` the preferred cache configuration for the current device. This is only a preference. The runtime will use the requested configuration if possible, but it is free to choose a different configuration if required to execute functions. 

This will return a `pCacheConfig` of [cudaFuncCachePreferNone](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49b3b4b8c65376ce1ca107be413e15981bc) on devices where the size of the L1 cache and shared memory are fixed. 

The supported cache configurations are: 

  * [cudaFuncCachePreferNone](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49b3b4b8c65376ce1ca107be413e15981bc): no preference for shared memory or L1 (default) 

  * [cudaFuncCachePreferShared](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49b84725d25c531f9bafc61ae329afe5b2b): prefer larger shared memory and smaller L1 cache 

  * [cudaFuncCachePreferL1](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49b8ecb48ccbc2230c81528a2c7c695100e): prefer larger L1 cache and smaller shared memory 

  * [cudaFuncCachePreferEqual](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49bd151ac8d667150c601de4b9542887a3b): prefer equal size L1 cache and shared memory 


Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaDeviceSetCacheConfig](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g6c9cc78ca80490386cf593b4baa35a15 "Sets the preferred cache configuration for the current device."), [cudaFuncSetCacheConfig ( C API)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g6699ca1943ac2655effa0d571b2f4f15 "Sets the preferred cache configuration for a device function."), [cudaFuncSetCacheConfig ( C++ API)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g7d9cc996fe45b6260ebb086caff1c685 "\[C++ API\] Sets the preferred cache configuration for a device function"), [cuCtxGetCacheConfig](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetDefaultMemPool ( [cudaMemPool_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gd7121b19b2347b777c07223c6ff4cdad)* memPool, int  device ) 
    

Returns the default mempool of a device. 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)[cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164)

###### Description

The default mempool of a device contains device memory from that device.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cuDeviceGetDefaultMemPool](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2), [cudaMallocAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb "Allocate from a pool."), [cudaMemPoolTrimTo](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g0faf526bcfffd835aa95a4514fb2f7d5 "Tries to release memory back to the OS."), [cudaMemPoolGetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g4ff47ef59413a4ed9d760c0841ce4a99 "Gets attributes of a memory pool."), [cudaDeviceSetMemPool](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb50d2e2af83e5f24d0b2ab9dadc82 "Sets the current memory pool of a device."), [cudaMemPoolSetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g0229135f7ef724b4f479a435ca300af5 "Sets attributes of a memory pool."), [cudaMemPoolSetAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g4210da54ee5a810945da586e00cb3019 "Controls visibility of pools between devices.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetHostAtomicCapabilities ( unsigned int* capabilities, const [cudaAtomicOperation *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g097391b0c5ed60f698c6b5d729b901fc)* operations, unsigned int  count, int  device ) 
    

Queries details about atomic operations supported between the device and host. 

######  Parameters 

`capabilities`
    \- Returned capability details of each requested operation 
`operations`
    \- Requested operations 
`count`
    \- Count of requested operations and size of capabilities 
`device`
    

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Returns in `*capabilities` the details about requested atomic `*operations` over the the link between `dev` and the host. The allocated size of `*operations` and `*capabilities` must be `count`. 

For each [cudaAtomicOperation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g097391b0c5ed60f698c6b5d729b901fc) in `*operations`, the corresponding result in `*capabilities` will be a bitmask indicating which of [cudaAtomicOperationCapability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3ded8497f09d41455ea64acb7a880868) the link supports natively. 

Returns [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a) if `dev` is not valid. 

Returns [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c) if `*capabilities` or `*operations` is NULL, if `count` is 0, or if any of `*operations` is not valid. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151 "Returns information about the device."), [cudaDeviceGetP2PAtomicCapabilities](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1ga608cadee2598ca942b362db73267c2b "Queries details about atomic operations supported between two devices."), cuDeviceGeHostAtomicCapabilities 

__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetLimit ( size_t* pValue, [cudaLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4c4b34c054d383b0e9a63ab0ffc93651) limit ) 
    

Return resource limits. 

######  Parameters 

`pValue`
    \- Returned size of the limit
`limit`
    \- Limit to query 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorUnsupportedLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c3b950b6f8668f7282fae25bfcefd13a), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Returns in `*pValue` the current size of `limit`. The following [cudaLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4c4b34c054d383b0e9a63ab0ffc93651) values are supported. 

  * [cudaLimitStackSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc93651fc8f54e641c9b133f1b57703d22ce656) is the stack size in bytes of each GPU thread. 

  * [cudaLimitPrintfFifoSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc9365123b80a6221a6853e918c2816bb76742c) is the size in bytes of the shared FIFO used by the printf() device system call. 

  * [cudaLimitMallocHeapSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc93651b399716bf0a592bc42055473c1273881) is the size in bytes of the heap used by the malloc() and free() device system calls. 

  * [cudaLimitDevRuntimeSyncDepth](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc9365123c4900be4af436cb769e4c72d07be11) is the maximum grid depth at which a thread can isssue the device runtime call [cudaDeviceSynchronize()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d "Wait for compute device to finish.") to wait on child grid launches to complete. This functionality is removed for devices of compute capability >= 9.0, and hence will return error [cudaErrorUnsupportedLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c3b950b6f8668f7282fae25bfcefd13a) on such devices. 

  * [cudaLimitDevRuntimePendingLaunchCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc9365118712cb05d2c3efaeea73afba823d916) is the maximum number of outstanding device runtime launches. 

  * [cudaLimitMaxL2FetchGranularity](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc9365168d3522068213e4ba46c4c99ebc4ce12) is the L2 cache fetch granularity. 

  * [cudaLimitPersistingL2CacheSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc93651fc5135f91c07d7aa2d1072db9854b113) is the persisting L2 cache size in bytes. 


Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaDeviceSetLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g05956f16eaa47ef3a4efee84563ccb7d "Set resource limits."), [cuCtxGetLimit](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetMemPool ( [cudaMemPool_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gd7121b19b2347b777c07223c6ff4cdad)* memPool, int  device ) 
    

Gets the current mempool for a device. 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)[cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164)

###### Description

Returns the last pool provided to [cudaDeviceSetMemPool](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb50d2e2af83e5f24d0b2ab9dadc82 "Sets the current memory pool of a device.") for this device or the device's default memory pool if [cudaDeviceSetMemPool](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb50d2e2af83e5f24d0b2ab9dadc82 "Sets the current memory pool of a device.") has never been called. By default the current mempool is the default mempool for a device, otherwise the returned pool must have been set with [cuDeviceSetMemPool](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0) or [cudaDeviceSetMemPool](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb50d2e2af83e5f24d0b2ab9dadc82 "Sets the current memory pool of a device."). 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cuDeviceGetMemPool](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b), [cudaDeviceGetDefaultMemPool](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69aee77b793f13fa79c5bc0b566845b7 "Returns the default mempool of a device."), [cudaDeviceSetMemPool](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb50d2e2af83e5f24d0b2ab9dadc82 "Sets the current memory pool of a device.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetNvSciSyncAttributes ( void* nvSciSyncAttrList, int  device, int  flags ) 
    

Return NvSciSync attributes that this device can support. 

######  Parameters 

`nvSciSyncAttrList`
    \- Return NvSciSync attributes supported. 
`device`
    \- Valid Cuda Device to get NvSciSync attributes for. 
`flags`
    \- flags describing NvSciSync usage.

###### Description

Returns in `nvSciSyncAttrList`, the properties of NvSciSync that this CUDA device, `dev` can support. The returned `nvSciSyncAttrList` can be used to create an NvSciSync that matches this device's capabilities. 

If NvSciSyncAttrKey_RequiredPerm field in `nvSciSyncAttrList` is already set this API will return [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c). 

The applications should set `nvSciSyncAttrList` to a valid NvSciSyncAttrList failing which this API will return cudaErrorInvalidHandle. 

The `flags` controls how applications intends to use the NvSciSync created from the `nvSciSyncAttrList`. The valid flags are: 

  * [cudaNvSciSyncAttrSignal](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g8157cb1b3c077ba222a56779a37d3bd1), specifies that the applications intends to signal an NvSciSync on this CUDA device. 

  * [cudaNvSciSyncAttrWait](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g407f1fe03a9f560b23220faa306e70cf), specifies that the applications intends to wait on an NvSciSync on this CUDA device. 


At least one of these flags must be set, failing which the API returns [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c). Both the flags are orthogonal to one another: a developer may set both these flags that allows to set both wait and signal specific attributes in the same `nvSciSyncAttrList`. 

Note that this API updates the input `nvSciSyncAttrList` with values equivalent to the following public attribute key-values: NvSciSyncAttrKey_RequiredPerm is set to 

  * NvSciSyncAccessPerm_SignalOnly if [cudaNvSciSyncAttrSignal](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g8157cb1b3c077ba222a56779a37d3bd1) is set in `flags`. 

  * NvSciSyncAccessPerm_WaitOnly if [cudaNvSciSyncAttrWait](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g407f1fe03a9f560b23220faa306e70cf) is set in `flags`. 

  * NvSciSyncAccessPerm_WaitSignal if both [cudaNvSciSyncAttrWait](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g407f1fe03a9f560b23220faa306e70cf) and [cudaNvSciSyncAttrSignal](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g8157cb1b3c077ba222a56779a37d3bd1) are set in `flags`. NvSciSyncAttrKey_PrimitiveInfo is set to 

  * NvSciSyncAttrValPrimitiveType_SysmemSemaphore on any valid `device`. 

  * NvSciSyncAttrValPrimitiveType_Syncpoint if `device` is a Tegra device. 

  * NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b if `device` is GA10X+. NvSciSyncAttrKey_GpuId is set to the same UUID that is returned in `[cudaDeviceProp.uuid](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1626c20637498c7be1381db55a6261308)` from cudaDeviceGetProperties for this `device`. 


[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorDeviceUninitialized](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00389e95927732f0d5a68151c72296581504), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), cudaErrorInvalidHandle, [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a), [cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164), [cudaErrorMemoryAllocation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f210f50ae7f17f655e0504929606add9)

**See also:**

[cudaImportExternalSemaphore](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP_1g64f7155c1fc4459a746db31b4aae263b "Imports an external semaphore."), [cudaDestroyExternalSemaphore](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP_1gd596384c606982e5e5a8b9a1e4dcec15 "Destroys an external semaphore."), [cudaSignalExternalSemaphoresAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP_1gb0806d36d4bc72b82afcf9315f11330c "Signals a set of external semaphore objects."), [cudaWaitExternalSemaphoresAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP_1g52ecbdf57e75f0fde9fa44a2f9532c2d "Waits on a set of external semaphore objects.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetP2PAtomicCapabilities ( unsigned int* capabilities, const [cudaAtomicOperation *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g097391b0c5ed60f698c6b5d729b901fc)* operations, unsigned int  count, int  srcDevice, int  dstDevice ) 
    

Queries details about atomic operations supported between two devices. 

######  Parameters 

`capabilities`
    \- Returned capability details of each requested operation 
`operations`
    \- Requested operations 
`count`
    \- Count of requested operations and size of capabilities 
`srcDevice`
    \- The source device of the target link 
`dstDevice`
    \- The destination device of the target link

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Returns in `*capabilities` the details about requested atomic `*operations` over the the link between `srcDevice` and `dstDevice`. The allocated size of `*operations` and `*capabilities` must be `count`. 

For each [cudaAtomicOperation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g097391b0c5ed60f698c6b5d729b901fc) in `*operations`, the corresponding result in `*capabilities` will be a bitmask indicating which of [cudaAtomicOperationCapability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3ded8497f09d41455ea64acb7a880868) the link supports natively. 

Returns [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a) if `srcDevice` or `dstDevice` are not valid or if they represent the same device. 

Returns [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c) if `*capabilities` or `*operations` is NULL, if `count` is 0, or if any of `*operations` is not valid. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaDeviceGetP2PAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gc63e5bf168e53b2daf71904eab048fa9 "Queries attributes of the link between two devices."), [cuDeviceGetP2PAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g4c55c60508f8eba4546b51f2ee545393), [cuDeviceGetP2PAtomicCapabilities](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1gfd989876c8fd3291b520c0b561d5282d)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetP2PAttribute ( int* value, [cudaDeviceP2PAttr](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g2f597e2acceab33f60bd61c41fea0c1b) attr, int  srcDevice, int  dstDevice ) 
    

Queries attributes of the link between two devices. 

######  Parameters 

`value`
    \- Returned value of the requested attribute 
`attr`
    
`srcDevice`
    \- The source device of the target link. 
`dstDevice`
    \- The destination device of the target link.

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Returns in `*value` the value of the requested attribute `attrib` of the link between `srcDevice` and `dstDevice`. The supported attributes are: 

  * [cudaDevP2PAttrPerformanceRank](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg2f597e2acceab33f60bd61c41fea0c1bdca8430f659738ac2ebfbfc4e0899d3b): A relative value indicating the performance of the link between two devices. Lower value means better performance (0 being the value used for most performant link). 

  * [cudaDevP2PAttrAccessSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg2f597e2acceab33f60bd61c41fea0c1b0fa6e51b6472b6ea6ea0cd27fea05a3c): 1 if peer access is enabled. 

  * [cudaDevP2PAttrNativeAtomicSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg2f597e2acceab33f60bd61c41fea0c1b8513982962e4439fa60f2a24348be587): 1 if all native atomic operations over the link are supported. 

  * [cudaDevP2PAttrCudaArrayAccessSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg2f597e2acceab33f60bd61c41fea0c1bc11dffbfb7a6d8872dfaeca4b971c11e): 1 if accessing CUDA arrays over the link is supported. 

  * [cudaDevP2PAttrOnlyPartialNativeAtomicSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg2f597e2acceab33f60bd61c41fea0c1be2a84f673603b20a1a3c41f2ab96621f): 1 if some CUDA-valid atomic operations over the link are supported. Information about specific operations can be retrieved with [cudaDeviceGetP2PAtomicCapabilities](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1ga608cadee2598ca942b362db73267c2b "Queries details about atomic operations supported between two devices."). 


Returns [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a) if `srcDevice` or `dstDevice` are not valid or if they represent the same device. 

Returns [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c) if `attrib` is not valid or if `value` is a null pointer. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaDeviceEnablePeerAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g2b0adabf90db37e5cfddc92cbb2589f3 "Enables direct access to memory allocations on a peer device."), [cudaDeviceDisablePeerAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g9663734ad02653207ad6836053bf572e "Disables direct access to memory allocations on a peer device."), [cudaDeviceCanAccessPeer](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g4db0d04e44995d5c1c34be4ecc863f22 "Queries if a device may directly access a peer device's memory."), [cuDeviceGetP2PAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g4c55c60508f8eba4546b51f2ee545393)[cudaDeviceGetP2PAtomicCapabilities](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1ga608cadee2598ca942b362db73267c2b "Queries details about atomic operations supported between two devices.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetPCIBusId ( char* pciBusId, int  len, int  device ) 
    

Returns a PCI Bus Id string for the device. 

######  Parameters 

`pciBusId`
    \- Returned identifier string for the device in the following format [domain]:[bus]:[device].[function] where `domain`, `bus`, `device`, and `function` are all hexadecimal values. pciBusId should be large enough to store 13 characters including the NULL-terminator. 
`len`
    \- Maximum length of string to store in `name`
`device`
    \- Device to get identifier string for

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a)

###### Description

Returns an ASCII string identifying the device `dev` in the NULL-terminated string pointed to by `pciBusId`. `len` specifies the maximum length of the string that may be returned. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaDeviceGetByPCIBusId](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g65f57fb8d0981ca03f6f9b20031c3e5d "Returns a handle to a compute device."), [cuDeviceGetPCIBusId](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g85295e7d9745ab8f0aa80dd1e172acfc)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetStreamPriorityRange ( int* leastPriority, int* greatestPriority ) 
    

Returns numerical values that correspond to the least and greatest stream priorities. 

######  Parameters 

`leastPriority`
    \- Pointer to an int in which the numerical value for least stream priority is returned 
`greatestPriority`
    \- Pointer to an int in which the numerical value for greatest stream priority is returned

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591)

###### Description

Returns in `*leastPriority` and `*greatestPriority` the numerical values that correspond to the least and greatest stream priorities respectively. Stream priorities follow a convention where lower numbers imply greater priorities. The range of meaningful stream priorities is given by [`*greatestPriority`, `*leastPriority`]. If the user attempts to create a stream with a priority value that is outside the the meaningful range as specified by this API, the priority is automatically clamped down or up to either `*leastPriority` or `*greatestPriority` respectively. See [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540 "Create an asynchronous stream with the specified priority.") for details on creating a priority stream. A NULL may be passed in for `*leastPriority` or `*greatestPriority` if the value is not desired. 

This function will return '0' in both `*leastPriority` and `*greatestPriority` if the current context's device does not support stream priorities (see [cudaDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151 "Returns information about the device.")). 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540 "Create an asynchronous stream with the specified priority."), [cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6 "Query the priority of a stream."), [cuCtxGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceGetTexture1DLinearMaxWidth ( size_t* maxWidthInElements, const [cudaChannelFormatDesc](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaChannelFormatDesc.html#structcudaChannelFormatDesc)* fmtDesc, int  device ) 
    

Returns the maximum number of elements allocatable in a 1D linear texture for a given element size. 

######  Parameters 

`maxWidthInElements`
    \- Returns maximum number of texture elements allocatable for given `fmtDesc`. 
`fmtDesc`
    \- Texture format description.
`device`
    

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorUnsupportedLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c3b950b6f8668f7282fae25bfcefd13a), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Returns in `maxWidthInElements` the maximum number of elements allocatable in a 1D linear texture for given format descriptor `fmtDesc`. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cuDeviceGetTexture1DLinearMaxWidth](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gb41b3a675bae9932bffa1c0ae969b1e0)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceRegisterAsyncNotification ( int  device, cudaAsyncCallback callbackFunc, void* userData, [cudaAsyncCallbackHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf5d1b31c73cce10a9ae82d3cc11157d2)* callback ) 
    

Registers a callback function to receive async notifications. 

######  Parameters 

`device`
    \- The device on which to register the callback 
`callbackFunc`
    \- The function to register as a callback 
`userData`
    \- A generic pointer to user data. This is passed into the callback function. 
`callback`
    \- A handle representing the registered callback instance

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591)[cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164)[cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a)[cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)[cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3)[cudaErrorUnknown](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00382e491daacef266c7b3e3c1e140a6133c)

###### Description

Registers `callbackFunc` to receive async notifications. 

The `userData` parameter is passed to the callback function at async notification time. Likewise, `callback` is also passed to the callback function to distinguish between multiple registered callbacks. 

The callback function being registered should be designed to return quickly (~10ms). Any long running tasks should be queued for execution on an application thread. 

Callbacks may not call cudaDeviceRegisterAsyncNotification or cudaDeviceUnregisterAsyncNotification. Doing so will result in [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3). Async notification callbacks execute in an undefined order and may be serialized. 

Returns in `*callback` a handle representing the registered callback instance. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaDeviceUnregisterAsyncNotification](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1geeeae22bfbfc38915ed9b7b58376e64d "Unregisters an async notification callback.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceReset ( void ) 
    

Destroy all allocations and reset all state on the current device in the current process. 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591)

###### Description

Explicitly destroys and cleans up all resources associated with the current device in the current process. It is the caller's responsibility to ensure that the resources are not accessed or passed in subsequent API calls and doing so will result in undefined behavior. These resources include CUDA types [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a), [cudaEvent_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247), [cudaArray_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gaf8b3ba752727d996074a71ee997ce68), [cudaMipmappedArray_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g1b03ad14fc0ab86f8b033837f5562d8a), [cudaPitchedPtr](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaPitchedPtr.html#structcudaPitchedPtr), [cudaTextureObject_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g83eb271ebc4cb2817e66d7c752f0c29b), [cudaSurfaceObject_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gbe57cf2ccbe7f9d696f18808dd634c0a), textureReference, surfaceReference, [cudaExternalMemory_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g79a2bf6a3a2aa9011a96dc17d476faf7), [cudaExternalSemaphore_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf8bc7ab93d638c5835e5289ac7d683c4) and [cudaGraphicsResource_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf58dd8d3c7a65714ff7f5459adbf7e6f). These resources also include memory allocations by [cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356 "Allocate memory on the device."), [cudaMallocHost](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gd5c991beb38e2b8419f50285707ae87e "\[C++ API\] Allocates page-locked memory on the host"), [cudaMallocManaged](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gcf6b9b1019e73c5bc2b39b39fe90816e "Allocates memory that will be automatically managed by the Unified Memory system.") and [cudaMallocPitch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c "Allocates pitched memory on the device."). Any subsequent API call to this device will reinitialize the device. 

Note that this function will reset the device immediately. It is the caller's responsibility to ensure that the device is not being accessed by any other host threads from the process when this function is called. 

Note:

  * [cudaDeviceReset()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gef69dd5c6d0206c2b8d099abac61f217 "Destroy all allocations and reset all state on the current device in the current process.") will not destroy memory allocations by [cudaMallocAsync()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1gbbf70065888d61853c047513baa14081 "Allocates memory with stream ordered semantics.") and [cudaMallocFromPoolAsync()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g871003f518e27ec92f7b331307fa32d4 "Allocates memory from a specified pool with stream ordered semantics."). These memory allocations need to be destroyed explicitly. 

  * If a non-primary [CUcontext](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9) is current to the thread, [cudaDeviceReset()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gef69dd5c6d0206c2b8d099abac61f217 "Destroy all allocations and reset all state on the current device in the current process.") will destroy only the internal CUDA RT state for that [CUcontext](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9). 


Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaDeviceSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d "Wait for compute device to finish.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceSetCacheConfig ( [cudaFuncCache](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gb980f35ed69ee7991704de29a13de49b) cacheConfig ) 
    

Sets the preferred cache configuration for the current device. 

######  Parameters 

`cacheConfig`
    \- Requested cache configuration

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591)

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through `cacheConfig` the preferred cache configuration for the current device. This is only a preference. The runtime will use the requested configuration if possible, but it is free to choose a different configuration if required to execute the function. Any function preference set via [cudaFuncSetCacheConfig ( C API)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g6699ca1943ac2655effa0d571b2f4f15 "Sets the preferred cache configuration for a device function.") or [cudaFuncSetCacheConfig ( C++ API)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g7d9cc996fe45b6260ebb086caff1c685 "\[C++ API\] Sets the preferred cache configuration for a device function") will be preferred over this device-wide setting. Setting the device-wide cache configuration to [cudaFuncCachePreferNone](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49b3b4b8c65376ce1ca107be413e15981bc) will cause subsequent kernel launches to prefer to not change the cache configuration unless required to launch the kernel. 

This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.

Launching a kernel with a different preference than the most recent preference setting may insert a device-side synchronization point. 

The supported cache configurations are: 

  * [cudaFuncCachePreferNone](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49b3b4b8c65376ce1ca107be413e15981bc): no preference for shared memory or L1 (default) 

  * [cudaFuncCachePreferShared](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49b84725d25c531f9bafc61ae329afe5b2b): prefer larger shared memory and smaller L1 cache 

  * [cudaFuncCachePreferL1](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49b8ecb48ccbc2230c81528a2c7c695100e): prefer larger L1 cache and smaller shared memory 

  * [cudaFuncCachePreferEqual](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggb980f35ed69ee7991704de29a13de49bd151ac8d667150c601de4b9542887a3b): prefer equal size L1 cache and shared memory 


Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaDeviceGetCacheConfig](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gd9bf5eae6d464de05aa3840df9f5deeb "Returns the preferred cache configuration for the current device."), [cudaFuncSetCacheConfig ( C API)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g6699ca1943ac2655effa0d571b2f4f15 "Sets the preferred cache configuration for a device function."), [cudaFuncSetCacheConfig ( C++ API)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g7d9cc996fe45b6260ebb086caff1c685 "\[C++ API\] Sets the preferred cache configuration for a device function"), [cuCtxSetCacheConfig](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceSetLimit ( [cudaLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4c4b34c054d383b0e9a63ab0ffc93651) limit, size_t value ) 
    

Set resource limits. 

######  Parameters 

`limit`
    \- Limit to set 
`value`
    \- Size of limit

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorUnsupportedLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c3b950b6f8668f7282fae25bfcefd13a), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorMemoryAllocation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f210f50ae7f17f655e0504929606add9)

###### Description

Setting `limit` to `value` is a request by the application to update the current limit maintained by the device. The driver is free to modify the requested value to meet h/w requirements (this could be clamping to minimum or maximum values, rounding up to nearest element size, etc). The application can use [cudaDeviceGetLimit()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2 "Return resource limits.") to find out exactly what the limit has been set to. 

Setting each [cudaLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4c4b34c054d383b0e9a63ab0ffc93651) has its own specific restrictions, so each is discussed here. 

  * [cudaLimitStackSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc93651fc8f54e641c9b133f1b57703d22ce656) controls the stack size in bytes of each GPU thread. 


  * [cudaLimitPrintfFifoSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc9365123b80a6221a6853e918c2816bb76742c) controls the size in bytes of the shared FIFO used by the printf() device system call. Setting [cudaLimitPrintfFifoSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc9365123b80a6221a6853e918c2816bb76742c) must not be performed after launching any kernel that uses the printf() device system call - in such case [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c) will be returned. 


  * [cudaLimitMallocHeapSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc93651b399716bf0a592bc42055473c1273881) controls the size in bytes of the heap used by the malloc() and free() device system calls. Setting [cudaLimitMallocHeapSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc93651b399716bf0a592bc42055473c1273881) must not be performed after launching any kernel that uses the malloc() or free() device system calls - in such case [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c) will be returned. 


  * [cudaLimitDevRuntimeSyncDepth](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc9365123c4900be4af436cb769e4c72d07be11) controls the maximum nesting depth of a grid at which a thread can safely call [cudaDeviceSynchronize()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d "Wait for compute device to finish."). Setting this limit must be performed before any launch of a kernel that uses the device runtime and calls [cudaDeviceSynchronize()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d "Wait for compute device to finish.") above the default sync depth, two levels of grids. Calls to [cudaDeviceSynchronize()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d "Wait for compute device to finish.") will fail with error code [cudaErrorSyncDepthExceeded](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038265dbf94c45903cd582cfc40f93a176a) if the limitation is violated. This limit can be set smaller than the default or up the maximum launch depth of 24. When setting this limit, keep in mind that additional levels of sync depth require the runtime to reserve large amounts of device memory which can no longer be used for user allocations. If these reservations of device memory fail, [cudaDeviceSetLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g05956f16eaa47ef3a4efee84563ccb7d "Set resource limits.") will return [cudaErrorMemoryAllocation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f210f50ae7f17f655e0504929606add9), and the limit can be reset to a lower value. This limit is only applicable to devices of compute capability < 9.0. Attempting to set this limit on devices of other compute capability will results in error [cudaErrorUnsupportedLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c3b950b6f8668f7282fae25bfcefd13a) being returned. 


  * [cudaLimitDevRuntimePendingLaunchCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc9365118712cb05d2c3efaeea73afba823d916) controls the maximum number of outstanding device runtime launches that can be made from the current device. A grid is outstanding from the point of launch up until the grid is known to have been completed. Device runtime launches which violate this limitation fail and return [cudaErrorLaunchPendingCountExceeded](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00382372902b9ffd65825d138e16125b1376) when [cudaGetLastError()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g3529f94cb530a83a76613616782bd233 "Returns the last error from a runtime call.") is called after launch. If more pending launches than the default (2048 launches) are needed for a module using the device runtime, this limit can be increased. Keep in mind that being able to sustain additional pending launches will require the runtime to reserve larger amounts of device memory upfront which can no longer be used for allocations. If these reservations fail, [cudaDeviceSetLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g05956f16eaa47ef3a4efee84563ccb7d "Set resource limits.") will return [cudaErrorMemoryAllocation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f210f50ae7f17f655e0504929606add9), and the limit can be reset to a lower value. This limit is only applicable to devices of compute capability 3.5 and higher. Attempting to set this limit on devices of compute capability less than 3.5 will result in the error [cudaErrorUnsupportedLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c3b950b6f8668f7282fae25bfcefd13a) being returned. 


  * [cudaLimitMaxL2FetchGranularity](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc9365168d3522068213e4ba46c4c99ebc4ce12) controls the L2 cache fetch granularity. Values can range from 0B to 128B. This is purely a performance hint and it can be ignored or clamped depending on the platform. 


  * [cudaLimitPersistingL2CacheSize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg4c4b34c054d383b0e9a63ab0ffc93651fc5135f91c07d7aa2d1072db9854b113) controls size in bytes available for persisting L2 cache. This is purely a performance hint and it can be ignored or clamped depending on the platform. 


Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaDeviceGetLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2 "Return resource limits."), [cuCtxSetLimit](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceSetMemPool ( int  device, [cudaMemPool_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gd7121b19b2347b777c07223c6ff4cdad) memPool ) 
    

Sets the current memory pool of a device. 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)[cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a)[cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164)

###### Description

The memory pool must be local to the specified device. Unless a mempool is specified in the [cudaMallocAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb "Allocate from a pool.") call, [cudaMallocAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb "Allocate from a pool.") allocates from the current mempool of the provided stream's device. By default, a device's current memory pool is its default memory pool. 

Note:

Use [cudaMallocFromPoolAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g871003f518e27ec92f7b331307fa32d4 "Allocates memory from a specified pool with stream ordered semantics.") to specify asynchronous allocations from a device different than the one the stream runs on. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cuDeviceSetMemPool](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0), [cudaDeviceGetMemPool](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g9001cd1b2301101e81508f4b7714c97e "Gets the current mempool for a device."), [cudaDeviceGetDefaultMemPool](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69aee77b793f13fa79c5bc0b566845b7 "Returns the default mempool of a device."), [cudaMemPoolCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g8158cc4b2c0d2c2c771f9d1af3cf386e "Creates a memory pool."), [cudaMemPoolDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g709113128c1c52c3bf170022dc7723dd "Destroys the specified memory pool."), [cudaMallocFromPoolAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1g871003f518e27ec92f7b331307fa32d4 "Allocates memory from a specified pool with stream ordered semantics.")

__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceSynchronize ( void ) 
    

Wait for compute device to finish. 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorStreamCaptureUnsupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038868e03e02390c8d77f8f60e47038b563)

###### Description

Blocks until the device has completed all preceding requested tasks. [cudaDeviceSynchronize()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d "Wait for compute device to finish.") returns an error if one of the preceding tasks has failed. If the [cudaDeviceScheduleBlockingSync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g057e6912c52708b6aa86e79dd83d007c) flag was set for this device, the host thread will block until the device has finished its work. 

Note:

  * Use of cudaDeviceSynchronize in device code was deprecated in CUDA 11.6 and removed for compute_90+ compilation. For compute capability < 9.0, compile-time opt-in by specifying -D CUDA_FORCE_CDP1_IF_SUPPORTED is required to continue using [cudaDeviceSynchronize()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d "Wait for compute device to finish.") in device code for now. Note that this is different from host-side cudaDeviceSynchronize, which is still supported. 

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaDeviceReset](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gef69dd5c6d0206c2b8d099abac61f217 "Destroy all allocations and reset all state on the current device in the current process."), [cuCtxSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaDeviceUnregisterAsyncNotification ( int  device, [cudaAsyncCallbackHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf5d1b31c73cce10a9ae82d3cc11157d2) callback ) 
    

Unregisters an async notification callback. 

######  Parameters 

`device`
    \- The device from which to remove `callback`. 
`callback`
    \- The callback instance to unregister from receiving async notifications.

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591)[cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164)[cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a)[cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)[cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3)[cudaErrorUnknown](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00382e491daacef266c7b3e3c1e140a6133c)

###### Description

Unregisters `callback` so that the corresponding callback function will stop receiving async notifications. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaDeviceRegisterAsyncNotification](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gcff9794f21aa34d3b5ccc5b6b245da4a "Registers a callback function to receive async notifications.")

__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaGetDevice ( int* device ) 
    

Returns which device is currently being used. 

######  Parameters 

`device`
    \- Returns the device on which the active host thread executes the device code.

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), cudaErrorDeviceUnavailable, 

###### Description

Returns in `*device` the current device for the calling host thread. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f "Returns the number of compute-capable devices."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0 "Returns information about the compute-device."), [cudaChooseDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf61f9ae0fe2d93b5b968756684a49460 "Select compute-device which best matches criteria."), [cuCtxGetCurrent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0)

__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaGetDeviceCount ( int* count ) 
    

Returns the number of compute-capable devices. 

######  Parameters 

`count`
    \- Returns the number of devices with compute capability greater or equal to 2.0

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591)

###### Description

Returns in `*count` the number of devices with compute capability greater or equal to 2.0 that are available for execution. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78 "Returns which device is currently being used."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0 "Returns information about the compute-device."), [cudaChooseDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf61f9ae0fe2d93b5b968756684a49460 "Select compute-device which best matches criteria."), [cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419 "Initialize device to be used for GPU executions."), [cuDeviceGetCount](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaGetDeviceFlags ( unsigned int* flags ) 
    

Gets the flags for the current device. 

######  Parameters 

`flags`
    \- Pointer to store the device flags

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a)

###### Description

Returns in `flags` the flags for the current device. If there is a current device for the calling thread, the flags for the device are returned. If there is no current device, the flags for the first device are returned, which may be the default flags. Compare to the behavior of [cudaSetDeviceFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69e73c7dda3fc05306ae7c811a690fac "Sets flags to be used for device executions."). 

Typically, the flags returned should match the behavior that will be seen if the calling thread uses a device after this call, without any change to the flags or current device inbetween by this or another thread. Note that if the device is not initialized, it is possible for another thread to change the flags for the current device before it is initialized. Additionally, when using exclusive mode, if this thread has not requested a specific device, it may use a device other than the first device, contrary to the assumption made by this function. 

If a context has been created via the driver API and is current to the calling thread, the flags for that context are always returned. 

Flags returned by this function may specifically include [cudaDeviceMapHost](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3762be9cccdd809a4ca128354fd134b0) even though it is not accepted by [cudaSetDeviceFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69e73c7dda3fc05306ae7c811a690fac "Sets flags to be used for device executions.") because it is implicit in runtime API flags. The reason for this is that the current context may have been created via the driver API in which case the flag is not implicit and may be unset. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78 "Returns which device is currently being used."), [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0 "Returns information about the compute-device."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaSetDeviceFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69e73c7dda3fc05306ae7c811a690fac "Sets flags to be used for device executions."), [cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419 "Initialize device to be used for GPU executions."), [cuCtxGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d), [cuDevicePrimaryCtxGetState](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g65f3e018721b6d90aa05cfb56250f469)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaGetDeviceProperties ( [cudaDeviceProp](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp)* prop, int  device ) 
    

Returns information about the compute-device. 

######  Parameters 

`prop`
    \- Properties for the specified device 
`device`
    \- Device number to get properties for

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a)

###### Description

Returns in `*prop` the properties of device `dev`. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f "Returns the number of compute-capable devices."), [cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78 "Returns which device is currently being used."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaChooseDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf61f9ae0fe2d93b5b968756684a49460 "Select compute-device which best matches criteria."), [cudaDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151 "Returns information about the device."), [cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419 "Initialize device to be used for GPU executions."), [cuDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266), [cuDeviceGetName](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaInitDevice ( int  device, unsigned int  deviceFlags, unsigned int  flags ) 
    

Initialize device to be used for GPU executions. 

######  Parameters 

`device`
    \- Device on which the runtime will initialize itself. 
`deviceFlags`
    \- Parameters for device operation. 
`flags`
    \- Flags for controlling the device initialization.

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a), 

###### Description

This function will initialize the CUDA Runtime structures and primary context on `device` when called, but the context will not be made current to `device`. 

When [cudaInitDeviceFlagsAreValid](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g2ff861a477ec064dcf01977194158e49) is set in `flags`, deviceFlags are applied to the requested device. The values of deviceFlags match those of the flags parameters in [cudaSetDeviceFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69e73c7dda3fc05306ae7c811a690fac "Sets flags to be used for device executions."). The effect may be verified by [cudaGetDeviceFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf830794caf068b71638c6182bba8f77a "Gets the flags for the current device."). 

This function will return an error if the device is in [cudaComputeModeExclusiveProcess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg7eb25f5413a962faad0956d92bae10d02cd032834fecbec513ea1018145b111d) and is occupied by another process or if the device is in [cudaComputeModeProhibited](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg7eb25f5413a962faad0956d92bae10d0fc71b88518e4501544d6e65b5f3671b6). 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f "Returns the number of compute-capable devices."), [cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78 "Returns which device is currently being used."), [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0 "Returns information about the compute-device."), [cudaChooseDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf61f9ae0fe2d93b5b968756684a49460 "Select compute-device which best matches criteria."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions.")[cuCtxSetCurrent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaIpcCloseMemHandle ( void* devPtr ) 
    

Attempts to close memory mapped with cudaIpcOpenMemHandle. 

######  Parameters 

`devPtr`
    \- Device pointer returned by [cudaIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9 "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.")

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorMapBufferObjectFailed](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038a2625b63c0940c54ed07f2986b12e0f1), [cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Decrements the reference count of the memory returnd by [cudaIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9 "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.") by 1. When the reference count reaches 0, this API unmaps the memory. The original allocation in the exporting process as well as imported mappings in other processes will be unaffected. 

Any resources used to enable peer access will be freed if this is the last mapping using them.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling [cudaDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151 "Returns information about the device.") with [cudaDevAttrIpcEventSupport](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cd0ab012a7c597ffbe86090ee2f9e1758c)

Note:

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356 "Allocate memory on the device."), [cudaFree](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094 "Frees memory on the device."), [cudaIpcGetEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g89a3abe1e9a11d08c665176669109784 "Gets an interprocess handle for a previously allocated event."), [cudaIpcOpenEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g9691446ab0aec1d6e528357387ed87b2 "Opens an interprocess event handle for use in the current process."), [cudaIpcGetMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539 "Gets an interprocess memory handle for an existing device memory allocation."), [cudaIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9 "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process."), [cuIpcCloseMemHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gd6f5d5bcf6376c6853b64635b0157b9e)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaIpcGetEventHandle ( [cudaIpcEventHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaIpcEventHandle__t.html#structcudaIpcEventHandle__t)* handle, [cudaEvent_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247) event ) 
    

Gets an interprocess handle for a previously allocated event. 

######  Parameters 

`handle`
    \- Pointer to a user allocated cudaIpcEventHandle in which to return the opaque event handle 
`event`
    \- Event allocated with [cudaEventInterprocess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49ec9cd742f8a3f6fde4ee72a66326f6) and [cudaEventDisableTiming](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ga5d3eff7c3623e2be533968d9cc1ee7e) flags. 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273), [cudaErrorMemoryAllocation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f210f50ae7f17f655e0504929606add9), [cudaErrorMapBufferObjectFailed](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038a2625b63c0940c54ed07f2986b12e0f1), [cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Takes as input a previously allocated event. This event must have been created with the [cudaEventInterprocess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49ec9cd742f8a3f6fde4ee72a66326f6) and [cudaEventDisableTiming](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ga5d3eff7c3623e2be533968d9cc1ee7e) flags set. This opaque handle may be copied into other processes and opened with [cudaIpcOpenEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g9691446ab0aec1d6e528357387ed87b2 "Opens an interprocess event handle for use in the current process.") to allow efficient hardware synchronization between GPU work in different processes. 

After the event has been been opened in the importing process, [cudaEventRecord](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446 "Records an event."), [cudaEventSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g949aa42b30ae9e622f6ba0787129ff22 "Waits for an event to complete."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event.") and [cudaEventQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7 "Queries an event's status.") may be used in either process. Performing operations on the imported event after the exported event has been freed with [cudaEventDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2cb6baa0830a1cd0bd957bfd8705045b "Destroys an event object.") will result in undefined behavior. 

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling [cudaDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151 "Returns information about the device.") with [cudaDevAttrIpcEventSupport](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cd0ab012a7c597ffbe86090ee2f9e1758c)

Note:

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaEventCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g4b5fdb19d7fb5f6f8862559f9279f6c3 "\[C++ API\] Creates an event object with the specified flags"), [cudaEventDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2cb6baa0830a1cd0bd957bfd8705045b "Destroys an event object."), [cudaEventSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g949aa42b30ae9e622f6ba0787129ff22 "Waits for an event to complete."), [cudaEventQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7 "Queries an event's status."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaIpcOpenEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g9691446ab0aec1d6e528357387ed87b2 "Opens an interprocess event handle for use in the current process."), [cudaIpcGetMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539 "Gets an interprocess memory handle for an existing device memory allocation."), [cudaIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9 "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process."), [cudaIpcCloseMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb3632b5d223db6acae5f8744e2c91 "Attempts to close memory mapped with cudaIpcOpenMemHandle."), [cuIpcGetEventHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gea02eadd12483de5305878b13288a86c)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaIpcGetMemHandle ( [cudaIpcMemHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaIpcMemHandle__t.html#structcudaIpcMemHandle__t)* handle, void* devPtr ) 
    

Gets an interprocess memory handle for an existing device memory allocation. 

######  Parameters 

`handle`
    \- Pointer to user allocated cudaIpcMemHandle to return the handle in. 
`devPtr`
    \- Base pointer to previously allocated device memory

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorMemoryAllocation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f210f50ae7f17f655e0504929606add9), [cudaErrorMapBufferObjectFailed](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038a2625b63c0940c54ed07f2986b12e0f1), [cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Takes a pointer to the base of an existing device memory allocation created with [cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356 "Allocate memory on the device.") and exports it for use in another process. This is a lightweight operation and may be called multiple times on an allocation without adverse effects. 

If a region of memory is freed with [cudaFree](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094 "Frees memory on the device.") and a subsequent call to [cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356 "Allocate memory on the device.") returns memory with the same device address, [cudaIpcGetMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539 "Gets an interprocess memory handle for an existing device memory allocation.") will return a unique handle for the new memory. 

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling [cudaDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151 "Returns information about the device.") with [cudaDevAttrIpcEventSupport](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cd0ab012a7c597ffbe86090ee2f9e1758c)

Note:

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356 "Allocate memory on the device."), [cudaFree](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094 "Frees memory on the device."), [cudaIpcGetEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g89a3abe1e9a11d08c665176669109784 "Gets an interprocess handle for a previously allocated event."), [cudaIpcOpenEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g9691446ab0aec1d6e528357387ed87b2 "Opens an interprocess event handle for use in the current process."), [cudaIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9 "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process."), [cudaIpcCloseMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb3632b5d223db6acae5f8744e2c91 "Attempts to close memory mapped with cudaIpcOpenMemHandle."), [cuIpcGetMemHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6f1b5be767b275f016523b2ac49ebec1)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaIpcOpenEventHandle ( [cudaEvent_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247)* event, [cudaIpcEventHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaIpcEventHandle__t.html#structcudaIpcEventHandle__t) handle ) 
    

Opens an interprocess event handle for use in the current process. 

######  Parameters 

`event`
    \- Returns the imported event 
`handle`
    \- Interprocess handle to open

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorMapBufferObjectFailed](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038a2625b63c0940c54ed07f2986b12e0f1), [cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorDeviceUninitialized](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00389e95927732f0d5a68151c72296581504)

###### Description

Opens an interprocess event handle exported from another process with [cudaIpcGetEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g89a3abe1e9a11d08c665176669109784 "Gets an interprocess handle for a previously allocated event."). This function returns a [cudaEvent_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247) that behaves like a locally created event with the [cudaEventDisableTiming](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ga5d3eff7c3623e2be533968d9cc1ee7e) flag specified. This event must be freed with [cudaEventDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2cb6baa0830a1cd0bd957bfd8705045b "Destroys an event object."). 

Performing operations on the imported event after the exported event has been freed with [cudaEventDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2cb6baa0830a1cd0bd957bfd8705045b "Destroys an event object.") will result in undefined behavior. 

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling [cudaDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151 "Returns information about the device.") with [cudaDevAttrIpcEventSupport](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cd0ab012a7c597ffbe86090ee2f9e1758c)

Note:

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaEventCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g4b5fdb19d7fb5f6f8862559f9279f6c3 "\[C++ API\] Creates an event object with the specified flags"), [cudaEventDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2cb6baa0830a1cd0bd957bfd8705045b "Destroys an event object."), [cudaEventSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g949aa42b30ae9e622f6ba0787129ff22 "Waits for an event to complete."), [cudaEventQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7 "Queries an event's status."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaIpcGetEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g89a3abe1e9a11d08c665176669109784 "Gets an interprocess handle for a previously allocated event."), [cudaIpcGetMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539 "Gets an interprocess memory handle for an existing device memory allocation."), [cudaIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9 "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process."), [cudaIpcCloseMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb3632b5d223db6acae5f8744e2c91 "Attempts to close memory mapped with cudaIpcOpenMemHandle."), [cuIpcOpenEventHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf1d525918b6c643b99ca8c8e42e36c2e)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaIpcOpenMemHandle ( void** devPtr, [cudaIpcMemHandle_t](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaIpcMemHandle__t.html#structcudaIpcMemHandle__t) handle, unsigned int  flags ) 
    

Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process. 

######  Parameters 

`devPtr`
    \- Returned device pointer 
`handle`
    \- cudaIpcMemHandle to open 
`flags`
    \- Flags for this operation. Must be specified as [cudaIpcMemLazyEnablePeerAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g60f28a5142ee7ae0336dfa83fd54e006)

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorMapBufferObjectFailed](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038a2625b63c0940c54ed07f2986b12e0f1), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273), [cudaErrorDeviceUninitialized](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00389e95927732f0d5a68151c72296581504), [cudaErrorTooManyPeers](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00388f0bc63a488221933dbf7cd67305b666), [cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Maps memory exported from another process with [cudaIpcGetMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539 "Gets an interprocess memory handle for an existing device memory allocation.") into the current device address space. For contexts on different devices [cudaIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9 "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.") can attempt to enable peer access between the devices as if the user called [cudaDeviceEnablePeerAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g2b0adabf90db37e5cfddc92cbb2589f3 "Enables direct access to memory allocations on a peer device."). This behavior is controlled by the [cudaIpcMemLazyEnablePeerAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g60f28a5142ee7ae0336dfa83fd54e006) flag. [cudaDeviceCanAccessPeer](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g4db0d04e44995d5c1c34be4ecc863f22 "Queries if a device may directly access a peer device's memory.") can determine if a mapping is possible. 

[cudaIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9 "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.") can open handles to devices that may not be visible in the process calling the API. 

Contexts that may open cudaIpcMemHandles are restricted in the following way. cudaIpcMemHandles from each device in a given process may only be opened by one context per device per other process. 

If the memory handle has already been opened by the current context, the reference count on the handle is incremented by 1 and the existing device pointer is returned. 

Memory returned from [cudaIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9 "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.") must be freed with [cudaIpcCloseMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb3632b5d223db6acae5f8744e2c91 "Attempts to close memory mapped with cudaIpcOpenMemHandle."). 

Calling [cudaFree](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094 "Frees memory on the device.") on an exported memory region before calling [cudaIpcCloseMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb3632b5d223db6acae5f8744e2c91 "Attempts to close memory mapped with cudaIpcOpenMemHandle.") in the importing context will result in undefined behavior. 

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling [cudaDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151 "Returns information about the device.") with [cudaDevAttrIpcEventSupport](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cd0ab012a7c597ffbe86090ee2f9e1758c)

Note:

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 

  * No guarantees are made about the address returned in `*devPtr`. In particular, multiple processes may not receive the same address for the same `handle`. 


**See also:**

[cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356 "Allocate memory on the device."), [cudaFree](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094 "Frees memory on the device."), [cudaIpcGetEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g89a3abe1e9a11d08c665176669109784 "Gets an interprocess handle for a previously allocated event."), [cudaIpcOpenEventHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g9691446ab0aec1d6e528357387ed87b2 "Opens an interprocess event handle for use in the current process."), [cudaIpcGetMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539 "Gets an interprocess memory handle for an existing device memory allocation."), [cudaIpcCloseMemHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb3632b5d223db6acae5f8744e2c91 "Attempts to close memory mapped with cudaIpcOpenMemHandle."), [cudaDeviceEnablePeerAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g2b0adabf90db37e5cfddc92cbb2589f3 "Enables direct access to memory allocations on a peer device."), [cudaDeviceCanAccessPeer](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g4db0d04e44995d5c1c34be4ecc863f22 "Queries if a device may directly access a peer device's memory."), [cuIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaSetDevice ( int  device ) 
    

Set device to be used for GPU executions. 

######  Parameters 

`device`
    \- Device on which the active host thread should execute the device code.

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a), cudaErrorDeviceUnavailable, 

###### Description

Sets `device` as the current device for the calling host thread. Valid device id's are 0 to ([cudaGetDeviceCount()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f "Returns the number of compute-capable devices.") \- 1). 

Any device memory subsequently allocated from this host thread using [cudaMalloc()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356 "Allocate memory on the device."), [cudaMallocPitch()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c "Allocates pitched memory on the device.") or [cudaMallocArray()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g6728eb7dc25f332f50bdb16a19620d3d "Allocate an array on the device.") will be physically resident on `device`. Any host memory allocated from this host thread using [cudaMallocHost()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gab84100ae1fa1b12eaca660207ef585b "Allocates page-locked memory on the host.") or [cudaHostAlloc()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902 "Allocates page-locked memory on the host.") or [cudaHostRegister()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8d5c17670f16ac4fc8fcb4181cb490c "Registers an existing host memory range for use by CUDA.") will have its lifetime associated with `device`. Any streams or events created from this host thread will be associated with `device`. Any kernels launched from this host thread using the <<<>>> operator or [cudaLaunchKernel()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g5064cdf5d8e6741ace56fd8be951783c "Launches a device function.") will be executed on `device`. 

This call may be made from any host thread, to any device, and at any time. This function will do no synchronization with the previous or new device, and should only take significant time when it initializes the runtime's context state. This call will bind the primary context of the specified device to the calling thread and all the subsequent memory allocations, stream and event creations, and kernel launches will be associated with the primary context. This function will also immediately initialize the runtime state on the primary context, and the context will be current on `device` immediately. This function will return an error if the device is in [cudaComputeModeExclusiveProcess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg7eb25f5413a962faad0956d92bae10d02cd032834fecbec513ea1018145b111d) and is occupied by another process or if the device is in [cudaComputeModeProhibited](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg7eb25f5413a962faad0956d92bae10d0fc71b88518e4501544d6e65b5f3671b6). 

It is not required to call [cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419 "Initialize device to be used for GPU executions.") before using this function. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f "Returns the number of compute-capable devices."), [cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78 "Returns which device is currently being used."), [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0 "Returns information about the compute-device."), [cudaChooseDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf61f9ae0fe2d93b5b968756684a49460 "Select compute-device which best matches criteria."), [cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419 "Initialize device to be used for GPU executions."), [cuCtxSetCurrent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaSetDeviceFlags ( unsigned int  flags ) 
    

Sets flags to be used for device executions. 

######  Parameters 

`flags`
    \- Parameters for device operation

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Records `flags` as the flags for the current device. If the current device has been set and that device has already been initialized, the previous flags are overwritten. If the current device has not been initialized, it is initialized with the provided flags. If no device has been made current to the calling thread, a default device is selected and initialized with the provided flags. 

The three LSBs of the `flags` parameter can be used to control how the CPU thread interacts with the OS scheduler when waiting for results from the device. 

  * [cudaDeviceScheduleAuto](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3ade1dbaf4b222b22733cdfdcc026075): The default value if the `flags` parameter is zero, uses a heuristic based on the number of active CUDA contexts in the process `C` and the number of logical processors in the system `P`. If `C` > `P`, then CUDA will yield to other OS threads when waiting for the device, otherwise CUDA will not yield while waiting for results and actively spin on the processor. Additionally, on Tegra devices, [cudaDeviceScheduleAuto](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3ade1dbaf4b222b22733cdfdcc026075) uses a heuristic based on the power profile of the platform and may choose [cudaDeviceScheduleBlockingSync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g057e6912c52708b6aa86e79dd83d007c) for low-powered devices. 

  * [cudaDeviceScheduleSpin](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf01347c3dafebf07e1a0b4321a030a63): Instruct CUDA to actively spin when waiting for results from the device. This can decrease latency when waiting for the device, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread. 

  * [cudaDeviceScheduleYield](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gbc13c52d342c67ebf0f1f7af937735a8): Instruct CUDA to yield its thread when waiting for results from the device. This can increase latency when waiting for the device, but can increase the performance of CPU threads performing work in parallel with the device. 

  * [cudaDeviceScheduleBlockingSync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g057e6912c52708b6aa86e79dd83d007c): Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the device to finish work. 

  * [cudaDeviceBlockingSync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g130ddae663f1873258fee5a6e0808b71): Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the device to finish work. 

[Deprecated:](https://docs.nvidia.com/cuda/cuda-runtime-api/deprecated.html#deprecated) This flag was deprecated as of CUDA 4.0 and replaced with [cudaDeviceScheduleBlockingSync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g057e6912c52708b6aa86e79dd83d007c). 

  * [cudaDeviceMapHost](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3762be9cccdd809a4ca128354fd134b0): This flag enables allocating pinned host memory that is accessible to the device. It is implicit for the runtime but may be absent if a context is created using the driver API. If this flag is not set, [cudaHostGetDevicePointer()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc00502b44e5f1bdc0b424487ebb08db0 "Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.") will always return a failure code. 

  * [cudaDeviceLmemResizeToMax](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gda5f97298bf704dd3b04cbac4819e6e3): Instruct CUDA to not reduce local memory after resizing local memory for a kernel. This can prevent thrashing by local memory allocations when launching many kernels with high local memory usage at the cost of potentially increased memory usage. 

[Deprecated:](https://docs.nvidia.com/cuda/cuda-runtime-api/deprecated.html#deprecated) This flag is deprecated and the behavior enabled by this flag is now the default and cannot be disabled. 

  * [cudaDeviceSyncMemops](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gb5cbd9be7931e6c68d11e1175aa14c23): Ensures that synchronous memory operations initiated on this context will always synchronize. See further documentation in the section titled "API Synchronization behavior" to learn more about cases when synchronous memory operations can exhibit asynchronous behavior. 


Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDeviceFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf830794caf068b71638c6182bba8f77a "Gets the flags for the current device."), [cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f "Returns the number of compute-capable devices."), [cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78 "Returns which device is currently being used."), [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0 "Returns information about the compute-device."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaSetValidDevices](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1b9336c70f2299405f67a4f8496d7cfe "Set a list of devices that can be used for CUDA."), [cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419 "Initialize device to be used for GPU executions."), [cudaChooseDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf61f9ae0fe2d93b5b968756684a49460 "Select compute-device which best matches criteria."), [cuDevicePrimaryCtxSetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaSetValidDevices ( int* device_arr, int  len ) 
    

Set a list of devices that can be used for CUDA. 

######  Parameters 

`device_arr`
    \- List of devices to try 
`len`
    \- Number of devices in specified list

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a)

###### Description

Sets a list of devices for CUDA execution in priority order using `device_arr`. The parameter `len` specifies the number of elements in the list. CUDA will try devices from the list sequentially until it finds one that works. If this function is not called, or if it is called with a `len` of 0, then CUDA will go back to its default behavior of trying devices sequentially from a default list containing all of the available CUDA devices in the system. If a specified device ID in the list does not exist, this function will return [cudaErrorInvalidDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038938c6e8b96ecde62e3ab5137156f739a). If `len` is not 0 and `device_arr` is NULL or if `len` exceeds the number of devices in the system, then [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c) is returned. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f "Returns the number of compute-capable devices."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0 "Returns information about the compute-device."), [cudaSetDeviceFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69e73c7dda3fc05306ae7c811a690fac "Sets flags to be used for device executions."), [cudaChooseDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf61f9ae0fe2d93b5b968756684a49460 "Select compute-device which best matches criteria.")

* * *

![](https://docs.nvidia.com/cuda/common/formatting/NVIDIA-LogoBlack.svg)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Manage My Privacy](https://www.nvidia.com/en-us/privacy-center/) | [Do Not Sell or Share My Data](https://www.nvidia.com/en-us/preferences/email-preferences/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2026 NVIDIA Corporation

![](https://docs.nvidia.com/akam/13/pixel_6b113425?a=dD1jM2VhNTQ4NDliMmJjYjU4NWIwZmJkMWJkZmUzZmExZTBhYjdlNDkyJmpzPW9mZg==)