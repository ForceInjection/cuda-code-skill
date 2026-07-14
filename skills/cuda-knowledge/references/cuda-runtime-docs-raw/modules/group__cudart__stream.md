# Stream Management

**Source:** [group__CUDART__STREAM.html](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)

---

### Classes

struct 

[cudaGraphRecaptureCallbackData](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphRecaptureCallbackData.html#structcudaGraphRecaptureCallbackData)

     [](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphRecaptureCallbackData.html#structcudaGraphRecaptureCallbackData)

### Typedefs

typedef [cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6)* ( *[cudaGraphRecaptureCallback_t](#group__CUDART__STREAM_1g97022470645289577f928e9e485975af) )( void*  data,  cudaGraphNode_t node, const cudaGraphNodeParams*  originalParams, const cudaGraphNodeParams*  recaptureParams,  enum cudaGraphRecaptureStatus status ) 
    
typedef void(CUDART_CB* [cudaStreamCallback_t](#group__CUDART__STREAM_1g11c9452045db759adb77a40d7c98f648) )( cudaStream_t stream,  cudaError_t status, void*  userData ) 
    

### Functions

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaCtxResetPersistingL2Cache](#group__CUDART__STREAM_1g37ef93f921871331188f90fb2eb20e5e) ( void ) 
     Resets all persisting lines in cache to normal status. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamAddCallback](#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCallback_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g11c9452045db759adb77a40d7c98f648) callback, void* userData, unsigned int  flags ) 
     Add a callback to a compute stream. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamAttachMemAsync](#group__CUDART__STREAM_1gc3bb7ccb325219073183a629d7c2756a) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, void* devPtr, size_t length = 0, unsigned int  flags = cudaMemAttachSingle ) 
     Attach memory to a stream asynchronously. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamBeginCapture](#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g24ae5ae00cc50614957ff8eba43e560e) mode ) 
     Begins graph capture on a stream. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamBeginCaptureToGraph](#group__CUDART__STREAM_1g52d4a730019358ef25b721f959543f23) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaGraph_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5707132c494c91db57db5a6da0beba4c) graph, const [cudaGraphNode_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eed9297e04a8e4b5100200d5e69c124)* dependencies, const [cudaGraphEdgeData](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphEdgeData.html#structcudaGraphEdgeData)* dependencyData, size_t numDependencies, [cudaStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g24ae5ae00cc50614957ff8eba43e560e) mode ) 
     Begins graph capture on a stream to an existing graph. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamBeginRecaptureToGraph](#group__CUDART__STREAM_1g980baa726cb9a77b21ed8f58a1e75b97) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g24ae5ae00cc50614957ff8eba43e560e) mode, [cudaGraph_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5707132c494c91db57db5a6da0beba4c) graph, [cudaGraphRecaptureCallbackData](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphRecaptureCallbackData.html#structcudaGraphRecaptureCallbackData)* callbackData = 0 ) 
     Begin graph capture on a stream to an existing graph. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamCopyAttributes](#group__CUDART__STREAM_1g3bc9fe4af9b3eef5ad453c92e237da1c) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) dst, [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) src ) 
     Copies attributes from source stream to destination stream. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamCreate](#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a)* pStream ) 
     Create an asynchronous stream. 
__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamCreateWithFlags](#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a)* pStream, unsigned int  flags ) 
     Create an asynchronous stream. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamCreateWithPriority](#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a)* pStream, unsigned int  flags, int  priority ) 
     Create an asynchronous stream with the specified priority. 
__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamDestroy](#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream ) 
     Destroys and cleans up an asynchronous stream. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamEndCapture](#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaGraph_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5707132c494c91db57db5a6da0beba4c)* pGraph ) 
     Ends capture on a stream, returning the captured graph. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamGetAttribute](#group__CUDART__STREAM_1g0842be3b57a279a9c20ddcfb7c5419b9) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out ) 
     Queries stream attribute. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamGetCaptureInfo](#group__CUDART__STREAM_1g8d9312f1098c45e2ed43c949cfccf1f7) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCaptureStatus *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g0ec4aacc66fe76c145252d12b189e233)* captureStatus_out, unsigned long long* id_out = 0, [cudaGraph_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5707132c494c91db57db5a6da0beba4c)* graph_out = 0, const [cudaGraphNode_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eed9297e04a8e4b5100200d5e69c124)** dependencies_out = 0, const [cudaGraphEdgeData](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphEdgeData.html#structcudaGraphEdgeData)** edgeData_out = 0, size_t* numDependencies_out = 0 ) 
     Query a stream's capture state. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamGetDevice](#group__CUDART__STREAM_1g4eeb32402810fb7b1d3b1d0cff34aede) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, int* device ) 
     Query the device of a stream. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamGetFlags](#group__CUDART__STREAM_1ga7f311f88126d751b9a7d3302ad6d0f8) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, unsigned int* flags ) 
     Query the flags of a stream. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamGetId](#group__CUDART__STREAM_1g5799ae8dd744e561dfdeda02c53e82df) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, unsigned long long* streamId ) 
     Query the Id of a stream. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamGetPriority](#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, int* priority ) 
     Query the priority of a stream. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamIsCapturing](#group__CUDART__STREAM_1ge564e8434c67d716367931c4bc7db1cc) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCaptureStatus *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g0ec4aacc66fe76c145252d12b189e233)* pCaptureStatus ) 
     Returns a stream's capture status. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamQuery](#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream ) 
     Queries an asynchronous stream for completion status. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamSetAttribute](#group__CUDART__STREAM_1g0d4f304ced0c3d4786c77d313bebef80) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value ) 
     Sets stream attribute. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamSynchronize](#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream ) 
     Waits for stream tasks to complete. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamUpdateCaptureDependencies](#group__CUDART__STREAM_1g5d24e83040683a297f2d160bedf25175) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaGraphNode_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eed9297e04a8e4b5100200d5e69c124)* dependencies, const [cudaGraphEdgeData](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphEdgeData.html#structcudaGraphEdgeData)* dependencyData, size_t numDependencies, unsigned int  flags = 0 ) 
     Update the set of dependencies in a capturing stream. 
__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaStreamWaitEvent](#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9) ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaEvent_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247) event, unsigned int  flags = 0 ) 
     Make a compute stream wait on an event. 
__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaThreadExchangeStreamCaptureMode](#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85) ( [cudaStreamCaptureMode *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g24ae5ae00cc50614957ff8eba43e560e)* mode ) 
     Swaps the stream capture interaction mode for a thread. 

### Typedefs

cudaError_t* ( *cudaGraphRecaptureCallback_t )( void*  data,  cudaGraphNode_t node, const cudaGraphNodeParams*  originalParams, const cudaGraphNodeParams*  recaptureParams,  enum cudaGraphRecaptureStatus status ) 
    

Callback function invoked when node parameter mismatches are detected while recapturing to an existing graph. Parameter struct pointers are only valid within the callback. 

######  Parameters 

`data`
    User parameter provided at beginning of recapture 
`cudaGraphNode_t node`
    
`originalParams`
    The original node parameters from the graph 
`recaptureParams`
    The node parameters received during the recapture 
`enum cudaGraphRecaptureStatus status`
    

###### Returns

Error code for the callback. Anything other than cudaSuccess will cause the recapture to fail immediately. 

void(CUDART_CB* cudaStreamCallback_t )( cudaStream_t stream,  cudaError_t status, void*  userData ) 
    

Type of stream callback functions. 

######  Parameters 

`stream`
    The stream as passed to [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream."), may be NULL. 
`cudaError_t status`
    
`userData`
    User parameter provided at registration. 

### Functions

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaCtxResetPersistingL2Cache ( void ) 
    

Resets all persisting lines in cache to normal status. 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), 

###### Description

Resets all persisting lines in cache to normal status. Takes effect on function return.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaAccessPolicyWindow](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaAccessPolicyWindow.html#structcudaAccessPolicyWindow)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamAddCallback ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCallback_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g11c9452045db759adb77a40d7c98f648) callback, void* userData, unsigned int  flags ) 
    

Add a callback to a compute stream. 

######  Parameters 

`stream`
    \- Stream to add callback to 
`callback`
    \- The function to call once preceding stream operations are complete 
`userData`
    \- User specified data to be passed to the callback function 
`flags`
    \- Reserved for future use, must be 0

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164)

###### Description

Note:

This function is slated for eventual deprecation and removal. If you do not require the callback to execute in case of a device error, consider using [cudaLaunchHostFunc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f "Enqueues a host function call in a stream."). Additionally, this function is not supported with [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream.") and [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph."), unlike [cudaLaunchHostFunc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f "Enqueues a host function call in a stream."). 

Adds a callback to be called on the host after all currently enqueued items in the stream have completed. For each cudaStreamAddCallback call, a callback will be executed exactly once. The callback will block later work in the stream until it is finished. 

The callback may be passed [cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591) or an error code. In the event of a device error, all subsequently executed callbacks will receive an appropriate [cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6). 

Callbacks must not make any CUDA API calls. Attempting to use CUDA APIs may result in [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3). Callbacks must not perform any synchronization that may depend on outstanding device work or other callbacks that are not mandated to run earlier. Callbacks without a mandated order (in independent streams) execute in undefined order and may be serialized. 

For the purposes of Unified Memory, callback execution makes a number of guarantees: 

  * The callback stream is considered idle for the duration of the callback. Thus, for example, a callback may always use memory attached to the callback stream. 

  * The start of execution of a callback has the same effect as synchronizing an event recorded in the same stream immediately prior to the callback. It thus synchronizes streams which have been "joined" prior to the callback. 

  * Adding device work to any stream does not have the effect of making the stream active until all preceding callbacks have executed. Thus, for example, a callback might use global attached memory even if work has been added to another stream, if it has been properly ordered with an event. 

  * Completion of a callback does not cause a stream to become active except as described above. The callback stream will remain idle if no device work follows the callback, and will remain idle across consecutive callbacks without device work in between. Thus, for example, stream synchronization can be done by signaling from a callback at the end of the stream. 


Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435 "Queries an asynchronous stream for completion status."), [cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e "Waits for stream tasks to complete."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaStreamDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62 "Destroys and cleans up an asynchronous stream."), [cudaMallocManaged](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gcf6b9b1019e73c5bc2b39b39fe90816e "Allocates memory that will be automatically managed by the Unified Memory system."), [cudaStreamAttachMemAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g496353d630c29c44a2e33f531a3944d1 "Attach memory to a stream asynchronously."), [cudaLaunchHostFunc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f "Enqueues a host function call in a stream."), [cuStreamAddCallback](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamAttachMemAsync ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, void* devPtr, size_t length = 0, unsigned int  flags = cudaMemAttachSingle ) 
    

Attach memory to a stream asynchronously. 

######  Parameters 

`stream`
    \- Stream in which to enqueue the attach operation 
`devPtr`
    \- Pointer to memory (must be a pointer to managed memory or to a valid host-accessible region of system-allocated memory) 
`length`
    \- Length of memory (defaults to zero) 
`flags`
    \- Must be one of [cudaMemAttachGlobal](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4808e47eba73eb94622ec70a9f9b91ff), [cudaMemAttachHost](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4f9a428d18fdd89a99441d0dd27131c0) or [cudaMemAttachSingle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gdc191442d08fc3a9de4cf055edfd2dbe) (defaults to [cudaMemAttachSingle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gdc191442d08fc3a9de4cf055edfd2dbe)) 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorNotReady](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038285d0c852ab65b8925505e1065563f6d), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)

###### Description

Enqueues an operation in `stream` to specify stream association of `length` bytes of memory starting from `devPtr`. This function is a stream-ordered operation, meaning that it is dependent on, and will only take effect when, previous work in stream has completed. Any previous association is automatically replaced. 

`devPtr` must point to an one of the following types of memories: 

  * managed memory declared using the __managed__ keyword or allocated with [cudaMallocManaged](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gcf6b9b1019e73c5bc2b39b39fe90816e "Allocates memory that will be automatically managed by the Unified Memory system."). 

  * a valid host-accessible region of system-allocated pageable memory. This type of memory may only be specified if the device associated with the stream reports a non-zero value for the device attribute [cudaDevAttrPageableMemoryAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cddc80992427a92713e699953a6d249d6f). 


For managed allocations, `length` must be either zero or the entire allocation's size. Both indicate that the entire allocation's stream association is being changed. Currently, it is not possible to change stream association for a portion of a managed allocation. 

For pageable allocations, `length` must be non-zero. 

The stream association is specified using `flags` which must be one of [cudaMemAttachGlobal](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4808e47eba73eb94622ec70a9f9b91ff), [cudaMemAttachHost](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4f9a428d18fdd89a99441d0dd27131c0) or [cudaMemAttachSingle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gdc191442d08fc3a9de4cf055edfd2dbe). The default value for `flags` is [cudaMemAttachSingle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gdc191442d08fc3a9de4cf055edfd2dbe) If the [cudaMemAttachGlobal](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4808e47eba73eb94622ec70a9f9b91ff) flag is specified, the memory can be accessed by any stream on any device. If the [cudaMemAttachHost](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4f9a428d18fdd89a99441d0dd27131c0) flag is specified, the program makes a guarantee that it won't access the memory on the device from any stream on a device that has a zero value for the device attribute [cudaDevAttrConcurrentManagedAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cdc88178f29891f2c18fe67361cc80de09). If the [cudaMemAttachSingle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gdc191442d08fc3a9de4cf055edfd2dbe) flag is specified and `stream` is associated with a device that has a zero value for the device attribute [cudaDevAttrConcurrentManagedAccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cdc88178f29891f2c18fe67361cc80de09), the program makes a guarantee that it will only access the memory on the device from `stream`. It is illegal to attach singly to the NULL stream, because the NULL stream is a virtual global stream and not a specific stream. An error will be returned in this case. 

When memory is associated with a single stream, the Unified Memory system will allow CPU access to this memory region so long as all operations in `stream` have completed, regardless of whether other streams are active. In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream activity instead of whole-GPU activity. 

Accessing memory on the device from streams that are not associated with it will produce undefined results. No error checking is performed by the Unified Memory system to ensure that kernels launched into other streams do not access this region. 

It is a program's responsibility to order calls to [cudaStreamAttachMemAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g496353d630c29c44a2e33f531a3944d1 "Attach memory to a stream asynchronously.") via events, synchronization or other means to ensure legal access to memory at all times. Data visibility and coherency will be changed appropriately for all kernels which follow a stream-association change. 

If `stream` is destroyed while data is associated with it, the association is removed and the association reverts to the default visibility of the allocation as specified at [cudaMallocManaged](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gcf6b9b1019e73c5bc2b39b39fe90816e "Allocates memory that will be automatically managed by the Unified Memory system."). For __managed__ variables, the default association is always [cudaMemAttachGlobal](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4808e47eba73eb94622ec70a9f9b91ff). Note that destroying a stream is an asynchronous operation, and as a result, the change to default association won't happen until all work in the stream has completed. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e "Waits for stream tasks to complete."), [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream."), [cudaStreamDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62 "Destroys and cleans up an asynchronous stream."), [cudaMallocManaged](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gcf6b9b1019e73c5bc2b39b39fe90816e "Allocates memory that will be automatically managed by the Unified Memory system."), [cuStreamAttachMemAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamBeginCapture ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g24ae5ae00cc50614957ff8eba43e560e) mode ) 
    

Begins graph capture on a stream. 

######  Parameters 

`stream`
    \- Stream in which to initiate capture 
`mode`
    \- Controls the interaction of this capture sequence with other API calls that are potentially unsafe. For more details see [cudaThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85 "Swaps the stream capture interaction mode for a thread."). 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Begin graph capture on `stream`. When a stream is in capture mode, all operations pushed into the stream will not be executed, but will instead be captured into a graph, which will be returned via [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph."). Capture may not be initiated if `stream` is [cudaStreamLegacy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246). Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in capture mode. The capture mode may be queried via [cudaStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge564e8434c67d716367931c4bc7db1cc "Returns a stream's capture status."). A unique id representing the capture sequence may be queried via [cudaStreamGetCaptureInfo](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g8d9312f1098c45e2ed43c949cfccf1f7 "Query a stream's capture state."). 

If `mode` is not cudaStreamCaptureModeRelaxed, [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph.") must be called on this stream from the same thread. 

Note:

Kernels captured using this API must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge564e8434c67d716367931c4bc7db1cc "Returns a stream's capture status."), [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph."), [cudaThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85 "Swaps the stream capture interaction mode for a thread.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamBeginCaptureToGraph ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaGraph_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5707132c494c91db57db5a6da0beba4c) graph, const [cudaGraphNode_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eed9297e04a8e4b5100200d5e69c124)* dependencies, const [cudaGraphEdgeData](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphEdgeData.html#structcudaGraphEdgeData)* dependencyData, size_t numDependencies, [cudaStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g24ae5ae00cc50614957ff8eba43e560e) mode ) 
    

Begins graph capture on a stream to an existing graph. 

######  Parameters 

`stream`
    \- Stream in which to initiate capture. 
`graph`
    \- Graph to capture into. 
`dependencies`
    \- Dependencies of the first node captured in the stream. Can be NULL if numDependencies is 0. 
`dependencyData`
    \- Optional array of data associated with each dependency. 
`numDependencies`
    \- Number of dependencies. 
`mode`
    \- Controls the interaction of this capture sequence with other API calls that are potentially unsafe. For more details see [cudaThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85 "Swaps the stream capture interaction mode for a thread."). 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Begin graph capture on `stream`. When a stream is in capture mode, all operations pushed into the stream will not be executed, but will instead be captured into `graph`, which will be returned via [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph."). 

Capture may not be initiated if `stream` is [cudaStreamLegacy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246). Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in capture mode. The capture mode may be queried via [cudaStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge564e8434c67d716367931c4bc7db1cc "Returns a stream's capture status."). A unique id representing the capture sequence may be queried via [cudaStreamGetCaptureInfo](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g8d9312f1098c45e2ed43c949cfccf1f7 "Query a stream's capture state."). 

If `mode` is not cudaStreamCaptureModeRelaxed, [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph.") must be called on this stream from the same thread. 

Note:

Kernels captured using this API must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge564e8434c67d716367931c4bc7db1cc "Returns a stream's capture status."), [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph."), [cudaThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85 "Swaps the stream capture interaction mode for a thread.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamBeginRecaptureToGraph ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g24ae5ae00cc50614957ff8eba43e560e) mode, [cudaGraph_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5707132c494c91db57db5a6da0beba4c) graph, [cudaGraphRecaptureCallbackData](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphRecaptureCallbackData.html#structcudaGraphRecaptureCallbackData)* callbackData = 0 ) 
    

Begin graph capture on a stream to an existing graph. 

######  Parameters 

`stream`
    \- Stream in which to initiate capture 
`mode`
    \- Controls the interaction of this capture sequence with other API calls that are potentially unsafe. For more details see [cudaThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85 "Swaps the stream capture interaction mode for a thread."). 
`graph`
    \- Existing CUDA graph to be captured into 
`callbackData`
    \- Optional struct of callback data that will be invoked for all parameter mismatches from the original graph

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), cudaErrorDeinitialized, cudaErrorNotInitialized, [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), 

###### Description

Begin graph capture on `stream` to the existing `graph`. The node creation order while recapturing the graph must be identical to the original graph. The recapture will fail immediately for: * Topology mismatches between the existing graph and the recaptured graph * Parameter mismatches for memory allocation or free nodes 

Any other node parameter mismatches during recapture can be configured to call the function provided in `callbackFunc`. The recapture will fail immediately if the callback returns anything other than cudaSuccess. 

If the recapture fails for any reason, the `graph` will be in an undefined state and should be destroyed. 

See cudaStreamBeginCapture for additional detail on beginning the capture.

Note:

Any user objects associated with `graph` will be released prior to the recapture. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream."), [cudaStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge564e8434c67d716367931c4bc7db1cc "Returns a stream's capture status."), [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph."), [cudaThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85 "Swaps the stream capture interaction mode for a thread.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamCopyAttributes ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) dst, [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) src ) 
    

Copies attributes from source stream to destination stream. 

######  Parameters 

`dst`
    Destination stream 
`src`
    Source stream For attributes see cudaStreamAttrID

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorNotSupported](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d846fd9f2e8ba5e2fb4f1695b7ab6164)

###### Description

Copies attributes from source stream `src` to destination stream `dst`. Both streams must have the same context. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaAccessPolicyWindow](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaAccessPolicyWindow.html#structcudaAccessPolicyWindow)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamCreate ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a)* pStream ) 
    

Create an asynchronous stream. 

######  Parameters 

`pStream`
    \- Pointer to new stream identifier

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)[cudaErrorExternalDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383ac6fbff2f7876645240e789d126d2bf)

###### Description

Creates a new asynchronous stream on the context that is current to the calling host thread. If no context is current to the calling host thread, then the primary context for a device is selected, made current to the calling thread, and initialized before creating a stream on it. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540 "Create an asynchronous stream with the specified priority."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6 "Query the priority of a stream."), [cudaStreamGetFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ga7f311f88126d751b9a7d3302ad6d0f8 "Query the flags of a stream."), [cudaStreamGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g4eeb32402810fb7b1d3b1d0cff34aede "Query the device of a stream."), [cudaStreamGetDevResource](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html#group__CUDART__EXECUTION__CONTEXT_1g55c60bf05fec3cf837d96520c91b8396 "Get stream resources."), [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435 "Queries an asynchronous stream for completion status."), [cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e "Waits for stream tasks to complete."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaStreamDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62 "Destroys and cleans up an asynchronous stream."), [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4)

__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamCreateWithFlags ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a)* pStream, unsigned int  flags ) 
    

Create an asynchronous stream. 

######  Parameters 

`pStream`
    \- Pointer to new stream identifier 
`flags`
    \- Parameters for stream creation

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)[cudaErrorExternalDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383ac6fbff2f7876645240e789d126d2bf)

###### Description

Creates a new asynchronous stream on the context that is current to the calling host thread. If no context is current to the calling host thread, then the primary context for a device is selected, made current to the calling thread, and initialized before creating a stream on it. The `flags` argument determines the behaviors of the stream. Valid values for `flags` are 

  * [cudaStreamDefault](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ged347a89ec289c787faa116b851708fc): Default stream creation flag. 

  * [cudaStreamNonBlocking](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5dbd11a1723d9f5938a133cedbc525e3): Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream), and that the created stream should perform no implicit synchronization with stream 0. 


Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540 "Create an asynchronous stream with the specified priority."), [cudaStreamGetFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ga7f311f88126d751b9a7d3302ad6d0f8 "Query the flags of a stream."), [cudaStreamGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g4eeb32402810fb7b1d3b1d0cff34aede "Query the device of a stream."), [cudaStreamGetDevResource](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html#group__CUDART__EXECUTION__CONTEXT_1g55c60bf05fec3cf837d96520c91b8396 "Get stream resources."), [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435 "Queries an asynchronous stream for completion status."), [cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e "Waits for stream tasks to complete."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaStreamDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62 "Destroys and cleans up an asynchronous stream."), [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamCreateWithPriority ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a)* pStream, unsigned int  flags, int  priority ) 
    

Create an asynchronous stream with the specified priority. 

######  Parameters 

`pStream`
    \- Pointer to new stream identifier 
`flags`
    \- Flags for stream creation. See [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream.") for a list of valid flags that can be passed 
`priority`
    \- Priority of the stream. Lower numbers represent higher priorities. See [cudaDeviceGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gfdb79818f7c0ee7bc585648c91770275 "Returns numerical values that correspond to the least and greatest stream priorities.") for more information about the meaningful stream priorities that can be passed. 

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)[cudaErrorExternalDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383ac6fbff2f7876645240e789d126d2bf)

###### Description

Creates a stream with the specified priority and returns a handle in `pStream`. The stream is created on the context that is current to the calling host thread. If no context is current to the calling host thread, then the primary context for a device is selected, made current to the calling thread, and initialized before creating a stream on it. This affects the scheduling priority of work in the stream. Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order. 

`priority` follows a convention where lower numbers represent higher priorities. '0' represents default priority. The range of meaningful numerical priorities can be queried using [cudaDeviceGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gfdb79818f7c0ee7bc585648c91770275 "Returns numerical values that correspond to the least and greatest stream priorities."). If the specified priority is outside the numerical range returned by [cudaDeviceGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gfdb79818f7c0ee7bc585648c91770275 "Returns numerical values that correspond to the least and greatest stream priorities."), it will automatically be clamped to the lowest or the highest number in the range. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 

  * Stream priorities are supported only on GPUs with compute capability 3.5 or higher.

  * In the current implementation, only compute kernels launched in priority streams are affected by the stream's priority. Stream priorities have no effect on host-to-device and device-to-host memory operations. 


**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaDeviceGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gfdb79818f7c0ee7bc585648c91770275 "Returns numerical values that correspond to the least and greatest stream priorities."), [cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6 "Query the priority of a stream."), [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435 "Queries an asynchronous stream for completion status."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream."), [cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e "Waits for stream tasks to complete."), [cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaStreamDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62 "Destroys and cleans up an asynchronous stream."), [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471)

__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamDestroy ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream ) 
    

Destroys and cleans up an asynchronous stream. 

######  Parameters 

`stream`
    \- Stream identifier

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)[cudaErrorExternalDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383ac6fbff2f7876645240e789d126d2bf)

###### Description

Destroys and cleans up the asynchronous stream specified by `stream`. 

In case the device is still doing work in the stream `stream` when [cudaStreamDestroy()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62 "Destroys and cleans up an asynchronous stream.") is called, the function will return immediately and the resources associated with `stream` will be released automatically once the device has completed all work in `stream`. 

Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 

  * Use of the handle after this call is undefined behavior.


**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435 "Queries an asynchronous stream for completion status."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e "Waits for stream tasks to complete."), [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream."), [cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamEndCapture ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaGraph_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5707132c494c91db57db5a6da0beba4c)* pGraph ) 
    

Ends capture on a stream, returning the captured graph. 

######  Parameters 

`stream`
    \- Stream to query 
`pGraph`
    \- The captured graph

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorStreamCaptureWrongThread](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00385627c9161ac543c5b473c64e2e6a6eb4)

###### Description

End capture on `stream`, returning the captured graph via `pGraph`. Capture must have been initiated on `stream` via a call to [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream."). If capture was invalidated, due to a violation of the rules of stream capture, then a NULL graph will be returned. 

If the `mode` argument to [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream.") was not cudaStreamCaptureModeRelaxed, this call must be from the same thread as [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream."). 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream."), [cudaStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge564e8434c67d716367931c4bc7db1cc "Returns a stream's capture status."), [cudaGraphDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga351557d4d9ecab23d56395599b0e069 "Destroys a graph.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamGetAttribute ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out ) 
    

Queries stream attribute. 

######  Parameters 

`hStream`
    
`attr`
    
`value_out`
    

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)

###### Description

Queries attribute `attr` from `hStream` and stores it in corresponding member of `value_out`. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaAccessPolicyWindow](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaAccessPolicyWindow.html#structcudaAccessPolicyWindow)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamGetCaptureInfo ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCaptureStatus *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g0ec4aacc66fe76c145252d12b189e233)* captureStatus_out, unsigned long long* id_out = 0, [cudaGraph_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5707132c494c91db57db5a6da0beba4c)* graph_out = 0, const [cudaGraphNode_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eed9297e04a8e4b5100200d5e69c124)** dependencies_out = 0, const [cudaGraphEdgeData](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphEdgeData.html#structcudaGraphEdgeData)** edgeData_out = 0, size_t* numDependencies_out = 0 ) 
    

Query a stream's capture state. 

######  Parameters 

`stream`
    \- The stream to query 
`captureStatus_out`
    \- Location to return the capture status of the stream; required 
`id_out`
    \- Optional location to return an id for the capture sequence, which is unique over the lifetime of the process 
`graph_out`
    \- Optional location to return the graph being captured into. All operations other than destroy and node removal are permitted on the graph while the capture sequence is in progress. This API does not transfer ownership of the graph, which is transferred or destroyed at [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph."). Note that the graph handle may be invalidated before end of capture for certain errors. Nodes that are or become unreachable from the original stream at [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph.") due to direct actions on the graph do not trigger [cudaErrorStreamCaptureUnjoined](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce9e4b4b58c1abf9faa1ce8b1079076d). 
`dependencies_out`
    \- Optional location to store a pointer to an array of nodes. The next node to be captured in the stream will depend on this set of nodes, absent operations such as event wait which modify this set. The array pointer is valid until the next API call which operates on the stream or until the capture is terminated. The node handles may be copied out and are valid until they or the graph is destroyed. The driver-owned array may also be passed directly to APIs that operate on the graph (not the stream) without copying. 
`edgeData_out`
    \- Optional location to store a pointer to an array of graph edge data. This array parallels `dependencies_out`; the next node to be added has an edge to `dependencies_out`[i] with annotation `edgeData_out`[i] for each `i`. The array pointer is valid until the next API call which operates on the stream or until the capture is terminated. 
`numDependencies_out`
    \- Optional location to store the size of the array returned in dependencies_out.

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorStreamCaptureImplicit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038cf775033deb76dbde57b8df7bd9244e7), [cudaErrorLossyQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d484d9d8e8f269cf93d4c111b646f908)

###### Description

Query stream state related to stream capture.

If called on [cudaStreamLegacy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246) (the "null stream") while a stream not created with [cudaStreamNonBlocking](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g5dbd11a1723d9f5938a133cedbc525e3) is capturing, returns [cudaErrorStreamCaptureImplicit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038cf775033deb76dbde57b8df7bd9244e7). 

Valid data (other than capture status) is returned only if both of the following are true: 

  * the call returns cudaSuccess

  * the returned capture status is [cudaStreamCaptureStatusActive](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg0ec4aacc66fe76c145252d12b189e233b165b0917377d0d7283ca5ac8013568b)


If `edgeData_out` is non-NULL then `dependencies_out` must be as well. If `dependencies_out` is non-NULL and `edgeData_out` is NULL, but there is non-zero edge data for one or more of the current stream dependencies, the call will return [cudaErrorLossyQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038d484d9d8e8f269cf93d4c111b646f908). 

Note:

  * Graph objects are not threadsafe. [More here](https://docs.nvidia.com/cuda/cuda-runtime-api/graphs-thread-safety.html#graphs-thread-safety). 

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream."), [cudaStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge564e8434c67d716367931c4bc7db1cc "Returns a stream's capture status."), [cudaStreamUpdateCaptureDependencies](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g5d24e83040683a297f2d160bedf25175 "Update the set of dependencies in a capturing stream.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamGetDevice ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, int* device ) 
    

Query the device of a stream. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`device`
    \- Returns the device to which the stream belongs

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), cudaErrorDeviceUnavailable, 

###### Description

Returns in `*device` the device of the stream. 

Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb "Set device to be used for GPU executions."), [cudaGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78 "Returns which device is currently being used."), [cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6 "Query the priority of a stream."), [cudaStreamGetFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ga7f311f88126d751b9a7d3302ad6d0f8 "Query the flags of a stream."), [cuStreamGetId](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5dafd2b6f48caeb13d5110a7f21e60e3)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamGetFlags ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, unsigned int* flags ) 
    

Query the flags of a stream. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`flags`
    \- Pointer to an unsigned integer in which the stream's flags are returned

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)

###### Description

Query the flags of a stream. The flags are returned in `flags`. See [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream.") for a list of valid flags. 

Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540 "Create an asynchronous stream with the specified priority."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6 "Query the priority of a stream."), [cudaStreamGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g4eeb32402810fb7b1d3b1d0cff34aede "Query the device of a stream."), [cuStreamGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamGetId ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, unsigned long long* streamId ) 
    

Query the Id of a stream. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`streamId`
    \- Pointer to an unsigned long long in which the stream Id is returned

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)

###### Description

Query the Id of a stream. The Id is returned in `streamId`. The Id is unique for the life of the program. 

The stream handle `hStream` can refer to any of the following: 

  * a stream created via any of the CUDA runtime APIs such as [cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream.") and [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540 "Create an asynchronous stream with the specified priority."), or their driver API equivalents such as [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4) or [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471). Passing an invalid handle will result in undefined behavior. 

  * any of the special streams such as the NULL stream, [cudaStreamLegacy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246) and [cudaStreamPerThread](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197) respectively. The driver API equivalents of these are also accepted which are NULL, [CU_STREAM_LEGACY](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa) and [CU_STREAM_PER_THREAD](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c). 


Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540 "Create an asynchronous stream with the specified priority."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6 "Query the priority of a stream."), [cudaStreamGetFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ga7f311f88126d751b9a7d3302ad6d0f8 "Query the flags of a stream."), [cuStreamGetId](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5dafd2b6f48caeb13d5110a7f21e60e3)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamGetPriority ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, int* priority ) 
    

Query the priority of a stream. 

######  Parameters 

`hStream`
    \- Handle to the stream to be queried 
`priority`
    \- Pointer to a signed integer in which the stream's priority is returned

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)

###### Description

Query the priority of a stream. The priority is returned in in `priority`. Note that if the stream was created with a priority outside the meaningful numerical range returned by [cudaDeviceGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gfdb79818f7c0ee7bc585648c91770275 "Returns numerical values that correspond to the least and greatest stream priorities."), this function returns the clamped priority. See [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540 "Create an asynchronous stream with the specified priority.") for details about priority clamping. 

Note:

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540 "Create an asynchronous stream with the specified priority."), [cudaDeviceGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gfdb79818f7c0ee7bc585648c91770275 "Returns numerical values that correspond to the least and greatest stream priorities."), [cudaStreamGetFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ga7f311f88126d751b9a7d3302ad6d0f8 "Query the flags of a stream."), [cudaStreamGetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g4eeb32402810fb7b1d3b1d0cff34aede "Query the device of a stream."), [cudaStreamGetDevResource](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html#group__CUDART__EXECUTION__CONTEXT_1g55c60bf05fec3cf837d96520c91b8396 "Get stream resources."), [cuStreamGetPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamIsCapturing ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaStreamCaptureStatus *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g0ec4aacc66fe76c145252d12b189e233)* pCaptureStatus ) 
    

Returns a stream's capture status. 

######  Parameters 

`stream`
    \- Stream to query 
`pCaptureStatus`
    \- Returns the stream's capture status

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorStreamCaptureImplicit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038cf775033deb76dbde57b8df7bd9244e7)

###### Description

Return the capture status of `stream` via `pCaptureStatus`. After a successful call, `*pCaptureStatus` will contain one of the following: 

  * [cudaStreamCaptureStatusNone](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg0ec4aacc66fe76c145252d12b189e233dbbe6269245e62cd99e6c95206008e50): The stream is not capturing. 

  * [cudaStreamCaptureStatusActive](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg0ec4aacc66fe76c145252d12b189e233b165b0917377d0d7283ca5ac8013568b): The stream is capturing. 

  * [cudaStreamCaptureStatusInvalidated](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg0ec4aacc66fe76c145252d12b189e2334c61e87e6268a6f9b6c928c574c12c76): The stream was capturing but an error has invalidated the capture sequence. The capture sequence must be terminated with [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph.") on the stream where it was initiated in order to continue using `stream`. 


Note that, if this is called on [cudaStreamLegacy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246) (the "null stream") while a blocking stream on the same device is capturing, it will return [cudaErrorStreamCaptureImplicit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038cf775033deb76dbde57b8df7bd9244e7) and `*pCaptureStatus` is unspecified after the call. The blocking stream capture is not invalidated. 

When a blocking stream is capturing, the legacy stream is in an unusable state until the blocking stream capture is terminated. The legacy stream is not supported for stream capture, but attempted use would have an implicit dependency on the capturing stream(s). 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream."), [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph.")

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamQuery ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream ) 
    

Queries an asynchronous stream for completion status. 

######  Parameters 

`stream`
    \- Stream identifier

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorNotReady](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038285d0c852ab65b8925505e1065563f6d), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)

###### Description

Returns [cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591) if all operations in `stream` have completed, or [cudaErrorNotReady](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038285d0c852ab65b8925505e1065563f6d) if not. 

For the purposes of Unified Memory, a return value of [cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591) is equivalent to having called [cudaStreamSynchronize()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e "Waits for stream tasks to complete."). 

Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e "Waits for stream tasks to complete."), [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream."), [cudaStreamDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62 "Destroys and cleans up an asynchronous stream."), [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamSetAttribute ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value ) 
    

Sets stream attribute. 

######  Parameters 

`hStream`
    
`attr`
    
`value`
    

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)

###### Description

Sets attribute `attr` on `hStream` from corresponding attribute of `value`. The updated attribute will be applied to subsequent work submitted to the stream. It will not affect previously submitted work. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaAccessPolicyWindow](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaAccessPolicyWindow.html#structcudaAccessPolicyWindow)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamSynchronize ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream ) 
    

Waits for stream tasks to complete. 

######  Parameters 

`stream`
    \- Stream identifier

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)

###### Description

Blocks until `stream` has completed all operations. If the [cudaDeviceScheduleBlockingSync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g057e6912c52708b6aa86e79dd83d007c) flag was set for this device, the host thread will block until the stream is finished with all of its tasks. 

Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435 "Queries an asynchronous stream for completion status."), [cudaStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9 "Make a compute stream wait on an event."), [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream."), [cudaStreamDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62 "Destroys and cleans up an asynchronous stream."), [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamUpdateCaptureDependencies ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaGraphNode_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eed9297e04a8e4b5100200d5e69c124)* dependencies, const [cudaGraphEdgeData](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphEdgeData.html#structcudaGraphEdgeData)* dependencyData, size_t numDependencies, unsigned int  flags = 0 ) 
    

Update the set of dependencies in a capturing stream. 

######  Parameters 

`stream`
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

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorIllegalState](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00380604022302a2ed15d31bc33546b2e963)

###### Description

Modifies the dependency set of a capturing stream. The dependency set is the set of nodes that the next captured node in the stream will depend on. 

Valid flags are [cudaStreamAddCaptureDependencies](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg36421316e18c4a2f66905cf41bf8ce6f2d9c20d80bfa97109439be472c67d9c4) and [cudaStreamSetCaptureDependencies](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg36421316e18c4a2f66905cf41bf8ce6fda9c5ad0afcd24d659a48769b931563a). These control whether the set passed to the API is added to the existing set or replaces it. A flags value of 0 defaults to [cudaStreamAddCaptureDependencies](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg36421316e18c4a2f66905cf41bf8ce6f2d9c20d80bfa97109439be472c67d9c4). 

Nodes that are removed from the dependency set via this API do not result in [cudaErrorStreamCaptureUnjoined](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce9e4b4b58c1abf9faa1ce8b1079076d) if they are unreachable from the stream at [cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph."). 

Returns [cudaErrorIllegalState](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00380604022302a2ed15d31bc33546b2e963) if the stream is not capturing. 

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream."), [cudaStreamGetCaptureInfo](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g8d9312f1098c45e2ed43c949cfccf1f7 "Query a stream's capture state."), 

__host__ ​ __device__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaStreamWaitEvent ( [cudaStream_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a) stream, [cudaEvent_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247) event, unsigned int  flags = 0 ) 
    

Make a compute stream wait on an event. 

######  Parameters 

`stream`
    \- Stream to wait 
`event`
    \- Event to wait on 
`flags`
    \- Parameters for the operation(See above)

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c), [cudaErrorInvalidResourceHandle](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273)

###### Description

Makes all future work submitted to `stream` wait for all work captured in `event`. See [cudaEventRecord()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446 "Records an event.") for details on what is captured by an event. The synchronization will be performed efficiently on the device when applicable. `event` may be from a different device than `stream`. 

flags include: 

  * [cudaEventWaitDefault](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf8f19058a7cc2c3994c8f71590b8747e): Default event creation flag. 

  * [cudaEventWaitExternal](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g0c23426b7252eaa9cef695859991304e): Event is captured in the graph as an external event node when performing stream capture. 


Note:

  * This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics. 

  * Note that this function may also return error codes from previous, asynchronous launches. 

  * Note that this function may also return [cudaErrorInitializationError](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038ce7993a88ecf2c57b8102d55d997a18c), [cudaErrorInsufficientDriver](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038f5e52d1774934b77ba55d2aa2c063067) or [cudaErrorNoDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e942e4cbbd2bef6e92e293253f055613) if this call tries to initialize internal CUDA RT state. 

  * Note that as specified by [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream.") no CUDA function may be called from callback. [cudaErrorNotPermitted](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e003867b6095ab719b21659a400b553963eb3) may, but is not guaranteed to, be returned as a diagnostic in such case. 


**See also:**

[cudaStreamCreate](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da "Create an asynchronous stream."), [cudaStreamCreateWithFlags](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3 "Create an asynchronous stream."), [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435 "Queries an asynchronous stream for completion status."), [cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e "Waits for stream tasks to complete."), [cudaStreamAddCallback](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498 "Add a callback to a compute stream."), [cudaStreamDestroy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62 "Destroys and cleans up an asynchronous stream."), [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f)

__host__ ​[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) cudaThreadExchangeStreamCaptureMode ( [cudaStreamCaptureMode *](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g24ae5ae00cc50614957ff8eba43e560e)* mode ) 
    

Swaps the stream capture interaction mode for a thread. 

######  Parameters 

`mode`
    \- Pointer to mode value to swap with the current mode

###### Returns

[cudaSuccess](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038e355f04607d824883b4a50662830d591), [cudaErrorInvalidValue](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c)

###### Description

Sets the calling thread's stream capture interaction mode to the value contained in `*mode`, and overwrites `*mode` with the previous mode for the thread. To facilitate deterministic behavior across function or module boundaries, callers are encouraged to use this API in a push-pop fashion: 
    
    
    ‎     [cudaStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g24ae5ae00cc50614957ff8eba43e560e) mode = desiredMode;
               [cudaThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85 "Swaps the stream capture interaction mode for a thread.")(&mode);
               ...
               [cudaThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85 "Swaps the stream capture interaction mode for a thread.")(&mode); // restore previous mode

During stream capture (see [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream.")), some actions, such as a call to [cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356 "Allocate memory on the device."), may be unsafe. In the case of [cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356 "Allocate memory on the device."), the operation is not enqueued asynchronously to a stream, and is not observed by stream capture. Therefore, if the sequence of operations captured via [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream.") depended on the allocation being replayed whenever the graph is launched, the captured graph would be invalid. 

Therefore, stream capture places restrictions on API calls that can be made within or concurrently to a [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream.")-[cudaStreamEndCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e "Ends capture on a stream, returning the captured graph.") sequence. This behavior can be controlled via this API and flags to [cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream."). 

A thread's mode is one of the following: 

  * `cudaStreamCaptureModeGlobal:` This is the default mode. If the local thread has an ongoing capture sequence that was not initiated with `cudaStreamCaptureModeRelaxed` at `cuStreamBeginCapture`, or if any other thread has a concurrent capture sequence initiated with `cudaStreamCaptureModeGlobal`, this thread is prohibited from potentially unsafe API calls. 

  * `cudaStreamCaptureModeThreadLocal:` If the local thread has an ongoing capture sequence not initiated with `cudaStreamCaptureModeRelaxed`, it is prohibited from potentially unsafe API calls. Concurrent capture sequences in other threads are ignored. 

  * `cudaStreamCaptureModeRelaxed:` The local thread is not prohibited from potentially unsafe API calls. Note that the thread is still prohibited from API calls which necessarily conflict with stream capture, for example, attempting [cudaEventQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7 "Queries an event's status.") on an event that was last recorded inside a capture sequence. 


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3 "Begins graph capture on a stream.")

* * *

![](https://docs.nvidia.com/cuda/common/formatting/NVIDIA-LogoBlack.svg)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Manage My Privacy](https://www.nvidia.com/en-us/privacy-center/) | [Do Not Sell or Share My Data](https://www.nvidia.com/en-us/preferences/email-preferences/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2026 NVIDIA Corporation

![](https://docs.nvidia.com/akam/13/pixel_6b113425?a=dD1jM2VhNTQ4NDliMmJjYjU4NWIwZmJkMWJkZmUzZmExZTBhYjdlNDkyJmpzPW9mZg==)