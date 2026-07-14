# Device-Initiated Communication

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html

---

# Device-Initiated Communication[](#device-initiated-communication "Permalink to this heading")

Starting with version 2.28, NCCL provides a device-side communication API, making it possible to use communication primitives directly from user CUDA kernels.

## Device API[](#device-api "Permalink to this heading")

Device API consists of the following modules:

>   * **LSA (Load/Store Accessible)** – for communication between devices accessible via memory load/store operations, using CUDA P2P. This includes devices connected over NVLink and some devices connected over PCIe, so long as they have P2P connectivity with each other (as indicated by `nvidia-smi topo -p2p p`). Up to NCCL 2.28.3, the availability of LSA was also subject to the [NCCL_P2P_LEVEL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#env-nccl-p2p-level) distance check, but that is no longer the case with newer versions. See [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa).
> 
>   * **Multimem** – for communication between devices using the hardware multicast feature provided by NVLink SHARP (available on some datacenter GPUs since the Hopper generation).
> 
>   * **GIN (GPU-Initiated Networking)** – for communication over the network (since NCCL 2.28.7).
> 
>   * **Reduce, Broadcast, and Fused Building Blocks** — Building Blocks for Computation-Fused Kernels: reduce, copy (broadcast), and reduce-then-copy (see [Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#device-api-reducecopy) in the API reference).
> 
> 


## Requirements[](#requirements "Permalink to this heading")

The device API relies on symmetric memory (see [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg)), which in turn depends on GPU virtual memory management (see [NCCL_CUMEM_ENABLE](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#env-nccl-cumem-enable)) and optionally – for multimem support – on NVLink SHARP (see [NCCL_NVLS_ENABLE](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#env-nccl-nvls-enable)).

GIN has the following requirements:

  * CUDA 12.2 or later when compiling the GPU code

  * NVIDIA GPUs: Volta or newer. NVIDIA GPU drivers >= 510.40.3

  * NVIDIA NICs: CX4 or newer. rdma-core >= 44.0

  * GPU Direct RDMA: GIN host proxy requires DMA-BUF or nvidia-peermem support. GIN GDAKI requires DMA-BUF with kernel version >= 6.1 or nvidia-peermem support

  * Network topology: Requires full NIC connectivity. Does not support topologies where NICs cannot communicate across rails. Also does not support `NCCL_CROSS_NIC=0`.

  * Fused NICs are not supported. To use GIN on dual-port NICs, set `NCCL_IB_MERGE_NICS=0`

  * Using GIN for buffers that are backed by multiple cuMem segments requires DMA-BUF


When using host-backed buffers, the following additional limitations apply:

  * Host segments must be allocated with `CU_MEM_LOCATION_TYPE_HOST_NUMA`.

  * DirectNIC is not supported.

  * LSA Multimem is not supported.

  * Host RMA APIs are not supported.


Using the host RMA API requires CUDA 12.5 or greater.

Building with EMIT_LLVM_IR=1 (to generate readable LLVM intermediate representation code) requires CUDA 12.

## Cross-Version Compatibility[](#cross-version-compatibility "Permalink to this heading")

NCCL assumes the compile-time version of the device code is the same as the compile-time version of the corresponding host code (i.e., the call to [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate")). Starting with NCCL 2.29, the host-side structures are versioned, to enable cross-version compatibility checks. In general, the compile-time version cannot be newer than the runtime version (e.g., the version of `libnccl.so`). As of NCCL 2.29, backwards compatibility is supported for kernels utilizing LSA and multimem, i.e., a kernel compiled with NCCL 2.29.2/2.29.3 should continue to work when running with NCCL 2.29.7. Kernels utilizing GIN are currently not backwards compatible and need to be recompiled when NCCL is upgraded.

## Host-Side Setup[](#host-side-setup "Permalink to this heading")

To perform communication from the device kernel, a device communicator needs to be created first, using [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate"). Data transfer operations on buffers require symmetric memory windows (see [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg)). A custom communication kernel can then be launched using the standard CUDA syntax. The code excerpt below demonstrates these steps:
    
    
    int main() {
      [...]
      NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
    
      /* Buffer initialization and window creation */
      char* buffer;
      size_t size = 256*1048576;
      NCCLCHECK(ncclMemAlloc((void**)&buffer, size));
      ncclWindow_t win;
      NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));
    
      /* Get device communicator */
      ncclDevComm devComm;
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      int nCTAs = 16;
      reqs.lsaBarrierCount = nCTAs;
      NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    
      /* Launch user kernel */
      customKernel<<<nCTAs, 512>>>(devComm, win);
      [...]
    }
    

Depending on the kernel and application requirements, the same window can be used for input and output, or multiple windows may be needed. When creating a device communicator, the resources that the kernel will need should be specified via the requirements list (see [`ncclDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommRequirements "ncclDevCommRequirements")). In the above example we specify just the number of barriers that our LSA kernel will need, in this case one for each CTA the kernel is to be launched on (16, each CTA running 512 threads).

## Simple LSA Kernel[](#simple-lsa-kernel "Permalink to this heading")
    
    
    template <typename T>
    __global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
      ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x };
      bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
    
      const int rank = devComm.lsaRank, nRanks = devComm.lsaSize;
      const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
      const int globalNthreads = blockDim.x * gridDim.x * nRanks;
    
      for (size_t o = globalTid; o < count; o += globalNthreads) {
        T v = 0;
        for (int peer = 0; peer < nRanks; peer++) {
          T* inputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
          v += inputPtr[o];
        }
        for (int peer = 0; peer < nRanks; peer++) {
          T* outputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
          outputPtr[o] = v;
        }
      }
    
      bar.sync(ncclCoopCta(), cuda::memory_order_release);
    }
    

The above code excerpt shows a simple device kernel – an in-place variant (the input buffer is reused for the output) of AllReduce, utilizing LSA support (data is transferred via memory load/store instructions).

The start of the buffer is specified as a (byte-based) _offset_ within the previously registered window _win_ (see [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg)); the buffer consists of _count_ elements of type _T_.

Before the kernel can start processing data, it needs to ensure that all participants are ready. It creates a memory barrier session _bar_ (see `ncclLsaBarrierSession`) and uses it to synchronize across all the threads of the CTA (_ncclCoopCta()_ ; see [Thread Groups](#devapi-coops)) and the ranks of the communicator (_devComm_). _ncclTeamTagLsa_ indicates the subset of ranks the barrier will apply to (see [Teams](#devapi-teams)) – this kernel assumes that all ranks are LSA-connected. _blockIdx.x_ is the CTA’s local index, used to select the barrier.

The kernel then calculates a globally unique index for each thread as well as the overall thread count, and can finally start processing data, using an all-to-all communication pattern. In each iteration of the outer loop, every participating thread loads a single input element from each communicator rank (the first inner loop). `ncclGetLsaPointer()` is used to calculate the locally-accessible address of the start of the buffer within each rank (remote device memory was previously mapped into the local address space – see [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg)). Extracted input data is accumulated and the result is stored back at each rank (the second inner loop). Before the kernel terminates, another memory synchronization needs to take place to ensure that all participants have finished processing their data.

Note that this simple implementation would likely fall short of achieving the peak bandwidth, as it utilizes neither vectorization nor loop unrolling. For optimized LSA reduce, copy, and fused reduce-then-copy building blocks (e.g. for AllReduce, AllGather, ReduceScatter), see [Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#device-api-reducecopy) in the Device API reference.

## Multimem Device Kernel[](#multimem-device-kernel "Permalink to this heading")
    
    
    int main() {
      [...]
      reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      int nCTAs = 16;
      reqs.lsaBarrierCount = nCTAs;
      reqs.lsaMultimem = true;
      NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
      [...]
    }
    
    template <typename T>
    __global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
      ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, /*multimem*/true };
      [...]
      T* mmPtr = (T*)ncclGetLsaMultimemPointer(win, offset, devComm);
      for (size_t o = globalTid; o < count; o += globalNthreads) {
        T v = multimem_sum(mmPtr+o);
        multimem_st(mmPtr+o, v);
      }
      [...]
    }
    

The above code excerpt demonstrates modifications needed to the earlier code segments to enable multimem support (the lines with critical changes are highlighted). On the host side, `lsaMultimem` needs to be set in the requirements prior to creating the device communicator ([`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate") will fail if the necessary hardware support is unavailable).

Within the device kernel, we can switch the memory barrier to a multimem-optimized variant by adding an extra argument to the constructor. The processing loop is actually simpler with multimem: `ncclGetLsaMultimemPointer()` needs to be invoked just once per kernel. The returned multicast memory pointer enables access to the device memory of all the ranks of the communicator without having to iterate over them, and the data can be reduced in hardware. To keep this example simple, the implementations of `multimem_sum` and `multimem_st` are not included; they need to be implemented using PTX, e.g., `multimem.ld_reduce.global.add` and `multimem.st.global`.

## Thread Groups[](#thread-groups "Permalink to this heading")

Many functions in the device API take a thread cooperative group as input to indicate which threads within the CTA will take part in the operation. NCCL provides three predefined ones: `ncclCoopThread()`, `ncclCoopWarp()`, and (the most commonly used) `ncclCoopCta()`.

Users may also pass CUDA cooperative groups, or any class which provides `thread_rank()`, `size()`, and `sync()` methods.

## Teams[](#teams "Permalink to this heading")

To address remote ranks or perform barriers, NCCL refers to subsets of ranks within a communicator as “teams”. NCCL provides three predefined ones:

>   * `ncclTeamWorld()` – the “world” team, encompassing all the ranks of a given communicator.
> 
>   * `ncclTeamLsa()` – all the peers accessible from the local rank using load/store operations.
> 
>   * `ncclTeamRail()` – the set of peers that have the same rank number within their LSA team (a rail team is orthogonal to an LSA team).
> 
> 


The `ncclTeam` structure contains fairly self-explanatory elements `nRanks`, `rank`, and `stride`. The device API contains functions to verify team membership, convert rank numbers between teams, etc. The world and LSA teams are always contiguous (stride `1`), whereas the rail team is typically not – its stride equals the size of the LSA team (the assumption is thus that each rank _n_ within the local LSA team has direct network connectivity with corresponding ranks _n_ of all remote LSA teams).

## Segment Types[](#segment-types "Permalink to this heading")

The `SegmentType` template parameter of [`ncclGin::put()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE11SegmentType "ncclGin::put") and [`ncclGin::get()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4N7ncclGin3getE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t4Coop14DescriptorSmem8uint32_t11SegmentType "ncclGin::get") describes the physical memory composition of the source and destination virtual addresses. Three tag types are defined:

  * `ncclGin_SegmentDevice` (default) — the virtual addresses only contain cuMem segments of type `CU_MEM_LOCATION_TYPE_DEVICE`.

  * `ncclGin_SegmentHostNuma` — the virtual addresses only contain cuMem segments of type `CU_MEM_LOCATION_TYPE_HOST_NUMA`.

  * `ncclGin_SegmentMixed` — the virtual addresses contain a mix of `CU_MEM_LOCATION_TYPE_DEVICE` and `CU_MEM_LOCATION_TYPE_HOST_NUMA` segments.


## Host-Accessible Device Pointer Functions[](#host-accessible-device-pointer-functions "Permalink to this heading")

Starting with version 2.29, NCCL provides host-accessible functions that enable host code to obtain pointers to LSA memory regions.

The four functions are [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer") (multimem base pointer), [`ncclGetMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetMultimemDevicePointer "ncclGetMultimemDevicePointer") (multimem base pointer with custom handle), [`ncclGetLsaDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaDevicePointer "ncclGetLsaDevicePointer") (LSA peer pointer), and [`ncclGetPeerDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetPeerDevicePointer "ncclGetPeerDevicePointer") (world rank peer pointer). Functions automatically discover the associated communicator from the window object and return `ncclResult_t` error codes.

Usage Example:
    
    
    int main() {
      [...]
      // Allocate symmetric memory buffer
      char* buffer;
      size_t size = 256 * 1024 * 1024;  // 256 MB buffer
      NCCLCHECK(ncclMemAlloc((void**)&buffer, size));
    
      // Create window with the allocated buffer
      ncclWindow_t win;
      NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));
    
      // Get host-accessible pointers
      void* multimemPtr;
      void* lsaPtr;
      void* peerPtr;
    
      // Get multimem pointer (returns nullptr if multimem not supported)
      NCCLCHECK(ncclGetLsaMultimemDevicePointer(win, 0, &multimemPtr));
      if (multimemPtr == nullptr) {
          // Multimem not available, use fallback
      }
    
      // Get LSA pointer for peer 1
      NCCLCHECK(ncclGetLsaDevicePointer(win, 0, 1, &lsaPtr));
    
      // Get peer pointer for world rank 2
      NCCLCHECK(ncclGetPeerDevicePointer(win, 0, 2, &peerPtr));
    
      // Use pointers in custom kernels or legacy code
      customKernel<<<nCTAs, 256>>>(multimemPtr, lsaPtr, peerPtr);
    
      // Cleanup
      NCCLCHECK(ncclCommWindowDeregister(comm, &win));
      // Device pointers are invalidated after window deregistration
      NCCLCHECK(ncclMemFree(buffer));
      [...]
    }
    

Important notes: Pointer lifetime is limited to the shorter of Window and Communicator lifetime. Functions should be called once and pointers cached for reuse. For detailed function documentation, see [Host-Accessible Device Pointer Functions](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#device-api-host-functions).

## GIN Device Kernel[](#gin-device-kernel "Permalink to this heading")

The following illustrates pure GIN AlltoAll: all peer data moves over the network. The host creates a [`ncclDevComm`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevComm "ncclDevComm") with GIN-specific resources, registers symmetric memory windows (see [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg)), and launches a kernel that performs the collective using GIN.
    
    
    // Grid width (CTAs). Must match reqs.worldGinBarrierCount and reqs.ginSignalCount.
    #define NCCL_DEVICE_CTA_COUNT 16
    #define NCCL_DEVICE_THREADS_PER_CTA 512
    
    int main() {
      [...]
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      reqs.worldGinBarrierCount = NCCL_DEVICE_CTA_COUNT;
      reqs.ginSignalCount = NCCL_DEVICE_CTA_COUNT;
      reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
      NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
      [...]
    }
    
    template <typename T>
    __global__ void PureGinAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset,
                                          ncclWindow_t recvwin, size_t recvoffset,
                                          size_t count, struct ncclDevComm devComm) {
      int ginContext = 0; // single context for simplicity
      unsigned int signalIndex = blockIdx.x;
      ncclGin gin { devComm, ginContext };
      uint64_t signalValue = gin.readSignal(signalIndex);
    
      ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x };
      bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);
    
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      int nthreads = blockDim.x * gridDim.x;
    
      const size_t size = count * sizeof(T);
      for (int r = tid; r < devComm.nRanks; r += nthreads) {
        gin.put(ncclTeamWorld(devComm), r,
            recvwin, recvoffset + devComm.rank * size,
            sendwin, sendoffset + r * size,
            size, ncclGin_WeakSignalInc{signalIndex});
      }
    
      // Wait only on the CTA whose blockIdx.x (signalIndex) accumulates all puts to this rank.
      int receivingCta = (devComm.rank % nthreads) / blockDim.x;
      if (blockIdx.x == receivingCta)
        gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
    
      gin.flush(ncclCoopCta());
      bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
    }
    

The above code excerpt shows the GIN-related host setup for NCCL 2.30 and later (highlighted lines) together with the `PureGinAlltoAllKernel` kernel definition. GPU-initiated networking is available since NCCL 2.28.7. Version-specific host and kernel changes for older NCCL builds are summarized under [Compatibility adjustments](#deviceapi-gin-compat) at the end of this section.

In [`ncclDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommRequirements "ncclDevCommRequirements"), `worldGinBarrierCount` reserves slots for [`ncclGinBarrierSession`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession") (network-side barriers) and `ginSignalCount` reserves per-CTA signals for completion. Both are set to the number of CTAs in the launch grid (here `NCCL_DEVICE_CTA_COUNT`), matching `gridDim.x`, so each thread block uses `blockIdx.x` as its barrier index and signal index. GIN relies on these barriers and signals for cross-rank synchronization and for tracking asynchronous work. Set `ginConnectionType` to `NCCL_GIN_CONNECTION_FULL` to connect each rank to all peers (see [`ncclGinConnectionType_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGinConnectionType_t "ncclGinConnectionType_t")). [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate") fails if GIN cannot be provided.

On the device, GIN barriers synchronize across ranks over the network. Each thread block uses `blockIdx.x` to select its barrier so blocks can coordinate with corresponding blocks on other nodes. A single GIN context is used here. Construct `ncclGin` with context index `0`. Each thread block reads its own per-CTA signal slot (`signalIndex == blockIdx.x`) before `bar.sync()` at kernel entry. The [`ncclGinBarrierSession`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession") uses `ncclTeamTagWorld()` and `blockIdx.x`. The barrier ensures all ranks are ready before the AlltoAll exchange (`bar.sync()` at kernel entry).

Unlike AllReduce-style kernels, for AlltoAll the per-thread index only needs to be unique _within this rank_. That index then selects the destination peer. The main data transfer is performed using the one-sided `put()`, launched in parallel on all participating threads with one `put()` per destination peer. The loop is needed whenever the communicator size exceeds the number of threads that take part in the loop (here, `threadIdx.x + blockIdx.x * blockDim.x` stepping by `blockDim.x * gridDim.x`). `put()` takes the usual arguments: destination rank, destination and source windows and offsets, transfer size, and optional actions. This example passes `ncclGin_WeakSignalInc{signalIndex}` as _remoteAction_ so the destination rank receives one completion increment once the payload is settled. The receiver waits for the count of completed incoming puts for that signal slot.

Each CTA uses `signalIndex = blockIdx.x` on its outgoing [`ncclGin::put()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE11SegmentType "ncclGin::put") operations. On the destination rank, each peer’s [`ncclGin::put()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE11SegmentType "ncclGin::put") contributes one increment to the signal slot indexed by that sender CTA’s `blockIdx.x`. All CTAs participate in issuing [`ncclGin::put()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE11SegmentType "ncclGin::put"), but only the _receiving CTA_ , a single thread block on this rank, must observe completion for that rank’s signal slot. The kernel sets `receivingCta = (devComm.rank % nthreads) / blockDim.x` so that exactly that thread block runs `waitSignal()` for `signalIndex == receivingCta`. Every other CTA skips `waitSignal()` and only issues [`ncclGin::put()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE11SegmentType "ncclGin::put") and later `flush()`.

Once the signal watched by `receivingCta` has been incremented `nRanks` times, every peer has deposited its contribution into this rank’s receive buffer and the buffer is ready for consumption. That CTA’s `waitSignal()` blocks until that threshold using `signalValue + devComm.nRanks`, because each peer issues one inbound [`ncclGin::put()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE11SegmentType "ncclGin::put") that advances this rank’s counter for that `signalIndex`. Before terminating, the kernel still calls `flush()` on all CTAs to commit outstanding outgoing `put()` operations. While `flush()` does not guarantee full remote completion of every side effect, it does ensure the local send buffer is safe to reuse from this kernel’s perspective. After `waitSignal()` and `flush()`, `bar.sync()` runs again. The barrier is added so that all ranks complete the collective before any rank exits the kernel.

### Compatibility adjustments[](#compatibility-adjustments "Permalink to this heading")

The host setup, kernel, and explanation above reflect the NCCL 2.30 version and later. When targeting an older build, use the following as needed.

  * **GPU-initiated networking (GIN) baseline** — GIN is available since NCCL 2.28.7. [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate") and `ncclGin` require a communicator that supports the device API and GIN.

  * **Before NCCL 2.30 — no** `worldGinBarrierCount` — [`ncclGinBarrierSession`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession") was only usable for rail connectivity, with the corresponding `railGinBarrierCount`. For world-team GIN barriers, set `barrierCount` to the number of CTAs (same as `gridDim.x`). In the kernel, use the hybrid `ncclBarrierSession` with `ncclTeamTagWorld()` together with `ncclGin` instead of [`ncclGinBarrierSession`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession") with `ncclTeamTagWorld()`.

  * **Before NCCL 2.29.7 — no** `ginConnectionType` — Set `ginForceEnable` to `true` to enable full GIN connectivity (equivalent to `NCCL_GIN_CONNECTION_FULL` once `ginConnectionType` exists). The `ginConnectionType` field is available starting with NCCL 2.29.7 (see [`ncclDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommRequirements "ncclDevCommRequirements") in [Device API – Host-Side Setup](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#device-api-setup)).

  * **Deprecated** `ginForceEnable` — Prefer `ginConnectionType` on NCCL 2.29.7 and later. `ginForceEnable` is deprecated since NCCL 2.29.7.