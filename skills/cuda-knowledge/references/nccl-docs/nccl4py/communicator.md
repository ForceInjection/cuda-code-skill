# Communicator

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator.html

---

# Communicator[](#communicator "Permalink to this heading")

The [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.Communicator") class and its methods, organized by lifecycle stage and operation kind:

  * [Communicator Class](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html) — the class itself, its constructor, and per-instance properties for identity and device-API capability.

  * [Creation and Lifecycle Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html) — creating, splitting, growing, and tearing down communicators.

  * [Collective Communication Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html) — collective communication methods (allreduce, broadcast, gather, …).

  * [Point-to-Point and Signal Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html) — point-to-point and signal methods (send / recv / signal / wait_signal / put_signal).

  * [Memory Registration Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html) — buffer and window registration for zero-copy and RMA.

  * [Device Communicator Setup](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html) — host-side bootstrap of a device communicator.

  * [Status and Utility Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html) — error queries and resource cleanup.


  * [Communicator Class](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html)
    * [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator)
    * [Properties](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#properties)
  * [Creation and Lifecycle Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html)
    * [Construction](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#construction)
    * [Bootstrap identifier](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#bootstrap-identifier)
    * [Splitting and growing](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#splitting-and-growing)
    * [Teardown](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#teardown)
    * [Pause and resume](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#pause-and-resume)
    * [Flag enums](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#flag-enums)
  * [Collective Communication Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html)
    * [allreduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#allreduce)
    * [broadcast](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#broadcast)
    * [reduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#reduce)
    * [allgather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#allgather)
    * [reduce_scatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#reduce-scatter)
    * [alltoall](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#alltoall)
    * [gather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#gather)
    * [scatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#scatter)
    * [create_pre_mul_sum](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#create-pre-mul-sum)
  * [Point-to-Point and Signal Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html)
    * [send](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html#send)
    * [recv](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html#recv)
    * [signal](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html#signal)
    * [wait_signal](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html#wait-signal)
    * [put_signal](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html#put-signal)
    * [WaitSignalDesc](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/p2p.html#waitsignaldesc)
  * [Memory Registration Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html)
    * [register_buffer](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#register-buffer)
    * [register_window](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#register-window)
    * [WindowFlag](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/registration.html#windowflag)
  * [Device Communicator Setup](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html)
    * [create_dev_comm](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html#create-dev-comm)
    * [GIN type enums](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html#gin-type-enums)
  * [Status and Utility Methods](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html)
    * [close_all_resources](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html#close-all-resources)
    * [get_last_error](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html#get-last-error)
    * [get_async_error](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html#get-async-error)
    * [get_mem_stat](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html#get-mem-stat)
    * [NcclCommMemStat](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html#ncclcommmemstat)
    * [get_error_string](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html#get-error-string)