# Creation and Lifecycle Methods

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html

---

# Creation and Lifecycle Methods[](#creation-and-lifecycle-methods "Permalink to this heading")

Methods on [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.Communicator") for creation, splitting, growing, and teardown.

## Construction[](#construction "Permalink to this heading")

_classmethod _Communicator.init(_nranks : int_, _rank : int_, _unique_id : [UniqueId](#nccl.core.UniqueId "nccl.core.utils.UniqueId") | Sequence[[UniqueId](#nccl.core.UniqueId "nccl.core.utils.UniqueId")]_, _config : [NCCLConfig](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html#nccl.core.NCCLConfig "nccl.core.communicator.NCCLConfig") | None = None_) → [Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.communicator.Communicator")[](#nccl.core.Communicator.init "Permalink to this definition")
    

Initializes a new NCCL communicator.

Creates a communicator that connects multiple ranks. This is a collective operation: all ranks must call this method with the same `nranks` and `unique_id` but with different `rank` values.

Parameters:
    

  * **nranks** – Total number of ranks in the communicator.

  * **rank** – This rank (must be between 0 and `nranks - 1`).

  * **unique_id** – Unique identifier(s) shared by all ranks. A sequence may be passed to use [`ncclCommInitRankScalable()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommInitRankScalable "ncclCommInitRankScalable").

  * **config** – NCCL configuration options. Defaults to `None`.


Returns:
    

A new Communicator instance.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If `unique_id` has an invalid type.

_classmethod _Communicator.init_all(_devices : int | Sequence[int] | None = None_) → list[[Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.communicator.Communicator")][](#nccl.core.Communicator.init_all "Permalink to this definition")
    

Initializes multiple NCCL communicators for single-process multi-GPU operations.

Creates an array of NCCL communicators, one for each device, within a single process. This is optimized for single-machine scenarios where all GPUs are controlled by the same process. Unlike [`init()`](#nccl.core.Communicator.init "nccl.core.Communicator.init"), which requires multi-process coordination (e.g. via MPI), [`init_all()`](#nccl.core.Communicator.init_all "nccl.core.Communicator.init_all") handles all coordination internally.

Each communicator is bound to its corresponding device and has its rank equal to its index in the returned list. The current device context is preserved by the underlying NCCL API. All communicators must be manually destroyed via [`destroy()`](#nccl.core.Communicator.destroy "nccl.core.Communicator.destroy") on each one.

Parameters:
    

**devices** – Specifies which devices to initialize. `None` (the default) initializes all visible CUDA devices. An int creates communicators for devices `[0, 1, ..., devices - 1]`. A sequence of ints uses the explicit device IDs. If the resulting device list is empty (`devices=0`, an empty sequence, or no visible devices), returns an empty list without calling into NCCL.

Returns:
    

List of initialized communicators, one per device. Rank `i` uses `devices[i]` (or device `i` when `devices` is an int).

Raises:
    

**TypeError** – If `devices` is not an int, sequence of ints, or `None`.

Communicator.initialize(_nranks : int_, _rank : int_, _unique_id : [UniqueId](#nccl.core.UniqueId "nccl.core.utils.UniqueId") | Sequence[[UniqueId](#nccl.core.UniqueId "nccl.core.utils.UniqueId")]_, _config : [NCCLConfig](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html#nccl.core.NCCLConfig "nccl.core.communicator.NCCLConfig") | None = None_) → None[](#nccl.core.Communicator.initialize "Permalink to this definition")
    

Initializes this communicator in-place.

Instance-method counterpart of the [`init()`](#nccl.core.Communicator.init "nccl.core.Communicator.init") classmethod. Allows creating a null communicator first (via `Communicator()`) and initializing it later. This is a collective operation; all ranks must call this method.

Parameters:
    

  * **nranks** – Total number of ranks in the communicator.

  * **rank** – This rank (must be between 0 and `nranks - 1`).

  * **unique_id** – Unique identifier(s) shared by all ranks.

  * **config** – NCCL configuration options. Defaults to `None`.


Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If `unique_id` has an invalid type or this communicator is already initialized.

## Bootstrap identifier[](#bootstrap-identifier "Permalink to this heading")

A [`UniqueId`](#nccl.core.UniqueId "nccl.core.UniqueId") is generated by one rank (typically rank 0) and broadcast to all participating ranks; all ranks then pass it to [`Communicator.init()`](#nccl.core.Communicator.init "nccl.core.Communicator.init").

_class _nccl.core.UniqueId(__internal : _nccl_bindings.UniqueId | None = None_)[](#nccl.core.UniqueId "Permalink to this definition")
    

Bases: `object`

NCCL unique identifier for communicator initialization.

A UniqueId is used to coordinate communicator initialization across multiple ranks. All ranks must use the same UniqueId to form a communicator. Typically one rank generates the UniqueId via [`get_unique_id()`](#nccl.core.get_unique_id "nccl.core.get_unique_id") and broadcasts it to all other ranks. Three serialization paths are supported:

  * **Bytes** : `bytes(uid)` (or [`as_bytes`](#nccl.core.UniqueId.as_bytes "nccl.core.UniqueId.as_bytes")) on the producer, [`from_bytes()`](#nccl.core.UniqueId.from_bytes "nccl.core.UniqueId.from_bytes") on receivers. The bytes of unique ID can be transmitted through any byte-oriented channel — a TCP socket, a shared filesystem, etc.

  * **NumPy** : [`as_ndarray`](#nccl.core.UniqueId.as_ndarray "nccl.core.UniqueId.as_ndarray") returns an in-place view of the underlying buffer, suitable for NumPy-aware buffer transports such as `mpi4py.MPI.Comm.Bcast` (uppercase `B`).

  * **Pickle** : instances are picklable directly, so higher level object broadcast helpers like `mpi4py.MPI.Comm.bcast` (lowercase `b`) work out of the box.


_static _from_bytes(_b : bytes | bytearray | memoryview_) → [UniqueId](#nccl.core.UniqueId "nccl.core.utils.UniqueId")[](#nccl.core.UniqueId.from_bytes "Permalink to this definition")
    

Reconstructs a UniqueId from a bytes-like buffer.

Parameters:
    

**b** – Bytes representation of a UniqueId, typically obtained via the [`as_bytes`](#nccl.core.UniqueId.as_bytes "nccl.core.UniqueId.as_bytes") property on the producing rank.

Returns:
    

Reconstructed [`UniqueId`](#nccl.core.UniqueId "nccl.core.UniqueId").

_property _as_ndarray _: numpy.ndarray_[](#nccl.core.UniqueId.as_ndarray "Permalink to this definition")
    

NumPy array view of the unique ID data.

_property _as_bytes _: bytes_[](#nccl.core.UniqueId.as_bytes "Permalink to this definition")
    

Bytes representation of the unique ID, suitable for serialization or broadcast.

nccl.core.get_unique_id(_empty : bool = False_) → [UniqueId](#nccl.core.UniqueId "nccl.core.utils.UniqueId")[](#nccl.core.get_unique_id "Permalink to this definition")
    

Generates a new NCCL unique identifier for communicator initialization.

Should be called by one rank (typically rank 0); the resulting [`UniqueId`](#nccl.core.UniqueId "nccl.core.UniqueId") must then be broadcast (e.g. via MPI) to all other ranks.

Parameters:
    

**empty** – If True, return an empty [`UniqueId`](#nccl.core.UniqueId "nccl.core.UniqueId") without calling NCCL. Useful when the bytes will be filled in later via [`UniqueId.from_bytes()`](#nccl.core.UniqueId.from_bytes "nccl.core.UniqueId.from_bytes"). Defaults to False.

Returns:
    

A new [`UniqueId`](#nccl.core.UniqueId "nccl.core.UniqueId") to be shared across ranks.

## Splitting and growing[](#splitting-and-growing "Permalink to this heading")

Communicator.split(_color : int | None = None_, _key : int = 0_, _config : [NCCLConfig](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html#nccl.core.NCCLConfig "nccl.core.communicator.NCCLConfig") | None = None_) → [Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.communicator.Communicator")[](#nccl.core.Communicator.split "Permalink to this definition")
    

Splits this communicator into sub-communicators based on color values.

Ranks that pass the same `color` value will be part of the same group. If `color` is `None`, the rank will not be part of any group and receives a null communicator (a [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.Communicator") instance with `ptr=0`). The `key` value determines rank ordering; smaller `key` means smaller rank in the new communicator. If keys are equal, the rank in the original communicator determines ordering.

This is a collective operation: all ranks in the communicator must call this method, even ranks that pass `color=None`. There must be no outstanding NCCL operations on the communicator to avoid deadlock.

Parameters:
    

  * **color** – Non-negative color value for grouping ranks. Pass `None` to exclude this rank from all groups. Defaults to `None`.

  * **key** – Ordering key within the color group. Defaults to 0.

  * **config** – Configuration for the new communicator. If `None`, inherits the parent’s configuration. Defaults to `None`.


Returns:
    

New sub-communicator, or a null communicator if `color` is `None`.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

See also

[`ncclCommSplit()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommSplit "ncclCommSplit")

Communicator.shrink(_exclude_ranks : Sequence[int] | None = None_, _config : [NCCLConfig](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html#nccl.core.NCCLConfig "nccl.core.communicator.NCCLConfig") | None = None_, _flag : [CommShrinkFlag](#nccl.core.CommShrinkFlag "nccl.core.constants.CommShrinkFlag") = CommShrinkFlag.DEFAULT_) → [Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.communicator.Communicator")[](#nccl.core.Communicator.shrink "Permalink to this definition")
    

Creates a new communicator by removing specified ranks from this one.

Ranks listed in `exclude_ranks` are excluded from the new communicator; the remaining ranks are renumbered to a contiguous `[0, n)` range.

This is a collective operation. All non-excluded ranks must call this method; excluded ranks must NOT call it. With [`DEFAULT`](#nccl.core.CommShrinkFlag.DEFAULT "nccl.core.CommShrinkFlag.DEFAULT") there must be no outstanding NCCL operations to avoid deadlock; combine with `config.shrink_share=True` to reuse parent communicator resources. With [`ABORT`](#nccl.core.CommShrinkFlag.ABORT "nccl.core.CommShrinkFlag.ABORT") outstanding operations are automatically aborted and no resources are shared with the parent.

Parameters:
    

  * **exclude_ranks** – Ranks to exclude from the new communicator. Defaults to `None` (no exclusions).

  * **config** – Configuration for the new communicator. If `None`, inherits the parent’s configuration. Defaults to `None`.

  * **flag** – Shrink behavior. Use [`DEFAULT`](#nccl.core.CommShrinkFlag.DEFAULT "nccl.core.CommShrinkFlag.DEFAULT") for normal operation or [`ABORT`](#nccl.core.CommShrinkFlag.ABORT "nccl.core.CommShrinkFlag.ABORT") after errors. Defaults to [`DEFAULT`](#nccl.core.CommShrinkFlag.DEFAULT "nccl.core.CommShrinkFlag.DEFAULT").


Returns:
    

New communicator without the excluded ranks.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

See also

[`ncclCommShrink()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommShrink "ncclCommShrink")

Communicator.get_unique_id() → [UniqueId](#nccl.core.UniqueId "nccl.core.utils.UniqueId")[](#nccl.core.Communicator.get_unique_id "Permalink to this definition")
    

Returns a per-communicator unique ID for use with [`grow()`](#nccl.core.Communicator.grow "nccl.core.Communicator.grow").

Generates a unique identifier bound to this communicator that can be shared with new ranks joining via [`grow()`](#nccl.core.Communicator.grow "nccl.core.Communicator.grow"). This is distinct from the global [`get_unique_id()`](#nccl.core.get_unique_id "nccl.core.get_unique_id") used for initial communicator creation. Only one existing rank (the grow root) should call this method.

A new UID cannot be generated while a previous UID is unconsumed; each UID can be used only once and the user must wait for the corresponding grow operation to complete before calling again.

Returns:
    

[`UniqueId`](#nccl.core.UniqueId "nccl.core.UniqueId") for grow operations.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.grow(_nranks : int_, _unique_id : [UniqueId](#nccl.core.UniqueId "nccl.core.utils.UniqueId") | None = None_, _rank : int | None = None_, _config : [NCCLConfig](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html#nccl.core.NCCLConfig "nccl.core.communicator.NCCLConfig") | None = None_) → [Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.communicator.Communicator")[](#nccl.core.Communicator.grow "Permalink to this definition")
    

Grows the communicator by adding new ranks.

Creates a new communicator that includes both existing ranks from this communicator and new ranks joining the group. There are three roles:

>   * Existing root: the one existing rank that called [`get_unique_id()`](#nccl.core.get_unique_id "nccl.core.get_unique_id").
> 
>   * Existing non-root: all other existing ranks.
> 
>   * New ranks: ranks joining via a null communicator (`Communicator()`).
> 
> 


This is a collective operation. All ranks (existing and new) must call this method. Usage by role:

>   * Existing root: `new_comm = existing_comm.grow(nranks, uid)`
> 
>   * Existing non-root: `new_comm = existing_comm.grow(nranks)`
> 
>   * New rank: `new_comm = Communicator().grow(nranks, uid, rank=assigned_rank)`
> 
> 


The UID is consumed upon successful grow and cannot be reused.

Parameters:
    

  * **nranks** – Total number of ranks in the new communicator (existing plus new). All roles must pass the same value.

  * **unique_id** – Unique identifier from [`get_unique_id()`](#nccl.core.get_unique_id "nccl.core.get_unique_id"). Existing root and new ranks must pass the [`UniqueId`](#nccl.core.UniqueId "nccl.core.UniqueId"); existing non-root must pass `None`. Defaults to `None`.

  * **rank** – This rank’s ID in the new communicator. New ranks must pass their assigned rank, which must be `>=` the parent communicator size. Existing ranks must pass `None`. Defaults to `None`.

  * **config** – Configuration for the new communicator. Defaults to `None`.


Returns:
    

New [`Communicator`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/class.html#nccl.core.Communicator "nccl.core.Communicator") containing all ranks.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If a new rank is given an initialized communicator, or an existing rank is given a null communicator.

## Teardown[](#teardown "Permalink to this heading")

Communicator.destroy() → None[](#nccl.core.Communicator.destroy "Permalink to this definition")
    

Destroys the communicator and frees local resources.

If [`finalize()`](#nccl.core.Communicator.finalize "nccl.core.Communicator.finalize") has not been called explicitly, [`destroy()`](#nccl.core.Communicator.destroy "nccl.core.Communicator.destroy") will call it internally. If [`finalize()`](#nccl.core.Communicator.finalize "nccl.core.Communicator.finalize") is called explicitly, users must ensure the communicator state becomes `ncclSuccess` before calling [`destroy()`](#nccl.core.Communicator.destroy "nccl.core.Communicator.destroy"). The communicator should not be accessed after [`destroy()`](#nccl.core.Communicator.destroy "nccl.core.Communicator.destroy") returns.

All resources (registered buffers, windows, custom operators) owned by this communicator are automatically closed before destruction. This is an intra-node collective call: all ranks on the same node must call it to avoid hanging. The recommended pattern is [`finalize()`](#nccl.core.Communicator.finalize "nccl.core.Communicator.finalize") followed by [`destroy()`](#nccl.core.Communicator.destroy "nccl.core.Communicator.destroy").

Errors during cleanup are suppressed for safety.

See also

[`ncclCommDestroy()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommDestroy "ncclCommDestroy")

Communicator.abort() → None[](#nccl.core.Communicator.abort "Permalink to this definition")
    

Aborts the communicator and frees resources, terminating in-flight operations.

Should be called when an unrecoverable error occurs. Unlike [`destroy()`](#nccl.core.Communicator.destroy "nccl.core.Communicator.destroy"), this immediately aborts uncompleted operations. All active ranks must call this function in order to abort the NCCL communicator successfully.

All resources (registered buffers, windows, custom operators) owned by this communicator are automatically closed before aborting. Errors during cleanup are suppressed for safety. For more details, see the Fault Tolerance section in the NCCL documentation.

See also

[`ncclCommAbort()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommAbort "ncclCommAbort")

Communicator.finalize() → None[](#nccl.core.Communicator.finalize "Permalink to this definition")
    

Finalizes the communicator, flushing uncompleted operations and network resources.

Typically called before [`destroy()`](#nccl.core.Communicator.destroy "nccl.core.Communicator.destroy") to ensure all operations complete. This is a collective operation that must be called by all ranks.

For nonblocking communicators this is itself nonblocking: success sets the communicator state to `ncclInProgress` to indicate finalization is in progress. Once all NCCL operations complete, the communicator transitions to `ncclSuccess`. Users can query the state with [`get_async_error()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/status.html#nccl.core.Communicator.get_async_error "nccl.core.Communicator.get_async_error").

See also

[`ncclCommFinalize()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommFinalize "ncclCommFinalize")

## Pause and resume[](#pause-and-resume "Permalink to this heading")

Communicator.revoke(_flags : int = 0_) → None[](#nccl.core.Communicator.revoke "Permalink to this definition")
    

Revokes the communicator.

Stops all in-flight operations and marks the communicator state as `ncclInProgress`. The state transitions to `ncclSuccess` when the communicator becomes quiescent, after which management operations ([`destroy()`](#nccl.core.Communicator.destroy "nccl.core.Communicator.destroy"), [`split()`](#nccl.core.Communicator.split "nccl.core.Communicator.split"), [`shrink()`](#nccl.core.Communicator.shrink "nccl.core.Communicator.shrink")) can proceed safely.

Calling [`finalize()`](#nccl.core.Communicator.finalize "nccl.core.Communicator.finalize") after [`revoke()`](#nccl.core.Communicator.revoke "nccl.core.Communicator.revoke") is invalid. Resource sharing via split-share / shrink-share is disabled while revoked.

Parameters:
    

**flags** – Reserved for future use. Currently must be 0.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.suspend(_flags : [CommSuspendFlag](#nccl.core.CommSuspendFlag "nccl.core.constants.CommSuspendFlag") = CommSuspendFlag.MEM_) → None[](#nccl.core.Communicator.suspend "Permalink to this definition")
    

Suspends communicator operations to free resources.

The communicator cannot be used for communication while suspended. Call [`resume()`](#nccl.core.Communicator.resume "nccl.core.Communicator.resume") to restore it.

Parameters:
    

**flags** – Suspend flags controlling what resources to release. [`MEM`](#nccl.core.CommSuspendFlag.MEM "nccl.core.CommSuspendFlag.MEM") releases dynamic GPU memory allocations.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

Communicator.resume() → None[](#nccl.core.Communicator.resume "Permalink to this definition")
    

Resumes all previously suspended communicator resources.

Restores a communicator that was suspended with [`suspend()`](#nccl.core.Communicator.suspend "nccl.core.Communicator.suspend") so that it can be used for communication again.

Raises:
    

[**NcclInvalid**](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the communicator is not initialized.

## Flag enums[](#flag-enums "Permalink to this heading")

### CommShrinkFlag[](#commshrinkflag "Permalink to this heading")

_class _nccl.core.CommShrinkFlag(_value_ , _names= <not given>_, _*values_ , _module=None_ , _qualname=None_ , _type=None_ , _start=1_ , _boundary=None_)[](#nccl.core.CommShrinkFlag "Permalink to this definition")
    

Bases: `IntEnum`

Behavior flag for [`Communicator.shrink()`](#nccl.core.Communicator.shrink "nccl.core.Communicator.shrink").

DEFAULT _ = 0_[](#nccl.core.CommShrinkFlag.DEFAULT "Permalink to this definition")
    

Shrink the parent communicator normally; outstanding NCCL operations must already be quiesced.

ABORT _ = 1_[](#nccl.core.CommShrinkFlag.ABORT "Permalink to this definition")
    

First terminate ongoing parent operations, then shrink. No resources are shared with the parent.

### CommSuspendFlag[](#commsuspendflag "Permalink to this heading")

_class _nccl.core.CommSuspendFlag(_value_ , _names= <not given>_, _*values_ , _module=None_ , _qualname=None_ , _type=None_ , _start=1_ , _boundary=None_)[](#nccl.core.CommSuspendFlag "Permalink to this definition")
    

Bases: `IntFlag`

Behavior flag for [`Communicator.suspend()`](#nccl.core.Communicator.suspend "nccl.core.Communicator.suspend").

MEM _ = 1_[](#nccl.core.CommSuspendFlag.MEM "Permalink to this definition")
    

Suspend memory by releasing dynamic GPU memory allocations held by the communicator.