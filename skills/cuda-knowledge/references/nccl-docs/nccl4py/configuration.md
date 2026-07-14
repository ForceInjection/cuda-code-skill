# Configuration

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/configuration.html

---

# Configuration[](#configuration "Permalink to this heading")

Configuration objects passed to communicator creation methods, plus the flag enums they consume.

## NCCLConfig[](#ncclconfig "Permalink to this heading")

Used by [`Communicator.init()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.init "nccl.core.Communicator.init"), [`Communicator.split()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.split "nccl.core.Communicator.split"), [`Communicator.shrink()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.shrink "nccl.core.Communicator.shrink"), and [`Communicator.grow()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/lifecycle.html#nccl.core.Communicator.grow "nccl.core.Communicator.grow"). Fields left unset (`None`) remain at NCCL’s internal default; values are validated by the C library when the config is consumed.

_class _nccl.core.NCCLConfig(_*_ , _blocking : bool | None = None_, _cga_cluster_size : int | None = None_, _min_ctas : int | None = None_, _max_ctas : int | None = None_, _net_name : str | None = None_, _split_share : bool | None = None_, _traffic_class : int | None = None_, _comm_name : str | None = None_, _collnet_enable : bool | None = None_, _cta_policy : [CTAPolicy](#nccl.core.CTAPolicy "nccl.core.constants.CTAPolicy") | None = None_, _shrink_share : bool | None = None_, _nvls_ctas : int | None = None_, _n_channels_per_net_peer : int | None = None_, _nvlink_centric_sched : bool | None = None_, _graph_usage_mode : int | None = None_, _num_rma_ctx : int | None = None_, _max_p2p_peers : int | None = None_, _graph_stream_ordering : int | None = None_)[](#nccl.core.NCCLConfig "Permalink to this definition")
    

Bases: `object`

NCCL configuration for communicator initialization.

Provides configuration options for NCCL communicators, allowing fine-tuning of performance and behavior characteristics. Fields not set in the constructor remain at NCCL’s internal default; values are validated by the C library when the config is consumed.

See also

[`ncclConfig_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclConfig_t "ncclConfig_t") for the description of each field.

blocking _: bool | None_ _ = None_[](#nccl.core.NCCLConfig.blocking "Permalink to this definition")
    

Blocking (True) or non-blocking (False) communicator behavior. If unset, NCCL uses True.

cga_cluster_size _: int | None_ _ = None_[](#nccl.core.NCCLConfig.cga_cluster_size "Permalink to this definition")
    

Cooperative Group Array (CGA) size for kernels (0-8). If unset, NCCL uses 4 for sm90+, 0 otherwise.

min_ctas _: int | None_ _ = None_[](#nccl.core.NCCLConfig.min_ctas "Permalink to this definition")
    

Minimum number of CTAs per kernel; positive integer up to 32. If unset, NCCL uses 1.

max_ctas _: int | None_ _ = None_[](#nccl.core.NCCLConfig.max_ctas "Permalink to this definition")
    

Maximum number of CTAs per kernel; positive integer up to 32. If unset, NCCL uses 32.

net_name _: str | None_ _ = None_[](#nccl.core.NCCLConfig.net_name "Permalink to this definition")
    

Network module name (e.g. ‘IB’, ‘Socket’). Case-insensitive. If unset, NCCL auto-selects.

split_share _: bool | None_ _ = None_[](#nccl.core.NCCLConfig.split_share "Permalink to this definition")
    

Share resources with the child communicator during split. If unset, NCCL uses False.

traffic_class _: int | None_ _ = None_[](#nccl.core.NCCLConfig.traffic_class "Permalink to this definition")
    

Traffic class (TC) for network operations (>= 0). Network-specific meaning.

comm_name _: str | None_ _ = None_[](#nccl.core.NCCLConfig.comm_name "Permalink to this definition")
    

User-defined communicator name for logging and profiling.

collnet_enable _: bool | None_ _ = None_[](#nccl.core.NCCLConfig.collnet_enable "Permalink to this definition")
    

Enable (True) or disable (False) IB SHARP. If unset, NCCL uses False.

cta_policy _: [CTAPolicy](#nccl.core.CTAPolicy "nccl.core.constants.CTAPolicy") | None_ _ = None_[](#nccl.core.NCCLConfig.cta_policy "Permalink to this definition")
    

CTA scheduling policy. If unset, NCCL uses CTAPolicy.DEFAULT.

shrink_share _: bool | None_ _ = None_[](#nccl.core.NCCLConfig.shrink_share "Permalink to this definition")
    

Share resources with the child communicator during shrink. If unset, NCCL uses False.

nvls_ctas _: int | None_ _ = None_[](#nccl.core.NCCLConfig.nvls_ctas "Permalink to this definition")
    

Total number of CTAs for NVLS kernels (positive integer). If unset, NCCL auto-determines.

n_channels_per_net_peer _: int | None_ _ = None_[](#nccl.core.NCCLConfig.n_channels_per_net_peer "Permalink to this definition")
    

Number of network channels for pairwise communication. Positive integer, rounded up to power of 2. If unset, NCCL uses an AlltoAll-optimized value.

nvlink_centric_sched _: bool | None_ _ = None_[](#nccl.core.NCCLConfig.nvlink_centric_sched "Permalink to this definition")
    

Enable NVLink-centric scheduling. If unset, NCCL uses False.

graph_usage_mode _: int | None_ _ = None_[](#nccl.core.NCCLConfig.graph_usage_mode "Permalink to this definition")
    

Graph usage mode (NCCL 2.29+). Supported values are 0 (no graphs), 1 (one graph), 2 (multiple graphs or mix of graph and non-graph). If unset, NCCL uses 2.

num_rma_ctx _: int | None_ _ = None_[](#nccl.core.NCCLConfig.num_rma_ctx "Permalink to this definition")
    

Number of RMA contexts (NCCL 2.29+). Positive integer. If unset, NCCL uses 1.

max_p2p_peers _: int | None_ _ = None_[](#nccl.core.NCCLConfig.max_p2p_peers "Permalink to this definition")
    

Maximum number of peers any rank will concurrently communicate with using P2P (NCCL 2.30+). Positive integer. If unset, NCCL uses the communicator size.

graph_stream_ordering _: int | None_ _ = None_[](#nccl.core.NCCLConfig.graph_stream_ordering "Permalink to this definition")
    

Whether NCCL preserves stream-ordering semantics for collectives captured into CUDA graphs. Supported values are 0 (disabled) or 1 (enabled). Cannot be combined with `graph_usage_mode=2`. Also controllable via the `NCCL_GRAPH_STREAM_ORDERING` environment variable. If unset, NCCL uses 1.

## NCCLDevCommRequirements[](#nccldevcommrequirements "Permalink to this heading")

Used by [`Communicator.create_dev_comm()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html#nccl.core.Communicator.create_dev_comm "nccl.core.Communicator.create_dev_comm"). Fields left unset (`None`) remain at NCCL’s internal default.

_class _nccl.core.NCCLDevCommRequirements(_*_ , _lsa_multimem : bool | None = None_, _barrier_count : int | None = None_, _lsa_barrier_count : int | None = None_, _rail_gin_barrier_count : int | None = None_, _lsa_ll_a2a_block_count : int | None = None_, _lsa_ll_a2a_slot_count : int | None = None_, _gin_force_enable : bool | None = None_, _gin_context_count : int | None = None_, _gin_signal_count : int | None = None_, _gin_counter_count : int | None = None_, _gin_connection_type : [NcclGinConnectionType](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html#nccl.core.NcclGinConnectionType "nccl.core.typing.NcclGinConnectionType") | None = None_, _gin_exclusive_contexts : bool | None = None_, _gin_queue_depth : int | None = None_, _world_gin_barrier_count : int | None = None_, _gin_strong_signals_required : bool | None = None_, _gin_va_signals_required : bool | None = None_)[](#nccl.core.NCCLDevCommRequirements "Permalink to this definition")
    

Bases: `object`

NCCL device communicator requirements configuration.

Provides configuration for device communicator creation, allowing fine-tuning of resource allocation and device-side communication behavior. Fields not set in the constructor remain at NCCL’s internal default; values are validated by the C library when the requirements are consumed by [`Communicator.create_dev_comm()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html#nccl.core.Communicator.create_dev_comm "nccl.core.Communicator.create_dev_comm").

See also

[`ncclDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommRequirements "ncclDevCommRequirements") for the description of each field.

lsa_multimem _: bool | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.lsa_multimem "Permalink to this definition")
    

Enable multimem on the LSA team. If unset, NCCL uses False.

barrier_count _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.barrier_count "Permalink to this definition")
    

Number of barriers required. If unset, NCCL uses 0.

lsa_barrier_count _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.lsa_barrier_count "Permalink to this definition")
    

Number of LSA barriers. If unset, NCCL uses 0.

rail_gin_barrier_count _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.rail_gin_barrier_count "Permalink to this definition")
    

Number of railed GIN barriers. If unset, NCCL uses 0.

lsa_ll_a2a_block_count _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.lsa_ll_a2a_block_count "Permalink to this definition")
    

LSA low-latency all-to-all block count. If unset, NCCL uses 0.

lsa_ll_a2a_slot_count _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.lsa_ll_a2a_slot_count "Permalink to this definition")
    

LSA low-latency all-to-all slot count. If unset, NCCL uses 0.

gin_force_enable _: bool | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.gin_force_enable "Permalink to this definition")
    

Force-enable GPU Interconnect Network. If unset, NCCL uses False.

gin_context_count _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.gin_context_count "Permalink to this definition")
    

Number of GIN contexts (hint; actual count may differ). If unset, NCCL uses 4.

gin_signal_count _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.gin_signal_count "Permalink to this definition")
    

Number of GIN signals (guaranteed to start at id=0). If unset, NCCL uses 0.

gin_counter_count _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.gin_counter_count "Permalink to this definition")
    

Number of GIN counters (guaranteed to start at id=0). If unset, NCCL uses 0.

gin_connection_type _: [NcclGinConnectionType](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/device_setup.html#nccl.core.NcclGinConnectionType "nccl.core.typing.NcclGinConnectionType") | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.gin_connection_type "Permalink to this definition")
    

GIN connection type. If unset, NCCL uses NcclGinConnectionType.NONE.

gin_exclusive_contexts _: bool | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.gin_exclusive_contexts "Permalink to this definition")
    

Use exclusive GIN contexts. If unset, NCCL uses False.

gin_queue_depth _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.gin_queue_depth "Permalink to this definition")
    

GIN queue depth. If unset, NCCL uses 0.

world_gin_barrier_count _: int | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.world_gin_barrier_count "Permalink to this definition")
    

Number of world GIN barriers. If unset, NCCL uses 0.

gin_strong_signals_required _: bool | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.gin_strong_signals_required "Permalink to this definition")
    

Whether GIN strong signals are required by kernels using this devComm. When False, using GIN strong signals results in undefined behavior. If unset, NCCL uses True.

gin_va_signals_required _: bool | None_ _ = None_[](#nccl.core.NCCLDevCommRequirements.gin_va_signals_required "Permalink to this definition")
    

Whether GIN VA signals are required by kernels using this devComm. When False, using GIN VA signals results in undefined behavior. If unset, NCCL uses True.

## CTAPolicy[](#ctapolicy "Permalink to this heading")

_class _nccl.core.CTAPolicy(_value_ , _names= <not given>_, _*values_ , _module=None_ , _qualname=None_ , _type=None_ , _start=1_ , _boundary=None_)[](#nccl.core.CTAPolicy "Permalink to this definition")
    

Bases: `IntFlag`

NCCL performance policy for CTA scheduling, used by [`NCCLConfig.cta_policy`](#nccl.core.NCCLConfig.cta_policy "nccl.core.NCCLConfig.cta_policy").

DEFAULT _ = 0_[](#nccl.core.CTAPolicy.DEFAULT "Permalink to this definition")
    

Default CTA policy.

EFFICIENCY _ = 1_[](#nccl.core.CTAPolicy.EFFICIENCY "Permalink to this definition")
    

Optimize for efficiency.

ZERO _ = 2_[](#nccl.core.CTAPolicy.ZERO "Permalink to this definition")
    

Zero-CTA optimization.