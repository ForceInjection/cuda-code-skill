# Types and Constants

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/types.html

---

# Types and Constants[](#types-and-constants "Permalink to this heading")

Type wrappers, predefined value constants, and type aliases that appear in public method signatures.

## Data type[](#data-type "Permalink to this heading")

### NcclDataType[](#nccldatatype "Permalink to this heading")

_class _nccl.core.NcclDataType(_value_ , _names= <not given>_, _*values_ , _module=None_ , _qualname=None_ , _type=None_ , _start=1_ , _boundary=None_)[](#nccl.core.NcclDataType "Permalink to this definition")
    

Bases: `IntEnum`

NCCL data type, mirroring [`ncclDataType_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t "ncclDataType_t").

Used as the `dtype` of buffer specs and as the `datatype` argument of NCCL collective operations. Supports conversion to/from NumPy dtypes via [`from_numpy_dtype()`](#nccl.core.NcclDataType.from_numpy_dtype "nccl.core.NcclDataType.from_numpy_dtype") and [`numpy_dtype`](#nccl.core.NcclDataType.numpy_dtype "nccl.core.NcclDataType.numpy_dtype").

INT8 _ = 0_[](#nccl.core.NcclDataType.INT8 "Permalink to this definition")
    

Signed 8-bit integer.

CHAR _ = 0_[](#nccl.core.NcclDataType.CHAR "Permalink to this definition")
    

Alias of [`INT8`](#nccl.core.NcclDataType.INT8 "nccl.core.NcclDataType.INT8").

UINT8 _ = 1_[](#nccl.core.NcclDataType.UINT8 "Permalink to this definition")
    

Unsigned 8-bit integer.

INT32 _ = 2_[](#nccl.core.NcclDataType.INT32 "Permalink to this definition")
    

Signed 32-bit integer.

INT _ = 2_[](#nccl.core.NcclDataType.INT "Permalink to this definition")
    

Alias of [`INT32`](#nccl.core.NcclDataType.INT32 "nccl.core.NcclDataType.INT32").

UINT32 _ = 3_[](#nccl.core.NcclDataType.UINT32 "Permalink to this definition")
    

Unsigned 32-bit integer.

INT64 _ = 4_[](#nccl.core.NcclDataType.INT64 "Permalink to this definition")
    

Signed 64-bit integer.

UINT64 _ = 5_[](#nccl.core.NcclDataType.UINT64 "Permalink to this definition")
    

Unsigned 64-bit integer.

FLOAT16 _ = 6_[](#nccl.core.NcclDataType.FLOAT16 "Permalink to this definition")
    

IEEE half-precision floating point (2 bytes).

HALF _ = 6_[](#nccl.core.NcclDataType.HALF "Permalink to this definition")
    

Alias of [`FLOAT16`](#nccl.core.NcclDataType.FLOAT16 "nccl.core.NcclDataType.FLOAT16").

FLOAT32 _ = 7_[](#nccl.core.NcclDataType.FLOAT32 "Permalink to this definition")
    

IEEE single-precision floating point (4 bytes).

FLOAT _ = 7_[](#nccl.core.NcclDataType.FLOAT "Permalink to this definition")
    

Alias of [`FLOAT32`](#nccl.core.NcclDataType.FLOAT32 "nccl.core.NcclDataType.FLOAT32").

FLOAT64 _ = 8_[](#nccl.core.NcclDataType.FLOAT64 "Permalink to this definition")
    

IEEE double-precision floating point (8 bytes).

DOUBLE _ = 8_[](#nccl.core.NcclDataType.DOUBLE "Permalink to this definition")
    

Alias of [`FLOAT64`](#nccl.core.NcclDataType.FLOAT64 "nccl.core.NcclDataType.FLOAT64").

BFLOAT16 _ = 9_[](#nccl.core.NcclDataType.BFLOAT16 "Permalink to this definition")
    

Brain floating-point (16-bit truncated single precision; CUDA 11+).

FLOAT8E4M3 _ = 10_[](#nccl.core.NcclDataType.FLOAT8E4M3 "Permalink to this definition")
    

8-bit floating point, 4 exponent + 3 mantissa bits (CUDA >= 11.8, SM >= 90).

FLOAT8E5M2 _ = 11_[](#nccl.core.NcclDataType.FLOAT8E5M2 "Permalink to this definition")
    

8-bit floating point, 5 exponent + 2 mantissa bits (CUDA >= 11.8, SM >= 90).

_classmethod _from_numpy_dtype(_dtype : numpy.dtype_) → [NcclDataType](#nccl.core.NcclDataType "nccl.core.typing.NcclDataType")[](#nccl.core.NcclDataType.from_numpy_dtype "Permalink to this definition")
    

Maps a NumPy dtype to its NCCL equivalent.

Parameters:
    

**dtype** – A NumPy dtype. Mapped first by name (for `ml-dtypes` like `bfloat16`, `float8_e4m3fn`, `float8_e5m2`) and then by `(kind, itemsize)` for standard types.

Returns:
    

Corresponding [`NcclDataType`](#nccl.core.NcclDataType "nccl.core.NcclDataType") member.

Raises:
    

[**NcclInvalid**](#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If the dtype has no NCCL equivalent.

_property _itemsize _: int_[](#nccl.core.NcclDataType.itemsize "Permalink to this definition")
    

Size in bytes of a single element of this data type.

_property _numpy_dtype _: numpy.dtype_[](#nccl.core.NcclDataType.numpy_dtype "Permalink to this definition")
    

Equivalent NumPy dtype.

Returns:
    

NumPy dtype corresponding to this NCCL data type. For `BFLOAT16` and the float8 variants, `ml-dtypes` must be installed.

Raises:
    

[**NcclInvalid**](#nccl.core.NcclInvalid "nccl.core.NcclInvalid") – If `ml-dtypes` is required but not installed.

### Predefined data type constants[](#predefined-data-type-constants "Permalink to this heading")

Module-level [`NcclDataType`](#nccl.core.NcclDataType "nccl.core.NcclDataType") instances for use as the `dtype` argument of buffer specs.

Constant | Maps to  
---|---  
`nccl.core.INT8` | [`NcclDataType.INT8`](#nccl.core.NcclDataType.INT8 "nccl.core.NcclDataType.INT8")  
`nccl.core.CHAR` | [`NcclDataType.CHAR`](#nccl.core.NcclDataType.CHAR "nccl.core.NcclDataType.CHAR")  
`nccl.core.UINT8` | [`NcclDataType.UINT8`](#nccl.core.NcclDataType.UINT8 "nccl.core.NcclDataType.UINT8")  
`nccl.core.INT32` | [`NcclDataType.INT32`](#nccl.core.NcclDataType.INT32 "nccl.core.NcclDataType.INT32")  
`nccl.core.INT` | [`NcclDataType.INT`](#nccl.core.NcclDataType.INT "nccl.core.NcclDataType.INT")  
`nccl.core.UINT32` | [`NcclDataType.UINT32`](#nccl.core.NcclDataType.UINT32 "nccl.core.NcclDataType.UINT32")  
`nccl.core.INT64` | [`NcclDataType.INT64`](#nccl.core.NcclDataType.INT64 "nccl.core.NcclDataType.INT64")  
`nccl.core.UINT64` | [`NcclDataType.UINT64`](#nccl.core.NcclDataType.UINT64 "nccl.core.NcclDataType.UINT64")  
`nccl.core.FLOAT16` | [`NcclDataType.FLOAT16`](#nccl.core.NcclDataType.FLOAT16 "nccl.core.NcclDataType.FLOAT16")  
`nccl.core.HALF` | [`NcclDataType.HALF`](#nccl.core.NcclDataType.HALF "nccl.core.NcclDataType.HALF")  
`nccl.core.FLOAT32` | [`NcclDataType.FLOAT32`](#nccl.core.NcclDataType.FLOAT32 "nccl.core.NcclDataType.FLOAT32")  
`nccl.core.FLOAT` | [`NcclDataType.FLOAT`](#nccl.core.NcclDataType.FLOAT "nccl.core.NcclDataType.FLOAT")  
`nccl.core.FLOAT64` | [`NcclDataType.FLOAT64`](#nccl.core.NcclDataType.FLOAT64 "nccl.core.NcclDataType.FLOAT64")  
`nccl.core.DOUBLE` | [`NcclDataType.DOUBLE`](#nccl.core.NcclDataType.DOUBLE "nccl.core.NcclDataType.DOUBLE")  
`nccl.core.BFLOAT16` | [`NcclDataType.BFLOAT16`](#nccl.core.NcclDataType.BFLOAT16 "nccl.core.NcclDataType.BFLOAT16")  
`nccl.core.FLOAT8E4M3` | [`NcclDataType.FLOAT8E4M3`](#nccl.core.NcclDataType.FLOAT8E4M3 "nccl.core.NcclDataType.FLOAT8E4M3")  
`nccl.core.FLOAT8E5M2` | [`NcclDataType.FLOAT8E5M2`](#nccl.core.NcclDataType.FLOAT8E5M2 "nccl.core.NcclDataType.FLOAT8E5M2")  
  
## Reduction operator[](#reduction-operator "Permalink to this heading")

### NcclRedOp[](#ncclredop "Permalink to this heading")

_class _nccl.core.NcclRedOp(_value_ , _names= <not given>_, _*values_ , _module=None_ , _qualname=None_ , _type=None_ , _start=1_ , _boundary=None_)[](#nccl.core.NcclRedOp "Permalink to this definition")
    

Bases: `IntEnum`

NCCL reduction operator, mirroring [`ncclRedOp_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclRedOp_t "ncclRedOp_t").

Used as the `op` argument of reduction collectives ([`Communicator.allreduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#nccl.core.Communicator.allreduce "nccl.core.Communicator.allreduce"), [`Communicator.reduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#nccl.core.Communicator.reduce "nccl.core.Communicator.reduce"), [`Communicator.reduce_scatter()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#nccl.core.Communicator.reduce_scatter "nccl.core.Communicator.reduce_scatter")).

SUM _ = 0_[](#nccl.core.NcclRedOp.SUM "Permalink to this definition")
    

Element-wise sum (`+`).

PROD _ = 1_[](#nccl.core.NcclRedOp.PROD "Permalink to this definition")
    

Element-wise product (`*`).

MAX _ = 2_[](#nccl.core.NcclRedOp.MAX "Permalink to this definition")
    

Element-wise maximum.

MIN _ = 3_[](#nccl.core.NcclRedOp.MIN "Permalink to this definition")
    

Element-wise minimum.

AVG _ = 4_[](#nccl.core.NcclRedOp.AVG "Permalink to this definition")
    

Sum across all ranks divided by the number of ranks.

### Predefined reduction operators[](#predefined-reduction-operators "Permalink to this heading")

Module-level [`NcclRedOp`](#nccl.core.NcclRedOp "nccl.core.NcclRedOp") instances for use as the `op` argument of reduction collectives. User-defined operators are created via [`Communicator.create_pre_mul_sum()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/communicator/collectives.html#nccl.core.Communicator.create_pre_mul_sum "nccl.core.Communicator.create_pre_mul_sum").

Constant | Maps to  
---|---  
`nccl.core.SUM` | [`NcclRedOp.SUM`](#nccl.core.NcclRedOp.SUM "nccl.core.NcclRedOp.SUM")  
`nccl.core.PROD` | [`NcclRedOp.PROD`](#nccl.core.NcclRedOp.PROD "nccl.core.NcclRedOp.PROD")  
`nccl.core.MAX` | [`NcclRedOp.MAX`](#nccl.core.NcclRedOp.MAX "nccl.core.NcclRedOp.MAX")  
`nccl.core.MIN` | [`NcclRedOp.MIN`](#nccl.core.NcclRedOp.MIN "nccl.core.NcclRedOp.MIN")  
`nccl.core.AVG` | [`NcclRedOp.AVG`](#nccl.core.NcclRedOp.AVG "nccl.core.NcclRedOp.AVG")  
  
## Exceptions[](#exceptions "Permalink to this heading")

### NcclInvalid[](#ncclinvalid "Permalink to this heading")

Python-side validation exception, raised when a public API receives a malformed argument before it reaches NCCL itself.

_exception _nccl.core.NcclInvalid(_msg_)[](#nccl.core.NcclInvalid "Permalink to this definition")
    

Bases: `Exception`

Raised when an argument provided to an NCCL4Py API is invalid.

Used for argument validation errors that the Python layer detects before forwarding the call to NCCL (e.g. unsupported dtype, mismatched buffer counts, wrong device). Errors raised by NCCL itself are reported as NCCLError from the bindings layer.