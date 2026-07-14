#!/usr/bin/env python3
"""Generic CUDA kernel benchmark (with optional correctness validation).

When --ref is provided:
  1. Validates kernel correctness against the reference implementation.
     Exits immediately if validation fails.
  2. Benchmarks the reference implementation.
  3. Benchmarks the CUDA kernel.
  4. Prints a combined summary with speedup.

When --ref is omitted:
  Benchmarks the CUDA kernel only.

The CUDA kernel is compiled to PTX and loaded via the CUDA Driver API
(cuModuleLoadData + cuLaunchKernel) in the same process, avoiding
cross-process CUDA memory isolation.

Usage:
    python benchmark.py <solution.cu> [--DIM=VALUE ...] [options]
    python benchmark.py <solution.cu> --ref=<ref.py> [--DIM=VALUE ...] [options]

ref.py format
-------------
    import torch

    def reference(*, A, B, C, M, K, N, **kwargs):
        C[:] = (A.reshape(M, K) @ B.reshape(K, N)).reshape(-1)

    # Optional tolerance overrides
    atol = 1e-4
    rtol = 1e-3
"""

import argparse
import csv
import ctypes
import importlib.util
import io
import os
import re
import subprocess
import sys
import threading
import time

import torch

# ---------------------------------------------------------------------------
# Type tables
# ---------------------------------------------------------------------------

SUPPORTED_TYPES = {
    "float*":          ("float*",          ctypes.c_void_p),
    "double*":         ("double*",         ctypes.c_void_p),
    "unsigned char*":  ("unsigned char*",  ctypes.c_void_p),
    "unsigned short*": ("unsigned short*", ctypes.c_void_p),
    "unsigned int*":   ("unsigned int*",   ctypes.c_void_p),
    "unsigned long long*": ("unsigned long long*", ctypes.c_void_p),
    "char*":           ("char*",           ctypes.c_void_p),
    "short*":          ("short*",          ctypes.c_void_p),
    "long*":           ("long*",           ctypes.c_void_p),
    "int*":            ("int*",            ctypes.c_void_p),
    "int":             ("int",             ctypes.c_int),
    "long":            ("long",            ctypes.c_long),
    "size_t":          ("size_t",          ctypes.c_size_t),
    "unsigned int":    ("unsigned int",    ctypes.c_uint),
    "unsigned short":  ("unsigned short",  ctypes.c_ushort),
    "unsigned char":   ("unsigned char",   ctypes.c_ubyte),
    "char":            ("char",            ctypes.c_char),
    "short":           ("short",           ctypes.c_short),
}

DTYPE_MAP = {
    "float*":          torch.float32,
    "double*":         torch.float64,
    "int*":            torch.int32,
    "long*":           torch.int64,
    "short*":          torch.int16,
    "char*":           torch.int8,
    "unsigned char*":  torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*":   getattr(torch, "uint32", torch.int32),
}

# Pointer types that are NOT torch tensors — allocated as pinned host memory.
# Used for profiling output buffers (e.g., clock64 phase timing arrays).
RAW_POINTER_TYPES = {"unsigned long long*"}

INT_TYPES = {"int", "long", "size_t", "unsigned int"}


# ---------------------------------------------------------------------------
# GPU hardware state monitor (background nvidia-smi sampler)
# ---------------------------------------------------------------------------

class GPUMonitor:
    """Background nvidia-smi sampler that captures GPU state during benchmarks.

    Usage:
        mon = GPUMonitor(gpu_id=0)
        mon.start()
        # ... run benchmark ...
        csv_data = mon.stop()
        for w in GPUMonitor.check_issues(csv_data):
            print(f"[gpu-monitor] WARNING: {w}")
    """

    QUERY = (
        "timestamp,clocks.gr,clocks.mem,pstate,power.draw,"
        "pcie.link.gen.current,pcie.link.width.current,temperature.gpu,"
        "utilization.gpu,utilization.memory"
    )

    def __init__(self, gpu_id: int = 0, interval: int = 1):
        self.gpu_id = gpu_id
        self.interval = interval
        self._process = None

    def start(self):
        """Launch background nvidia-smi sampling."""
        cmd = [
            "nvidia-smi", "-i", str(self.gpu_id),
            "--query-gpu=" + self.QUERY,
            "--format=csv", "-l", str(self.interval),
        ]
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except FileNotFoundError:
            print("[gpu-monitor] nvidia-smi not found — skipping hardware monitoring",
                  file=sys.stderr)

    def stop(self) -> str:
        """Stop sampling and return collected CSV text."""
        if not self._process:
            return ""
        try:
            self._process.terminate()
            stdout, _ = self._process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            stdout, _ = self._process.communicate()
        self._process = None
        return stdout or ""

    @staticmethod
    def check_issues(csv_text: str) -> list:
        """Parse nvidia-smi CSV output and return a list of human-readable
        warning strings for any detected anomalies (throttling, PCIe downgrade,
        thermal issues, etc.).  Returns empty list if all clear."""
        if not csv_text.strip():
            return ["nvidia-smi produced no output — unable to verify GPU state"]

        try:
            reader = csv.DictReader(io.StringIO(csv_text))
            rows = [r for r in reader]
        except Exception:
            return ["Could not parse nvidia-smi CSV output"]

        if not rows:
            return ["nvidia-smi produced no data rows"]

        warnings = []
        seen = {}

        for r in rows:
            # nvidia-smi CSV pads column names with spaces; try both forms
            def _val(*keys):
                for k in keys:
                    v = r.get(k, "").strip()
                    if v:
                        return v
                return ""

            # --- pstate ---
            ps = _val("pstate", " pstate")
            if ps and ps not in ("P0", "") and "pstate" not in seen:
                seen["pstate"] = ps
                warnings.append(
                    f"GPU pstate = {ps} (expected P0) — clock may be throttled"
                )

            # --- PCIe link generation ---
            gen = _val("pcie.link.gen.current", " pcie.link.gen.current")
            if gen and gen not in ("4", "5", "") and "pcie_gen" not in seen:
                seen["pcie_gen"] = gen
                warnings.append(
                    f"PCIe link gen = {gen} (expected 4 or 5) — possible downgrade"
                )

            # --- PCIe link width ---
            width = _val("pcie.link.width.current", " pcie.link.width.current")
            if width and width not in ("16", "") and "pcie_width" not in seen:
                seen["pcie_width"] = width
                warnings.append(
                    f"PCIe link width = x{width} (expected x16) — possible lane failure"
                )

            # --- thermal ---
            temp_str = _val("temperature.gpu", " temperature.gpu")
            try:
                temp = float(temp_str)
                if temp > 85:
                    warnings.append(
                        f"GPU temperature = {temp}°C — thermal throttling risk"
                    )
            except (ValueError, TypeError):
                pass

        return warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_solve_signature(cu_file: str):
    """Extract parameter list from `extern "C" ... void solve(...)` in a .cu file."""
    with open(cu_file, "r") as f:
        content = f.read()

    # Strip C-style comments to avoid matching signatures in comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//[^\n]*', '', content)

    # The capture group [^#]*? prevents matching across #ifdef/#else/#endif
    # boundaries.  When no preprocessor guards are present this is equivalent
    # to [\s\S]*?.
    pattern = r'extern\s+"C"\s+(?:__global__\s+)?void\s+solve\s*\(([^#]*?)\)[\s\S]*?\{'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(
            f'Cannot find \'extern "C" void solve(...)\' in {cu_file}'
        )

    raw = match.group(1)
    raw = re.sub(r"/\*.*?\*/", "", raw)
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())

    params = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        is_const = "const" in token
        token_clean = re.sub(r"\s+", " ", token.replace("const", "").strip())
        matched = False
        for key in sorted(SUPPORTED_TYPES.keys(), key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            m = re.match(rf"({base})\s+(\w+)", token_clean)
            if m:
                params.append((key, m.group(2), is_const))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse parameter: '{token.strip()}'")

    return params


def detect_arch() -> str:
    """Auto-detect GPU compute capability and return sm_XX string."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    return "sm_80"


_STRIP_INCLUDES = re.compile(
    r'^\s*#\s*include\s*<__clang_cuda[^>]*>\s*$', re.MULTILINE
)


def _preprocess_cu(cu_file: str) -> str:
    """Strip clang-specific includes. Returns path to clean file."""
    with open(cu_file, "r") as f:
        src = f.read()
    cleaned = _STRIP_INCLUDES.sub("", src)
    if cleaned == src:
        return cu_file
    tmp = cu_file + ".nvcc_clean.cu"
    with open(tmp, "w") as f:
        f.write(cleaned)
    return tmp


def _ensure_global(cu_file: str) -> str:
    """If the solve function lacks __global__, rewrite with __global__ added."""
    with open(cu_file, "r") as f:
        src = f.read()

    # If __global__ is already there, return as-is
    if re.search(r'extern\s+"C"\s+__global__\s+void\s+solve', src):
        return cu_file

    # Add __global__ before void solve
    new_src = re.sub(
        r'(extern\s+"C"\s+)void\s+solve',
        r'\1__global__ void solve',
        src
    )
    if new_src == src:
        return cu_file

    tmp = cu_file + ".global.cu"
    with open(tmp, "w") as f:
        f.write(new_src)
    return tmp


# ---------------------------------------------------------------------------
# Compile .cu → load with torch
# ---------------------------------------------------------------------------

def _compile_and_load(cu_file: str, arch: str, force_recompile: bool = False,
                      extra_nvcc_flags: list = None):
    """Compile .cu → .ptx, load with PyTorch, return kernel launcher.

    Caches the PTX file: if the .ptx exists and is newer than the .cu source,
    skips recompilation (unless force_recompile=True).  This avoids spawning
    nvcc subprocesses, which is essential for NCU profiling since NCU
    disconnects when child processes exit.

    extra_nvcc_flags: optional list of additional flags passed to nvcc
        (e.g. ['-DKERNEL_PROFILE']).
    """
    extra_nvcc_flags = extra_nvcc_flags or []
    ptx_file = os.path.splitext(cu_file)[0] + ".ptx"

    need_compile = force_recompile or not os.path.exists(ptx_file)
    if not need_compile:
        ptx_mtime = os.path.getmtime(ptx_file)
        cu_mtime = os.path.getmtime(cu_file)
        need_compile = cu_mtime > ptx_mtime

    if need_compile:
        clean_file = _preprocess_cu(cu_file)
        global_file = _ensure_global(clean_file)

        cmd = ["nvcc", "-ptx", f"-arch={arch}", "-O3"] + extra_nvcc_flags + \
              ["-o", ptx_file, global_file]
        print(f"[compile] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        for tmp in [clean_file, global_file]:
            if tmp != cu_file and os.path.exists(tmp):
                os.remove(tmp)

        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(f"[compile] -> {ptx_file}")
    else:
        print(f"[compile] using cached {ptx_file}")

    # Load PTX with PyTorch
    with open(ptx_file, "r") as f:
        ptx_src = f.read()

    # Create a CUDA module from PTX
    # Use the low-level ctypes approach to load PTX
    import ctypes
    libnames = ("libcuda.so", "libcuda.so.1", "/usr/local/cuda/lib64/libcuda.so",
                "/usr/local/cuda-13/lib64/libcuda.so", "/usr/local/cuda-12/lib64/libcuda.so")
    cuda = None
    for name in libnames:
        try:
            cuda = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if cuda is None:
        raise RuntimeError("Cannot load libcuda.so")

    # CUDA Driver API types
    CUresult = ctypes.c_int
    CUDA_SUCCESS = 0

    cuda.cuInit(0)
    cuda.cuCtxGetCurrent.restype = CUresult

    # Get current context
    ctx = ctypes.c_void_p()
    err = cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetCurrent failed: {err}")

    # Load module from PTX
    module = ctypes.c_void_p()
    ptx_bytes = ptx_src.encode("utf-8")
    err = cuda.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(ptx_bytes))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleLoadData failed: {err}")

    # Get kernel function
    kernel = ctypes.c_void_p()
    err = cuda.cuModuleGetFunction(ctypes.byref(kernel), module, b"solve")
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleGetFunction failed: {err} (err={err})")

    return cuda, module, kernel, ptx_file


def _launch_kernel(cuda, kernel, params: list, dim_values: dict,
                   kernel_tensors: dict, raw_pointers: dict = None):
    """Launch the kernel via CUDA Driver API with the given tensors.

    raw_pointers: dict mapping param name → torch.Tensor (pinned CPU memory)
        for RAW_POINTER_TYPES parameters. Passed in parameter order.
    """
    raw_pointers = raw_pointers or {}
    CUDA_SUCCESS = 0

    # Build kernel arguments in function-signature order
    args = []
    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            args.append(ctypes.c_void_p(kernel_tensors[pname].data_ptr()))
        elif ptype in RAW_POINTER_TYPES:
            args.append(ctypes.c_void_p(raw_pointers[pname].data_ptr()))
        elif ptype in INT_TYPES:
            val = dim_values[pname]
            if ptype in ("int", "unsigned int"):
                args.append(ctypes.c_int(val))
            elif ptype in ("long", "size_t"):
                args.append(ctypes.c_long(val))
            else:
                args.append(ctypes.c_int(val))

    args_array = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])

    # Determine grid/block dims from the total element count
    total = 256  # default
    for ptype, pname, _ in params:
        if ptype in INT_TYPES:
            total = dim_values[pname]
            break

    threads = 256
    blocks = (total + threads - 1) // threads

    err = cuda.cuLaunchKernel(
        kernel,
        blocks, 1, 1,        # grid dims
        threads, 1, 1,       # block dims
        0,                    # shared memory bytes
        ctypes.c_void_p(0),  # stream (0 = default)
        args_array,
        None                  # extra options
    )
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {err}")


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def load_reference(ref_file: str):
    """Import a Python reference file and return its module."""
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Reference file not found: {ref_file}")
    spec = importlib.util.spec_from_file_location("_ref_module", ref_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "reference"):
        raise AttributeError(
            f"'{ref_file}' must define a `reference(**kwargs)` function."
        )
    return mod


def _determine_ptr_elems(int_values: list, ptr_size_override: int) -> int:
    if ptr_size_override > 0:
        return ptr_size_override
    elif len(int_values) == 0:
        return 1024 * 1024
    elif len(int_values) == 1:
        return int_values[0]
    else:
        sv = sorted(int_values, reverse=True)
        return min(sv[0] * sv[1], 256 * 1024 * 1024)


def _fmt_vals(vals, width=10):
    return "[" + ", ".join(f"{v:>{width}.4f}" for v in vals) + "]"


def _color(text: str, ok: bool) -> str:
    if not sys.stdout.isatty():
        return text
    code = "\033[92m" if ok else "\033[91m"
    return f"{code}{text}\033[0m"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_iterations(fn, warmup: int, repeat: int) -> list:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / repeat
    return [avg_ms] * repeat


def _stats(times_ms: list):
    avg = sum(times_ms) / len(times_ms)
    med = sorted(times_ms)[len(times_ms) // 2]
    return avg, med, min(times_ms), max(times_ms)


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def _print_results(label, avg, med, mn, mx, total_ptr_bytes, ptr_elems,
                   cu_file, dim_values, arch, ref_avg=None):
    print()
    print("=" * 55)
    print(f"  {label}")
    print(f"  Kernel       : {os.path.basename(cu_file)}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Arch         : {arch}")
    print(f"  Dims         : {dim_values}")
    print(f"  Buf/ptr      : {ptr_elems} elems")
    print("-" * 55)
    print(f"  Average      : {avg:>10.4f} ms")
    print(f"  Median       : {med:>10.4f} ms")
    print(f"  Min          : {mn:>10.4f} ms")
    print(f"  Max          : {mx:>10.4f} ms")
    if avg > 0:
        bw = total_ptr_bytes / (avg / 1000) / 1e9
        print(f"  ~Bandwidth   : {bw:>10.2f} GB/s  (all ptrs, rough)")
    if ref_avg is not None and avg > 0:
        speedup = ref_avg / avg
        print(f"  Speedup      : {speedup:>10.2f}x  vs reference")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_outputs(kernel_tensors, ref_tensors, output_params, atol, rtol):
    PREVIEW = 8
    print(f"\n[validate] {len(output_params)} output tensor(s)\n")

    all_pass = True
    for pname, ptype in output_params:
        kt = kernel_tensors[pname].float()
        rt = ref_tensors[pname].float()

        match = torch.allclose(kt, rt, atol=atol, rtol=rtol)
        if not match:
            all_pass = False

        max_diff  = (kt - rt).abs().max().item()
        mean_diff = (kt - rt).abs().mean().item()
        rel_err   = ((kt - rt).abs() / rt.abs().clamp(min=1e-8)).mean().item()

        status_str = _color("PASS" if match else "FAIL", match)
        print(f"  [{status_str}]  {pname}  ({ptype})")
        print(f"         max |Δ|   = {max_diff:.6e}")
        print(f"         mean |Δ|  = {mean_diff:.6e}")
        print(f"         mean rel  = {rel_err:.6e}")

        if not match:
            diff_mask = ~torch.isclose(kt, rt, atol=atol, rtol=rtol)
            bad_idx   = diff_mask.nonzero(as_tuple=True)[0]
            n_bad     = bad_idx.numel()
            print(f"         mismatches: {n_bad} / {kt.numel()}")
            if n_bad > 0:
                idx = bad_idx[0].item()
                print(f"         first bad   @ idx={idx}:  "
                      f"kernel={kt[idx].item():.6f}  ref={rt[idx].item():.6f}")

        k_preview = kernel_tensors[pname][:PREVIEW].float().cpu().tolist()
        r_preview = ref_tensors[pname][:PREVIEW].float().cpu().tolist()
        print(f"         kernel[:{PREVIEW}] = {_fmt_vals(k_preview)}")
        print(f"         ref   [:{PREVIEW}] = {_fmt_vals(r_preview)}")
        print()

    return all_pass


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _setup(cu_file, dim_values, ptr_size_override, arch, seed=None,
           force_recompile=False, extra_nvcc_flags=None,
           profile_phases=False):
    params = parse_solve_signature(cu_file)
    # When not profiling, strip raw-pointer params (they only exist under
    # #ifdef KERNEL_PROFILE guards).  When profiling, keep them.
    if not profile_phases:
        params = [(t, n, c) for t, n, c in params if t not in RAW_POINTER_TYPES]
    sig_str = ", ".join(f"{'const ' if c else ''}{t} {n}" for t, n, c in params)
    print(f"[signature] solve({sig_str})\n")

    cuda, module, kernel, ptx_file = _compile_and_load(
        cu_file, arch, force_recompile=force_recompile,
        extra_nvcc_flags=extra_nvcc_flags,
    )

    for ptype, pname, _ in params:
        if ptype in INT_TYPES and pname not in dim_values:
            raise ValueError(
                f"Missing dimension: --{pname}=<value>  (required by kernel signature)"
            )

    int_vals = [dim_values[pname]
                for ptype, pname, _ in params if ptype in INT_TYPES]
    ptr_elems = _determine_ptr_elems(int_vals, ptr_size_override)

    if seed is not None:
        torch.manual_seed(seed)

    kernel_tensors: dict = {}
    raw_pointers: dict = {}        # pinned host memory for profiling buffers
    output_params = []

    # Pre-compute number of CUDA blocks for profiling buffer allocation.
    # Uses the same heuristic as _launch_kernel: first INT param / 256.
    _total = 256
    for ptype, pname, _ in params:
        if ptype in INT_TYPES:
            _total = dim_values[pname]
            break
    _num_blocks = (_total + 255) // 256

    print("[buffers]")
    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            dtype = DTYPE_MAP[ptype]
            if dtype.is_floating_point:
                t = torch.randn(ptr_elems, device="cuda", dtype=dtype)
            else:
                t = torch.zeros(ptr_elems, device="cuda", dtype=dtype).random_()
            kernel_tensors[pname] = t
            if not is_const:
                output_params.append((pname, ptype))
            role = "input" if is_const else "output"
            eb   = t.element_size()
            print(
                f"  {pname:>10s} : {ptype:<16s} [{role:>6s}] "
                f"{ptr_elems} elems  ({ptr_elems * eb / 1024 / 1024:.1f} MB)"
            )
        elif ptype in RAW_POINTER_TYPES:
            # Profiling output buffer — allocate pinned host memory.
            # One slot per block × 4 phases (load, compute, store, total).
            num_entries = _num_blocks * 4
            t = torch.zeros(num_entries, dtype=torch.int64, device="cpu")
            t = t.pin_memory()
            raw_pointers[pname] = t
            kb = t.element_size() * t.numel() / 1024
            print(
                f"  {pname:>10s} : {ptype:<16s} [output] "
                f"{t.numel()} elems ({_num_blocks} blocks × 4 phases, "
                f"{kb:.1f} KB, pinned host)"
            )
        elif ptype in SUPPORTED_TYPES:
            val = dim_values[pname]
            print(f"  {pname:>10s} : {ptype:<16s} = {val}")

    total_ptr_bytes = sum(t.nelement() * t.element_size()
                          for t in kernel_tensors.values())

    return (cuda, kernel, params, kernel_tensors, raw_pointers, output_params,
            ptr_elems, total_ptr_bytes)


def _print_phase_timing(raw_pointers: dict, params: list, dim_values: dict):
    """Print per-block phase timing breakdown from clock64 profiling output.

    Expects raw_pointers to contain an `unsigned long long*` parameter whose
    pinned tensor holds per-block [load, compute, store, total] cycle counts.
    """
    for ptype, pname, _ in params:
        if ptype not in RAW_POINTER_TYPES:
            continue
        if pname not in raw_pointers:
            continue
        data = raw_pointers[pname].numpy()
        allocated_blocks = len(data) // 4
        if allocated_blocks == 0:
            print(f"\n[phase-timing] {pname}: no data (0 blocks)")
            continue

        # Auto-detect active blocks: find the last block with non-zero total.
        n_blocks = allocated_blocks
        for b in range(allocated_blocks - 1, -1, -1):
            if data[b * 4 + 3] != 0:
                n_blocks = b + 1
                break
        if n_blocks == 0:
            print(f"\n[phase-timing] {pname}: all blocks have zero total cycles")
            continue

        cyc_per_ns = 1.98  # GPU-dependent (Hopper ~1.98, Ampere ~1.41 GHz)
        print(f"\n[phase-timing] {pname}  ({n_blocks} active blocks, "
              f"{allocated_blocks} allocated, clock64 cycles)")
        print("-" * 72)
        print(f"{'Block':>6s}  {'load':>12s}  {'compute':>12s}  "
              f"{'store':>12s}  {'total':>12s}")
        print("-" * 72)

        load_sum = comp_sum = store_sum = total_sum = 0
        for b in range(min(n_blocks, 32)):  # show first 32 blocks
            lc = int(data[b * 4 + 0])
            cc = int(data[b * 4 + 1])
            sc = int(data[b * 4 + 2])
            tc = int(data[b * 4 + 3])
            load_sum += lc; comp_sum += cc; store_sum += sc; total_sum += tc
            print(f"{b:>6d}  {lc:>12,d}  {cc:>12,d}  {sc:>12,d}  {tc:>12,d}")

        if n_blocks > 32:
            # accumulate remaining in summary
            for b in range(32, n_blocks):
                load_sum += int(data[b * 4 + 0])
                comp_sum += int(data[b * 4 + 1])
                store_sum += int(data[b * 4 + 2])
                total_sum += int(data[b * 4 + 3])
            print(f"  ... ({n_blocks - 32} more blocks omitted)")

        print("-" * 72)
        def _us(cyc):
            return cyc / (cyc_per_ns * 1000.0)
        avg_load = load_sum / n_blocks
        avg_comp = comp_sum / n_blocks
        avg_store = store_sum / n_blocks
        avg_total = total_sum / n_blocks
        print(f"{'avg':>6s}  {avg_load:>12.1f}  {avg_comp:>12.1f}  "
              f"{avg_store:>12.1f}  {avg_total:>12.1f}")
        print(f"{'avg(us)':>6s}  {_us(avg_load):>12.2f}  "
              f"{_us(avg_comp):>12.2f}  {_us(avg_store):>12.2f}  "
              f"{_us(avg_total):>12.2f}")
        pct_load = load_sum / max(total_sum, 1) * 100
        pct_comp = comp_sum / max(total_sum, 1) * 100
        pct_store = store_sum / max(total_sum, 1) * 100
        print(f"{'%':>6s}  {pct_load:>11.1f}%  {pct_comp:>11.1f}%  "
              f"{pct_store:>11.1f}%")
        print("-" * 72)
        if pct_load > 60:
            print("  → Load-dominated — consider DRAM/L1 bandwidth optimizations")
        elif pct_comp > 60:
            print("  → Compute-dominated — consider ILP / Tensor Core optimizations")
        elif pct_store > 40:
            print("  → Store-heavy — check for uncoalesced writes")
        else:
            print("  → Balanced pipeline")


def run(cu_file, ref_file, dim_values, warmup, repeat,
        ptr_size_override, arch, atol, rtol, seed,
        force_recompile=False, monitor=False, profile_phases=False):
    has_ref = bool(ref_file)

    # --- profile_phases → compile with -DKERNEL_PROFILE -----------------------
    extra_nvcc_flags = []
    if profile_phases:
        extra_nvcc_flags.append("-DKERNEL_PROFILE")
        print("[profile-phases] compiling with -DKERNEL_PROFILE")

    # --- GPU hardware monitor -------------------------------------------------
    gpu_mon = None
    if monitor:
        gpu_mon = GPUMonitor(gpu_id=torch.cuda.current_device())
        gpu_mon.start()
        print("[gpu-monitor] started background nvidia-smi sampling")

    ref_fn = None
    ref_kwargs = None
    _atol = atol
    _rtol = rtol

    if has_ref:
        ref_mod = load_reference(ref_file)
        ref_fn  = ref_mod.reference
        _atol   = float(getattr(ref_mod, "atol", atol))
        _rtol   = float(getattr(ref_mod, "rtol", rtol))
        print(f"[reference] {ref_file}  (atol={_atol}, rtol={_rtol})\n")

    # -- compile + allocate ---------------------------------------------------
    (cuda, kernel, params, kernel_tensors, raw_pointers, output_params,
     ptr_elems, total_ptr_bytes) = _setup(
        cu_file, dim_values, ptr_size_override, arch,
        seed=seed if has_ref else None, force_recompile=force_recompile,
        extra_nvcc_flags=extra_nvcc_flags,
        profile_phases=profile_phases,
    )

    # Only tensor output params count for validation; raw pointers are
    # profiling buffers, not validation targets.
    tensor_output_params = [(n, t) for n, t in output_params if t not in RAW_POINTER_TYPES]
    if not tensor_output_params and has_ref:
        print("\n[warn] No output tensors detected (all pointer params are const). "
              "Nothing to validate.", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Step 1: correctness check
    # -------------------------------------------------------------------------
    if has_ref:
        ref_tensors = {pname: t.clone() for pname, t in kernel_tensors.items()}

        ref_kwargs = {}
        for ptype, pname, _ in params:
            if ptype in DTYPE_MAP:
                ref_kwargs[pname] = ref_tensors[pname]
            else:
                ref_kwargs[pname] = dim_values[pname]

        print("\n[kernel]    running ... ", end="", flush=True)
        _launch_kernel(cuda, kernel, params, dim_values,
                       kernel_tensors, raw_pointers)
        torch.cuda.synchronize()
        print("done")

        print("[reference] running ... ", end="", flush=True)
        ref_fn(**ref_kwargs)
        torch.cuda.synchronize()
        print("done")

        validation_passed = _validate_outputs(
            kernel_tensors, ref_tensors, tensor_output_params, _atol, _rtol
        )

        print("=" * 60)
        print(f"  Kernel    : {os.path.basename(cu_file)}")
        print(f"  Reference : {os.path.basename(ref_file)}")
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        print(f"  Arch      : {arch}")
        print(f"  Dims      : {dim_values}")
        print(f"  Buf/ptr   : {ptr_elems} elems")
        print(f"  Tolerance : atol={_atol}  rtol={_rtol}")
        print("-" * 60)
        result_str = "ALL PASS ✓" if validation_passed else "FAILED ✗"
        print(f"  Result    : {_color(result_str, validation_passed)}")
        print("=" * 60)

        if not validation_passed:
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 2: benchmark reference
    # -------------------------------------------------------------------------
    times_ref = None
    if has_ref:
        print(f"\n[warmup] reference  {warmup} iterations ...")
        times_ref = _time_iterations(lambda: ref_fn(**ref_kwargs), warmup, repeat)
        print(f"[bench]  reference  {repeat} iterations ... done")

    # -------------------------------------------------------------------------
    # Step 3: benchmark kernel (with optional phase profiling on last iter)
    # -------------------------------------------------------------------------
    if not has_ref:
        PREVIEW = 8
        tensor_info = [
            (pname, ptype, "input" if is_const else "output", kernel_tensors[pname])
            for ptype, pname, is_const in params if ptype in DTYPE_MAP
        ]
        print(f"\n[preview] first {PREVIEW} elements before kernel call:")
        for name, ptype, role, t in tensor_info:
            tag = "IN " if role == "input" else "OUT"
            print(f"  {tag} {name:>6s} = {_fmt_vals(t[:PREVIEW].cpu().tolist())}")

        _launch_kernel(cuda, kernel, params, dim_values,
                       kernel_tensors, raw_pointers)
        torch.cuda.synchronize()

        print(f"\n[preview] first {PREVIEW} elements after 1 kernel call:")
        for name, ptype, role, t in tensor_info:
            tag = "IN " if role == "input" else "OUT"
            print(f"  {tag} {name:>6s} = {_fmt_vals(t[:PREVIEW].cpu().tolist())}")

    print(f"\n[warmup] kernel  {warmup} iterations ...")
    for _ in range(warmup):
        _launch_kernel(cuda, kernel, params, dim_values,
                       kernel_tensors, raw_pointers)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        _launch_kernel(cuda, kernel, params, dim_values,
                       kernel_tensors, raw_pointers)
    end_event.record()
    torch.cuda.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / repeat
    times_kernel = [avg_ms] * repeat
    print(f"[bench]  kernel  {repeat} iterations ... done")

    # -------------------------------------------------------------------------
    # Step 4: print phase timing (if profiling enabled)
    # -------------------------------------------------------------------------
    if profile_phases and raw_pointers:
        _print_phase_timing(raw_pointers, params, dim_values)

    # -------------------------------------------------------------------------
    # Step 5: stop GPU monitor and report
    # -------------------------------------------------------------------------
    if gpu_mon:
        csv_data = gpu_mon.stop()
        print("[gpu-monitor] stopped")
        warnings = GPUMonitor.check_issues(csv_data)
        if warnings:
            print("\n[gpu-monitor] ⚠️  GPU STATE WARNINGS:")
            for w in warnings:
                print(f"  • {w}")
        else:
            print("[gpu-monitor] ✓ GPU state normal — no throttling or degradation")

    # -------------------------------------------------------------------------
    # Step 6: print summary
    # -------------------------------------------------------------------------
    avg_k, med_k, mn_k, mx_k = _stats(times_kernel)

    if has_ref:
        avg_r, med_r, mn_r, mx_r = _stats(times_ref)
        _print_results(
            "CUDA Kernel", avg_k, med_k, mn_k, mx_k,
            total_ptr_bytes, ptr_elems, cu_file, dim_values, arch, ref_avg=avg_r,
        )
        _print_results(
            f"Reference ({os.path.basename(ref_file)})",
            avg_r, med_r, mn_r, mx_r,
            total_ptr_bytes, ptr_elems, cu_file, dim_values, arch,
        )
    else:
        _print_results(
            "CUDA Kernel", avg_k, med_k, mn_k, mx_k,
            total_ptr_bytes, ptr_elems, cu_file, dim_values, arch,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generic CUDA kernel benchmark (with optional validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cu_file", help="Path to .cu solution file")
    parser.add_argument("--ref", type=str, default="",
                        help="Path to reference .py file")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--ptr-size", type=int, default=0)
    parser.add_argument("--arch", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompile", action="store_true",
                        help="Force PTX recompilation even if cached")
    parser.add_argument("--monitor", action="store_true",
                        help="Run nvidia-smi in background during benchmark "
                             "and report GPU state anomalies")
    parser.add_argument("--profile-phases", action="store_true",
                        help="Compile with -DKERNEL_PROFILE and print per-phase "
                             "clock64 timing after benchmark.  Requires the .cu "
                             "kernel to have #ifdef KERNEL_PROFILE guards with "
                             "an `unsigned long long*` output parameter.")

    args, unknown = parser.parse_known_args()

    dim_values: dict = {}
    for u in unknown:
        if u.startswith("--") and "=" in u:
            key, val = u[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{u}'", file=sys.stderr)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else detect_arch()

    run(
        cu_file           = args.cu_file,
        ref_file          = args.ref,
        dim_values        = dim_values,
        warmup            = args.warmup,
        repeat            = args.repeat,
        ptr_size_override = args.ptr_size,
        arch              = arch,
        atol              = args.atol,
        rtol              = args.rtol,
        seed              = args.seed,
        force_recompile   = args.force_recompile,
        monitor           = args.monitor,
        profile_phases    = args.profile_phases,
    )


if __name__ == "__main__":
    main()
