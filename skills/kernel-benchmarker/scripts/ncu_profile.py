#!/usr/bin/env python3
"""Build a self-contained profiling executable for a CUDA kernel.

Parses the kernel signature from the .cu file and generates a C host wrapper
with matching parameter types (including FP16/BF16). Produces a standalone
executable suitable for direct NCU profiling — no subprocess, no Python
dependency at profile time.

Usage:
    # Build
    python3 ncu_profile.py solution.cu --M=4096 --N=4096 --K=4096 --build-only

    # Profile with NCU
    ncu --kernel-name solve --launch-skip 10 --launch-count 1 --set full \
        -o report -f ./solution_bench --M=4096 --N=4096 --K=4096 \
        --warmup=10 --repeat=22
"""

import argparse
import os
import re
import subprocess
import sys


# ---------------------------------------------------------------------------
# Type tables (mirrors benchmark.py to keep ncu_profile.py self-contained)
# ---------------------------------------------------------------------------

# C type name, sizeof(), optional extra #include
TYPE_TO_C = {
    "float*":          ("float",          4,  ""),
    "double*":         ("double",         8,  ""),
    "half*":           ("__half",         2,  "#include <cuda_fp16.h>"),
    "__half*":         ("__half",         2,  "#include <cuda_fp16.h>"),
    "__nv_bfloat16*":  ("__nv_bfloat16",  2,  "#include <cuda_bf16.h>"),
    "int*":            ("int",            4,  ""),
    "long*":           ("long",           8,  ""),
    "short*":          ("short",          2,  ""),
    "char*":           ("char",           1,  ""),
    "unsigned char*":  ("unsigned char",  1,  ""),
    "unsigned short*": ("unsigned short", 2,  ""),
    "unsigned int*":   ("unsigned int",   4,  ""),
    "unsigned long long*": ("unsigned long long", 8, ""),
}

SCALAR_C_TYPES = {
    "int":           "int",
    "long":          "long",
    "size_t":        "size_t",
    "unsigned int":  "unsigned int",
    "unsigned short":"unsigned short",
    "unsigned char": "unsigned char",
    "char":          "char",
    "short":         "short",
}

SUPPORTED_TYPES = {**{k: True for k in TYPE_TO_C}, **{k: True for k in SCALAR_C_TYPES}}


def parse_solve_signature(cu_file: str):
    """Extract parameter list from extern \"C\" ... void solve(...) in a .cu file."""
    with open(cu_file, "r") as f:
        content = f.read()

    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//[^\n]*', '', content)

    pattern = r'extern\s+"C"\s+(?:__global__\s+)?void\s+solve\s*\(([^#]*?)\)[\s\S]*?\{'
    m = re.search(pattern, content)
    if not m:
        raise ValueError(f"Cannot find 'extern \"C\" void solve(...)' in {cu_file}")

    raw = m.group(1)
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
            m2 = re.match(rf"({base})\s+(\w+)", token_clean)
            if m2:
                params.append((key, m2.group(2), is_const))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse parameter: '{token}' in signature of {cu_file}")
    return params


def _ensure_global(cu_file: str) -> str:
    """If solve lacks __global__, return path to a temp copy with __global__ added."""
    with open(cu_file, "r") as f:
        src = f.read()

    if re.search(r'extern\s+"C"\s+__global__\s+void\s+solve', src):
        return cu_file

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


def detect_arch() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            mj, mn = torch.cuda.get_device_capability(0)
            return f"sm_{mj}{mn}"
    except ImportError:
        pass
    return "sm_80"


# ---------------------------------------------------------------------------
# C wrapper generator
# ---------------------------------------------------------------------------

def _indent(text: str, n: int = 4) -> str:
    prefix = " " * n
    return prefix + text.replace("\n", "\n" + prefix)


def _gen_alloc(ptr_params: list, dim_var: str) -> str:
    """Generate cudaMalloc calls for each pointer param."""
    lines = []
    for ptype, pname, _ in ptr_params:
        ctype, elsize, _inc = TYPE_TO_C[ptype]
        lines.append(f"size_t bytes_{pname} = (size_t){dim_var} * {elsize};")
        lines.append(f"{ctype} *d_{pname};")
        lines.append(f"cudaMalloc(&d_{pname}, bytes_{pname});")
    return "\n    ".join(lines)


def _gen_init(ptr_params: list, dim_var: str) -> str:
    """Generate buffer initialization: non-zero pattern for inputs, zero for outputs."""
    lines = []
    filler_byte = 0xCD  # deterministic non-zero

    for ptype, pname, is_const in ptr_params:
        if is_const:
            lines.append(f"cudaMemset(d_{pname}, {filler_byte}, bytes_{pname});")
        else:
            lines.append(f"cudaMemset(d_{pname}, 0, bytes_{pname});")
    return "\n    ".join(lines)


def _gen_kernel_args(params: list) -> str:
    """Generate argument list for solve() call."""
    args = []
    for ptype, pname, _ in params:
        if "*" in ptype:
            args.append(f"d_{pname}")
        else:
            args.append(pname)
    return ", ".join(args)


def _gen_arg_parser(params: list) -> str:
    """Generate CLI argument parsing for scalar params."""
    lines = ["int warmup = 10, repeat = 22;"]
    # Set defaults from scalar params with N as primary
    for ptype, pname, _ in params:
        if "*" not in ptype:
            default = 1000000 if pname == "N" else 4096
            lines.append(f"int {pname} = {default};")

    lines.append("")
    lines.append("for (int i = 1; i < argc; i++) {")
    for ptype, pname, _ in params:
        if "*" not in ptype:
            lines.append(f'    if (strncmp(argv[i], "--{pname}=", {len(pname)+3}) == 0) '
                         f'{pname} = atoi(argv[i] + {len(pname)+3});')
    lines.append('    if (strncmp(argv[i], "--warmup=", 9) == 0) warmup = atoi(argv[i] + 9);')
    lines.append('    if (strncmp(argv[i], "--repeat=", 9) == 0) repeat = atoi(argv[i] + 9);')
    lines.append("}")
    return "\n    ".join(lines)


def _gen_grid_dims(params: list) -> tuple:
    """Determine total element count and grid/block dims from first INT param."""
    total = 256
    for ptype, pname, _ in params:
        if "*" not in ptype:
            total = pname
            break
    return f"int threads = 256, blocks = ({total} + 255) / 256;"


def generate_wrapper(cu_file_abs: str, params: list) -> str:
    """Generate self-contained C host wrapper from parsed kernel params."""
    ptr_params = [(t, n, c) for t, n, c in params if "*" in t]
    scalar_params = [(t, n, c) for t, n, c in params if "*" not in t]

    # Collect extra includes for half-precision types
    extra_includes = set()
    for ptype, _, _ in ptr_params:
        _, _, inc = TYPE_TO_C.get(ptype, ("", 0, ""))
        if inc:
            extra_includes.add(inc)

    # Primary dimension variable (first scalar param)
    dim_var = scalar_params[0][1] if scalar_params else "N"

    lines = []
    lines.append("// Auto-generated NCU profiling bench — self-contained, no subprocess.")
    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>")
    lines.append("#include <string.h>")
    lines.append("#include <cuda_runtime.h>")
    for inc in sorted(extra_includes):
        lines.append(inc)
    lines.append("")
    lines.append(f'// User kernel source')
    lines.append(f'#include "{cu_file_abs}"')
    lines.append("")
    lines.append("int main(int argc, char **argv) {")

    # Arg parsing
    lines.append(_indent(_gen_arg_parser(params)))
    lines.append("")

    # Grid/block setup
    lines.append(_indent(_gen_grid_dims(params)))
    lines.append("")

    # Allocate
    lines.append(_indent("// Allocate device buffers"))
    lines.append(_indent(_gen_alloc(ptr_params, dim_var)))
    lines.append("")

    # Initialize
    lines.append(_indent("// Initialize: non-zero pattern for inputs, zero for outputs"))
    lines.append(_indent(_gen_init(ptr_params, dim_var)))
    lines.append("")
    lines.append(_indent("cudaDeviceSynchronize();"))
    lines.append("")

    # Warmup
    lines.append(_indent("// Warmup"))
    lines.append(_indent("for (int i = 0; i < warmup; i++)"))
    lines.append(_indent(f"solve<<<blocks, threads>>>({_gen_kernel_args(params)});", 8))
    lines.append(_indent("cudaDeviceSynchronize();"))
    lines.append("")

    # Timed iterations
    lines.append(_indent("// Timed iterations"))
    lines.append(_indent("cudaEvent_t start, stop;"))
    lines.append(_indent("cudaEventCreate(&start); cudaEventCreate(&stop);"))
    lines.append(_indent("double total = 0.0;"))
    lines.append(_indent("for (int i = 0; i < repeat; i++) {"))
    lines.append(_indent("cudaEventRecord(start, 0);", 8))
    lines.append(_indent(f"solve<<<blocks, threads>>>({_gen_kernel_args(params)});", 8))
    lines.append(_indent("cudaEventRecord(stop, 0);", 8))
    lines.append(_indent("cudaEventSynchronize(stop);", 8))
    lines.append(_indent("float ms; cudaEventElapsedTime(&ms, start, stop);", 8))
    lines.append(_indent("total += ms;", 8))
    lines.append(_indent("if (i < 2 || i >= repeat - 2) printf(\"  iter %d: %.4f ms\\n\", i, ms);", 8))
    lines.append(_indent("}"))
    lines.append(_indent("printf(\"  avg: %.4f ms\\n\", total / repeat);"))
    lines.append("")

    # Cleanup
    lines.append(_indent("// Cleanup"))
    lines.append(_indent("cudaEventDestroy(start); cudaEventDestroy(stop);"))
    for _, pname, _ in ptr_params:
        lines.append(_indent(f"cudaFree(d_{pname});"))
    lines.append(_indent('printf("done\\n");'))
    lines.append(_indent("return 0;"))
    lines.append("}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cu_file")
    parser.add_argument("--arch", type=str, default="")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=22)

    args, unknown = parser.parse_known_args()
    dims = {}
    for u in unknown:
        if u.startswith("--") and "=" in u:
            k, v = u[2:].split("=", 1)
            try:
                dims[k] = int(v)
            except ValueError:
                pass

    # Preprocess: add __global__ if missing (some kernels omit it)
    cu_file = _ensure_global(args.cu_file)
    cu_abs = os.path.abspath(cu_file)
    arch = args.arch or detect_arch()

    # Parse kernel signature (from original file since _ensure_global
    # only adds __global__, doesn't change the parameter list)
    try:
        params = parse_solve_signature(args.cu_file)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    scalar_params = [(t, n) for t, n, _ in params if "*" not in t]
    ptr_params = [(t, n, c) for t, n, c in params if "*" in t]

    print(f"[signature] solve(" + ", ".join(
        f"{'const ' if c else ''}{t} {n}" for t, n, c in params) + ")")
    print(f"  pointer params: {len(ptr_params)}")
    print(f"  scalar params:  {len(scalar_params)}")

    # Ensure all required dims are provided or use defaults
    for _, pname in scalar_params:
        if pname not in dims and pname != "warmup" and pname != "repeat":
            dims[pname] = 1000000 if pname == "N" else 4096

    # Generate and write wrapper
    wrapper_src = generate_wrapper(cu_abs, params)
    exe_path = os.path.splitext(args.cu_file)[0] + "_bench"
    wrapper_path = exe_path + "_wrapper.cu"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_src)

    cmd = ["nvcc", f"-arch={arch}", "-O3", "-lineinfo", "-o", exe_path, wrapper_path]
    print(f"[compile] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(wrapper_path)
    # Clean up temp file from _ensure_global if one was created
    if cu_file != args.cu_file and os.path.exists(cu_file):
        os.remove(cu_file)

    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[compile] -> {exe_path}", flush=True)

    if args.build_only:
        dim_args = " ".join(f"--{n}={dims[n]}" for _, n in scalar_params)
        print(f"\nReady for NCU:\n"
              f"  ncu --kernel-name solve --launch-skip {args.warmup} "
              f"--launch-count 1 --set launch -o report.ncu-rep -f "
              f"{exe_path} {dim_args} --warmup={args.warmup} "
              f"--repeat={args.repeat}\n"
              f"\n"
              f"  # For detailed metrics (requires host PMU access: "
              f"perf_event_paranoid=0):\n"
              f"  # ncu ... --set full ...")
        return

    exe_args = [exe_path] + [f"--{n}={dims[n]}" for _, n in scalar_params] + \
               [f"--warmup={args.warmup}", f"--repeat={args.repeat}"]
    subprocess.run(exe_args)


if __name__ == "__main__":
    main()
