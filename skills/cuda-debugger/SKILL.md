---
name: cuda-debugger
description: Debugs CUDA kernel errors using compute-sanitizer (memcheck, racecheck, initcheck, synccheck), cuda-gdb (non-interactive batch backtrace), and cuobjdump (PTX/SASS/register analysis). Use when a CUDA kernel crashes (segfault, illegal memory access), produces wrong results, hangs, or when the user asks to debug or diagnose GPU kernel issues. Triggers on CUDA crash, segfault, cudaErrorIllegalAddress, invalid memory access, race condition, wrong kernel results, hanging GPU program, cuda-gdb, compute-sanitizer, cuobjdump, register usage, shared memory overflow.
---

# CUDA Debugger

Debugs CUDA kernel errors by selecting and executing the correct diagnostic tool, then parsing output to locate and explain the root cause.

All commands MUST be executed in the project **root directory**.

## Workflow

### Progress Tracking

Copy the following checklist and update it in real-time:

```text
Task Progress:
- [ ] Step 1: Gather Context (error type, input files, reproduction)
- [ ] Step 2: Compile with Debug Flags (-g -G -lineinfo)
- [ ] Step 3: Run Diagnostic Tool
- [ ] Step 4: Parse & Report Findings
```

---

### Step 1: Gather Context

Determine what tool chain is needed based on the user's description:

| Symptom                            | Primary Tool                   | Secondary              |
| ---------------------------------- | ------------------------------ | ---------------------- |
| Segfault, illegal memory access    | `compute-sanitizer --tool memcheck` | cuda-gdb backtrace |
| Wrong results, non-deterministic   | `compute-sanitizer --tool racecheck` | —                  |
| Garbage/uninitialized values       | `compute-sanitizer --tool initcheck` | —                  |
| Kernel hangs, deadlock             | `compute-sanitizer --tool synccheck` | cuda-gdb           |
| Unknown crash (no sanitizer clue)  | `cuda-gdb` batch               | —                      |
| Occupancy / register investigation | `cuobjdump -res-usage`         | —                      |
| Verify PTX/SASS                    | `cuobjdump -ptx` / `-sass`     | —                      |

If the user provides a `.cu` source file but no executable, compile it first (Step 2). If they provide an already-compiled binary, skip Step 2.

---

### Step 2: Compile with Debug Flags

For best diagnostic output, **always** compile with `-g -G -lineinfo`:

```bash
# Standalone executable
nvcc -g -G -lineinfo -O0 -arch=$(python3 -c "
import torch; cc = torch.cuda.get_device_capability()
print(f'sm_{cc[0]}{cc[1]}')
") -o <program> <source.cu> -lcuda

# Or with benchmark.py (pass extra nvcc flags)
python3 skills/kernel-benchmarker/scripts/benchmark.py <source.cu> \
    --force-recompile --compiler-flags="-g -G -lineinfo -O0"
```

Key debug flags:
- `-g`: Host-side debug symbols
- `-G`: Device-side debug symbols (enables source-level reporting)
- `-lineinfo`: Line number information in backtraces
- `-O0`: Disable optimizations (prevents variable elimination)

**If compilation fails**: report the error and stop. Do not proceed to diagnostics.

---

### Step 3: Run Diagnostic Tool

Execute the tool selected in Step 1, capturing output for parsing.

#### 3a. compute-sanitizer memcheck (default)

```bash
compute-sanitizer --show-backtrace yes --print-limit 20 ./<program> [args]
```

Common options:

```bash
# Log to file for large output
compute-sanitizer --log-file errors.txt --show-backtrace yes ./<program>

# Continue past first error (collect all)
compute-sanitizer --error-exitcode 0 --print-limit 50 ./<program>
```

#### 3b. compute-sanitizer racecheck

```bash
compute-sanitizer --tool racecheck --show-backtrace yes ./<program>
```

#### 3c. compute-sanitizer initcheck

```bash
compute-sanitizer --tool initcheck --track-unused-memory yes ./<program>
```

#### 3d. compute-sanitizer synccheck

```bash
compute-sanitizer --tool synccheck ./<program>
```

#### 3e. cuda-gdb batch mode

```bash
cuda-gdb -batch \
  -ex "set cuda memcheck on" \
  -ex "run" \
  -ex "bt" \
  -ex "info cuda threads" \
  -ex "info locals" \
  ./<program>
```

For crash at a specific kernel, add breakpoint before run:

```bash
cuda-gdb -batch \
  -ex "set cuda memcheck on" \
  -ex "break <kernel_name>" \
  -ex "run" \
  -ex "bt" \
  -ex "info cuda threads" \
  ./<program>
```

#### 3f. cuobjdump

```bash
# Register and shared memory usage
cuobjdump -res-usage ./<program>

# Dump SASS for a specific arch
cuobjdump -sass -arch <sm_XX> ./<program>

# List embedded architectures
cuobjdump -lelf ./<program>
```

---

### Step 4: Parse & Report Findings

Parse the tool output using these interpretation rules:

#### compute-sanitizer Output Interpretation

Key patterns to extract:

| Output Pattern                                      | Meaning                                                    | Action                                                       |
| --------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| `Misaligned address`                                | Pointer not aligned to type boundary                       | Check pointer arithmetic; use `__align__` or padding         |
| `Out of bounds` with shared memory                  | Shared memory allocation too small                         | Verify `extern __shared__` size vs. launch `sharedMem` bytes |
| `Invalid __global__ write` at calculated index      | Index computation overflow / negative index                | Add bounds-check guards in kernel                            |
| `WAR` hazard (racecheck)                            | Write-after-read: one thread writes before another reads   | Add `__syncthreads()` between phases                         |
| `RAW` hazard (racecheck)                            | Read-after-write: data read before producer writes         | Add `__syncthreads()` or ensure write-complete barrier       |
| `WAW` hazard (racecheck)                            | Write-after-write: two threads write same location         | Restructure to single-writer per location                    |
| Uninitialized device memory (initcheck)             | Kernel reads memory that was never written to              | Add `cudaMemset` after `cudaMalloc`; init local arrays       |
| `__syncthreads()` in divergent code (synccheck)     | Barrier inside conditional that not all threads reach       | Hoist `__syncthreads()` out of `if/else` blocks              |
| Thread IDs in errors (e.g., only threads ≥ 64 fail) | Warp-boundary issue                                        | Check warp-level logic; look for lane-id assumptions         |

For `compute-sanitizer --show-backtrace yes` output, extract:
- **Source file and line number**
- **Thread/block/warp coordinates** that triggered the error
- **Access type** (read/write) and **address**

#### Interpreting Thread Coordinates

```text
# Example: error at thread [64,0,0] block [2,0,0]
# → Only the second warp of block 2; suggests a warp-boundary issue
# Example: error at thread [0,0,0] block [0,0,0]  
# → Thread-zero failure; suggests logic error, not data-dependent
# Example: scattered blocks and threads
# → Data-dependent bug (out-of-bounds for some inputs)
```

#### cuda-gdb Output Interpretation

Extract the backtrace frame that mentions the `.cu` source file and line number. Ignore CUDA runtime internals in the backtrace — focus on the first frame from user code.

---

## Output Template

```markdown
## CUDA Debugging Report

### Tool Used
{compute-sanitizer --tool memcheck | racecheck | initcheck | synccheck | cuda-gdb | cuobjdump}

**Command**: `{exact command executed}`

### Error Summary
- **Type**: {memory error | race condition | uninitialized memory | sync violation | crash}
- **Location**: `{file.cu}:{line}` (or "unknown — recompile with -lineinfo")
- **Severity**: {CRITICAL | HIGH | MEDIUM}

### Root Cause
{1-2 sentence explanation of what went wrong}

### Source Location (if available)
```
{file.cu}:{line} — in kernel `{kernel_name}`
Thread [{x},{y},{z}], Block [{x},{y},{z}]
Access type: {read|write}, Address: {0x...}
```

### Fix Recommendation

1. **Immediate fix**: {specific code change to make at the reported line}
2. **Prevention**: {guard or pattern to prevent recurrence}
3. **Verification**: `compute-sanitizer` re-run command to confirm fix
```

---

## Reference: Debugging Knowledge Base

For detailed interpretation patterns, consult `../cuda-knowledge/references/debugging-tools.md`. Key sections:

| Section                    | When to Consult                                              |
| -------------------------- | ------------------------------------------------------------ |
| compute-sanitizer          | Parsing sanitizer output, understanding hazard types         |
| cuda-gdb                   | Batch mode patterns, breakpoint strategies                   |
| cuobjdump                  | Register/shared memory usage, PTX/SASS inspection            |
| Debugging Strategy §3      | Interpreting thread coordinates in error reports             |
| Debugging Strategy §5      | "Stare at the Diff" — when tools fail, minimize changeset    |
| Debugging Strategy §6      | Incremental testing: isolate components, use identity inputs |

---

## Common Pitfalls

1. **Forgetting `-lineinfo`**: Without it, backtraces show `Unknown` instead of source lines. Always compile with `-g -G -lineinfo` for debug builds.

2. **Optimization hiding variables**: `-O3` can eliminate variables you want to inspect. Use `-O0` for debugging, re-enable optimization after fixing.

3. **racecheck overhead**: Can slow execution 10-100×. Use on the smallest reproducer possible.

4. **Shared memory "out of bounds" ≠ wrong index**: Often the shared memory allocation itself is correct but the kernel launch didn't pass enough dynamic shared memory. Check the third `<<< >>>` parameter.

5. **Tools require a real GPU**: compute-sanitizer and cuda-gdb only work on machines with NVIDIA GPUs. If unavailable, use static analysis (code review, printf) as fallback.
