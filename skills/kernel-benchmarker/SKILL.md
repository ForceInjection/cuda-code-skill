---
name: kernel-benchmarker
description: Compiles, validates, and benchmarks a CUDA kernel (.cu file) against a Python reference (*_ref.py). Auto-detects GPU arch and infers dimension args from the `extern "C" void solve(...)` signature. Runs benchmark.py to: compile with nvcc, optionally validate outputs (exits on failure), benchmark both kernel and reference with CUDA-event timing, and print a latency/bandwidth/speedup summary. Use when the user wants to validate or benchmark a .cu file.
---

# Kernel Benchmarker

Executes correctness validation and performance benchmarking on a CUDA kernel, then summarizes the results.

## Execution Workflow

All commands MUST be executed in the project **root directory**.

### Progress Tracking

Copy the following checklist and update it in real-time:

```text
Task Progress:
- [ ] Step 1: Correctness Validation + Performance Benchmarking (benchmark.py)
- [ ] Step 2: Summarize Output Results
```

---

### Step 1: Correctness Validation + Performance Benchmarking

If `--ref` is provided, `benchmark.py` will perform correctness validation first. **If it fails, it will exit immediately (non-zero exit code)**. If it passes, it will benchmark both the reference and the kernel, then print a summary.

```bash
python3 skills/kernel-benchmarker/scripts/benchmark.py <cu_file> \
    --ref=<ref_file> [--PARAM=VALUE ...] --repeat=20
```

**Example** (Matrix Transpose):

```bash
python3 skills/kernel-benchmarker/scripts/benchmark.py kernel/MatrixTranspose/solution.cu \
    --ref=kernel/MatrixTranspose/transpose_ref.py --M=10000 --N=1000 --repeat=20
```

- If validation **fails** (non-zero exit code or output contains `FAIL`), stop subsequent steps.
  - **MANDATORY REQUIREMENT**: If the failure reason is a VRAM error like Segfault or Illegal Memory Access, you MUST consult the `../cuda-knowledge/references/debugging-tools.md` workflow, and consider executing `compute-sanitizer --tool memcheck` or `cuda-gdb` to obtain the exact error line number before providing feedback and suggestions to the user.
- If validation **passes** (`ALL PASS ✓`), proceed to Step 2.

---

## Parameter Inference Rules

| Parameter                      | Inference Method                                                                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<cu_file>`                    | The `.cu` file path provided by the user.                                                                                                                           |
| `<ref_file>`                   | Provided by the user; if unspecified, look for `*_ref.py` (e.g., `matmul_ref.py`, `vector_add_ref.py`, `transpose_ref.py`) in the same directory as the `.cu` file. |
| Dimension Params (`--M`, etc.) | Infer parameter names from the `extern "C" void solve(...)` signature; if unspecified, use reasonable defaults (MatMul: M=K=N=4096, Vector Add: N=1000000).         |
| `--repeat`                     | Default is 20.                                                                                                                                                      |

---

## Step 2: Summarize Output

```markdown
## Kernel Validation Report

### Basic Information

- **Kernel File**: `<cu_file>`
- **Reference Implementation**: `<ref_file>`
- **Dimension Parameters**: M=..., N=... (etc.)
- **GPU**: <GPU name>

### 1. Correctness Validation

- **Result**: ✅ ALL PASS / ❌ FAILED
- (If failed, attach error message)

### 2. Performance Benchmarking

| Metric     | Kernel       | Reference  |
| ---------- | ------------ | ---------- |
| Average    | X.XXXX ms    | X.XXXX ms  |
| Median     | X.XXXX ms    | X.XXXX ms  |
| Min        | X.XXXX ms    | X.XXXX ms  |
| Max        | X.XXXX ms    | X.XXXX ms  |
| ~Bandwidth | XX.XX GB/s   | XX.XX GB/s |
| Speedup    | X.XXx vs ref | —          |
```
