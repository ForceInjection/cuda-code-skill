# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-skill monorepo for CUDA kernel development assistance, combining scraped NVIDIA documentation (offline RAG knowledge base) with Agent skills that form an automated profile→analyze→optimize loop for GPU kernels. Designed for AI IDE integration (Claude Code, Trae, Qoder). A parallel `AGENTS.md` provides equivalent guidance for Qoder (qoder.com).

## Commands

### Documentation Scraper

The scraper is a single `uv` script with inline PEP 723 dependencies — no virtualenv or `pip install` needed:

```bash
uv run nvidia_doc_sync/scrape_cuda_docs.py ptx
uv run nvidia_doc_sync/scrape_cuda_docs.py runtime
uv run nvidia_doc_sync/scrape_cuda_docs.py driver
uv run nvidia_doc_sync/scrape_cuda_docs.py math
uv run nvidia_doc_sync/scrape_cuda_docs.py cublas
uv run nvidia_doc_sync/scrape_cuda_docs.py nccl

# Re-run cleanup only (no re-download):
uv run nvidia_doc_sync/scrape_cuda_docs.py driver --skip-download

# Force re-download ignoring cache:
uv run nvidia_doc_sync/scrape_cuda_docs.py driver --force

# Custom output directory:
uv run nvidia_doc_sync/scrape_cuda_docs.py ptx --output-dir /path/to/output
```

### Kernel Benchmarking

```bash
# Validate + benchmark kernel against a Python reference
python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> \
    --ref=<ref.py> --M=4096 --N=4096 --K=4096 --repeat=20

# Benchmark only (no validation)
python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> --N=1000000 --repeat=20

# Override GPU arch detection (e.g., for cross-compilation)
python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> --N=1000000 --arch=sm_89

# Force recompilation (bypass PTX cache)
python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> --N=1000000 --force-recompile

# Enable GPU hardware state monitoring (runs nvidia-smi in parallel, warns on throttling)
python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> --N=1000000 --monitor

# Enable kernel-internal phase timing (compiles with -DKERNEL_PROFILE, prints per-phase clock64)
python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> --N=1000000 --profile-phases
```

Reference files must define `def reference(*, <tensors>, <dims>, **kwargs):` with optional module-level `atol`/`rtol`.

`benchmark.py` compiles kernels via `nvcc -ptx` and loads/launches them via CUDA Driver API (`cuLaunchKernel`). PTX files are **auto-cached** (`*.ptx` alongside the `.cu` source) — subsequent runs skip compilation. Delete the `.ptx` file to force recompilation.

### Kernel Internal Phase Timing (conditional profiling)

Kernels can use `%%clock64` PTX instructions with `#ifdef KERNEL_PROFILE` guards so the same `.cu` file works as both a zero-overhead production build and a per-phase timing diagnostic build:

```bash
# Production build — zero profiling overhead
nvcc -O3 -arch=sm_89 -o kernel solution.cu

# Diagnostic build — prints per-phase clock64 cycle counts (load/compute/store)
nvcc -O3 -arch=sm_89 -DKERNEL_PROFILE -o kernel_profile solution.cu
```

Or use `benchmark.py --profile-phases` to auto-compile with `-DKERNEL_PROFILE` and print per-block phase timing alongside the normal benchmark summary.

The separate `ncu_profile.py` script generates a self-contained C host program that compiles together with the kernel into a standalone executable — purpose-built for NCU profiling without subprocess interference.

### NCU Profiling

Two approaches exist. **Prefer `ncu_profile.py`** (self-contained executable) — it avoids NCU disconnection caused by `benchmark.py`'s `nvcc` subprocess. The wrapping approach only works for quick `--set launch` profiles.

**Recommended: Standalone executable via `ncu_profile.py`**

```bash
# Step 1: Build self-contained profiling executable
python3 skills/kernel-benchmarker/scripts/ncu_profile.py <solution.cu> \
    --N=1000000 --build-only

# Step 2: Profile directly with NCU (no subprocess)
# Default (--set launch): works everywhere, including containers, no PMU needed
ncu --kernel-name solve --launch-skip 10 --launch-count 1 \
    --set launch -o report -f ./<solution>_bench --N=1000000

# Full metrics (--set full): requires host PMU access (perf_event_paranoid=0)
ncu --kernel-name solve --launch-skip 10 --launch-count 1 \
    --set full -o report -f ./<solution>_bench --N=1000000 --warmup=10 --repeat=22
```

**Alternative: Wrap benchmark.py directly** (may disconnect due to nvcc subprocess)

```bash
ncu --target-processes all --profile-from-start on \
    --launch-skip 20 --launch-count 1 --set full \
    -o <output_stem> -f \
    python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> \
    --PARAM=VALUE --repeat=22
```

Sample NCU reports are available at `examples/ncu-profile/` (`.ncu-rep` binary + text export).

### Searching the Knowledge Base

The knowledge base has a **two-layer structure**:

| Layer               | Location                            | What it is                                                                                                                             |
| ------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Search guides**   | `cuda-knowledge/references/*.md`    | Top-level index files with grep patterns, TOC, and navigation aids — read these first to understand what's available and how to search |
| **Doc directories** | `cuda-knowledge/references/*-docs/` | The actual scraped markdown content organized by chapter/module                                                                        |

Search guides: `ptx-isa.md`, `cublas.md`, `cuda-runtime.md`, `cuda-driver.md`, `cuda-math.md`, `nccl.md`, `debugging-tools.md`, `ncu-guide.md`, `nsys-guide.md`, `nvtx-patterns.md`, `performance-traps.md`.

```bash
# cuBLAS
grep -r "cublasLtMatmul" skills/cuda-knowledge/references/cublas-docs/3-using-the-cublaslt-api/
grep -r "CUBLASLT_EPILOGUE_" skills/cuda-knowledge/references/cublas-docs/

# NCCL
grep -r "ncclAllReduce" skills/cuda-knowledge/references/nccl-docs/api/

# CUDA Math API
grep -r "__nv_fp8_e4m3\|__nv_fp8_e5m2" skills/cuda-knowledge/references/cuda-math-docs/
```

### Finding Code Patterns (cuda-samples)

```bash
# Find samples by pattern or API (search reference files for code)
grep -r "cublasSgemm\|cudaStreamBeginCapture\|__shfl_down_sync" skills/cuda-samples/references/

# Find samples by name or architecture in the quick reference index
grep -r "reduction\|GEMM\|CUDA Graph" skills/cuda-samples/SKILL.md

# Find samples requiring specific GPU architecture
grep -r "SM 8.0\|SM 9.0\|Hopper\|Ampere" skills/cuda-samples/SKILL.md

# Verify all sample paths exist in the submodule
uv run scripts/check_links.py

# Clone the submodule for full source access (shallow clone, ~220 MB)
git submodule update --init
```

### Validation

```bash
# Check cross-skill interface consistency
python3 scripts/check_skills.py

# Check all documented counts match the filesystem
python3 scripts/check_counts.py

# Verify cuda-samples skill paths against the submodule
# Requires: git submodule update --init (shallow clone, ~220 MB)
uv run scripts/check_links.py
```

## Architecture

### Skills Pipeline (skills/)

Six skills form a complete optimization loop, orchestrated by `cuda-optimizer`:

```text
cuda-knowledge (API reference docs)  +  cuda-samples (code pattern index)
         ↓                                    ↓
cuda-optimizer (orchestrator)
    ├── kernel-benchmarker   → compile, validate, benchmark
    ├── cuda-debugger        → crash diagnosis (compute-sanitizer, cuda-gdb, cuobjdump)
    ├── ncu-rep-analyzer     → NCU profile, diagnose bottleneck, suggest fixes
    └── cuda-code-generator  → generate/rewrite .cu files with optimizations
```

`cuda-knowledge` (~1420 markdown files) provides API reference based on CUDA Toolkit 13.3 (PTX ISA 9.3, cuBLAS 13.3, Runtime/Driver/Math API 13.3); `cuda-samples` (SKILL.md ~13 KB, 10 reference files 48 KB; ~50 curated entries) provides concrete working code patterns from official NVIDIA samples with GitHub permalinks and key snippets. Both follow progressive disclosure: lightweight SKILL.md stays in context, detailed references load on demand. All three action skills ground their work in both to reduce hallucination. The optimizer drives this loop: **benchmark → evaluate exit conditions → NCU profile → implement optimizations → repeat** until performance converges (<2% improvement over 2 consecutive rounds).

### Kernel Interface Convention

All `.cu` kernels follow a strict interface:

```cuda
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // ...
}
```

- Function name is always `solve`, with `extern "C"` linkage.
- Pointer parameters: `const` prefix means input; no `const` means output.
- Supported types: `float*`, `double*`, `int*`, `unsigned char*`, `unsigned short*`, `unsigned long long*` (for clock64 profiling), plus scalar int types.
- `benchmark.py` auto-parses this signature to infer dimension parameter names and allocate tensors.

### Scraper Design (nvidia_doc_sync/scrape_cuda_docs.py)

A single-file script with three scraper classes for different NVIDIA doc formats:

| Format             | Class                    | Examples                  | Strategy                                                                                                                                       |
| ------------------ | ------------------------ | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Sphinx single-page | `SphinxScraper`          | PTX ISA, cuBLAS           | Split monolithic HTML by heading hierarchy into per-section .md files                                                                          |
| Sphinx multi-page  | `SphinxMultiPageScraper` | NCCL                      | Crawl toctree links, convert each page                                                                                                         |
| Doxygen multi-page | `APIScraper`             | Runtime, Driver, Math API | Two-phase: download raw → clean (strip TOC, footer, boilerplate, URLs; 76-83% size reduction). `--skip-download` re-runs only the clean phase. |

### Directory Structure

```text
skills/
  cuda-knowledge/references/     # ~1420 .md files, CUDA Toolkit 13.3 (PTX ISA 9.3)
    *-docs/                      # Scraped documentation: ptx(679), cublas(354), runtime(104), driver(129), math(41), nccl(57)
    *.md                         # 11 search guides with grep patterns per API
  cuda-samples/
    SKILL.md                     # Quick reference table, optimization mapping, arch compatibility
    references/                  # 10 topic files with detailed code snippets & GitHub permalinks:
                                 #   getting-started, tensor-core-gemm, reduction-scan-sort,
                                 #   cuda-graphs, streams-async, cuda-libraries, framework-interop,
                                 #   multi-gpu, performance, advanced-topics
  cuda-optimizer/SKILL.md        # Orchestrator — drives the full optimization loop
  cuda-code-generator/SKILL.md   # Generates .cu files, must consult cuda-knowledge + cuda-samples
    references/
      cuda-optimization-strategies.md  # Bottleneck → strategy mapping (Block Tiling, Vectorized Load, etc.)
  ncu-rep-analyzer/SKILL.md      # NCU profiling + bottleneck classification + optimization suggestions
  cuda-debugger/SKILL.md         # Crash diagnosis: compute-sanitizer, cuda-gdb, cuobjdump
  kernel-benchmarker/SKILL.md    # Compile, validate, benchmark via benchmark.py
examples/
  vectorAdd/                     # Compilable kernel + reference (use as template)
  ncu-profile/                   # Sample .ncu-rep (binary) and text export
nvidia_doc_sync/                 # Documentation scraper and its README
scripts/                         # Validation scripts (check_skills, check_counts, check_links)
```

### Key Design Decisions

- **Skills are independent but chainable** — each skill can be invoked standalone or as part of the optimizer loop.
- **Optimizer never stops mid-loop** — after each sub-skill returns, the orchestrator immediately proceeds to the next step. The output of one sub-skill is the input for the next.
- **New kernel versions get timestamped filenames** — `solution_opt_20260316_153045.cu`, never overwrite the original.
- **Knowledge-grounding is mandatory** — code-generator and ncu-rep-analyzer must consult both `cuda-knowledge/references/` (API signatures) and `cuda-samples/SKILL.md` (working code patterns) before generating code or recommendations involving complex APIs (cuBLASLt, Tensor Core, FP8 types).
- **PTX-cached benchmarks** — `benchmark.py` caches compiled PTX files (`*.ptx`) beside the `.cu` source. Delete the `.ptx` to force recompilation when the kernel changes.
- **Permissions** — `.claude/settings.local.json` pre-approves `python3`, `uv run`, `git submodule`, `sshpass`, `rsync`, and `git` commands. If adding new automation scripts, register them there to avoid permission prompts.
