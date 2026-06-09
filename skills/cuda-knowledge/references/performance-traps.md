# Common Performance Traps in CUDA

_Lessons from real-world GPU optimization projects_

Plans help you start, but profiling reveals the real bottlenecks. The problems below are frequently discovered through systematic profiling but often missed in initial designs.

## Bank Conflicts in Shared Memory

### Symptoms

- ncu shows high bank conflict rate (e.g., "16-way bank conflicts")
- Most cycles stalled on shared memory operations
- Low effective bandwidth despite using shared memory

### Common Causes

- Strided access patterns that map multiple threads to the same bank
- Transposing data in shared memory without accounting for bank layout
- Writing in one pattern, reading in another (both can't be conflict-free simultaneously)

### Investigation

```bash
# Check bank conflicts and wavefronts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
    ./program

# Divide conflicts by wavefronts to get average conflicts per operation
# >1 per operation indicates conflicts
```

### Solutions

- Pad shared memory arrays (e.g., `[32][33]` instead of `[32][32]`)
- Change thread-to-data mapping to avoid stride patterns
- Optimize for the more frequent operation if both read and write can't be conflict-free
- For transpose operations, accept conflicts on one dimension

Can give 2-3× speedup when memory-bound.

## Memory Coalescing

### Symptoms

- ncu shows high sector/request ratio (e.g., "32 sectors/request" vs optimal 1-4)
- Low global memory throughput despite high demand
- Memory-bound kernel with poor bandwidth utilization

### Common Causes

- Strided access patterns (every thread reads every Nth element)
- Transposed access patterns (reading column-major when stored row-major)
- Unaligned access or indirection through index arrays

### Investigation

```bash
# Check memory transactions
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./program
# Divide sectors by requests: 1-4 is good, 8-16 is poor, 32+ is essentially random
```

### Solutions

- Use vectorized loads (`float4`, `uint4`) when threads access adjacent memory
- Structure of Arrays (SoA) over Array of Structures (AoS)
- Transpose in shared memory if global access must be strided
- Ensure proper alignment (128-byte for vector loads)

Can give 1.5-2× speedup for severe coalescing issues.

## Scale-Dependent Optimizations

### The Problem

**Optimization techniques that work at large scale can hurt at small scale, and vice versa.**

### Common Examples

- **Warp specialization:** Fixed setup cost only amortizes with large workloads
- **Async operations:** Only hide latency if you have compute to overlap
- **Advanced features (TMA, etc.):** Benefit at high utilization, overhead at low

### Rule

**Always profile at YOUR target scale.** "Best practices" from papers may not apply to your problem size.

### Questions to Ask

- This optimization helped at scale X. What's my scale?
- What's the overhead, and does my workload amortize it?
- Should I verify this applies before implementing?

## Documenting What Doesn't Work

**Document negative results to prevent retrying failed approaches:**

```markdown
## Attempted Optimizations

### Warp Specialization (Stage 9)

- Context: 64×64 tiles
- Result: Slower than baseline
- Reason: Setup overhead > benefit at small scale
- Decision: Don't retry until workload >128×128
```

This prevents loops where you try the same optimization again after losing context. Failed experiments are valuable knowledge if documented.

## GPU Hardware State Monitoring

Performance anomalies (unexpectedly low bandwidth, high variance between runs, results that don't match NCU predictions) are often caused by GPU hardware state, not kernel code defects. Always rule out environmental factors before chasing code-level optimizations.

### nvidia-smi Real-time Sampling

Run alongside your benchmark to capture GPU state during execution:

```bash
nvidia-smi -i 0 --query-gpu=timestamp,clocks.gr,clocks.mem,pstate,power.draw,\
pcie.link.gen.current,pcie.link.width.current,temperature.gpu,\
utilization.gpu,utilization.memory --format=csv -l 1 > gpu_monitor.csv
```

**Key indicators to check**:

| Metric                    | Normal                              | Red flag                             | What it means                                     |
| ------------------------- | ----------------------------------- | ------------------------------------ | ------------------------------------------------- |
| `clocks.gr`               | Max boost (GPU-dependent) | Below max by >10%                    | Clock throttling — check thermal/power            |
| `pstate`                  | P0                                  | P3, P8                               | GPU not in performance mode                       |
| `pcie.link.gen.current`   | Gen4 or Gen5                        | Gen1, Gen2, Gen3                     | PCIe link downgrade — reseat card or check BIOS   |
| `pcie.link.width.current` | x16                                 | x4, x8                               | PCIe lane failure — physical issue                |
| `power.draw`              | ~idle + expected delta              | At power limit                       | Power throttling — check `nvidia-smi -q -d POWER` |
| `temperature.gpu`         | <80°C under load                    | >85°C or rapidly climbing            | Thermal throttling imminent                       |
| `utilization.gpu`         | >80% for compute-heavy              | <30% paired with high power          | Kernel is stalled, not computing                  |

### Environment Isolation Checklist

When benchmark results don't match expectations, verify these in order:

1. **GPU clock state** — `nvidia-smi -q -d CLOCK | grep -A5 "Graphics"` — confirm P0 and max frequency
2. **Power limits** — `nvidia-smi -q -d POWER | grep -E "Power Limit|Current Power"` — check if throttling is active
3. **PCIe link** — `nvidia-smi -q -d PCIE | grep -E "Link (Gen|Width)"` — confirm expected generation and width
4. **GPU persistence mode** — `nvidia-smi -q | grep "Persistence Mode"` — should be Enabled for consistent behavior
5. **ECC status** — `nvidia-smi -q | grep -A2 "ECC Errors"` — non-zero counts indicate hardware issues
6. **Other GPU consumers** — `nvidia-smi` — confirm no other processes are using the GPU during benchmarking
7. **Container overhead** — If running in Docker, compare bare-metal vs container benchmarks; Docker runtime overhead is usually <1% for GPU kernels
8. **CPU governor** — `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq` — should be `performance`, not `powersave`
9. **NUMA affinity** — `nvidia-smi topo -m` — confirm GPU is on the correct NUMA node for CPU memory access

### Quick Diagnostic Script

```bash
#!/bin/bash
# GPU health check before benchmarking
echo "=== GPU Clock ==="
nvidia-smi -q -d CLOCK | grep -A3 "Graphics"
echo "=== Power ==="
nvidia-smi -q -d POWER | grep -E "Power Limit|Current Power"
echo "=== PCIe ==="
nvidia-smi -q -d PCIE | grep -E "Link (Gen|Width)"
echo "=== Thermal ==="
nvidia-smi -q -d TEMPERATURE | grep "GPU Current Temp"
echo "=== ECC ==="
nvidia-smi -q | grep -A3 "ECC Errors" | grep -v "N/A"
echo "=== Processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Run this before and during benchmark runs. Any anomaly in the output is a stronger candidate for the performance gap than a kernel code issue.

## Summary

1. **Profile first, always** — Intuition about bottlenecks is usually wrong
2. **Measure at your scale** — Advice from papers may not apply to your problem size
3. **One change at a time** — Compound changes make diagnosis impossible
4. **Document failures** — Prevent retrying what already failed
5. **Verify with metrics** — "Should work" ≠ "does work"

The profile → hypothesis → fix → verify loop is the core optimization methodology.
