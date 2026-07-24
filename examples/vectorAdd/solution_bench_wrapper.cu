// Auto-generated NCU profiling bench — self-contained, no subprocess.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// User kernel source
#include "/Users/wangtianqing/Project/skills/cuda-code-skill/examples/vectorAdd/solution.cu"

int main(int argc, char **argv) {
    int warmup = 10, repeat = 22;
        int N = 1000000;
        
        for (int i = 1; i < argc; i++) {
            if (strncmp(argv[i], "--N=", 4) == 0) N = atoi(argv[i] + 4);
            if (strncmp(argv[i], "--warmup=", 9) == 0) warmup = atoi(argv[i] + 9);
            if (strncmp(argv[i], "--repeat=", 9) == 0) repeat = atoi(argv[i] + 9);
        }

    int threads = 256, blocks = (N + 255) / 256;

    // Allocate device buffers
    size_t bytes_A = (size_t)N * 4;
        float *d_A;
        cudaMalloc(&d_A, bytes_A);
        size_t bytes_B = (size_t)N * 4;
        float *d_B;
        cudaMalloc(&d_B, bytes_B);
        size_t bytes_C = (size_t)N * 4;
        float *d_C;
        cudaMalloc(&d_C, bytes_C);

    // Initialize: non-zero pattern for inputs, zero for outputs
    cudaMemset(d_A, 205, bytes_A);
        cudaMemset(d_B, 205, bytes_B);
        cudaMemset(d_C, 0, bytes_C);

    cudaDeviceSynchronize();

    // Warmup
    for (int i = 0; i < warmup; i++)
        solve<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    double total = 0.0;
    for (int i = 0; i < repeat; i++) {
        cudaEventRecord(start, 0);
        solve<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total += ms;
        if (i < 2 || i >= repeat - 2) printf("  iter %d: %.4f ms\n", i, ms);
    }
    printf("  avg: %.4f ms\n", total / repeat);

    // Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    printf("done\n");
    return 0;
}
