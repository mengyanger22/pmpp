#include <cuda_runtime.h>
#include <cstdio>

__global__ void vectorAddKernel(const float *A, const float *B, float *C, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

#include "../utils/common.cuh"
/*
void runExperiment(size_t n) {
    printf("\n------- data size N = %zu ----\n", n);
    size_t bytes = n * sizeof(float);
    float* A_h = (float*) malloc(bytes);
    float* B_h = (float*) malloc(bytes);
    float* C_h_cpu = (float*) malloc(bytes);
    float* C_h_gpu = (float*) malloc(bytes);
    
    fillRandomFloat(A_h, n);
    fillRandomFloat(B_h, n);

    CpuTimer cpuTimer;
    cpuTimer.start();
    vectorAddCPU(A_h, B_h, C_h_cpu, n);
    double cpuMs = cpuTimer.elapsedMs();

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    cudaMalloc(&C_d, bytes);

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);

    GpuTimer gpuTimer;
    gpuTimer.start();
    vectorAddKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, n);
    gpuTimer.stop();
    float gpuMs = gpuTimer.elapsedMs();

    cudaMemcpy(C_h_gpu, C_d, bytes, cudaMemcpyDeviceToHost);
    bool ok = compareArrays(C_h_cpu, C_h_gpu, n, 1e-4f, false);
    printf("CPU time: %.4f ms | GPU time: %.4f ms| SpeedUp: %.2fx | checkResult: %s\n",
            cpuMs, gpuMs, cpuMs / gpuMs, ok ? "pass" : "failed");
    free(A_h); free(B_h); free(C_h_cpu); free(C_h_gpu);
    cudaFree(A_d); cudaFree(B_h); cudaFree(C_d);
}
*/
void runExperiment(size_t n) {
    printf("\n ------------ data size N = %zu -------------\n", n);
    size_t bytes = n * sizeof(float);
    
    float* A_h = (float*)malloc(bytes);
    float* B_h = (float*)malloc(bytes);
    float* C_h_cpu = (float*)malloc(bytes);
    float* C_h_gpu = (float*)malloc(bytes);

    fillRandomFloat(A_h, n);
    fillRandomFloat(B_h, n);

    CpuTimer cpuTimer;
    cpuTimer.start();
    vectorAddCPU(A_h, B_h, C_h_cpu, n);
    double cpuMs = cpuTimer.elapsedMs();

    float* A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    GpuTimer gpuTimer;
    gpuTimer.start();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAddKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, n);
    gpuTimer.stop();
    double gpuMs = gpuTimer.elapsedMs();

    cudaMemcpy(C_h_gpu, C_d, bytes, cudaMemcpyDeviceToHost);
    bool ok = compareArrays(C_h_cpu, C_h_gpu, n, 1e-4f, false);
    printf("CPU time: %.4f ms | GPU time: %.4f ms | SpeedUp: %.2fx | Check: %s\n", cpuMs, gpuMs, cpuMs/gpuMs, ok ? "pass" : "failed");
    free(A_h); free(B_h); free(C_h_cpu); free(C_h_gpu);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

int main() {
    printDeviceBrief();
    size_t sizes[] = {1000, 10000, 1000000, 10000000, 50000000, 100000000, 500000000*2};
    for (size_t size : sizes) {
        runExperiment(size);
    }
    return 0;
}