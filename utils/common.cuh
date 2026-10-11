#ifndef PMPP_COMMON_CUH

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                   \
    do                                                                                                     \
    {                                                                                                      \
        cudaError_t err = (call);                                                                          \
        if (err != cudaSuccess)                                                                            \
        {                                                                                                  \
            fprintf(stderr, "CUDA error at: %s:%d --- %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                            \
        }                                                                                                  \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                                                      \
    do                                                                                                           \
    {                                                                                                            \
        cudaError_t err = cudaGetLastError();                                                                    \
        if (err != cudaSuccess)                                                                                  \
        {                                                                                                        \
            fprintf(stderr, "CUDA kernel error at %s:%d --- %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                  \
        }                                                                                                        \
    } while (0)

struct GpuTimer {
    cudaEvent_t startEvent, stopEvent;
    GpuTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }
    void start() {
        cudaEventRecord(startEvent);
    }
    void stop() {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
    }
    float elapsedMs() {
        float ms;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        return ms;
    }
};

#include <chrono>
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() {
        t0 = std::chrono::high_resolution_clock::now();
    }
    float elapsedMs() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count(); 
    }
};

inline void fillRandomFloat(float* arr, size_t n, unsigned int seed = 42, float lo = 0.0f, float hi = 1.0f) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        arr[i] = lo + (hi - lo) * (static_cast<float>(random()) / RAND_MAX);
    }
}

inline void fillRandomInt(int* arr, size_t n, unsigned int seed = 42, int lo = 0, int hi = 100) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        arr[i] = lo + random() % (hi - lo + 1);
    }
}

inline bool compareArrays(const float* a, const float* b, size_t n, float tol = 1e-3f, bool verbose = true) {
    float maxDiff = 0.0f;
    size_t mismatchCount = 0;
    size_t maxDiffIdx = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = abs(a[i] - b[i]);
        if (diff > tol) { mismatchCount++; }
        if (diff > maxDiff) { 
            maxDiff = diff;
            maxDiffIdx = i;
        }
    }
    if (verbose) {
        if (mismatchCount == 0) {
            printf("[arrays compare pass] maxDiff = %e (idx %zu), allowed diff %e\n", maxDiff, maxDiffIdx, tol);
        }
        else {
            printf("[arrays compare failed] %zu / %zu elements exceed the allowed diff, max diff = %e (idx %zu)\n", mismatchCount, n, maxDiff, maxDiffIdx);
        }
    }
    return mismatchCount == 0;
}

inline void vectorAddCPU(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) { c[i] = a[i] + b[i]; }
}

// MK * KN => MN
inline void matrixMulCPU(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float tol = 0.0f;
            for (int h = 0; h < K; h++) {
                tol += A[row * K + h] * B[h * N + col];
            }
            C[row * N + col] = tol;
        }
    }
}

inline void printDeviceBrief() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("== device: %s (compute capability %d.%d) | SM: %d | GlobalMemory: %.2f(GB)==\n", 
            prop.name, prop.major, prop.minor, prop.multiProcessorCount, 
            prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

}

#define PMPP_COMMON_CUH
#endif