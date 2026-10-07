#ifndef PMPP_COMMON_CUH
#define PMPP_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                             \
    do                                                                                               \
    {                                                                                                \
        cudaError_t err = (call);                                                                    \
        if (err != cudaSuccess)                                                                      \
        {                                                                                            \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                      \
        }                                                                                            \
    } while (0)

#define CUDA_CHECK_KERNEL()                                             \
    do                                                                  \
    {                                                                   \
        cudaError_t err = cudaGetLastError();                           \
        if (err != cudaSuccess)                                         \
        {                                                               \
            fprintf(stderr, "CUDA kernel launch error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

struct GpuTimer
{
    cudaEvent_t startEvent, stopEvent;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
    }
    ~GpuTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    void start() {
        CUDA_CHECK(cudaEventRecord(startEvent));
    }
    void stop() {
        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
    }
    float elapsedMs() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
        return ms;
    }
};

#include <chrono>
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() {
        t0 = std::chrono::high_resolution_clock::now();
    }
    double elapsedMs() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

inline void fillRandomFloat(float *arr, size_t n, unsigned int seed=42, float lo=0.0f, float hi=1.0f) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        arr[i] = lo + (hi - lo) * (static_cast<float>(rand()) / RAND_MAX);
    }
}

inline void fillRandomInt(int *arr, size_t n, unsigned int seed=42, int lo=0, int hi=100) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        arr[i] = lo + rand() % (hi - lo + 1);
    }
}

inline bool compareArrays(const float* a, const float *b, size_t n, float tol=1e-3f, bool verbose=true) {
    size_t mismatchCount = 0;
    float maxDiff = 0.0f;
    size_t maxDiffIdx = 0;

    for (size_t i = 0; i < n; i++) {
        float diff = fabs(a[i]-b[i]);
        if (diff > maxDiff) {maxDiff = diff; maxDiffIdx = i;}
        if (diff > tol) {mismatchCount++;}
    }
    if (verbose) {
        if (mismatchCount == 0) {
            printf("[校验通过] 最大误差 = %e (index %zu), 容差 = %e\n", maxDiff, maxDiffIdx, tol);
        } else {
            printf("[校验失败] 共有 %zu / %zu 个元素超出容差，最大误差 = %e (index %zu), 容差 = %e\n",
                    mismatchCount, n, maxDiff, maxDiffIdx, tol);
        }
    }
    return mismatchCount == 0;
}

inline void vectorAddCPU(const float* A, const float* B, float *C, size_t n) {
    for (size_t i = 0; i < n; i++) { C[i] = A[i] + B[i]; }
}

// M*K K*N => M*N
inline void matrixMulCPU(const float *A, const float* B, float *C, int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float total = 0.0f;
            for (int h = 0; h < K; h++) {
                total += A[row * K + h] * B[h * N + col];
            }
            C[row * N + col] = total;
        }
    }
}

inline void printfDeviceBrief() {
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("== 设备: %s (Compute Capability %d.%d) | SM数量: %d | 全局显存: %.2f GB ==\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

#endif