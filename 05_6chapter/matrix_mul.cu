#include "../utils/common.cuh"

__global__
void matrixMulNaiveKernel(const float *A, const float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float total = 0.0f;
        for (int i = 0; i < K; i++) {
            total += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = total;
    }
}

#define TILE_WIDTH 16

__global__
void matrixMulTiledKernel(const float *A, const float *B, float *C, int M, int K, int N) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float total = 0.0f;
    int tileNums = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int tileIdx = 0; tileIdx < tileNums; tileIdx++) {
        int colA = tileIdx * TILE_WIDTH + threadIdx.x;
        int rowB = tileIdx * TILE_WIDTH + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && colA < K) ? A[row * K + colA] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (rowB < K && col < M) ? B[rowB * N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {
            total += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = total;
}

void runExperiment(int M, int K, int N) {
    printf("\n-------Matrix Size: A(%dx%d) B(%dx%d) C(%dx%d)-------\n", M, K, K, N, M, N);
    
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *A_h = (float*)malloc(bytesA);
    float *B_h = (float*)malloc(bytesB);
    float *C_h_cpu = (float*)malloc(bytesC);
    float *C_h_naive = (float*)malloc(bytesC);
    float *C_h_tiled = (float*)malloc(bytesC);

    fillRandomFloat(A_h, M*K);
    fillRandomFloat(B_h, K*N);

    CpuTimer cpuTimer;
    cpuTimer.start();
    matrixMulCPU(A_h, B_h, C_h_cpu, M, K, N);
    float cpuMs = cpuTimer.elapsedMs();

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, bytesA);
    cudaMalloc(&B_d, bytesB);
    cudaMalloc(&C_d, bytesC);
    cudaMemcpy(A_d, A_h, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytesB, cudaMemcpyHostToDevice);

    dim3 gridSize(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH 
    );
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);

    GpuTimer timerNaive;
    timerNaive.start();
    matrixMulNaiveKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    timerNaive.stop();
    float naiveMs = timerNaive.elapsedMs();
    cudaMemcpy(C_h_naive, C_d, bytesC, cudaMemcpyDeviceToHost);

    GpuTimer timerTiled;
    timerTiled.start();
    matrixMulTiledKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, K, N);
    timerTiled.stop();
    float tiledMs = timerTiled.elapsedMs();
    cudaMemcpy(C_h_tiled, C_d, bytesC, cudaMemcpyDeviceToHost);

    bool naiveOk = compareArrays(C_h_cpu, C_h_naive, (size_t)M * N, 1e-2f, false);
    bool tiledOk = compareArrays(C_h_cpu, C_h_tiled, (size_t)M * N, 1e-2f, false);

    printf("CPU耗时:        %10.3f ms\n", cpuMs);
    printf("GPU Naive耗时:  %10.3f ms | 相对CPU加速: %7.2fx | 校验: %s\n",
           naiveMs, cpuMs / naiveMs, naiveOk ? "通过" : "失败");
    printf("GPU Tiled耗时:  %10.3f ms | 相对CPU加速: %7.2fx | 校验: %s\n",
           tiledMs, cpuMs / tiledMs, tiledOk ? "通过" : "失败");
    printf("Tiled相对Naive的加速比: %.2fx\n", naiveMs / tiledMs);

    free(A_h); free(B_h); free(C_h_cpu); free(C_h_naive); free(C_h_tiled);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

int main() {
    printDeviceBrief();
    printf("TILE_WIDTH = %d (可以在代码顶部修改这个值做对比实验)\n", TILE_WIDTH);

    // 规模建议: 先用小矩阵验证正确性(CPU baseline算得动)，
    // 再用大矩阵看naive和tiled的性能差异(CPU这一步会很慢，可以注释掉CPU部分单独测GPU)
    runExperiment(256, 256, 256);      // 小矩阵: 验证正确性
    runExperiment(1024, 1024, 1024);   // 中等矩阵: 比较贴近实际应用规模
    runExperiment(1000, 1000, 1000);   // 非TILE_WIDTH整除的尺寸: 测试边界条件处理
    runExperiment(1024, 1052, 1024);

    // 如果想测更大规模(比如4096x4096)，建议把CPU baseline那几行注释掉，
    // 因为CPU版本O(n^3)复杂度，矩阵一大CPU会跑很久，而且你的RTX 3050只有4GB显存，
    // 4096x4096的三个float矩阵大约需要 3 * 4096 * 4096 * 4 bytes ≈ 200MB，是安全的，
    // 但如果继续往上加规模要留意显存是否够用。

    printf("\n建议实验方向:\n");
    printf("  1. 改TILE_WIDTH(8/16/32)，看对性能的影响，想想为什么不是越大越好\n");
    printf("  2. 用 ncu 分析naive和tiled两个kernel的 global memory带宽利用率差异\n");
    printf("  3. 用 ncu 看shared memory bank conflict情况，尝试故意构造一个有conflict的访问模式再优化掉\n");
    return 0;
}