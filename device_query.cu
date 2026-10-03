#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable device detected.\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("========== Device %d: %s ==========\n", dev, prop.name);
        printf("Compute capability:                 %d.%d\n", prop.major, prop.minor);
        printf("\n--- Global / Constant Memory ---\n");
        printf("Total Memory:                       %.2f GB (%zu bytes)\n",
            prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), prop.totalGlobalMem);
        printf("Total constant memory:              %zu bytes (%.1f KB)\n", 
            prop.totalConstMem, prop.totalConstMem / 1024.0);
        printf("Memory bus width:                   %d bits\n", prop.memoryBusWidth);
        printf("Memory clock rate:                  %.0f MHz\n", prop.memoryClockRate / 1000.0);
        printf("L2 cache size:                      %d bytes (%.1f KB)\n",
            prop.l2CacheSize, prop.l2CacheSize / 1024.0);
        
        printf("\n--- Per-Block Limits ---\n");
        printf("Shared memory per block:            %zu bytes (%.1f KB)\n",
            prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
        printf("Register per block:                 %d\n", prop.regsPerBlock);
        printf("Max threads per SM:                 %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max resident blocks per SM:         %d\n", prop.maxBlocksPerMultiProcessor);

        printf("\n--- Warp / Scheduling ---\n");
        printf("Warp size:                          %d\n", prop.warpSize);
        printf("GPU clock rate:                     %.0f MHz\n", prop.clockRate / 1000.0);

        printf("\n--- Shared Memory Bank Info (useful for your bank-conflict analysis) ---\n");
        printf("Shared memory bank width (bytes):   4 (standard, unless configured to 8-byte mode)\n");
        printf("Number of shared memory banks:      32 (standard on all modern architectures)\n");

        printf("\n");
        
    }


    return 0;
}