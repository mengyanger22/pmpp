#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int devCount = 0;
    cudaError_t err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount Failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (devCount == 0) {
        printf("No CUDA-capable device detected.\n");
        return 1;
    }

    for (int dev = 0; dev < devCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("========== Device %d: %s ==========\n", dev, prop.name);
        printf("Compute capability:                 %d.%d\n", prop.major, prop.minor);

        printf("\n--- Global / Constant Memory ---\n");
        printf("Total memory:                       %.2f GB (%zu bytes)\n",
            prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), prop.totalGlobalMem);
        printf("Total constant memory:              %zu bytes (%.1f KB)\n",
            prop.totalConstMem, prop.totalConstMem / 1024.0);
        printf("Memory bus width:                   %d bits\n", prop.memoryBusWidth);
        printf("Memory clock rate:                  %.0f MHz\n", prop.memoryClockRate / 1000.0);
        printf("L2 cache size:                      %d bytes (%.1f KB)\n",
            prop.l2CacheSize, prop.l2CacheSize / 1024.0);
        
        
    }

    return 0;
}