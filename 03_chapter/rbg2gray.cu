#include "../utils/common.cuh"
#include "../utils/image_utils.h"

__global__
void rgb2grayKernel(const unsigned char *rgb, unsigned char *gray, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < height) {
        size_t grayIdx = row * width + col;
        size_t rgbIdx = grayIdx * 3;
        unsigned char r = rgb[rgbIdx + 0];
        unsigned char g = rgb[rgbIdx + 1];
        unsigned char b = rgb[rgbIdx + 2];
        gray[grayIdx] = (unsigned char) (0.21f * r + 0.71f *g + 0.07f * b);
    }
}

int main() {

    printDeviceBrief();

    const char* inputPath = "./images/test_input.ppm";
    FILE* probe = fopen(inputPath, "rb");
    if (!probe) {
        printf("cant find test image, automatic generate one...\n");
        generateTestImage(inputPath, 512, 512, 1);
    } else {
        fclose(probe);
    }

    Image img;
    if (readPPM(inputPath, img) != 1) return 1;

    int width = img.width, height = img.height;
    size_t grayBytes = (size_t)width * height;
    size_t rgbBytes = grayBytes * img.channels;

    unsigned char *gray_h_cpu = (unsigned char*)malloc(grayBytes);
    unsigned char *gray_h_gpu = (unsigned char*)malloc(grayBytes);
    
    CpuTimer cpuTimer;
    cpuTimer.start();
    rgb2grayCPU(img.data, gray_h_cpu, width, height);
    float cpuMs = cpuTimer.elapsedMs();

    unsigned char *rgb_d, *gray_d;
    cudaMalloc(&rgb_d, rgbBytes);
    cudaMalloc(&gray_d, grayBytes);
    cudaMemcpy(rgb_d, img.data, rgbBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    GpuTimer gpuTimer;
    gpuTimer.start();
    rgb2grayKernel<<<gridSize, blockSize>>>(rgb_d, gray_d, width, height);
    gpuTimer.stop();
    float gpuMs = gpuTimer.elapsedMs();

    cudaMemcpy(gray_h_gpu, gray_d, grayBytes, cudaMemcpyDeviceToHost);
    size_t mismatch = 0;
    for (size_t i = 0; i < grayBytes; i ++) {
        if (abs(gray_h_cpu[i] - gray_h_gpu[i]) > 1) mismatch++;
    }

    printf("图像尺寸: %d x %d | block配置: (%d,%d) | grid配置: (%d,%d)\n",
            width, height, blockSize.x, blockSize.y, gridSize.x, gridSize.y);
    printf("CPU耗时: %.4f ms | GPU kernel耗时: %.4f ms | 加速比: %.2fx\n", cpuMs, gpuMs, cpuMs / gpuMs);
    printf("校验: %zu / %zu 像素超出容差 -> %s\n", mismatch, grayBytes, mismatch == 0 ? "通过" : "有差异(检查一下)");

    writeGrayAsPPM("images/test_gray_cpu.ppm", gray_h_cpu, width, height);
    writeGrayAsPPM("images/test_gray_gpu.ppm", gray_h_gpu, width, height);

    free(gray_h_cpu); free(gray_h_gpu);
    cudaFree(rgb_d); cudaFree(gray_d);

    printf("\n建议实验: 把图片换成非32整除的尺寸(比如500x333), \n");
    printf("          观察边界条件(if (col < width && row < height))是否处理正确。\n");

    return 0;
}