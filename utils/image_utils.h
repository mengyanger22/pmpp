// image_utils.h
// 极简图像读写工具，专门配合 PMPP Ch3 (Multidimensional grids and data) 的
// RGB转灰度 / 图像模糊 这类例子使用。
//
// 为什么用PPM格式而不是PNG/JPG：
//   PPM(P6, binary)格式极其简单（没有压缩、没有复杂的header），
//   不需要任何第三方库就能自己读写，几十行代码搞定，
//   避免你为了跑一个教学例子还要去装libpng/libjpeg这类依赖。
//
// 怎么拿到测试图片（任选一种）：
//   1. 用ImageMagick把现成的jpg/png转成ppm（Linux下通常可以直接apt装）:
//        sudo apt install imagemagick
//        convert my_photo.jpg -compress none my_photo.ppm
//   2. 用本文件自带的 generateTestImage() 函数，生成一张合成的渐变/棋盘格测试图，
//      不需要任何外部图片，适合快速跑通代码逻辑。

#ifndef PMPP_IMAGE_UTILS_H
#define PMPP_IMAGE_UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = nullptr;

    ~Image() { if (data) free(data); }
};

inline bool readPPM(const char* filename, Image& img) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) { printf("unable to open the file: %s\n", filename); return false; }

    char magic[3] = {0};
    if (fscanf(fp, "%2s", magic) || strcmp(magic, "P6") != 0) {
        fprintf(stderr, "不是合法的P6 PPM文件: %s\n", filename);
        fclose(fp);
        return false;
    }
    int c = fgetc(fp);
    while (c == '#') {
        while (c != '\n') {
            c = fgetc(fp);
        }
    }
    ungetc(c, fp);

    img.channels = 3;
    size_t dataSize = (size_t)img.width * img.height * img.channels;
    img.data = (unsigned char*)malloc(dataSize);
    size_t readBytes = fread(img.data, 1, dataSize, fp);
    fclose(fp);

    if (readBytes != dataSize) {
        fprintf(stderr, "PPM数据读取不完整: 期望%zu字节, 实际读到%zu字节\n", dataSize, readBytes);
        return false;
    }

    printf("[读取成功] %s: %d x %d, %d通道\n", filename, img.width, img.height, img.channels);
    return true;
}

inline bool writePPM(const char* filename, const unsigned char* data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "unable to create file: %s\n", filename); return false; }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(data, 1, (size_t)width * height * 3, fp);
    fclose(fp);
    printf("saved PPM successfully: %s\n", filename);
    return true;
}

inline bool writeGrayAsPPM(const char* filename, const unsigned char* gray, int width, int height) {
    int channel = 3;
    unsigned char* rgb = (unsigned char *)malloc(width * height *channel);
    for (int i = 0; i < width * height; i++) {
        rgb[i*channel] = gray[i];
        rgb[i*channel+1] = gray[i];
        rgb[i*channel+2] = gray[i];
    }
    bool ok = writePPM(filename, rgb, width, height);
    free(rgb);
    return ok;
}

// 不依赖任何外部图片，直接生成一张合成测试图（棋盘格+渐变），用来快速跑通代码
// pattern: 0 = 渐变, 1 = 棋盘格
inline void generateTestImage(const char* filename, int width=512, int height=512, int pattern=1) {
    unsigned char* rgb = (unsigned char*)malloc(size_t(width) * height *3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = ((size_t)y * width + x) * 3;
            if (pattern == 0) {
                rgb[idx + 0] = (unsigned char)(255.0 * x / width);
                rgb[idx + 1] = (unsigned char)(255.0 * y / height);
                rgb[idx + 2] = 128;
            } else {
                int block = 32;
                bool isWhite = ((x / block) + (y / block)) % 2 == 0;
                unsigned char v = isWhite ? 230 : 25;
                rgb[idx + 0] = v;
                rgb[idx + 1] = v;
                rgb[idx + 2] = v;
            }
        }
    }
    writePPM(filename, rgb, width, height);
    free(rgb);
}

inline void rgbToGrayCPU(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        gray[i] = (unsigned char) (0.21f * rgb[3 * i] + 0.71f * rgb[3 * i + 1] + 0.07f * rgb[3 * i + 2]);
    }
}

#endif