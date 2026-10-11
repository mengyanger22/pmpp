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
    int channels = 0;       // 3 = RGB, 1 = 灰度
    unsigned char* data = nullptr;  // 行优先存储, 大小 = width*height*channels

    ~Image() { if (data) free(data); }
};

// 读取二进制PPM(P6)格式图片，结果是RGB交错存储(R,G,B,R,G,B,...)
inline bool readPPM(const char* filename, Image& img) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "无法打开文件: %s\n", filename); return false; }

    char magic[3] = {0};
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fprintf(stderr, "不是合法的P6 PPM文件: %s\n", filename);
        fclose(fp);
        return false;
    }

    // 跳过注释行
    int c = fgetc(fp);
    while (c == '#') {
        while (fgetc(fp) != '\n');
        c = fgetc(fp);
    }
    ungetc(c, fp);

    int maxVal;
    if (fscanf(fp, "%d %d %d", &img.width, &img.height, &maxVal) != 3) {
        fprintf(stderr, "PPM头信息解析失败\n");
        fclose(fp);
        return false;
    }
    fgetc(fp); // 跳过header后紧跟的单个空白符

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

// 保存为PPM(P6) RGB格式
inline bool writePPM(const char* filename, const unsigned char* data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "无法创建文件: %s\n", filename); return false; }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(data, 1, (size_t)width * height * 3, fp);
    fclose(fp);
    printf("[保存成功] %s: %d x %d\n", filename, width, height);
    return true;
}

// 保存单通道灰度图为PPM（复制成RGB三通道存，这样任何看图工具都能直接打开）
inline bool writeGrayAsPPM(const char* filename, const unsigned char* gray, int width, int height) {
    unsigned char* rgb = (unsigned char*)malloc((size_t)width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        rgb[i * 3 + 0] = gray[i];
        rgb[i * 3 + 1] = gray[i];
        rgb[i * 3 + 2] = gray[i];
    }
    bool ok = writePPM(filename, rgb, width, height);
    free(rgb);
    return ok;
}

// 不依赖任何外部图片，直接生成一张合成测试图（棋盘格+渐变），用来快速跑通代码
// pattern: 0 = 渐变, 1 = 棋盘格
inline void generateTestImage(const char* filename, int width = 512, int height = 512, int pattern = 1) {
    unsigned char* rgb = (unsigned char*)malloc((size_t)width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = ((size_t)y * width + x) * 3;
            if (pattern == 0) {
                // 渐变：方便肉眼观察模糊/边界处理是否正确
                rgb[idx + 0] = (unsigned char)(255.0 * x / width);
                rgb[idx + 1] = (unsigned char)(255.0 * y / height);
                rgb[idx + 2] = 128;
            } else {
                // 棋盘格：方便肉眼观察卷积/模糊的效果是否符合预期
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

// CPU参考实现: RGB转灰度 (标准亮度加权公式, 和Ch3例子常用公式一致)
inline void rgb2grayCPU(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        unsigned char r = rgb[i * 3 + 0];
        unsigned char g = rgb[i * 3 + 1];
        unsigned char b = rgb[i * 3 + 2];
        gray[i] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

#endif // PMPP_IMAGE_UTILS_H