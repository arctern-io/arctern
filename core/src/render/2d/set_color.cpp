#include <iostream>
#include <cmath>

namespace zilliz {
namespace render {

const double eps = 1e-6;

void
guassiankernel(float *kernel, int size, float sigma) {
    float sum = 0;
    float *data = kernel;

    for (int i = 0; i < size; ++i) {
        float index = (size >> 1) - i;
        if (size & 1)
            *(data + i) = exp(-(index * index) / (2 * sigma * sigma + eps));
        else {
            index -= 0.5;
            *(data + i) = exp(-(index * index) / (2 * sigma * sigma + eps));
        }
        sum += *(data + i);
    }

    for (int i = 0; i < size; ++i) {
        *(data + i) /= sum;
    }
}

void
matproduct(float a[], float b[], float c[], int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }
}

void
guassiankernel2d(float *kernel, int sizeX, int sizeY, float sigmaX, float sigmaY) {
    float *matX = (float *) malloc(sizeX * sizeof(float));
    float *matY = (float *) malloc(sizeY * sizeof(float));
    guassiankernel(matX, sizeX, sigmaX);
    guassiankernel(matY, sizeY, sigmaY);
    matproduct(matX, matY, kernel, sizeX, 1, sizeY);
    free(matX);
    free(matY);
}



void
HeatMapArray_cpu(float *in_count, float *out_count, float *kernel, int64_t kernel_size, int64_t width, int64_t height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int count_index = y * width + x;
            if (in_count[count_index] > 1e-8) {
                int r = kernel_size / 2;
                for (int m = -r; m <= r; m++) {
                    if (x + m < 0 || x + m >= width)
                        continue;
                    for (int n = -r; n <= r; n++) {
                        if (y + n < 0 || y + n >= height)
                            continue;
                        int kernel_index = (r + n) * (2 * r + 1) + (m + r);
                        int dev_index = (y + n) * width + (x + m);
                        out_count[dev_index] += in_count[count_index] * kernel[kernel_index];
                    }
                }
            }
        }
    }
}

void
MeanKernel_cpu(float *img_in, float *img_out, int64_t r, int64_t img_w, int64_t img_h) {
    for (int row = 0; row < img_h; row++) {
        for (int col = 0; col < img_w; col++) {
            float gradient = 0.0;
            if (r > 10) r = 10;
            int count = 0;
            if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w)) {
                for (int m = -r; m <= r; m++) {
                    if (row + m < 0 || row + m >= img_h)
                        continue;
                    for (int n = -r; n <= r; n++) {
                        if (col + n < 0 || col + n >= img_w)
                            continue;
                        int y = row + m;
                        int x = col + n;
                        gradient += img_in[y * img_w + x];
                        count++;
                    }
                }
                img_out[row * img_w + col] = gradient / count;
            }
        }
    }
}

} //mespace render
} //namespace zilliz