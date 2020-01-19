#include <iostream>
#include <memory>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "render/utils/color/color_gradient.h"
#include "render/2d/set_color.h"

namespace zilliz {
namespace render {

unsigned int
iDivUp(const unsigned int &a, const unsigned int &b) { return (a + b - 1) / b; }


template<typename T>
__global__ void SetCountValue_gpu(float *out,
                                  uint32_t *in_x,
                                  uint32_t *in_y,
                                  T *in_c,
                                  int64_t num,
                                  int64_t width,
                                  int64_t height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < num; i += blockDim.x * gridDim.x) {
        uint32_t vertice_x = in_x[i];
        uint32_t vertice_y = height - in_y[i] - 1;
        if (vertice_y > height || vertice_x > width)
            continue;
        int64_t index = vertice_y * width + vertice_x;
        if (index >= width * height)
            continue;
        out[index] += in_c[i];
    }
}

__global__ void
HeatMapArray_gpu(float *in_count, float *out_count, float *kernel, int64_t kernel_size, int64_t width, int64_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
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

__global__ void
MeanKernel_gpu(float *img_in, float *img_out, int64_t r, int64_t img_w, int64_t img_h) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    double gradient = 0.0;
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

template<typename T>
void set_colors_gpu(float *colors,
                    uint32_t* input_x,
                    uint32_t* input_y,
                    T* input_c,
                    int64_t num,
                    VegaHeatMap &vega_heat_map) {
    WindowParams window_params = vega_heat_map.window_params();
    int64_t width = window_params.width();
    int64_t height = window_params.height();
    int64_t window_size = width * height;

    float *pix_count;
    uint32_t *in_x, *in_y;
    T *in_c;
    cudaMalloc((void **) &pix_count, window_size * sizeof(float));
    cudaMalloc((void **) &in_x, num * sizeof(uint32_t));
    cudaMalloc((void **) &in_y, num * sizeof(uint32_t));
    cudaMalloc((void **) &in_c, num * sizeof(T));
    cudaMemset(pix_count, 0, window_size * sizeof(float));
    cudaMemcpy(in_x, input_x, num * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(in_y, input_y, num * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(in_c, input_c, num * sizeof(T), cudaMemcpyHostToDevice);
    SetCountValue_gpu<T> << < 256, 1024 >>
        > (pix_count, in_x, in_y, in_c, num, width, height);

    double scale = vega_heat_map.map_scale() * 0.4;
    int d = pow(2, scale);
    int64_t kernel_size = d * 2 + 3;

    float *kernel = (float *) malloc(kernel_size * kernel_size * sizeof(float));
    guassiankernel2d(kernel, kernel_size, kernel_size, kernel_size, kernel_size);
    float *dev_kernel;
    cudaMalloc((void **) &dev_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(dev_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    float *dev_count;
    cudaMalloc((void **) &dev_count, window_size * sizeof(float));
    cudaMemset(dev_count, 0, window_size * sizeof(float));

    const unsigned int blockW = 32;
    const unsigned int blockH = 32;
    const dim3 threadBlock(blockW, blockH);
    const dim3 grid(iDivUp(width, blockW), iDivUp(height, blockH));
    HeatMapArray_gpu << < grid, threadBlock >>
        > (pix_count, dev_count, dev_kernel, kernel_size, width, height);

    float *color_count;
    cudaMalloc((void **) &color_count, window_size * sizeof(float));
    cudaMemset(color_count, 0, window_size * sizeof(float));
    int64_t mean_radius = (int) (log((kernel_size - 3) / 2) / 0.4);

    MeanKernel_gpu << < grid, threadBlock >>
        > (dev_count, color_count, mean_radius + 1, width, height);
    MeanKernel_gpu << < grid, threadBlock >>
        > (color_count, dev_count, mean_radius / 2 + 1, width, height);

    auto host_count = (float *) malloc(window_size * sizeof(float));
    cudaMemcpy(host_count, dev_count, window_size * sizeof(float), cudaMemcpyDeviceToHost);
    float max_pix = 0;
    for (auto k = 0; k < window_size; k++) {
        if (max_pix < host_count[k])
            max_pix = host_count[k];
    }
    ColorGradient color_gradient;
    color_gradient.createDefaultHeatMapGradient();

    int64_t c_offset = 0;
    for (auto j = 0; j < window_size; j++) {
        float value = host_count[j] / max_pix;
        float color_r, color_g, color_b;
        color_gradient.getColorAtValue(value, color_r, color_g, color_b);
        colors[c_offset++] = color_r;
        colors[c_offset++] = color_g;
        colors[c_offset++] = color_b;
        colors[c_offset++] = value;
    }

    free(kernel);
    free(host_count);
    cudaFree(pix_count);
    cudaFree(dev_kernel);
    cudaFree(dev_count);
    cudaFree(color_count);
}

} //namespace render
} //namespace

#define TEMPLATE_GEN_PREFIX
#define T uint32_t
#include "render/2d/set_color.inl"

#define TEMPLATE_GEN_PREFIX
#define T float
#include "render/2d/set_color.inl"

#define TEMPLATE_GEN_PREFIX
#define T double
#include "render/2d/set_color.inl"
