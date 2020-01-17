#include <cmath>
#include "render/utils/color/color_gradient.h"
#include "render/2d/set_color.h"

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

template<typename T>
void SetCountValue_cpu(float *out,
                       uint32_t *in_x,
                       uint32_t *in_y,
                       T *in_c,
                       int64_t num,
                       int64_t width,
                       int64_t height) {
    for (int i = 0; i < num; i++) {
        uint32_t vertice_x = in_x[i];
        uint32_t vertice_y = in_y[i];
        int64_t index = vertice_y * width + vertice_x;
        if (index >= width * height)
            continue;
        out[index] += in_c[i];
    }
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

template<typename T>
void set_colors(float *colors,
                std::shared_ptr<uint32_t> input_x,
                std::shared_ptr<uint32_t> input_y,
                std::shared_ptr<T> input_c,
                int64_t num,
                VegaHeatMap &vega_heat_map) {
#ifdef CPU_ONLY
    set_colors_cpu<T>(colors, input_x, input_y, input_c, num, vega_heat_map);
#else
    set_colors_gpu(colors, input_x, input_y, input_c, num, vega_heat_map);
#endif
}

template<typename T>
void set_colors_cpu(float *colors,
                    std::shared_ptr<uint32_t> input_x,
                    std::shared_ptr<uint32_t> input_y,
                    std::shared_ptr<T> input_c,
                    int64_t num,
                    VegaHeatMap &vega_heat_map) {
    WindowParams window_params = vega_heat_map.window_params();
    int64_t width = window_params.width();
    int64_t height = window_params.height();
    int64_t window_size = width * height;

    float *pix_count = (float *) malloc(window_size * sizeof(float));
    memset(pix_count, 0, window_size * sizeof(float));
    SetCountValue_cpu<T>(pix_count, input_x.get(), input_y.get(), input_c.get(), num, width, height);

    double scale = vega_heat_map.map_scale() * 0.4;
    int d = pow(2, scale);
    float kernel_size = d * 2 + 3;

    float *kernel = (float *) malloc(kernel_size * kernel_size * sizeof(float));
    guassiankernel2d(kernel, kernel_size, kernel_size, kernel_size, kernel_size);

    float *heat_count = (float *) malloc(window_size * sizeof(float));
    HeatMapArray_cpu(pix_count, heat_count, kernel, kernel_size, width, height);

    float *color_count = (float *) malloc(window_size * sizeof(float));
    memset(color_count, 0, window_size * sizeof(float));
    int64_t mean_radius = (int) (log((kernel_size - 3) / 2) / 0.4);
    MeanKernel_cpu(heat_count, color_count, mean_radius / 2 + 1, width, height);
    MeanKernel_cpu(color_count, heat_count, mean_radius + 1, width, height);

    float max_pix = 0;
    for (auto k = 0; k < window_size; k++) {
        if (max_pix < heat_count[k])
            max_pix = heat_count[k];
    }
    ColorGradient color_gradient;
    color_gradient.createDefaultHeatMapGradient();
//    colors = (float *) malloc(window_size * 4 * sizeof(float));

    int64_t c_offset = 0;
    for (auto j = 0; j < window_size; j++) {
        float value = heat_count[j] / max_pix;
        float color_r, color_g, color_b;
        color_gradient.getColorAtValue(value, color_r, color_g, color_b);
        colors[c_offset++] = color_r;
        colors[c_offset++] = color_g;
        colors[c_offset++] = color_b;
        colors[c_offset++] = value;
    }

    free(pix_count);
    free(kernel);
    free(heat_count);
    free(color_count);
}

} //namespace render
} //namespace zilliz
