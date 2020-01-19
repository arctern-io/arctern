#pragma once
#include <iostream>
#include <memory>
#include "render/utils/vega/vega_heatmap/vega_heatmap.h"
#include "render/utils/color/color_gradient.h"
//#define USE_GPU

namespace zilliz {
namespace render {

void
HeatMapArray_cpu(float *in_count, float *out_count, float *kernel, int64_t kernel_size, int64_t width, int64_t height);

void
MeanKernel_cpu(float *img_in, float *img_out, int64_t r, int64_t img_w, int64_t img_h);

void
guassiankernel2d(float *kernel, int sizeX, int sizeY, float sigmaX, float sigmaY);

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
        uint32_t vertice_y = height - in_y[i] - 1;
        if (vertice_y > height || vertice_x > width)
            continue;
        int64_t index = vertice_y * width + vertice_x;
        if (index >= width * height)
            continue;
        out[index] += in_c[i];
    }
}

template<typename T>
void set_colors_gpu(float *colors,
                    uint32_t* input_x,
                    uint32_t* input_y,
                    T* input_c,
                    int64_t num,
                    VegaHeatMap &vega_heat_map);

template<typename T>
void set_colors_cpu(float *colors,
                    uint32_t* input_x,
                    uint32_t* input_y,
                    T* input_c,
                    int64_t num,
                    VegaHeatMap &vega_heat_map) {
    WindowParams window_params = vega_heat_map.window_params();
    int64_t width = window_params.width();
    int64_t height = window_params.height();
    int64_t window_size = width * height;

    float *pix_count = (float *) malloc(window_size * sizeof(float));
    memset(pix_count, 0, window_size * sizeof(float));
    SetCountValue_cpu<T>(pix_count, input_x, input_y, input_c, num, width, height);

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

template<typename T>
inline void set_colors(float *colors,
                       uint32_t* input_x,
                       uint32_t* input_y,
                       T* input_c,
                       int64_t num,
                       VegaHeatMap &vega_heat_map) {
#ifndef USE_GPU
    set_colors_cpu<T>(colors, input_x, input_y, input_c, num, vega_heat_map);
#else
    set_colors_gpu<T>(colors, input_x, input_y, input_c, num, vega_heat_map);
#endif
}

} //namespace render
} //namespace zilliz

#ifdef USE_GPU

#define TEMPLATE_GEN_PREFIX extern
#define T uint32_t
#include "set_color.inl"

#define TEMPLATE_GEN_PREFIX extern
#define T float
#include "set_color.inl"

#define TEMPLATE_GEN_PREFIX extern
#define T double
#include "set_color.inl"

#endif