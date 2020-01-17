#pragma once

#include <iostream>
#include <memory>
#include "render/utils/vega/vega_heatmap/vega_heatmap.h"
//#define CPU_ONLY

namespace zilliz {
namespace render {

void
guassiankernel2d(float *kernel, int sizeX, int sizeY, float sigmaX, float sigmaY);

template<typename T>
void set_colors(float *colors,
                std::shared_ptr<uint32_t> input_x,
                std::shared_ptr<uint32_t> input_y,
                std::shared_ptr<T> input_c,
                int64_t num,
                VegaHeatMap &vega_heat_map);

template<typename T>
void set_colors_cpu(float *colors,
                    std::shared_ptr<uint32_t> input_x,
                    std::shared_ptr<uint32_t> input_y,
                    std::shared_ptr<T> input_c,
                    int64_t num,
                    VegaHeatMap &vega_heat_map);

template<typename T>
void set_colors_gpu(float *colors,
                    std::shared_ptr<uint32_t> input_x,
                    std::shared_ptr<uint32_t> input_y,
                    std::shared_ptr<T> input_c,
                    int64_t num,
                    VegaHeatMap &vega_heat_map);

} //namespace render
} //namespace zilliz