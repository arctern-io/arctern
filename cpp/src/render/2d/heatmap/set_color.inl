/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "render/2d/heatmap/set_color.h"
#ifndef TEMPLATE_GEN_PREFIX
#define TEMPLATE_GEN_PREFIX extern
#endif

namespace zilliz {
namespace render {

#define T uint8_t
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

#define T uint16_t
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

#define T uint32_t
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

#define T uint64_t
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

#define T int8_t
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

#define T int16_t
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

#define T int32_t
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

#define T int64_t
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

#define T float
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

#define T double
TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
#undef T

} //render
} // namespace zilliz

