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
// dummy, to make IDE happy
#define TEMPLATE_GEN_PREFIX extern
#endif

#ifndef T
// dummy, to make IDE happy
#define T int
#endif

namespace zilliz {
namespace render {

TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
} //render
} // namespace zilliz

#undef T
//#undef TEMPLATE_GEN_PREFIX
