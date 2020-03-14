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
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "render/2d/choropleth_map/choropleth_map.h"
#include "render/2d/heatmap/heatmap.h"
#include "render/2d/scatter_plot/pointmap.h"

namespace arctern {
namespace render {

std::vector<std::string> coordinate_projection(const std::vector<std::string>& point_wkt,
                                               const std::string top_left,
                                               const std::string bottom_right,
                                               const int height, const int width);

std::pair<uint8_t*, int64_t> pointmap(uint32_t* arr_x, uint32_t* arr_y,
                                      int64_t num_vertices, const std::string& conf);

template <typename T>
std::pair<uint8_t*, int64_t> heatmap(uint32_t* arr_x, uint32_t* arr_y, T* arr_c,
                                     int64_t num_vertices, const std::string& conf);

template <typename T>
std::pair<uint8_t*, int64_t> choroplethmap(const std::vector<std::string>& arr_wkt,
                                           T* arr_c, int64_t num_buildings,
                                           const std::string& conf);

}  // namespace render
}  // namespace arctern

#include "render/render_builder_impl.h"
