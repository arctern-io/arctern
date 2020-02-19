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

namespace zilliz {
namespace render {

std::pair<uint8_t*, int64_t> pointmap(uint32_t* arr_x, uint32_t* arr_y, int64_t num,
                                      const std::string& conf) {
  PointMap point_map(arr_x, arr_y, num);

  VegaCircle2d vega_circle_2d(conf);
  point_map.mutable_point_vega() = vega_circle_2d;

  auto render = point_map.Render();
  auto ret_size = point_map.output_image_size();
  return std::make_pair(render, ret_size);
}

template <typename T>
std::pair<uint8_t*, int64_t> heatmap(uint32_t* arr_x, uint32_t* arr_y, T* arr_c,
                                     int64_t num_vertices, const std::string& conf) {
  HeatMap<T> heat_map(arr_x, arr_y, arr_c, num_vertices);

  VegaHeatMap vega_heat_map(conf);
  heat_map.mutable_heatmap_vega() = vega_heat_map;

  auto render = heat_map.Render();
  auto ret_size = heat_map.output_image_size();
  return std::make_pair(render, ret_size);
}
}  // namespace render
}  // namespace zilliz
