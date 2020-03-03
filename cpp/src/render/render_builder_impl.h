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

#include <ogr_geometry.h>
#include <string>
#include <utility>
#include <vector>

#include "utils/check_status.h"

namespace zilliz {
namespace render {

void pointXY_from_wkt(std::string wkt, double& x, double& y) {
  OGRGeometry* res_geo = nullptr;
  CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt.c_str(), nullptr, &res_geo));
  auto rst_pointer = reinterpret_cast<OGRPoint*>(res_geo);
  x = rst_pointer->getX();
  y = rst_pointer->getY();
  OGRGeometryFactory::destroyGeometry(res_geo);
}

std::vector<std::string> coordinate_projection(const std::vector<std::string>& point_wkt,
                                               const std::string top_left,
                                               const std::string bottom_right,
                                               const int height, const int width) {
  double top_left_x, top_left_y, bottom_right_x, bottom_right_y;
  pointXY_from_wkt(top_left, top_left_x, top_left_y);
  pointXY_from_wkt(bottom_right, bottom_right_x, bottom_right_y);

  int size = point_wkt.size();
  std::vector<std::string> output_point(size);
  double input_x, input_y;
  uint32_t output_x, output_y;
  for (int i = 0; i < size; i++) {
    pointXY_from_wkt(point_wkt[i], input_x, input_y);
    if (input_x < top_left_x || input_x > bottom_right_x || input_y > top_left_y ||
        input_y < bottom_right_y) {
      continue;
    }
    output_x =
        (uint32_t)(((input_x - top_left_x) * width) / (bottom_right_x - top_left_x));
    output_y =
        (uint32_t)(((input_y - bottom_right_y) * height) / (top_left_y - bottom_right_y));
    OGRPoint point(output_x, output_y);
    char* point_str = nullptr;
    CHECK_GDAL(point.exportToWkt(&point_str));
    std::string out_wkt(point_str);
    output_point.push_back(out_wkt);
  }

  return output_point;
}

std::pair<uint8_t*, int64_t> pointmap(uint32_t* arr_x, uint32_t* arr_y, int64_t num,
                                      const std::string& conf) {
  PointMap point_map(arr_x, arr_y, num);

  VegaCircle2d vega_circle_2d(conf);
  point_map.mutable_point_vega() = vega_circle_2d;

  const auto& render = point_map.Render();
  const auto& ret_size = point_map.output_image_size();
  return std::make_pair(render, ret_size);
}

template <typename T>
std::pair<uint8_t*, int64_t> heatmap(uint32_t* arr_x, uint32_t* arr_y, T* arr_c,
                                     int64_t num_vertices, const std::string& conf) {
  HeatMap<T> heat_map(arr_x, arr_y, arr_c, num_vertices);

  VegaHeatMap vega_heat_map(conf);
  heat_map.mutable_heatmap_vega() = vega_heat_map;

  const auto& render = heat_map.Render();
  const auto& ret_size = heat_map.output_image_size();
  return std::make_pair(render, ret_size);
}

template <typename T>
std::pair<uint8_t*, int64_t> choroplethmap(const std::vector<std::string>& arr_wkt,
                                           T* arr_c, int64_t num_buildings,
                                           const std::string& conf) {
  ChoroplethMap<T> choropleth_map(arr_wkt, arr_c, num_buildings);

  VegaChoroplethMap vega_choropleth_map(conf);
  choropleth_map.mutable_choroplethmap_vega() = vega_choropleth_map;

  const auto& render = choropleth_map.Render();
  const auto& ret_size = choropleth_map.output_image_size();

  return std::make_pair(render, ret_size);
}

}  // namespace render
}  // namespace zilliz
