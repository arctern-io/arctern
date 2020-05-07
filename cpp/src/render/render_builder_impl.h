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

#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "render/utils/render_utils.h"
#include "utils/check_status.h"

namespace arctern {
namespace render {

void Projection(const std::vector<OGRGeometry*>& geos, const std::string& bottom_right,
                const std::string& top_left, const int& height, const int& width) {
  double top_left_x, top_left_y, bottom_right_x, bottom_right_y;
  pointXY_from_wkt(top_left, top_left_x, top_left_y);
  pointXY_from_wkt(bottom_right, bottom_right_x, bottom_right_y);

  auto coordinate_width = bottom_right_x - top_left_x;
  auto coordinate_height = top_left_y - bottom_right_y;

  uint32_t output_x, output_y;
  for (auto geo : geos) {
    if (geo == nullptr) {
      continue;
    } else {
      // projection
      auto type = wkbFlatten(geo->getGeometryType());
      if (type == wkbPoint) {
        output_x = (uint32_t)(((geo->toPoint()->getX() - top_left_x) * width) /
                              coordinate_width);
        output_y = (uint32_t)(((geo->toPoint()->getY() - bottom_right_y) * height) /
                              coordinate_height);
        geo->toPoint()->setX(output_x);
        geo->toPoint()->setY(output_y);
      } else if (type == wkbPolygon) {
        auto ring = geo->toPolygon()->getExteriorRing();
        auto ring_size = ring->getNumPoints();
        for (int j = 0; j < ring_size; j++) {
          output_x =
              (uint32_t)(((ring->getX(j) - top_left_x) * width) / coordinate_width);
          output_y =
              (uint32_t)(((ring->getY(j) - bottom_right_y) * height) / coordinate_height);
          ring->setPoint(j, output_x, output_y);
        }
      } else {
        std::string err_msg = "unsupported geometry type, type = " + std::to_string(type);
        throw std::runtime_error(err_msg);
      }
    }
  }
}

void TransformAndProjection(const std::vector<OGRGeometry*>& geos,
                            const std::string& src_rs, const std::string& dst_rs,
                            const std::string& bottom_right, const std::string& top_left,
                            const int& height, const int& width) {
  OGRSpatialReference oSrcSRS;
  oSrcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  if (oSrcSRS.SetFromUserInput(src_rs.c_str()) != OGRERR_NONE) {
    std::string err_msg = "faild to tranform with sourceCRS = " + src_rs;
    throw std::runtime_error(err_msg);
  }

  OGRSpatialReference oDstS;
  oDstS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  if (oDstS.SetFromUserInput(dst_rs.c_str()) != OGRERR_NONE) {
    std::string err_msg = "faild to tranform with targetCRS = " + dst_rs;
    throw std::runtime_error(err_msg);
  }

  void* poCT = OCTNewCoordinateTransformation(&oSrcSRS, &oDstS);

  double min_x, max_y, max_x, min_y;
  pointXY_from_wkt_with_transform(top_left, min_x, max_y, poCT);
  pointXY_from_wkt_with_transform(bottom_right, max_x, min_y, poCT);

  auto coor_width = max_x - min_x;
  auto coor_height = max_y - min_y;

  int32_t output_x, output_y;
  for (auto geo : geos) {
    if (geo == nullptr) {
      continue;
    } else {
      // 1. transform
      CHECK_GDAL(OGR_G_Transform(geo, (OGRCoordinateTransformation*)poCT));
      // 2. projection
      auto type = wkbFlatten(geo->getGeometryType());
      if (type == wkbPoint) {
        auto x = geo->toPoint()->getX();
        auto y = geo->toPoint()->getY();
        output_x = (int32_t)(((x - min_x) * width) / coor_width);
        output_y = (int32_t)(((y - min_y) * height) / coor_height);
        geo->toPoint()->setX(output_x);
        geo->toPoint()->setY(output_y);
      } else if (type == wkbPolygon) {
        auto ring = geo->toPolygon()->getExteriorRing();
        auto ring_size = ring->getNumPoints();
        for (int j = 0; j < ring_size; j++) {
          auto x = ring->getX(j);
          auto y = ring->getY(j);
          output_x = (int32_t)(((x - min_x) * width) / coor_width);
          output_y = (int32_t)(((y - min_y) * height) / coor_height);
          ring->setPoint(j, output_x, output_y);
        }
      } else {
        std::string err_msg = "unsupported geometry type, type = " + std::to_string(type);
        throw std::runtime_error(err_msg);
      }
    }
  }

  OCTDestroyCoordinateTransformation(poCT);
}

std::pair<uint8_t*, int64_t> pointmap(uint32_t* arr_x, uint32_t* arr_y, int64_t num,
                                      const std::string& conf) {
  VegaPointmap vega_pointmap(conf);
  if (!vega_pointmap.is_valid()) {
    return std::make_pair(nullptr, -1);
  }

  PointMap point_map(arr_x, arr_y, num);
  point_map.mutable_point_vega() = vega_pointmap;

  const auto& render = point_map.Render();
  const auto& ret_size = point_map.output_image_size();
  return std::make_pair(render, ret_size);
}

template <typename T>
std::pair<uint8_t*, int64_t> weighted_pointmap(uint32_t* arr_x, uint32_t* arr_y,
                                               int64_t num_vertices,
                                               const std::string& conf) {
  VegaWeightedPointmap vega_weighted_pointmap(conf);
  if (!vega_weighted_pointmap.is_valid()) {
    return std::make_pair(nullptr, -1);
  }

  WeightedPointMap<T> weighted_pointmap(arr_x, arr_y, num_vertices);
  weighted_pointmap.mutable_weighted_point_vega() = vega_weighted_pointmap;

  const auto& render = weighted_pointmap.Render();
  const auto& ret_size = weighted_pointmap.output_image_size();
  return std::make_pair(render, ret_size);
}

template <typename T>
std::pair<uint8_t*, int64_t> weighted_pointmap(uint32_t* arr_x, uint32_t* arr_y, T* arr,
                                               int64_t num_vertices,
                                               const std::string& conf) {
  VegaWeightedPointmap vega_weighted_pointmap(conf);
  if (!vega_weighted_pointmap.is_valid()) {
    return std::make_pair(nullptr, -1);
  }

  WeightedPointMap<T> weighted_pointmap(arr_x, arr_y, arr, num_vertices);
  weighted_pointmap.mutable_weighted_point_vega() = vega_weighted_pointmap;

  const auto& render = weighted_pointmap.Render();
  const auto& ret_size = weighted_pointmap.output_image_size();
  return std::make_pair(render, ret_size);
}

template <typename T>
std::pair<uint8_t*, int64_t> weighted_pointmap(uint32_t* arr_x, uint32_t* arr_y, T* arr_c,
                                               T* arr_s, int64_t num_vertices,
                                               const std::string& conf) {
  VegaWeightedPointmap vega_weighted_pointmap(conf);
  if (!vega_weighted_pointmap.is_valid()) {
    return std::make_pair(nullptr, -1);
  }

  WeightedPointMap<T> weighted_pointmap(arr_x, arr_y, arr_c, arr_s, num_vertices);
  weighted_pointmap.mutable_weighted_point_vega() = vega_weighted_pointmap;

  const auto& render = weighted_pointmap.Render();
  const auto& ret_size = weighted_pointmap.output_image_size();
  return std::make_pair(render, ret_size);
}

template <typename T>
std::pair<uint8_t*, int64_t> heatmap(uint32_t* arr_x, uint32_t* arr_y, T* arr_c,
                                     int64_t num_vertices, const std::string& conf) {
  VegaHeatMap vega_heat_map(conf);
  if (!vega_heat_map.is_valid()) {
    return std::make_pair(nullptr, -1);
  }

  HeatMap<T> heat_map(arr_x, arr_y, arr_c, num_vertices);
  heat_map.mutable_heatmap_vega() = vega_heat_map;

  const auto& render = heat_map.Render();
  const auto& ret_size = heat_map.output_image_size();
  return std::make_pair(render, ret_size);
}

template <typename T>
std::pair<uint8_t*, int64_t> choroplethmap(const std::vector<OGRGeometry*>& arr_wkt,
                                           T* arr_c, int64_t num_buildings,
                                           const std::string& conf) {
  VegaChoroplethMap vega_choropleth_map(conf);
  if (!vega_choropleth_map.is_valid()) {
    return std::make_pair(nullptr, -1);
  }

  ChoroplethMap<T> choropleth_map(arr_wkt, arr_c, num_buildings);
  choropleth_map.mutable_choroplethmap_vega() = vega_choropleth_map;

  const auto& render = choropleth_map.Render();
  const auto& ret_size = choropleth_map.output_image_size();

  return std::make_pair(render, ret_size);
}

std::pair<uint8_t*, int64_t> iconviz(uint32_t* arr_x, uint32_t* arr_y, int64_t num,
                                     const std::string& conf) {
  VegaIcon vega_icon(conf);
  if (!vega_icon.is_valid()) {
    return std::make_pair(nullptr, -1);
  }

  IconViz icon_viz(arr_x, arr_y, num);
  icon_viz.mutable_icon_vega() = vega_icon;

  const auto& render = icon_viz.Render();
  const auto& ret_size = icon_viz.output_image_size();
  return std::make_pair(render, ret_size);
}

template <typename T>
std::pair<uint8_t*, int64_t> fishnetmap(uint32_t* arr_x, uint32_t* arr_y, T* arr,
                                        int64_t num_vertices, const std::string& conf) {
  VegaFishNetMap vega_fishnet_map(conf);
  if (!vega_fishnet_map.is_valid()) {
    return std::make_pair(nullptr, -1);
  }

  FishNetMap<T> fishnet_map(arr_x, arr_y, arr, num_vertices);
  fishnet_map.mutable_fishnet_vega() = vega_fishnet_map;
  const auto& render = fishnet_map.Render();
  const auto& ret_size = fishnet_map.output_image_size();
  return std::make_pair(render, ret_size);
}

}  // namespace render
}  // namespace arctern
