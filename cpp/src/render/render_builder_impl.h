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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/check_status.h"

namespace arctern {
namespace render {

void pointXY_from_wkt(const std::string& wkt, double& x, double& y, void* poCT) {
  OGRGeometry* res_geo = nullptr;
  CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt.c_str(), nullptr, &res_geo));
  CHECK_GDAL(OGR_G_Transform(res_geo, (OGRCoordinateTransformation*)poCT));
  auto rst_pointer = reinterpret_cast<OGRPoint*>(res_geo);
  x = rst_pointer->getX();
  y = rst_pointer->getY();
  OGRGeometryFactory::destroyGeometry(res_geo);
}

std::shared_ptr<arrow::Array> TransformAndProjection(
    const std::shared_ptr<arrow::Array>& geos, const std::string& src_rs,
    const std::string& dst_rs, const std::string& bottom_right,
    const std::string& top_left, const int& height, const int& width) {
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
  arrow::BinaryBuilder builder;

  auto len = geos->length();
  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geos);

  double top_left_x, top_left_y, bottom_right_x, bottom_right_y;
  pointXY_from_wkt(top_left, top_left_x, top_left_y, poCT);
  pointXY_from_wkt(bottom_right, bottom_right_x, bottom_right_y, poCT);
  auto coordinate_width = bottom_right_x - top_left_x;
  auto coordinate_height = top_left_y - bottom_right_y;
  uint32_t output_x, output_y;

  for (int32_t i = 0; i < len; i++) {
    if (wkt_geometries->IsNull(i)) {
      std::string nullstr = "";
      CHECK_ARROW(builder.Append(nullstr));
      continue;
    }
    OGRGeometry* geo = nullptr;
    auto err_code = OGRGeometryFactory::createFromWkt(
        wkt_geometries->GetString(i).c_str(), nullptr, &geo);
    if (err_code) continue;
    if (geo == nullptr) {
      CHECK_ARROW(builder.AppendNull());
    } else {
      // 1. transform
      CHECK_GDAL(OGR_G_Transform(geo, (OGRCoordinateTransformation*)poCT));

      // 2. projection
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

      auto sz = geo->WkbSize();
      std::vector<char> str(sz);
      err_code = geo->exportToWkb(OGRwkbByteOrder::wkbNDR, (uint8_t*)str.data());
      if (err_code != OGRERR_NONE) {
        std::string err_msg =
            "failed to export to wkt, error code = " + std::to_string(err_code);
        throw std::runtime_error(err_msg);
      }

      CHECK_ARROW(builder.Append(str.data(), str.size()));
      OGRGeometryFactory::destroyGeometry(geo);
    }
  }

  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  OCTDestroyCoordinateTransformation(poCT);

  return results;
}

template <typename T>
std::unordered_map<OGRGeometry*, T, hash_func> weight_agg(
    const std::shared_ptr<arrow::Array>& geos,
    const std::shared_ptr<arrow::Array>& arr_c) {
  auto geo_arr = std::static_pointer_cast<arrow::BinaryArray>(geos);
  auto c_arr = (T*)arr_c->data()->GetValues<T>(1);
  auto geos_size = geos->length();
  auto geo_type = geos->type_id();
  auto c_size = arr_c->length();
  assert(geo_type == arrow::Type::BINARY);
  assert(geos_size == c_size);

  std::unordered_map<OGRGeometry*, T, hash_func> results;
  for (size_t i = 0; i < geos_size; i++) {
    std::string geo_wkb = geo_arr->GetString(i);
    OGRGeometry* res_geo;
    CHECK_GDAL(OGRGeometryFactory::createFromWkb(geo_wkb.c_str(), nullptr, &res_geo));
    auto type = wkbFlatten(res_geo->getGeometryType());
    if (results.find(res_geo) == results.end()) {
      results[res_geo] = c_arr[i];
    } else {
      results[res_geo] += c_arr[i];
    }
  }
  return results;
}

std::pair<uint8_t*, int64_t> pointmap(uint32_t* arr_x, uint32_t* arr_y, int64_t num,
                                      const std::string& conf) {
  VegaPointmap vega_pointmap(conf);
  if (!vega_pointmap.is_valid()) {
    // TODO: add log here
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
    // TODO: add log here
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
    // TODO: add log here
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
    // TODO: add log here
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
    // TODO: add log here
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
    // TODO: add log here
    return std::make_pair(nullptr, -1);
  }

  ChoroplethMap<T> choropleth_map(arr_wkt, arr_c, num_buildings);
  choropleth_map.mutable_choroplethmap_vega() = vega_choropleth_map;

  const auto& render = choropleth_map.Render();
  const auto& ret_size = choropleth_map.output_image_size();

  return std::make_pair(render, ret_size);
}

}  // namespace render
}  // namespace arctern
