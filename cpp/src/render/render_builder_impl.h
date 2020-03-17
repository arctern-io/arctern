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

#include <ogrsf_frmts.h>
#include <ogr_api.h>
#include <string>
#include <utility>
#include <vector>

#include "utils/check_status.h"

namespace arctern {
namespace render {

void pointXY_from_wkt(const std::string& wkt, double& x, double& y) {
  OGRGeometry* res_geo = nullptr;
  CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt.c_str(), nullptr, &res_geo));
  auto rst_pointer = reinterpret_cast<OGRPoint*>(res_geo);
  x = rst_pointer->getX();
  y = rst_pointer->getY();
  OGRGeometryFactory::destroyGeometry(res_geo);
}

std::shared_ptr<arrow::Array> TransformAndProjection(const std::shared_ptr<arrow::Array> &geos,
                                                     const std::string &src_rs,
                                                     const std::string &dst_rs,
                                                     const std::string &bottom_right,
                                                     const std::string &top_left,
                                                     const int &height,
                                                     const int &width) {
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
  arrow::StringBuilder builder;

  auto len = geos->length();
  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geos);

  double top_left_x, top_left_y, bottom_right_x, bottom_right_y;
  pointXY_from_wkt(top_left, top_left_x, top_left_y);
  pointXY_from_wkt(bottom_right, bottom_right_x, bottom_right_y);
  auto coordinate_width = bottom_right_x - top_left_x;
  auto coordinate_height = top_left_y - bottom_right_y;
  uint32_t output_x, output_y;

  for (int32_t i = 0; i < len; i++) {
    if (wkt_geometries->IsNull(i)) continue;
    OGRGeometry* geo = nullptr;
    auto err_code =
        OGRGeometryFactory::createFromWkt(wkt_geometries->GetString(i).c_str(), nullptr, &geo);
    if (err_code) continue;
    if (geo == nullptr) {
      CHECK_ARROW(builder.AppendNull());
    } else {
      // 1. transform
      CHECK_GDAL(OGR_G_Transform(geo, (OGRCoordinateTransformation*)poCT));

      // 2. projection
      auto type = wkbFlatten(geo->getGeometryType());
      if (type == wkbPoint) {
        output_x = (uint32_t)(((geo->toPoint()->getX() - top_left_x) * width) / coordinate_width);
        output_y = (uint32_t)(((geo->toPoint()->getY() - bottom_right_y) * height) / coordinate_height);
        geo->toPoint()->setX(output_x);
        geo->toPoint()->setY(output_y);
      } else if (type == wkbPolygon) {
        auto ring = geo->toPolygon()->getExteriorRing();
        auto ring_size = ring->getNumPoints();
        for (int j = 0; j < ring_size; j++) {
          output_x = (uint32_t)(((ring->getX(j) - top_left_x) * width) / coordinate_width);
          output_y = (uint32_t)(((ring->getY(j) - bottom_right_y) * height) / coordinate_height);
          ring->setPoint(j, output_x, output_y);
        }
      } else {
        std::string err_msg =
            "unsupported geometry type, type = " + std::to_string(type);
        throw std::runtime_error(err_msg);
      }

      // 3. export to wkt
      char* str;
      err_code = OGR_G_ExportToWkt(&geo, &str);
      if (err_code != OGRERR_NONE) {
        std::string err_msg =
            "failed to export to wkt, error code = " + std::to_string(err_code);
        throw std::runtime_error(err_msg);
      }

      CHECK_ARROW(builder.Append(str));
      OGRGeometryFactory::destroyGeometry(geo);
      CPLFree(str);
    }
  }

  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  OCTDestroyCoordinateTransformation(poCT);

  return results;
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
    output_x =
        (uint32_t)(((input_x - top_left_x) * width) / (bottom_right_x - top_left_x));
    output_y =
        (uint32_t)(((input_y - bottom_right_y) * height) / (top_left_y - bottom_right_y));
    OGRPoint point(output_x, output_y);
    char* point_str = nullptr;
    CHECK_GDAL(point.exportToWkt(&point_str));
    std::string out_wkt(point_str);
    output_point[i] = out_wkt;
  }
  return output_point;
}

std::pair<uint8_t*, int64_t> pointmap(uint32_t* arr_x, uint32_t* arr_y, int64_t num,
                                      const std::string& conf) {
  VegaCircle2d vega_circle_2d(conf);
  if (!vega_circle_2d.is_valid()) {
    // TODO: add log here
    return std::make_pair(nullptr, -1);
  }

  PointMap point_map(arr_x, arr_y, num);
  point_map.mutable_point_vega() = vega_circle_2d;

  const auto& render = point_map.Render();
  const auto& ret_size = point_map.output_image_size();
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
std::pair<uint8_t*, int64_t> choroplethmap(const std::vector<std::string>& arr_wkt,
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
