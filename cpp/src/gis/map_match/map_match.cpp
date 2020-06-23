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

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "gis/map_match/map_match.h"

namespace arctern {
namespace gis {
namespace map_match {

struct Point {
  double x;
  double y;
};

struct Projection {
  Point point;
  int32_t road_id;
  double distance;
};

Projection projection_to_edge(const OGRGeometry* road, const OGRGeometry* gps_point) {
  Projection projection;
  double min_distance = std::numeric_limits<double>::max();
  Point nearest_point;

  const OGRPoint* gps_point_geo = dynamic_cast<const OGRPoint*>(gps_point);
  const OGRLineString* road_geo = dynamic_cast<const OGRLineString*>(road);

  if (road_geo == nullptr || gps_point_geo == nullptr) {
    projection.distance = min_distance;
  }

  double x = gps_point_geo->getX();
  double y = gps_point_geo->getY();
  int32_t num_points = road_geo->getNumPoints();

  for (int32_t i = 0; i < (num_points - 1); i++) {
    double x1 = road_geo->getX(i);
    double y1 = road_geo->getY(i);
    double x2 = road_geo->getX(i + 1);
    double y2 = road_geo->getY(i + 1);
    double L2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    if (L2 == 0.0) {
      continue;
    }
    double x1_x = x - x1;
    double y1_y = y - y1;
    double x1_x2 = x2 - x1;
    double y1_y2 = y2 - y1;
    double ratio = (x1_x * x1_x2 + y1_y * y1_y2) / L2;
    ratio = (ratio > 1) ? 1 : ratio;
    ratio = (ratio < 0) ? 0 : ratio;
    double prj_x = x1 + ratio * (x1_x2);
    double prj_y = y1 + ratio * (y1_y2);

    double distance = sqrt((x - prj_x) * (x - prj_x) + (y - prj_y) * (y - prj_y));

    if (min_distance >= distance) {
      min_distance = distance;
      nearest_point.x = prj_x;
      nearest_point.y = prj_y;
    }
  }

  projection.distance = min_distance;
  projection.point = nearest_point;

  return projection;
}

Projection nearest_edge(const std::vector<OGRGeometry*>& roads,
                        const OGRGeometry* gps_point) {
  double min_distance = std::numeric_limits<double>::max();
  Projection result;

  for (int32_t i = 0; i < roads.size(); i++) {
    Projection projection = projection_to_edge(roads[i], gps_point);
    if (min_distance >= projection.distance) {
      min_distance = projection.distance;
      result = projection;
      result.road_id = i;
    }
  }

  return result;
}

std::vector<std::shared_ptr<arrow::Array>> compute(
    const IndexTree& index_tree,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points, int32_t flag) {
  std::vector<std::shared_ptr<arrow::Array>> results;
  auto gps_points_geo = arctern::render::GeometryExtraction(gps_points);
  auto num_gps_points = gps_points_geo.size();

  int32_t offset = 0;
  for (int32_t i = 0; i < gps_points.size(); i++) {
    arrow::BinaryBuilder builder;
    auto len = gps_points[i]->length();
    for (int32_t j = 0; j < len; j++) {
      std::vector<OGRGeometry*> vector_road;
      if (gps_points_geo[offset] == nullptr) {
        offset++;
        builder.AppendNull();
      } else {
        auto geo_point = gps_points_geo[offset++].get();
        vector_road = index_tree.map_match_query(geo_point, true);
        if (vector_road.empty()) {
          builder.AppendNull();
        } else {
          Projection projection = nearest_edge(vector_road, geo_point);
          if (flag == 0) {
            OGRPoint point(projection.point.x, projection.point.y);
            std::vector<unsigned char> str(point.WkbSize());
            OGR_G_ExportToWkb(&point, OGRwkbByteOrder::wkbNDR, str.data());
            builder.Append(str.data(), str.size());
          } else {
            auto geo = vector_road[projection.road_id];
            std::vector<unsigned char> str(geo->WkbSize());
            OGR_G_ExportToWkb(geo, OGRwkbByteOrder::wkbNDR, str.data());
            builder.Append(str.data(), str.size());
          }
        }
      }
    }
    std::shared_ptr<arrow::BinaryArray> result;
    builder.Finish(&result);
    results.emplace_back(result);
  }
  return results;
}

std::vector<std::shared_ptr<arrow::Array>> nearest_location_on_road(
    const IndexTree& index_tree,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points) {
  return compute(index_tree, gps_points, 0);
}

std::vector<std::shared_ptr<arrow::Array>> nearest_road(
    const IndexTree& index_tree,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points) {
  return compute(index_tree, gps_points, 1);
}

std::vector<std::shared_ptr<arrow::Array>> near_road(
    const IndexTree& index_tree,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points, const double distance) {
  std::vector<std::shared_ptr<arrow::Array>> results;
  auto gps_points_geo = arctern::render::GeometryExtraction(gps_points);

  auto num_gps_points = gps_points_geo.size();

  arrow::BooleanBuilder builder;
  int32_t offset = 0;
  for (int i = 0; i < gps_points.size(); ++i) {
    int size = gps_points[i]->length();
    for (int j = 0; j < size; ++j) {
      auto index = offset + j;
      auto vector_road =
          index_tree.map_match_query(gps_points_geo[index].get(), false, distance);
      if (vector_road.empty()) {
        builder.Append(false);
      } else {
        builder.Append(true);
      }
    }
    std::shared_ptr<arrow::BooleanArray> result;
    builder.Finish(&result);
    results.emplace_back(result);
    offset += size;
  }

  return results;
}

}  // namespace map_match
}  // namespace gis
}  // namespace arctern
