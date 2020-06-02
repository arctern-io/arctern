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
#include <string>

#include "map_match/map_match.h"

namespace arctern {
namespace map_match {
using index::IndexTree;

struct Projection {
  unsigned char* geo_str = nullptr;
  int32_t size;
  double distance;
};

void destroy_geometry(std::vector<OGRGeometry*>& geos) {
  for (int32_t i = 0; i < geos.size(); i++) {
    OGRGeometryFactory::destroyGeometry(geos[i]);
  }
}

Projection projection_to_edge(const OGRGeometry* road, const OGRGeometry* gps_point) {
  Projection projection;
  double min_distance = std::numeric_limits<double>::max();
  auto nearest_point = std::make_shared<OGRPoint>();

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
      throw nullptr;
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
      nearest_point->setX(prj_x);
      nearest_point->setY(prj_y);
    }
  }

  auto wkb_size = nearest_point->WkbSize();
  auto wkb = static_cast<unsigned char*>(CPLMalloc(wkb_size));
  OGR_G_ExportToWkb(nearest_point.get(), OGRwkbByteOrder::wkbNDR, wkb);
  projection.geo_str = wkb;
  projection.distance = min_distance;
  projection.size = wkb_size;

  return projection;
}

Projection nearest_edge(const std::vector<OGRGeometry*>& roads,
                        const OGRGeometry* gps_point, int32_t flag) {
  double min_distance = std::numeric_limits<double>::max();
  Projection result, projection;

  for (int32_t i = 0; i < roads.size(); i++) {
    projection = projection_to_edge(roads[i], gps_point);
    if (min_distance >= projection.distance) {
      if (min_distance >= projection.distance) {
        min_distance = projection.distance;

        if (result.geo_str != nullptr) {
          free(result.geo_str);
        }

        if (flag == 0) {
          result = projection;
        } else {
          free(projection.geo_str);
          auto wkb_size = roads[i]->WkbSize();
          auto wkb = static_cast<unsigned char*>(CPLMalloc(wkb_size));
          OGR_G_ExportToWkb(roads[i], OGRwkbByteOrder::wkbNDR, wkb);
          result.geo_str = wkb;
          result.size = wkb_size;
        }
      }
    } else {
      free(projection.geo_str);
    }
  }

  return result;
}

const std::vector<OGRGeometry*> get_road(OGRGeometry*& gps_point,
                                         const IndexTree& index_tree) {
  std::vector<void*> matches;
  {
    OGREnvelope ogr_env;
    gps_point->getEnvelope(&ogr_env);
    geos::geom::Envelope env(ogr_env.MinX - 0.001, ogr_env.MaxX + 0.001,
                             ogr_env.MinY - 0.001, ogr_env.MaxY + 0.001);
    index_tree.get_tree()->query(&env, matches);
  }
  std::vector<OGRGeometry*> results;
  for (auto match : matches) {
    // match(void*) contains index as binary representation.
    auto index = reinterpret_cast<size_t>(match);
    auto geo = index_tree.get_geometry(index);
    results.emplace_back(geo);
  }

  return results;
}

std::vector<std::shared_ptr<arrow::Array>> compute(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points, int32_t flag) {
  std::vector<std::shared_ptr<arrow::Array>> result;

  auto gps_points_geo = arctern::render::GeometryExtraction(gps_points);
  auto num_gps_points = gps_points_geo.size();
  auto index_tree = index::IndexTree::Create(IndexType::kRTree);
  index_tree.Append(roads);

  Projection projection_point;

  arrow::BinaryBuilder builder;
  int32_t index = 0;
  int32_t offset = gps_points[index]->length();

  for (int32_t i = 0; i < num_gps_points; i++) {
    std::vector<OGRGeometry*> vector_road;
    if (gps_points_geo[i] != nullptr) {
      vector_road = get_road(gps_points_geo[i], index_tree);
    }
    if (vector_road.empty() || gps_points_geo[i] == nullptr) {
      if (i == (offset - 1)) {
        if (gps_points.size() > (index + 1)) {
          index++;
          offset += gps_points[index]->length();
        }
        std::shared_ptr<arrow::BinaryArray> projection_points;
        builder.Finish(&projection_points);
        result.emplace_back(projection_points);
      }
      continue;
    }
    projection_point = nearest_edge(vector_road, gps_points_geo[i], flag);
    builder.Append(projection_point.geo_str, projection_point.size);

    if (i == (offset - 1)) {
      if (gps_points.size() > (index + 1)) {
        index++;
        offset += gps_points[index]->length();
      }
      std::shared_ptr<arrow::BinaryArray> projection_points;
      builder.Finish(&projection_points);
      result.emplace_back(projection_points);
    }
  }

  destroy_geometry(gps_points_geo);

  return result;
}

std::vector<std::shared_ptr<arrow::Array>> is_near_road(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points) {
  std::vector<std::shared_ptr<arrow::Array>> results;

  auto gps_points_geo = arctern::render::GeometryExtraction(gps_points);

  auto num_gps_points = gps_points_geo.size();
  auto index_tree = index::IndexTree::Create(IndexType::kRTree);
  index_tree.Append(roads);

  arrow::BooleanBuilder builder;
  int32_t offset = 0;
  for (int i = 0; i < gps_points.size(); ++i) {
    int size = gps_points[i]->length();
    for (int j = 0; j < size; ++j) {
      auto index = offset + j;
      auto vector_road = get_road(gps_points_geo[index], index_tree);
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

  destroy_geometry(gps_points_geo);

  return results;
}

std::vector<std::shared_ptr<arrow::Array>> nearest_location_on_road(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points) {
  return compute(roads, gps_points, 0);
}

std::vector<std::shared_ptr<arrow::Array>> nearest_road(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points) {
  return compute(roads, gps_points, 1);
}

std::vector<std::shared_ptr<arrow::Array>> near_road(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points) {
  return is_near_road(roads, gps_points);
}

}  // namespace map_match
}  // namespace arctern
