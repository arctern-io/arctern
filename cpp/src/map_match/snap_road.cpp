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

#include <iostream>
#include <string>
#include <src/index/index.h>

#include "map_match/snap_road.h"

namespace arctern {
namespace map_match {

struct Projection {
  unsigned char* point_str;
  int32_t size;
  double distance;
};

Projection projection_to_edge(const OGRGeometry* road, const OGRGeometry* gps_point) {
  double min_distance = 1000000;
  auto nearest_point = std::make_shared<OGRPoint>();

  const OGRPoint* gps_point_geo = dynamic_cast<const OGRPoint*>(gps_point);
  const OGRLineString* road_geo = dynamic_cast<const OGRLineString*>(road);

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

    double distance = (x - prj_x) * (x - prj_x) + (y - prj_y) * (y - prj_y);

    if (min_distance >= distance) {
      min_distance = distance;
      nearest_point->setX(prj_x);
      nearest_point->setY(prj_y);
    }
  }

  Projection projection;
  auto wkb_size = nearest_point->WkbSize();
  auto wkb = static_cast<unsigned char*>(CPLMalloc(wkb_size));
  OGR_G_ExportToWkb(nearest_point.get(), OGRwkbByteOrder::wkbNDR, wkb);
  projection.point_str = wkb;
  projection.distance = min_distance;
  projection.size = wkb_size;

  return projection;
}

Projection nearest_projection(const std::vector<OGRGeometry*>& roads,
                              const OGRGeometry* gps_point) {
  double min_distance = 10000000;
  Projection result, projection_point;
  for (int32_t i = 0; i < roads.size(); i++) {
    projection_point = projection_to_edge(roads[i], gps_point);
    if (min_distance >= projection_point.distance) {
      min_distance = projection_point.distance;
      result = projection_point;
    }
  }

  return result;
}

const std::vector<OGRGeometry*> get_road(OGRGeometry* &gps_point,
                                         std::shared_ptr<RTree> tree){
    auto geo = reinterpret_cast<OGRPoint *>(gps_point);
    std::vector<void *> matches;
    OGREnvelope *envelope = new OGREnvelope();
    geo->getEnvelope(envelope);
    const geos::geom::Envelope *env = new geos::geom::Envelope(envelope->MinX - 0.01, envelope->MaxX+0.01, envelope->MinY-0.01,
                                                               envelope->MaxY+0.01);
    tree->query(env, matches);
    std::vector<OGRGeometry*> results;
    for (auto match: matches) {
        auto node = (IndexNode *) match;
        results.push_back(node->geometry().get());
    }
    return results;
}

std::vector<std::shared_ptr<arrow::Array>> snap_to_road(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points, int32_t num_thread) {
  std::vector<std::shared_ptr<arrow::Array>> result;

  auto roads_geo = arctern::render::GeometryExtraction(roads);
  auto gps_points_geo = arctern::render::GeometryExtraction(gps_points);
  auto num_gps_points = gps_points_geo.size();
  std::vector<Projection> projections_str(num_gps_points);

    auto index = std::static_pointer_cast<RTree>(index_builder(roads, IndexType::rTree));
//#pragma omp parallel for num_threads(num_thread)
  for (int32_t i = 0; i < num_gps_points; i++) {
//    auto vector_road = get_road(gps_points_geo[i], index);
    projections_str[i] = nearest_projection(roads_geo, gps_points_geo[i]);
//    projections_str[i] = nearest_projection(vector_road, gps_points_geo[i]);
  }

  arrow::BinaryBuilder builder;
  int32_t offset = 0;
  for (int32_t i = 0; i < gps_points.size(); i++) {
    std::shared_ptr<arrow::BinaryArray> projection_str;
    for (int32_t j = 0; j < gps_points[i]->length(); j++) {
      builder.Append(projections_str[j + offset].point_str,
		             projections_str[j + offset].size);
    }
    builder.Finish(&projection_str);
    result.emplace_back(projection_str);
    offset += gps_points[i]->length();
  }

  for (int32_t i = 0; i < gps_points_geo.size(); i++) {
    OGRGeometryFactory::destroyGeometry(gps_points_geo[i]);
  }

  for (int32_t i = 0; i < roads_geo.size(); i++) {
    OGRGeometryFactory::destroyGeometry(roads_geo[i]);
  }

  return result;
}

}  // namespace map_match
}  // namespace arctern
