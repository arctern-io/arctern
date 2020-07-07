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

#include "index/geo_index.h"
#include "gis/map_match/map_match.h"
#include "gis/spatial_join/st_indexed_within.h"

namespace arctern {
namespace geo_indexing {

GeosIndex::GeosIndex() {
  auto index_tree = new IndexTree(IndexType::kRTree);
  index_ = index_tree;
}

GeosIndex::~GeosIndex() { delete index_; }

void GeosIndex::append(const std::vector<std::shared_ptr<arrow::Array>>& geos) {
  index_->Append(geos);
}

std::vector<std::shared_ptr<arrow::Array>> GeosIndex::near_road(
    const std::vector<std::shared_ptr<arrow::Array>>& points, const double distance) {
  return arctern::gis::map_match::near_road(*index_, points, distance);
}

std::vector<std::shared_ptr<arrow::Array>> GeosIndex::nearest_location_on_road(
    const std::vector<std::shared_ptr<arrow::Array>>& points) {
  return arctern::gis::map_match::nearest_location_on_road(*index_, points);
}

std::vector<std::shared_ptr<arrow::Array>> GeosIndex::nearest_road(
    const std::vector<std::shared_ptr<arrow::Array>>& points) {
  return arctern::gis::map_match::nearest_road(*index_, points);
}

std::vector<std::shared_ptr<arrow::Array>> GeosIndex::ST_IndexedWithin(
    const std::vector<std::shared_ptr<arrow::Array>>& points) {
  return arctern::gis::spatial_join::ST_IndexedWithin(*index_, points);
}

std::vector<std::shared_ptr<arrow::Array>> GeosIndex::query(
        const std::vector<std::shared_ptr<arrow::Array>>& inputs) {
  auto gps_points_geo = arctern::render::GeometryExtraction(inputs);
  assert(gps_points_geo.size() == 1);
  return (*index_).query(gps_points_geo[0].get());
}

}  // namespace geo_indexing
}  // namespace arctern
