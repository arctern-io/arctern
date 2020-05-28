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

#include "gis/spatial_join/st_indexed_within.h"

#include <src/index/index.h>

#include <string>
#include <vector>

#include "index/index.h"
#include "render/utils/render_utils.h"

namespace arctern {
namespace gis {
namespace spatial_join {

template <typename T>
const std::vector<std::shared_ptr<arrow::Array>> join(
    const std::vector<std::shared_ptr<arrow::Array>>& geos, std::shared_ptr<T> index) {
  const auto& wkb_vec = render::GeometryExtraction(geos);
  auto num_of_point = wkb_vec.size();
  auto array_size = geos.size();
  auto tree = index.get();

  std::vector<std::shared_ptr<arrow::Array>> arrays(array_size);
  int size_per_array = num_of_point / array_size;
  array_size = num_of_point % array_size == 0 ? array_size : array_size + 1;
  for (int i = 0; i < array_size; i++) {
    arrow::Int32Builder builder;
    for (int j = i * size_per_array; j < (i + 1) * size_per_array && j < num_of_point;
         j++) {
      auto geo = reinterpret_cast<OGRPoint*>(wkb_vec[j]);
      std::vector<void*> matches;
      OGREnvelope* envelope = new OGREnvelope();
      geo->getEnvelope(envelope);
      const geos::geom::Envelope* env = new geos::geom::Envelope(
          envelope->MinX, envelope->MaxX, envelope->MinY, envelope->MaxY);
      tree->query(env, matches);
      IndexNode* res = nullptr;

      int32_t geo_index = -1;
      for (auto match : matches) {
        res = (IndexNode*)match;
        auto indexed_geo = res->geometry();
        if (geo->Within(indexed_geo.get())) {
          geo_index = res->index();
          break;
        }
      }
      CHECK_ARROW(builder.Append(geo_index));
      OGRGeometryFactory::destroyGeometry(wkb_vec[j]);
    }
    std::shared_ptr<arrow::Array> array;
    CHECK_ARROW(builder.Finish(&array));
    arrays[i] = array;
  }
  return arrays;
}

std::vector<std::shared_ptr<arrow::Array>> ST_IndexedWithin(
    const std::vector<std::shared_ptr<arrow::Array>>& points,
    const std::vector<std::shared_ptr<arrow::Array>>& polygons,
    const std::string index_type) {
  if (!index_type.compare("RTREE")) {
    auto index =
        std::static_pointer_cast<RTree>(index_builder(polygons, IndexType::rTree));
    return join<RTree>(points, index);
  } else if (!index_type.compare("QuadTREE")) {
    auto index =
        std::static_pointer_cast<QuadTree>(index_builder(polygons, IndexType::qTree));
    return join<QuadTree>(points, index);
  } else {
    std::string err_msg = "unknow index type";
    throw std::runtime_error(err_msg);
  }
}

}  // namespace spatial_join
}  // namespace gis
}  // namespace arctern