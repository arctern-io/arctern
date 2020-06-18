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
using geo_indexing::IndexTree;

Int32ArrayPtr left_join(const WkbArrayPtr& lefts, const IndexTree& index_tree) {
  arrow::Int32Builder builder;
  for (int s = 0; s < lefts->length(); ++s) {
    auto left_view = lefts->GetView(s);
    auto left_geo = render::GeometryExtraction(left_view);
    std::vector<void*> matches;
    {
      OGREnvelope ogr_env;
      left_geo->getEnvelope(&ogr_env);
      geos::geom::Envelope env(ogr_env.MinX, ogr_env.MaxX, ogr_env.MinY, ogr_env.MaxY);
      index_tree.get_tree()->query(&env, matches);
    }

    int32_t final_index = -1;
    for (auto match : matches) {
      // match(void*) contains index as binary representation.
      auto index = reinterpret_cast<size_t>(match);
      auto right_geo = index_tree.get_geometry(index);
      if (left_geo->Within(right_geo)) {
        final_index = static_cast<int>(index);
        break;
      }
    }
    CHECK_ARROW(builder.Append(final_index));
  }
  Int32ArrayPtr arr;
  CHECK_ARROW(builder.Finish(&arr));
  return arr;
}

std::vector<ArrayPtr> left_join(const std::vector<ArrayPtr>& left_vec,
                                const IndexTree& index_tree) {
  std::vector<ArrayPtr> results;
  for (const auto& arr_raw : left_vec) {
    auto arr = std::static_pointer_cast<arrow::BinaryArray>(arr_raw);
    results.emplace_back(left_join(arr, index_tree));
  }
  return results;
}

std::vector<std::shared_ptr<arrow::Array>> ST_IndexedWithin(
    const std::vector<std::shared_ptr<arrow::Array>>& points,
    const std::vector<std::shared_ptr<arrow::Array>>& polygons,
    const std::string index_type) {
  IndexType type;
  if (index_type == "RTREE") {
    type = IndexType::kRTree;
  } else if (index_type == "QuadTree") {
    type = IndexType::kQuadTree;
  } else {
    throw std::invalid_argument("wrong index_type: " + index_type);
  }

  auto index = geo_indexing::IndexTree::Create(type);
  index.Append(polygons);
  return left_join(points, index);
}

}  // namespace spatial_join
}  // namespace gis
}  // namespace arctern
