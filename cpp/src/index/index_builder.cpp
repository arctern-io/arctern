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

#include "index/index.h"
#include "render/utils/render_utils.h"

namespace arctern {
namespace index {

IndexNode::IndexNode(OGRGeometry* geo, int32_t index) {
  geometry_ = std::shared_ptr<OGRGeometry>(geo, [](OGRGeometry*) {});
  index_ = index;
}

template <typename T>
void gen_index(std::shared_ptr<T>& tree,
               const std::vector<std::shared_ptr<arrow::Array>>& geos) {
  const auto& geo_vec = render::GeometryExtraction(geos);

  int32_t offset = 0;
  for (auto geo : geo_vec) {
    auto rs_pointer = reinterpret_cast<OGRPolygon*>(geo);
    OGREnvelope* envelope = new OGREnvelope();
    rs_pointer->getEnvelope(envelope);
    const geos::geom::Envelope* env = new geos::geom::Envelope(
        envelope->MinX, envelope->MaxX, envelope->MinY, envelope->MaxY);
    IndexNode* node = new IndexNode(geo, offset++);
    tree->insert(env, node);
    free((void*)envelope);
  }
}

std::shared_ptr<SpatialIndex> index_builder(
    const std::vector<std::shared_ptr<arrow::Array>>& geo, IndexType index_type) {
  switch (index_type) {
    case IndexType::rTree: {
      auto tree = std::make_shared<RTree>();
      gen_index<RTree>(tree, geo);
      return tree;
    }
    case IndexType::qTree: {
      auto tree = std::make_shared<QuadTree>();
      gen_index<QuadTree>(tree, geo);
      return tree;
    }
  }
  return nullptr;
}

}  // namespace index
}  // namespace arctern
