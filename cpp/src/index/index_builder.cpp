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

template <typename TreeType>
std::shared_ptr<TreeType> CreateIndexTree(
    const std::vector<std::shared_ptr<arrow::Array>>& geos) {

  static_assert(std::is_base_of<SpatialIndex, TreeType>::value(), "mismatch");
  auto geo_vec = render::GeometryExtraction(geos);
  auto tree = std::make_shared<TreeType>();
  int32_t offset = 0;

  for (auto& geo: geo_vec) {
    auto rs_pointer = static_cast<const OGRPolygon*>(geo);
    auto envelope = std::make_unique<OGREnvelope>();
    rs_pointer->getEnvelope(envelope.get());
    const geos::geom::Envelope* env = new geos::geom::Envelope(
        envelope->MinX, envelope->MaxX, envelope->MinY, envelope->MaxY);
    IndexNode* node = new IndexNode(geo, offset++);
    tree->insert(env, node);
  }
}

}  // namespace index
}  // namespace arctern