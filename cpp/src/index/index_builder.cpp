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
#include <geos/indexBintree.h>
#include <geos/indexChain.h>
#include <geos/indexQuadtree.h>
#include <geos/indexStrtree.h>
#include <geos/indexSweepline.h>
#include <geos/spatialIndex.h>
#include <ogr_geometry.h>
#include <utils/arrow_alias.h>

#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "arrow/api.h"
#include "arrow/array.h"
#include "index/index.h"
#include "render/utils/render_utils.h"

namespace arctern {
namespace geo_indexing {

IndexTree IndexTree::Create(IndexType type) {
  using RTree = GEOS_DLL::geos::index::strtree::STRtree;
  using QuadTree = GEOS_DLL::geos::index::quadtree::Quadtree;
  IndexTree tree;
  switch (type) {
    case IndexType::kQuadTree: {
      tree.tree_ = std::make_unique<QuadTree>();
      break;
    }
    case IndexType::kRTree: {
      tree.tree_ = std::make_unique<RTree>();
      break;
    }
    default: {
      throw std::invalid_argument("IndexType is Invalid");
      break;
    }
  }
  return tree;
}

void IndexTree::Append(const WkbArrayPtr& right) {
  for (int i = 0; i < right->length(); ++i) {
    if (right->IsNull(i)) {
      envelopes_.emplace_back(0, 0, 0, 0);
      geometries_.emplace_back(nullptr);
      continue;
    }
    auto view = right->GetView(i);
    auto append_index = geometries_.size();
    geometries_.emplace_back(render::GeometryExtraction(view));
    auto& polygon = geometries_.back();
    {
      OGREnvelope envelope;
      polygon->getEnvelope(&envelope);
      envelopes_.emplace_back(envelope.MinX, envelope.MaxX, envelope.MinY, envelope.MaxY);
    }
    auto& envelope = envelopes_.back();
    void* node = reinterpret_cast<void*>(append_index);
    tree_->insert(&envelope, node);
  }
}

void IndexTree::Append(const std::vector<ArrayPtr>& right) {
  for (const auto& ptr_raw : right) {
    auto ptr = std::static_pointer_cast<arrow::BinaryArray>(ptr_raw);
    this->Append(ptr);
  }
}

}  // namespace geo_indexing
}  // namespace arctern
