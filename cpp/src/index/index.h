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

using RTree = GEOS_DLL::geos::index::strtree::STRtree;
using QuadTree = GEOS_DLL::geos::index::quadtree::Quadtree;
using SpatialIndex = GEOS_DLL::geos::index::SpatialIndex;
namespace arctern {
namespace index {

enum class IndexType {
  kInvalid,
  kRTree,
  kQuadTree,
};

class IndexNode {
 public:
  IndexNode() : geometry_(nullptr), index_(-1){};

  IndexNode(OGRGeometry* geo, int32_t index) : geometry_(geo), index_(index) {}

  OGRGeometry* geometry() const { return geometry_; }

  int32_t index() const { return index_; }

 private:
  OGRGeometry* geometry_;
  int32_t index_;
};

class IndexTree {
 public:
  static IndexTree Create(IndexType type) {
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

  void append(const WkbArrayPtr& right) {
    for (int i = 0; i < right->length(); ++i) {
      auto view = right->GetView(i);
      auto append_index = geometries_.size();
      geometries_.emplace_back(render::GeometryExtraction(view));
      auto& polygon = geometries_.back();
      {
        OGREnvelope envelope;
        polygon->getEnvelope(&envelope);
        envelopes_.emplace_back(envelope.MinX, envelope.MaxX, envelope.MinY,
                                envelope.MaxY);
      }
      auto& envelope = envelopes_.back();
      void* node = reinterpret_cast<void*>(append_index);
      tree_->insert(&envelope, node);
    }
  }

  void append(const std::vector<ArrayPtr>& right) {
    for (const auto& ptr_raw : right) {
      auto ptr = std::static_pointer_cast<arrow::BinaryArray>(ptr_raw);
      this->append(ptr);
    }
  }

  const geos::geom::Envelope& get_envelop(size_t index) const {
    return envelopes_[index];
  }

  OGRGeometry* get_geometry(size_t index) const { return geometries_[index].get(); }

  SpatialIndex* get_tree() const { return tree_.get(); }

 private:
  IndexTree() = default;

 private:
  // use deque instead of vector for validation of references
  std::deque<geos::geom::Envelope> envelopes_;
  std::deque<OGRGeometryUniquePtr> geometries_;
  std::unique_ptr<SpatialIndex> tree_;
};

}  // namespace index
}  // namespace arctern