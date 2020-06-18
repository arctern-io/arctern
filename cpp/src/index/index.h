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

#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "arrow/api.h"
#include "arrow/array.h"
#include "index/index.h"
#include "render/utils/render_utils.h"
#include "utils/arrow_alias.h"
namespace arctern {
namespace geo_indexing {
enum class IndexType {
  kInvalid,
  kRTree,
  kQuadTree,
};

class IndexTree {
 public:
  using SpatialIndex = GEOS_DLL::geos::index::SpatialIndex;
  static IndexTree Create(IndexType type);

  void Append(const WkbArrayPtr& right);

  void Append(const std::vector<ArrayPtr>& right);

  //  const geos::geom::Envelope& get_envelop(size_t index) const {
  //    return envelopes_[index];
  //  }

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

}  // namespace geo_indexing
}  // namespace arctern
