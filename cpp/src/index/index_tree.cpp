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
#include "index/index_tree.h"
#include "render/utils/render_utils.h"

namespace arctern {
namespace geo_indexing {

#define PI 3.14159
#define RAD2DEG(x) ((x)*180.0 / PI)

void IndexTree::Create(IndexType type) {
  using RTree = GEOS_DLL::geos::index::strtree::STRtree;
  using QuadTree = GEOS_DLL::geos::index::quadtree::Quadtree;
  switch (type) {
    case IndexType::kQuadTree: {
      tree_ = std::make_unique<QuadTree>();
      break;
    }
    case IndexType::kRTree: {
      tree_ = std::make_unique<RTree>();
      break;
    }
    default: {
      throw std::invalid_argument("IndexType is Invalid");
      break;
    }
  }
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

const std::vector<OGRGeometry*> IndexTree::map_match_query(const OGRGeometry* gps_point,
                                                           const bool greedy_search,
                                                           const double distance) const {
  auto deg_distance = RAD2DEG(distance / 6371251.46);
  std::vector<void*> matches;
  std::vector<OGRGeometry*> results;
  {
    OGREnvelope ogr_env;
    gps_point->getEnvelope(&ogr_env);
    do {
      results.clear();
      matches.clear();
      geos::geom::Envelope env(ogr_env.MinX - deg_distance, ogr_env.MaxX + deg_distance,
                               ogr_env.MinY - deg_distance, ogr_env.MaxY + deg_distance);
      get_tree()->query(&env, matches);
      for (auto match : matches) {
        // match(void*) contains index as binary representation.
        auto index = reinterpret_cast<size_t>(match);
        auto geo = get_geometry(index);
        results.emplace_back(geo);
      }
      deg_distance *= 2;
      if (!results.empty() || deg_distance > ogr_env.MinX + 90.0 ||
          deg_distance > 90.0 - ogr_env.MinX)
        break;
    } while (greedy_search);
  }

  return results;
}

const std::vector<std::shared_ptr<arrow::Array>> IndexTree::query(
        const OGRGeometry *input) const {
  std::vector<std::shared_ptr<arrow::Array>> results_arrow(1);
  std::vector<void*> matches;
  OGREnvelope ogr_env;
  input->getEnvelope(&ogr_env);
  matches.clear();
  geos::geom::Envelope env(ogr_env.MinX, ogr_env.MaxX,
                           ogr_env.MinY, ogr_env.MaxY);
  get_tree()->query(&env, matches);

  arrow::Int32Builder builder;

  for (auto match : matches) {
    // match(void*) contains index as binary representation.
    auto index = reinterpret_cast<size_t>(match);
    auto final_index = static_cast<int>(index);
    CHECK_ARROW(builder.Append(final_index));
  }
  CHECK_ARROW(builder.Finish(&(results_arrow[0])));

  return results_arrow;
}

}  // namespace geo_indexing
}  // namespace arctern
