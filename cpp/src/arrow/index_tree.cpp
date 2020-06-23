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

#include "arrow/index_tree.h"

namespace arctern {
namespace geo_indexing {

GeosIndex::GeosIndex() {
  auto index_tree = new IndexTree(IndexType::kRTree);
  index_ = index_tree;
}

void GeosIndex::append(const std::vector<std::shared_ptr<arrow::Array> > &geos) {
  std::cout << "into geosindex append" << std::endl;
  index_->Append(geos);
}

std::vector<std::shared_ptr<arrow::Array>> GeosIndex::near_road(const std::vector<std::shared_ptr<arrow::Array> > &gps_points, const double distance) {
  std::cout << "into geosindex index_near_road" << std::endl;
  return index_->near_road(gps_points, distance);
}

}
}