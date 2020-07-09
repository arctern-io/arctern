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

#include <memory>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "index/index_tree.h"

namespace arctern {
namespace geo_indexing {

class GeosIndex {
 public:
  GeosIndex();

  ~GeosIndex();

  void append(const std::vector<std::shared_ptr<arrow::Array>>& geos);

  std::vector<std::shared_ptr<arrow::Array>> near_road(
      const std::vector<std::shared_ptr<arrow::Array>>& points, const double distance);

  std::vector<std::shared_ptr<arrow::Array>> nearest_location_on_road(
      const std::vector<std::shared_ptr<arrow::Array>>& points);

  std::vector<std::shared_ptr<arrow::Array>> nearest_road(
      const std::vector<std::shared_ptr<arrow::Array>>& points);

  std::vector<std::shared_ptr<arrow::Array>> ST_IndexedWithin(
      const std::vector<std::shared_ptr<arrow::Array>>& points);

  std::vector<std::shared_ptr<arrow::Array>> query(
      const std::vector<std::shared_ptr<arrow::Array>>& inputs);

 private:
  IndexTree* index_;
};

}  // namespace geo_indexing
}  // namespace arctern
