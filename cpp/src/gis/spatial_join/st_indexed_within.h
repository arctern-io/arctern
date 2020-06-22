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
#include <memory>
#include <string>
#include <vector>

#include "index/index.h"
#include "utils/arrow_alias.h"

namespace arctern {
namespace gis {
namespace spatial_join {

using IndexType = arctern::geo_indexing::IndexType;

std::vector<std::shared_ptr<arrow::Array>> ST_IndexedWithin(
    const std::vector<std::shared_ptr<arrow::Array>>& points,
    const std::vector<std::shared_ptr<arrow::Array>>& polygons, std::string index_type);

}  // namespace spatial_join
}  // namespace gis
}  // namespace arctern
