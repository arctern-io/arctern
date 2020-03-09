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

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "gis/wkb_types.h"

namespace arctern {
namespace gis {

using GroupedWkbTypes = std::set<WkbTypes>;

struct GeometryTypeMasks {
  // This field contains masks for each geometry type.
  std::map<GroupedWkbTypes, std::vector<bool>> type_masks;
  // This field contains mask counts for each geometry type.
  std::map<GroupedWkbTypes, int64_t> group_mask_counts;
  // If the given geometries share identical type, this field will be set true.
  bool is_unique_group;
  // This field is valid only if 'is_unique_group' equals true.
  GroupedWkbTypes unique_group;
};

class GeometryTypeScanner {
 public:
  virtual std::shared_ptr<GeometryTypeMasks> Scan() = 0;

  const std::vector<GroupedWkbTypes>& types() { return types_; }

  std::vector<GroupedWkbTypes>& mutable_types() { return types_; }

 private:
  std::vector<GroupedWkbTypes> types_;
};

}  // namespace gis
}  // namespace arctern
