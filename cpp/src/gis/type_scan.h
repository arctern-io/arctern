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
  // helper function
  const auto& get_masks(const GroupedWkbTypes& grouped_types) const {
    auto iter = dict_.find(grouped_types);
    assert(iter != dict_.end());
    return iter->second.masks;
  }

  const auto& get_counts(const GroupedWkbTypes& grouped_types) const {
    auto iter = dict_.find(grouped_types);
    assert(iter != dict_.end());
    return iter->second.counts;
  }

 public:
  // If the given geometries share identical type, this field will be set true.
  bool is_unique_type;
  // This field is valid only if 'is_unique_type' equals true.
  GroupedWkbTypes unique_type;
  struct Info {
    // This field contains masks for each geometry type.
    std::vector<bool> masks;
    // This field contains mask counts for each geometry type.
    int64_t counts;
  };
  std::map<GroupedWkbTypes, Info> dict_;
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
