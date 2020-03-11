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
    auto iter = dict.find(grouped_types);
    assert(iter != dict.end());
    return iter->second.masks;
  }

  const auto& get_counts(const GroupedWkbTypes& grouped_types) const {
    auto iter = dict.find(grouped_types);
    assert(iter != dict.end());
    auto& record = iter->second;

    //    assert(record.mask_counts == record.indexes.size());
    return record.mask_counts;
  }

 public:
  // If the given geometries share identical type, this field will be set true.
  bool is_unique_type;
  // This field is valid only if 'is_unique_type' equals true.
  GroupedWkbTypes unique_type;
  struct Info {
    // This field contains masks[i] = is_matched(data[i]);
    std::vector<bool> masks;
    // TODO(dog): remove later since mask_count === indexes.size()
    // TODO(dog): now make unittest happy
    // This field contains counts of true in masks, or size of indexes
    int64_t mask_counts = 0;
    // This field contains index where is_matched(data[index]) holds
    std::vector<int> indexes;
  };

  // This contains Info for each geometry type, enable only if !unique_type
  std::map<GroupedWkbTypes, Info> dict;
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
