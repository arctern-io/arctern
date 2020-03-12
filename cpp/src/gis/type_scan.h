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

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "gis/wkb_types.h"

namespace arctern {
namespace gis {

using GroupedWkbTypes = std::set<WkbTypes>;

struct GeometryTypeMasks {
 public:
  using EncodeUid = uint32_t;
  struct Info {
   public:
    // TODO(dog): remove masks since encode_uids contains all the info
    // TODO(dog): now make unittest happy
    // This field contains masks[i] = is_matched(data[i]);
    std::vector<bool> masks;
    // TODO(dog): remove later since mask_count === indexes.size()
    // TODO(dog): now make unittest happy
    // This field contains counts of true in masks, or size of indexes
    int64_t mask_counts = 0;
    // This field contains unique id(uid) for each class
    EncodeUid encode_uid;
  };
  const auto& get_info(const GroupedWkbTypes& grouped_types) const {
    auto iter = dict.find(grouped_types);
    assert(iter != dict.end());
    return iter->second;
  }

  // helper function
  const auto& get_masks(const GroupedWkbTypes& grouped_types) const {
    return get_info(grouped_types).masks;
  }
  const auto& get_counts(const GroupedWkbTypes& grouped_types) const {
    return get_info(grouped_types).mask_counts;
  }

  EncodeUid get_encode_uid(const GroupedWkbTypes& grouped_types) {
    return get_info(grouped_types).encode_uid;
  }

 public:
  // If the given geometries share identical type, this field will be set true.
  bool is_unique_type;
  // This field is valid only if 'is_unique_type' equals true.
  GroupedWkbTypes unique_type;
  // extra fields for
 public:
  // This field contains uid for each data
  std::vector<EncodeUid> encode_uids;
  int num_scan_classes = 0;
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
