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
#include <set>
#include <vector>
#include <memory>

#include "gis/wkb_types.h"

namespace zilliz {
namespace gis {

using GroupedWkbTypes = std::set<WkbTypes>;

struct GeometryTypeMasks {
  // This field contains masks for each geometry type.
  std::map<WkbTypes, std::vector<bool>> type_masks;
  // This field contains mask counts for each geometry type.
  std::map<WkbTypes, int64_t> type_mask_counts;
  // This field contains masks for each grouped geometry type.
  std::map<GroupedWkbTypes, std::vector<bool>> grouped_type_masks;
  // This field contains mask counts for each grouped geometry type.
  std::map<GroupedWkbTypes, int64_t> grouped_type_mask_counts;
  // If the given geometries share identical type, this field will be set true.
  bool is_unique_type;
  // This field is valid only if 'is_unique_type' equals true.
  WkbTypes unique_type = WkbTypes::kUnknown;
  // If the given geometries share identical grouped types, this field will be set true.
  bool is_unique_grouped_types;
  // This field is valid only if 'is_unique_grouped_types' equals true.
  GroupedWkbTypes unique_grouped_types;
};

class GeometryTypeScanner {
 public:
  virtual std::shared_ptr<GeometryTypeMasks> Scan() = 0;

  const std::vector<WkbTypes>& types() { return types_; }

  const std::vector<GroupedWkbTypes>& grouped_types() { return grouped_types_; }

  std::vector<WkbTypes>& mutable_types() { return types_; }

  std::vector<GroupedWkbTypes>& mutable_grouped_types() { return grouped_types_; }

 private:
  std::vector<WkbTypes> types_;
  std::vector<GroupedWkbTypes> grouped_types_;
};

}  // namespace gis
}  // namespace zilliz
