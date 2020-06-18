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

#include <arrow/api.h>

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "gis/wkb_types.h"
#include "utils/arrow_alias.h"

namespace arctern {
namespace gis {
namespace dispatch {

using GroupedWkbTypes = std::set<WkbTypes>;

struct GeometryTypeMasks {
 public:
  using EncodeUid = uint32_t;
  struct Info {
   public:
    // TODO(dog): remove mask since encode_uids contains all the info
    // TODO(dog): now make unittest happy
    // This field contains mask[i] = is_matched(data[i]);
    std::vector<bool> mask;
    // TODO(dog): remove later since mask_count === indexes.size()
    // TODO(dog): now make unittest happy
    // This field contains counts of true in mask, or size of indexes
    int64_t mask_count = 0;
  };
  const auto& get_info(const GroupedWkbTypes& grouped_types) const& {
    auto iter = dict.find(grouped_types);
    if (iter == dict.end()) {
      throw std::runtime_error("check is_unique first");
    }
    return iter->second;
  }

  auto&& get_info(const GroupedWkbTypes& grouped_types) && {
    auto iter = dict.find(grouped_types);
    if (iter == dict.end()) {
      throw std::runtime_error("check is_unique first");
    }
    return std::move(iter->second);
  }

  // helper function
  const auto& get_mask(const GroupedWkbTypes& grouped_types) const& {
    return get_info(grouped_types).mask;
  }
  auto&& get_mask(const GroupedWkbTypes& grouped_types) && {
    return std::move(std::move(*this).get_info(grouped_types).mask);
  }

  auto get_count(const GroupedWkbTypes& grouped_types) const {
    return get_info(grouped_types).mask_count;
  }

 public:
  // If the given geometries share identical type, this field will be set true.
  bool is_unique_type;
  // This field is valid only if 'is_unique_type' equals true.
  GroupedWkbTypes unique_type;
  // extra fields for
 public:
  // This contains Info for each geometry type, enable only if !unique_type
  std::map<GroupedWkbTypes, Info> dict;
};

class GeometryTypeScanner {
 public:
  virtual std::shared_ptr<GeometryTypeMasks> Scan() const = 0;

  const std::vector<GroupedWkbTypes>& types() const { return types_; }

  std::vector<GroupedWkbTypes>& mutable_types() { return types_; }

 private:
  std::vector<GroupedWkbTypes> types_;
};

class MaskResult {
 public:
  enum class Status {
    kInvalid,
    kOnlyFalse,
    kMixed,
    kOnlyTrue,
  };

  explicit MaskResult(Status status) : status_(status) {
    assert(status != Status::kMixed);
  }

  explicit MaskResult(std::vector<bool>&& mask)
      : status_(Status::kMixed), mask_(std::move(mask)) {}

  MaskResult() = default;
  MaskResult(const std::shared_ptr<arrow::StringArray>& geometries,
             const GroupedWkbTypes& supported) {
    this->AppendFilter(geometries, supported);
  }

  MaskResult(const std::shared_ptr<arrow::BinaryArray>& geometries,
             const GroupedWkbTypes& supported) {
    this->AppendFilter(geometries, supported);
  }

  void AppendFilter(const std::shared_ptr<arrow::StringArray>& geometries,
                    const GroupedWkbTypes& supported_type);

  void AppendFilter(const std::shared_ptr<arrow::BinaryArray>& geometries,
                    const GroupedWkbTypes& supported_type);

  Status get_status() const { return status_; }
  const std::vector<bool>& get_mask() const { return mask_; }

 private:
  // bitwise append
  void AppendFilter(const GeometryTypeScanner& scanner,
                    const GroupedWkbTypes& supported_type);

 private:
  Status status_ = Status::kOnlyTrue;
  // valid only when status = kMixed
  std::vector<bool> mask_;
};

MaskResult RelateSelector(const WkbArrayPtr& left_geo, const WkbArrayPtr& right_geo);

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern
