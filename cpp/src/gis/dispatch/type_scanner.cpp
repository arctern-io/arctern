// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "gis/dispatch/type_scanner.h"

#include <arrow/api.h>

#include "gis/dispatch/wkb_type_scanner.h"
#include "gis/dispatch/wkt_type_scanner.h"

namespace arctern {
namespace gis {
namespace dispatch {

void MaskResult::AppendFilter(const GeometryTypeScanner& scanner,
                              const GroupedWkbTypes& supported) {
  auto type_masks = scanner.Scan();
  auto status =
      !type_masks->is_unique_type
          ? Status::kMixed
          : type_masks->unique_type == supported ? Status::kOnlyTrue : Status::kOnlyFalse;
  if ((int)this->status_ < (int)status) {
    return;
  } else if ((int)this->status_ > (int)status) {
    this->status_ = status;
    if (status == Status::kMixed) {
      this->mask_ = std::move(*type_masks).get_mask(supported);
      return;
    } else {
      this->mask_.clear();
      this->mask_.shrink_to_fit();
      return;
    }
  } else {
    if (status != Status::kMixed) {
      return;
    }
    const auto& mask = type_masks->get_mask(supported);
    assert(mask.size() == this->mask_.size());
    bool has_true = false;
    for (auto i = 0; i < mask.size(); ++i) {
      bool flag = this->mask_[i] && mask[i];
      this->mask_[i] = flag;
      has_true = has_true || flag;
    }

    // downgrade to last
    if (!has_true) {
      this->status_ = Status::kOnlyFalse;
      this->mask_.clear();
      this->mask_.shrink_to_fit();
    }
  }
}

void MaskResult::AppendFilter(const std::shared_ptr<arrow::StringArray>& geometries,
                              const GroupedWkbTypes& supported_types) {
  dispatch::WktTypeScanner scanner(geometries);
  scanner.mutable_types().push_back(supported_types);
  this->AppendFilter(scanner, supported_types);
}

void MaskResult::AppendFilter(const std::shared_ptr<arrow::BinaryArray>& geometries,
                              const GroupedWkbTypes& supported_types) {
  dispatch::WkbTypeScanner scanner(geometries);
  scanner.mutable_types().push_back(supported_types);
  this->AppendFilter(scanner, supported_types);
}

using string_view = decltype(WkbArrayPtr()->GetView(0));

static inline WkbTypes GetType(string_view view) {
  WkbTypes type;
  assert(view.size() >= sizeof(WkbByteOrder) + sizeof(WkbTypes));
  memcpy(&type, view.data() + sizeof(WkbByteOrder), sizeof(WkbTypes));
  return type;
}

static constexpr uint64_t bundle(WkbTypes left, WkbTypes right) {
  return ((uint64_t)left << 32) | ((uint64_t)right);
}

// select GPU-enabled
MaskResult RelateSelector(const WkbArrayPtr& left_geo, const WkbArrayPtr& right_geo) {
  assert(left_geo->length() == right_geo->length());
  auto length = left_geo->length();
  if (length == 0) {
    return MaskResult(MaskResult::Status::kOnlyFalse);
  }
  std::vector<bool> mask(length);
  int64_t count = 0;

  for (int64_t index = 0; index < length; ++index) {
    bool flag = false;
    if (left_geo->IsNull(index) || right_geo->IsNull(index)) {
      flag = false;
    } else {
      auto left_type = GetType(left_geo->GetView(index));
      auto right_type = GetType(right_geo->GetView(index));
      constexpr auto kPoint = WkbTypes::kPoint;
      constexpr auto kLineString = WkbTypes::kLineString;
      constexpr auto kPolygon = WkbTypes::kPolygon;
      switch (bundle(left_type, right_type)) {
        case bundle(kPoint, kPoint):
        case bundle(kPoint, kLineString):
        case bundle(kLineString, kPoint):
        case bundle(kLineString, kLineString):
        case bundle(kPoint, kPolygon):
        case bundle(kPolygon, kPoint): {
          flag = true;
          break;
        }
        default: {
          flag = false;
          break;
        }
      }
    }
    mask[index] = flag;
    count += flag ? 1 : 0;
  }

  if (count == length) {
    return MaskResult(MaskResult::Status::kOnlyTrue);
  } else if (count == 0) {
    return MaskResult(MaskResult::Status::kOnlyFalse);
  } else {
    return MaskResult(std::move(mask));
  }
}

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern
