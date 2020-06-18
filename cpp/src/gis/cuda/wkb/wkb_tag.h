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

#pragma once
#include <cstdint>

#include "gis/cuda/common/common.h"
#include "gis/wkb_types.h"
namespace arctern {
namespace gis {
namespace cuda {

using WkbSpaceType = WkbSpaceType;
using WkbCategory = WkbCategory;

struct WkbTag {
  WkbTag() = default;
  constexpr DEVICE_RUNNABLE WkbTag(WkbTypes type) : data((uint32_t)type) {}
  constexpr DEVICE_RUNNABLE WkbTag(WkbCategory category, WkbSpaceType group)
      : data((uint32_t)category + (uint32_t)group * kWkbSpaceTypeEncodeBase) {}

  constexpr explicit DEVICE_RUNNABLE WkbTag(uint32_t data) : data(data) {}
  constexpr DEVICE_RUNNABLE WkbCategory get_category() {
    return static_cast<WkbCategory>(data % kWkbSpaceTypeEncodeBase);
  }
  constexpr DEVICE_RUNNABLE WkbSpaceType get_space_type() {
    return static_cast<WkbSpaceType>(data / kWkbSpaceTypeEncodeBase);
  }
  constexpr DEVICE_RUNNABLE int get_dimension() {
    auto cat = get_category();
    switch (cat) {
      case WkbCategory::kPoint:
      case WkbCategory::kMultiPoint: {
        return 0;
      }
      case WkbCategory::kLineString:
      case WkbCategory::kMultiLineString: {
        return 1;
      }
      case WkbCategory::kPolygon:
      case WkbCategory::kMultiPolygon: {
        return 2;
      }
      default: {
        // unknown dimension
        return -1;
      }
    }
  }
  uint32_t data;
};

inline DEVICE_RUNNABLE WkbCategory RemoveMulti(WkbCategory category) {
  switch (category) {
    case WkbCategory::kMultiPoint: {
      return WkbCategory::kPoint;
    }
    case WkbCategory::kMultiLineString: {
      return WkbCategory::kLineString;
    }
    case WkbCategory::kMultiPolygon: {
      return WkbCategory::kPolygon;
    }
    default: {
      assert(false);
      return WkbCategory::kUnknown;
    }
  }
}

inline DEVICE_RUNNABLE WkbTag RemoveMulti(WkbTag tag) {
  return WkbTag(RemoveMulti(tag.get_category()), tag.get_space_type());
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
