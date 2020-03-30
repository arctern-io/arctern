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

#include <thrust/optional.h>

#include "gis/cuda/functor/bounding_box.h"

namespace arctern {
namespace gis {
namespace cuda {
namespace {

DEVICE_RUNNABLE int CalcSimplePointCount(WkbTag tag, const uint32_t*& meta_iter) {
  assert(tag.get_space_type() == WkbSpaceType::XY);
  switch (tag.get_category()) {
    case WkbCategory::kPoint: {
      return 1;
    }
    case WkbCategory::kLineString: {
      auto count = *meta_iter++;
      return count;
    }
    case WkbCategory::kPolygon: {
      int polygons = *meta_iter++;
      int total_count = 0;
      for (auto i = 0; i < polygons; ++i) {
        auto count = *meta_iter++;
        total_count += count;
      }
      return polygons;
    }
    default: {
      assert(false);
      return -1;
    }
  }
}
}  // namespace

DEVICE_RUNNABLE thrust::optional<int> CalcPointCount(WkbTag tag,
                                                     const uint32_t* meta_iter_raw) {
  assert(tag.get_space_type() == WkbSpaceType::XY);
  auto meta_iter = meta_iter_raw;
  auto multi_handler = [](WkbTag sub_tag, const uint32_t*& meta_iter) {

  };
  switch (tag.get_category()) {
    case WkbCategory::kPoint:
    case WkbCategory::kLineString:
    case WkbCategory::kPolygon: {
      return CalcSimplePointCount(tag, meta_iter);
    }
    case WkbCategory::kMultiPoint:
    case WkbCategory::kMultiPolygon:
    case WkbCategory::kMultiLineString: {
      auto size = *meta_iter++;
      auto total_count = 0;
      for (int i = 0; i < size; ++i) {
        auto fetched_tag = (WkbTag)*meta_iter++;
        auto sub_tag = RemoveMulti(tag);
        assert(sub_tag.data == fetched_tag.data);
        total_count += CalcSimplePointCount(sub_tag, meta_iter);
      }
      return total_count;
    }
    default: {
      return thrust::nullopt;
    }
  }
}

DEVICE_RUNNABLE BoundingBox CalcBoundingBox(WkbTag tag,
                                            ConstGpuContext::ConstIter& iter) {
  assert(tag.get_space_type() == WkbSpaceType::XY);
  auto point_count = CalcSimplePointCount();
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
