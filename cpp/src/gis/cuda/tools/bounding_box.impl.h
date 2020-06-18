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
#include "gis/cuda/tools/bounding_box.h"

namespace arctern {
namespace gis {
namespace cuda {
namespace internal {

DEVICE_RUNNABLE inline int CalcSimplePointCountImpl(WkbTag tag,
                                                    const uint32_t*& meta_iter) {
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
      return total_count;
    }
    default: {
      assert(false);
      return -1;
    }
  }
}

DEVICE_RUNNABLE inline int CalcPointCount(WkbTag tag, const uint32_t*& meta_iter) {
  assert(tag.get_space_type() == WkbSpaceType::XY);
  switch (tag.get_category()) {
    case WkbCategory::kPoint:
    case WkbCategory::kLineString:
    case WkbCategory::kPolygon: {
      return CalcSimplePointCountImpl(tag, meta_iter);
    }
    case WkbCategory::kMultiPoint:
    case WkbCategory::kMultiPolygon:
    case WkbCategory::kMultiLineString: {
      auto size = *meta_iter++;
      auto total_count = 0;
      for (int i = 0; i < size; ++i) {
        auto fetched_tag = (WkbTag)*meta_iter++;
        auto sub_tag = RemoveMulti(tag);
        (void)fetched_tag;  // used in assert, maybe unused
        assert(sub_tag.data == fetched_tag.data);
        total_count += CalcSimplePointCountImpl(sub_tag, meta_iter);
      }
      return total_count;
    }
    default: {
      assert(false);
      return 0;
    }
  }
}
}  // namespace internal

DEVICE_RUNNABLE inline BoundingBox CalcBoundingBox(WkbTag tag,
                                                   ConstGpuContext::ConstIter& iter) {
  assert(tag.get_space_type() == WkbSpaceType::XY);
  auto point_count = internal::CalcPointCount(tag, iter.metas);
  auto value2 = iter.read_value_ptr<double2>(point_count);

  BoundingBox bbox;
  for (auto i = 0; i < point_count; ++i) {
    bbox.Update(value2[i]);
  }
  assert(value2 + point_count == (const double2*)iter.values);
  return bbox;
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
