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
#include "gis/cuda/wkb/wkb_tag.h"

namespace arctern {
namespace gis {
namespace cuda {

template <typename WkbVisitorImpl>
struct WkbCodingVisitor : public WkbVisitorImpl {
  using WkbVisitorImpl::VisitByteOrder;
  using WkbVisitorImpl::VisitMetaInt;
  using WkbVisitorImpl::VisitMetaWkbTag;
  using WkbVisitorImpl::VisitValues;
  using WkbVisitorImpl::WkbVisitorImpl;

  __device__ void VisitPoint(int dimensions) { VisitValues(dimensions, 1); }

  __device__ void VisitLineString(int dimensions) {
    auto size = VisitMetaInt();
    VisitValues(dimensions, size);
  }

  __device__ void VisitPolygon(int dimensions) {
    auto size = VisitMetaInt();
    for (int i = 0; i < size; ++i) {
      //      auto polys = VisitMetaInt();
      //      VisitValues(dimensions, polys);
      VisitLineString(dimensions);
    }
  }

  __device__ void ConsumeHeader(WkbCategory expected_categroy, int expected_dimensions) {
    VisitByteOrder();
    WkbTag tag = VisitMetaWkbTag();
    assert(expected_dimensions == 2);
    assert(tag.get_space_type() == WkbSpaceType::XY);
  }

  __device__ void VisitMultiPoint(int dimensions) {
    auto size = VisitMetaInt();
    for (int i = 0; i < size; ++i) {
      ConsumeHeader(WkbCategory::kMultiPoint, dimensions);
      VisitPoint(dimensions);
    }
  }

  __device__ void VisitMultiLineString(int dimensions) {
    auto size = VisitMetaInt();
    for (int i = 0; i < size; ++i) {
      ConsumeHeader(WkbCategory::kMultiLineString, dimensions);
      VisitLineString(dimensions);
    }
  }
  __device__ void VisitMultiPolygon(int dimensions) {
    auto size = VisitMetaInt();
    for (int i = 0; i < size; ++i) {
      ConsumeHeader(WkbCategory::kMultiPolygon, dimensions);
      VisitPolygon(dimensions);
    }
  }

#define WKB_CATEGORY_CASE(category) \
  case WkbCategory::k##category: {  \
    Visit##category(dimensions);    \
    break;                          \
  }

  __device__ void VisitBody(WkbTag tag) {
    assert(tag.get_space_type() == WkbSpaceType::XY);
    constexpr auto dimensions = 2;
    switch (tag.get_category()) {
      WKB_CATEGORY_CASE(Point);
      WKB_CATEGORY_CASE(LineString);
      WKB_CATEGORY_CASE(Polygon);
      WKB_CATEGORY_CASE(MultiPoint);
      WKB_CATEGORY_CASE(MultiLineString);
      WKB_CATEGORY_CASE(MultiPolygon);
      default: {
        assert(false);
        break;
      }
    }
  }
#undef WKB_CATEGORY_CASE
};
}  // namespace cuda
}  // namespace gis
}  // namespace arctern
