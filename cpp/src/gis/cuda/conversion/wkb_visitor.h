#pragma once
#include "gis/cuda/wkb/wkb_tag.h"

namespace arctern {
namespace gis {
namespace cuda {
template <typename WkbVisitorImpl>
struct WkbCodingVisitor : public WkbVisitorImpl {
  using WkbVisitorImpl::WkbVisitorImpl;

  using WkbVisitorImpl::VisitMetaInt;

  using WkbVisitorImpl::VisitValues;

  __device__ void VisitPoint(int dimensions) { VisitValues(dimensions, 1); }

  __device__ void VisitLineString(int dimensions) {
    auto size = VisitMetaInt();
    VisitValues(dimensions, size);
  }

  __device__ void VisitPolygon(int dimensions) {
    auto size = VisitMetaInt();
    for (int i = 0; i < size; ++i) {
      VisitLineString(dimensions);
    }
  }

  __device__ void VisitMultiPoint(int dimensions) {
    auto size = VisitMetaInt();
    for (int i = 0; i < size; ++i) {
      VisitPoint(dimensions);
    }
  }

  __device__ void VisitMultiLineString(int dimensions) {
    auto size = VisitMetaInt();
    for (int i = 0; i < size; ++i) {
      VisitMultiLineString(dimensions);
    }
  }
  __device__ void VisitMultiPolygon(int dimensions) {
    auto size = VisitMetaInt();
    for (int i = 0; i < size; ++i) {
      VisitMultiLineString(dimensions);
    }
  }

#define WKB_CATEGORY_CASE(category) \
  case WkbCategory::k##category: { Visit##category(dimensions); break; }

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
