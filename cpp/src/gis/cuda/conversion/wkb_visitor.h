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
    auto polys = VisitMetaInt();
    for (int i = 0; i < polys; ++i) {
      VisitLineString(dimensions);
    }
  }
  __device__ void VisitBody(WkbTag tag) {
    assert(tag.get_space_type() == WkbSpaceType::XY);
    constexpr auto dimensions = 2;
    switch (tag.get_category()) {
      case WkbCategory::kPoint: {
        VisitPoint(dimensions);
        break;
      }
      case WkbCategory::kLineString: {
        VisitLineString(dimensions);
        break;
      }
      case WkbCategory::kPolygon: {
        VisitPolygon(dimensions);
        break;
      }
      default: {
        assert(false);
        break;
      }
    }
  }
};
}  // namespace cuda
}  // namespace gis
}  // namespace arctern
