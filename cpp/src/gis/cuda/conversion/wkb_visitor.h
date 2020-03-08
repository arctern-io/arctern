#pragma once

template<typename WkbVisitorImpl>
struct WkbVisitor: public WkbVisitorImpl {
  using WkbVisitorImpl::WkbVisitorImpl;

  using WkbVisitorImpl::VisitMetaInt;

  using WkbVisitorImpl::VisitValues;

  __device__ void VisitPoint(int demensions) { VisitValues(demensions, 1); }

  __device__ void VisitLineString(int demensions) {
    auto size = VisitMetaInt();
    VisitValues(demensions, size);
  }

  __device__ void VisitPolygon(int demensions) {
    auto polys = VisitMetaInt();
    for (int i = 0; i < polys; ++i) {
      VisitLineString(demensions);
    }
  }
};
