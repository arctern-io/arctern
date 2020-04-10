#include "gis/cuda/tools/relation.h"

namespace arctern {
namespace gis {
namespace cuda {
__device__ bool PointOnInnerLineString(double2 left_point, int right_size,
                                       const double2* right_points) {
  // endpoints not included
  bool point_in_line = false;
  for (int i = 0; i < right_size - 1; ++i) {
    auto u = right_points[i];
    auto v = right_points[i + 1];
    if (i != 0 && IsEqual(u, left_point)) {
      point_in_line = true;
      break;
    }
    if (IsPointInLine(left_point, u, v)) {
      point_in_line = true;
      break;
    }
  }
  return point_in_line;
}

__device__ bool LineStringOnLineString(int left_size, const double2* left_points,
                                       int right_size, const double2* right_points) {
  //
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
