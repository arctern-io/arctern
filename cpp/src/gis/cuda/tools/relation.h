#pragma once
#include <thrust/complex.h>

#include "gis/cuda/common/common.h"

namespace arctern {
namespace gis {
namespace cuda {
// TODO(dog): use global percision mode
DEVICE_RUNNABLE inline bool IsEqual(double2 a, double2 b) {
  return abs(a.x - b.x) == 0 && abs(a.y - b.y) == 0;
}

DEVICE_RUNNABLE inline auto to_complex(double2 point) {
  return thrust::complex<double>(point.x, point.y);
}

DEVICE_RUNNABLE inline bool IsPointInLine(double2 point_raw, double2 line_beg,
                                          double2 line_end) {
  thrust::complex<double> point = to_complex(point_raw);
  auto v0 = point - to_complex(line_beg);
  auto v1 = point - to_complex(line_end);
  auto result = v0 * thrust::conj(v1);

  return result.imag() == 0 && result.real() < 0;
}

__device__ bool PointOnInnerLineString(double2 left_point, int right_size,
                                       const double2* right_points);

}  // namespace cuda
}  // namespace gis
}  // namespace arctern