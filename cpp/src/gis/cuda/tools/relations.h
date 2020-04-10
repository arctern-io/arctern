#pragma once
#include "gis/cuda/common/common.h"

namespace arctern {
namespace gis {
namespace cuda {
// TODO(dog): use global percision mode
DEVICE_RUNNABLE inline bool IsEqual(double2 a, double2 b) {
  return abs(a.x - b.x) == 0 && abs(a.y - b.y) == 0;
}
double2 operator-(double2 u, double2 v) { return {u.x - v.x, u.y - v.y}; }

double2 complex_mul(double2 u, double2 v) {
  // complex multiply
  return {u.x * v.x - u.y * v.y, u.x * v.y + u.y * v.x};
}

double2 complex_conjugate(double2 u) {
  // negate the imaginary part
  return {u.x, -u.y};
}

DEVICE_RUNNABLE inline bool IsPointInLine(double2 point, double2 line_beg,
                                          double2 line_end) {
  auto v0 = point - line_beg;
  auto v1 = point - line_end;

  auto result = complex_mul(v0, complex_conjugate(v1));
  return result.y == 0 && result.x < 0;
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern