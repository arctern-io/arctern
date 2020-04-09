#pragma once
#include "gis/cuda/common/common.h"

namespace arctern {
namespace gis {
namespace cuda {
// TODO(dog): use global percision mode
DEVICE_RUNNABLE inline bool is_equal(double2 a, double2 b) {
  return abs(a.x - b.x) == 0 && abs(a.y - b.y) == 0;
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern