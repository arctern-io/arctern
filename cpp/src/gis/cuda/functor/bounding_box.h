#pragma once
#include <thrust/pair.h>

#include "gis/cuda/common/common.h"

namespace arctern {
namespace gis {
namespace cuda {

constexpr double inf = std::numeric_limits<double>::max();
struct MinMax {
  DEVICE_RUNNABLE MinMax() : min(+inf), max(-inf) {}
  DEVICE_RUNNABLE void update(double value) {
    min = value < min ? value : min;
    max = value > max ? value : max;
  }
  DEVICE_RUNNABLE bool is_trivial() const { return min == max; }
  DEVICE_RUNNABLE bool is_valid() const { return min < max; }

 public:
  double min;
  double max;
};

class BoundingBox {
 public:
  DEVICE_RUNNABLE MinMax get_xs() const { return xs_; }
  DEVICE_RUNNABLE MinMax get_ys() const { return ys_; }
  DEVICE_RUNNABLE void update(double2 value) {
    xs_.update(value.x);
    ys_.update(value.y);
  }

 private:
  MinMax xs_;
  MinMax ys_;
};



}  // namespace cuda
}  // namespace gis
}  // namespace arctern
