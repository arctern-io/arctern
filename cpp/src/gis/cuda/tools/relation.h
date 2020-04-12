#pragma once
#include <thrust/complex.h>
#include <thrust/pair.h>

#include "gis/cuda/common/common.h"
#include "gis/cuda/tools/de9im_matrix.h"

namespace arctern {
namespace gis {
namespace cuda {
DEVICE_RUNNABLE inline bool IsEqual(double2 a, double2 b);

}  // namespace cuda
}  // namespace gis
}  // namespace arctern

#include "relation.impl.h"