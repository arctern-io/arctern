#pragma once

#include "gis/cuda/common/gis_definitions.h"
namespace zilliz {
namespace gis {
namespace cuda {
void ST_Point(const double* xs, const double* ys, int size, GeometryVector& results);
}    // namespace cuda
}    // namespace gis
}    // namespace zilliz

