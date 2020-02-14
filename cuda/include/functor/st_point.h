#pragma once
#include "gis/gis_definitions.h"

namespace zilliz {
namespace gis {
namespace cuda {
void ST_point(const double* xs, const double ys, GeometryVector& results);
}    // namespace cuda
}    // namespace gis
}    // namespace zilliz

