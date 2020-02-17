#pragma once
#include "gis/gis_definitions.h"

namespace zilliz {
namespace gis {
namespace cuda {

void ST_Area(const GeometryVector& vec, double* host_results);

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
