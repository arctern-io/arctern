#pragma once
#include "gis/gis_definitions.h"

namespace zilliz {
namespace gis {
namespace cuda {

void ST_distance(const GeometryVector& left,
                 const GeometryVector& right,
                 double* host_results);

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
