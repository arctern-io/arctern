#pragma once
#include "gis/cuda/common/gis_definitions.h"

namespace zilliz {
namespace gis {
namespace cuda {

void ST_Distance(const GeometryVector& left_vec, const GeometryVector& right_vec,
                 double* host_results);

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
