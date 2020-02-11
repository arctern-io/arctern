#pragma once
#include "common/gis_definitions.h"

namespace zilliz {
namespace gis {
namespace cpp {

void ST_area(const GeometryVector& vec, double* host_results);

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
