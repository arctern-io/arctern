#pragma once
#include "common/gis_definitions.h"

namespace zilliz {
namespace gis {
namespace cpp {

void ST_distance(const GeometryVector& left, const GeometryVector& right, double* host_results);

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
