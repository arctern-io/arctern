#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <stdint.h>

#include "arrow/api.h"
#include "arrow/array.h"

#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

namespace zilliz {
namespace gis {
namespace cpp {
namespace gemetry {

std::shared_ptr<arrow::Array>
ST_make_point(const double *const ptr_x, const double *const ptr_y, const int64_t len);

std::shared_ptr<arrow::Array>
ST_make_point(std::shared_ptr<arrow::Array> arr_x,
              std::shared_ptr<arrow::Array> arr_y);

} // geometry
} // cpp
} // gis
} // zilliz