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
ST_point(const double *const ptr_x, const double *const ptr_y, const int64_t len);

std::shared_ptr<arrow::Array>
ST_point(std::shared_ptr<arrow::Array> ptr_x, std::shared_ptr<arrow::Array> ptr_y);

//points_list.size() % 2 ==0
//points_list.size() >= 4
std::shared_ptr<arrow::Array>
ST_line(const std::vector<const double *const> &points_list,
        const int64_t len);

//points_list.size() % 2 ==0
//points_list.size() >= 6
std::shared_ptr<arrow::Array>
ST_polygon(const std::vector<const double *const> &points_list,
           const int64_t len);

std::shared_ptr<arrow::Array>
ST_envelope(const double *const ptr_left_down_x,const double *const ptr_left_down_y,
            const double *const ptr_right_up_x,const double *const ptr_right_up_y,
            const int64_t len);

//arr_str_points lines of csv file
//123.456, 98.12
//45.98, 37.07
std::shared_ptr<arrow::Array>
ST_point(std::shared_ptr<arrow::Array> arr_str_points);

std::shared_ptr<arrow::Array>
ST_line(std::shared_ptr<arrow::Array> arr_str_points);

std::shared_ptr<arrow::Array>
ST_polygon(std::shared_ptr<arrow::Array> arr_str_points);

std::shared_ptr<arrow::Array>
ST_envelope(std::shared_ptr<arrow::Array> arr_str_points);

} // geometry
} // cpp
} // gis
} // zilliz