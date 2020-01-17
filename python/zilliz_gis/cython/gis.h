
#ifndef GIS_H
#define GIS_H

#include "arrow/api.h"

std::shared_ptr<arrow::Array>
point_map(std::shared_ptr<arrow::Array> arr_x,
           std::shared_ptr<arrow::Array> arr_y);

std::shared_ptr<arrow::Array>
make_point(std::shared_ptr<arrow::Array> arr_x,
           std::shared_ptr<arrow::Array> arr_y);


#endif //GIS_H
