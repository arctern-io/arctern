
#ifndef GIS_POINT_MAP_H
#define GIS_POINT_MAP_H

#include <string>
#include <iostream>
#include <stdint.h>

#include "arrow/api.h"

std::shared_ptr<arrow::Array>
point_map(std::shared_ptr<arrow::Array> arr_x,
           std::shared_ptr<arrow::Array> arr_y);


#endif //GIS_POINT_MAP_H
