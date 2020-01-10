
#ifndef GIS_MAKE_POINT_H
#define GIS_MAKE_POINT_H

#include <string>
#include <iostream>
#include <stdint.h>

//#include "ogrsf_frmts.h"
//#include "gdal_utils.h"
#include "arrow/api.h"

std::shared_ptr<arrow::Array>
make_point(std::shared_ptr<arrow::Array> arr_x,
           std::shared_ptr<arrow::Array> arr_y);


#endif //GIS_MAKE_POINT_H
