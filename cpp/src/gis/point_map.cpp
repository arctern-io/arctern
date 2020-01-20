#include <string>
#include <iostream>
#include <stdint.h>

//#include "ogrsf_frmts.h"
//#include "gdal_utils.h"
#include "arrow/api.h"
#include "make_point.h"

#define CHECK_STATUS(action)                        \
{                                                   \
    arrow::Status status = action;                  \
    if (!status.ok()) {                             \
        printf("%s\n", status.ToString().c_str());  \
        exit(0);                                    \
    }                                               \
}

std::shared_ptr<arrow::Array>
point_map(std::shared_ptr<arrow::Array> arr_x,
           std::shared_ptr<arrow::Array> arr_y) {
    return render
}
