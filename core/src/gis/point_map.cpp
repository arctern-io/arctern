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

    auto d_arr_x = std::static_pointer_cast<arrow::DoubleArray>(arr_x);
    auto d_arr_y = std::static_pointer_cast<arrow::DoubleArray>(arr_y);

    std::shared_ptr<arrow::Array> point_arr;



    // TODO: use cuda parallesim to improve this
    int64_t length = arr_x->length();
    assert(length == arr_y->length());

    arrow::StringBuilder builder;
    builder.Append("point_map");
    CHECK_STATUS(builder.Finish(&point_arr));

    return point_arr;
}
