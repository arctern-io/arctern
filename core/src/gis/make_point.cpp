#include <string>
#include <iostream>
#include <stdint.h>

#include "ogrsf_frmts.h"
#include "gdal_utils.h"
#include "arrow/api.h"

#define CHECK_STATUS(action)                        \
{                                                   \
    arrow::Status status = action;                  \
    if (!status.ok()) {                             \
        printf("%s\n", status.ToString().c_str());  \
        exit(0);                                    \
    }                                               \
}

std::shared_ptr<arrow::Array>
make_point(std::shared_ptr<arrow::Array> arr_x,
           std::shared_ptr<arrow::Array> arr_y) {

    auto d_arr_x = std::static_pointer_cast<arrow::DoubleArray>(arr_x);
    auto d_arr_y = std::static_pointer_cast<arrow::DoubleArray>(arr_y);

    std::shared_ptr<arrow::Array> point_arr;
    arrow::StringBuilder builder;
    // TODO: use cuda parallesim to improve this
    int64_t length = arr_x->length();
    assert(length == arr_y->length());
    for (int64_t i = 0; i < length; ++i) {
        auto x = d_arr_x->Value(i);
        auto y = d_arr_y->Value(i);
        OGRPoint point(x, y);
        CHECK_STATUS(builder.Append(point.exportToWkt()));
    }
    CHECK_STATUS(builder.Finish(&point_arr));

    return point_arr;
}
