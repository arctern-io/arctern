#include <string>
#include <iostream>
#include <stdint.h>

#include "arrow/render_api.h"
#include "make_point.h"

#define CHECK_STATUS(action)                            \
{                                                       \
    arrow::Status status = action;                      \
    if (!status.ok()) {                                 \
        printf("arrow status: %s, position: %s:%d\n",   \
            status.ToString().c_str(),                  \
            __FILE__,                                   \
            __LINE__ - 5);                              \
        exit(0);                                        \
    }                                                   \
}

std::shared_ptr<arrow::Array>
point_map(std::shared_ptr<arrow::Array> arr_x,
           std::shared_ptr<arrow::Array> arr_y) {
    return zilliz::render::get_pointmap(arr_x, arr_y);
}