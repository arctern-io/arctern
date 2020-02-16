#ifndef RENDER_H
#define RENDER_H

#include "arrow/api.h"

namespace zilliz {
namespace render {


std::shared_ptr<arrow::Array>
point_map(const std::shared_ptr<arrow::Array> &arr_x, const std::shared_ptr<arrow::Array> &arr_y);


std::shared_ptr<arrow::Array>
heat_map(const std::shared_ptr<arrow::Array> &arr_x, const std::shared_ptr<arrow::Array> &arr_y, const std::shared_ptr<arrow::Array> &arr_c);


} // render
} // zilliz

#endif