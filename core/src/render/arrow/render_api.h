#pragma once

#include "arrow/api.h"

namespace zilliz {
namespace render {

std::shared_ptr<arrow::Array>
get_pointmap(std::shared_ptr<arrow::Array> arr_x, std::shared_ptr<arrow::Array> arr_y);

std::shared_ptr<arrow::Array>
get_heatmap(std::shared_ptr<arrow::Array> arr_x, std::shared_ptr<arrow::Array> arr_y, std::shared_ptr<arrow::Array> arr_c);

//std::shared_ptr<arrow::Array>
//get_pointmap(std::shared_ptr<arrow::Array> arr_x);
}
}
