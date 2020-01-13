#pragma once
#include "pointmap.h"

namespace zilliz {
namespace render {

std::shared_ptr<arrow::Array>
get_pointmap(arrow::ArrayVector input_array);


} //namespace render
} //namespace zilliz