#pragma once


#include "arrow/api.h"
#include "render/utils/vega/vega.h"


namespace zilliz {
namespace render {

struct Input {
 public:
    arrow::ArrayVector array_vector;
    std::shared_ptr<Vega> vega;
};

using InputPtr = std::shared_ptr<Input>;

} // namespace render
} // namespace zilliz
