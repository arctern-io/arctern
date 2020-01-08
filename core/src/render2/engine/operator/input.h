#pragma once


#include "arrow/api.h"


namespace zilliz {
namespace render {
namespace engine {


struct Input {
 public:
    arrow::ArrayVector array_vector;
    std::shared_ptr<Vega> vega;
};


} // namespace engine
} // namespace render
} // namespace zilliz
