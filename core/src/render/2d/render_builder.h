#pragma once
#include "render/2d/pointmap.h"

namespace zilliz {
namespace render {

std::pair<std::shared_ptr<uint8_t>, int64_t>
pointmap(std::shared_ptr<uint32_t > arr_x, std::shared_ptr<uint32_t > arr_y, int64_t num_vertices);


} //namespace render
} //namespace zilliz