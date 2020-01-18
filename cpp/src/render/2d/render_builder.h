#pragma once
#include "render/2d/pointmap.h"
#include "render/2d/heatmap.h"

namespace zilliz {
namespace render {

std::pair<std::shared_ptr<uint8_t>, int64_t>
pointmap(std::shared_ptr<uint32_t > arr_x, std::shared_ptr<uint32_t > arr_y, int64_t num_vertices);

template<typename T>
std::pair<std::shared_ptr<uint8_t>, int64_t>
heatmap(std::shared_ptr<uint32_t > arr_x, std::shared_ptr<uint32_t > arr_y, std::shared_ptr<T> arr_c, int64_t num_vertices);

} //namespace render
} //namespace zilliz
#include "render/2d/render_builder_impl.h"