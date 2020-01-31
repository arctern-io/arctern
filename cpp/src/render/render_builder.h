#pragma once
#include "render/2d/pointmap.h"
#include "render/2d/heatmap.h"

namespace zilliz {
namespace render {

std::pair<uint8_t*, int64_t>
pointmap(uint32_t* arr_x, uint32_t* arr_y, int64_t num_vertices);

template<typename T>
std::pair<uint8_t*, int64_t>
heatmap(uint32_t* arr_x, uint32_t* arr_y, T* arr_c, int64_t num_vertices);

} //namespace render
} //namespace zilliz
#include "render_builder_impl.h"