#pragma once

#include "render/engine/plan/prim/circle.h"
#include "render/engine/operator/dataset.h"
#include "render/engine/plan/node/plan_node_weighted_color_circles_2d.h"


namespace zilliz {
namespace render {
namespace engine {


class ColorParser {
 public:
    static CircleParams2D
    GetCircleParams(ColorStyle color_style, double ratio);
};


} // namespace engine
} // namespace render
} // namespace zilliz


