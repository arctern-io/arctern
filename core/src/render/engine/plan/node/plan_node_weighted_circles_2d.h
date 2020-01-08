#pragma once

#include <map>
#include "render/engine/plan/plan/plan_node.h"
#include "render/engine/plan/prim/circle.h"
#include "render/engine/common/table_id.h"
#include "render/engine/common/window_params.h"
#include "chewie/grpc/client.h"
#include "chewie/cache/object/cache_hint.h"
#include "render/engine/plan/node/plan_node_weighted_color_circles_2d.h"


namespace zilliz {
namespace render {
namespace engine {

using CacheHint = zilliz::chewie::CacheHint;

class PlanNodeWeightedCircle2D : public PlanNode {
 public:
    void
    Apply(Visitor &visitor) override {visitor.Visit(*this);}

    const ColorStyle
    color_style() const { return color_style_; }

    ColorStyle &
    mutable_color_style() { return color_style_; }

    double &
    count_start() { return count_start_; }

    double &
    count_end() { return count_end_; }

    double &
    mutable_count_start() { return count_start_; }

    double &
    mutable_count_end() { return count_end_; }

    void
    set_color_style(ColorStyle color_style) { color_style_ = color_style; }

    void
    set_count_start(int64_t count_start) { count_start_ = count_start; }

    void
    set_count_end(int64_t count_end) { count_end_ = count_end; }

 private:
    ColorStyle color_style_;
    double count_start_;
    double count_end_;
};



using PlanNodeWeightedCircle2DPtr = std::shared_ptr<PlanNodeWeightedCircle2D>;

} // namespace engine
} // namespace render
} // namespace zilliz
