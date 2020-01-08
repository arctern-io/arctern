#pragma once

#include "render/engine/plan/plan/plan_node.h"
#include "render/engine/plan/prim/circle.h"
#include "chewie/cache/object/cache_hint.h"


namespace zilliz {
namespace render {
namespace engine {

using CacheHint = zilliz::chewie::CacheHint;


class PlanNodeCircle2D : public PlanNode {
 public:

    void
    Apply(Visitor &visitor) override { visitor.Visit(*this); }

    const CircleParams2D &
    render_params() const { return render_param_; }

    CircleParams2D &
    mutable_render_params() { return render_param_; }

    void
    set_circle_params(CircleParams2D render_param) { render_param_ = render_param; }

 private:
    CircleParams2D render_param_;
};

using PlanNodeCircle2DPtr = std::shared_ptr<PlanNodeCircle2D>;

} // namespace engine
} // namespace render
} // namespace zilliz
