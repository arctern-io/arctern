#pragma once

#include <map>
#include "render/engine/plan/plan/plan_node.h"
#include "render/engine/plan/prim/circle.h"
#include "render/engine/common/table_id.h"
#include "render/engine/common/window_params.h"
#include "chewie/grpc/client.h"
#include "chewie/cache/object/cache_hint.h"


namespace zilliz {
namespace render {
namespace engine {

using CacheHint = zilliz::chewie::CacheHint;

class PlanNodeWeightedPointSizeCircle2D : public PlanNode {
 public:
    void
    Apply(Visitor &visitor) override {visitor.Visit(*this);}

    const CircleParams2D &
    render_params() const { return render_param_; }

    CircleParams2D &
    mutable_render_params() { return render_param_; }

    void
    set_circle_params(CircleParams2D render_param) { render_param_ = render_param; }

 private:
    CircleParams2D render_param_;
};

using PlanNodeWeightedPointSizeCircle2DPtr = std::shared_ptr<PlanNodeWeightedPointSizeCircle2D>;

} // namespace engine
} // namespace render
} // namespace zilliz
