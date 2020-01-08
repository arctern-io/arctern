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

class PlanNodeHeatMap2D : public PlanNode {
 public:
    void
    Apply(Visitor &visitor) override {visitor.Visit(*this);}

    double
    map_scale_ratio() const { return map_scale_ratio_; }

    double &
    mutable_map_scale_ratio() { return map_scale_ratio_; }

    void
    set_map_scale_ratio(double map_scale_ratio) { map_scale_ratio_ = map_scale_ratio; }

 private:
    double map_scale_ratio_;
};



using PlanNodeHeatMap2DPtr = std::shared_ptr<PlanNodeHeatMap2D>;

} // namespace engine
} // namespace render
} // namespace zilliz

