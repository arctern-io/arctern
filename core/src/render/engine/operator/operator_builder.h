#pragma once

#include <render/engine/plan/node/plan_node_building_weighted_2d.h>
#include "zcommon/plan/plan/fused_node.h"
#include "zcommon/plan/expr/func_expr.h"
#include "render/engine/operator/operator.h"
#include "op_circles_2d.h"


namespace zilliz {
namespace render {
namespace engine {


class OperatorBuilder : public PlanNode::Visitor {
 public:
    using FusedNode = zilliz::plan::FusedNode;
    using FusedNodePtr = zilliz::plan::FusedNodePtr;

 public:
    OperatorPtr
    GetOperator(const zilliz::plan::FusedNodePtr &fused_node);

    OperatorPtr
    GetOperator(const zilliz::plan::FusedNodePtr &fused_node, int devid);

 public: // PlanNode::Visitor functions
    void
    Visit(PlanNodeIcon2D &plan_node) override;

    void
    Visit(PlanNodeCircle2D &plan_node) override;

    void
    Visit(PlanNodeWeightedCircle2D &plan_node) override;

    void
    Visit(PlanNodeWeightedColorCircle2D &plan_node) override;

    void
    Visit(PlanNodeWeightedPointSizeCircle2D &plan_node) override;

    void
    Visit(PlanNodeMultiColorCircle2D &plan_node) override;

    void
    Visit(PlanNodeBuildingWeighted2D &plan_node) override;

    void
    Visit(PlanNodeHeatMap2D &plan_node) override;

    void
    Visit(PlanNodeCursorInter &plan_node) override;

 private:
    bool
    IsOpIcon2D(const RenderPlanPtr &render_plan);

    bool
    IsOpCircles2D(const RenderPlanPtr &render_plan);

    bool
    IsOpMultiColorCircles2D(const RenderPlanPtr &render_plan);

    bool
    IsOpWeightedColorCircles2D(const RenderPlanPtr &render_plan);

    bool
    IsOpBuildingWeighted2D(const RenderPlanPtr &render_plan);

    bool
    IsOpWeightedPointSizeCircles2D(const RenderPlanPtr &render_plan);

    bool
    IsOpWeightedCircles2D(const RenderPlanPtr &render_plan);

    bool
    IsOpHeatMap2D(const RenderPlanPtr &render_plan);

    bool
    IsOpCursorInter(const RenderPlanPtr &render_plan);

 private:
    zilliz::plan::FuncExprPtr render_func_expr_;
    size_t arg_offset_;
};


} // namespace engine
} // namespace render
} // namespace zilliz



