#pragma once

#include "render/engine/common/table_id.h"
#include "render/engine/lib/standard_types.h"


namespace zilliz {
namespace render {
namespace engine {

enum class PlanNodeType : int {
    kUnknown,
    kHeatMap,
    kCircle2D,
    kIcon2D,
    kWeightedCircle2D,
    kWeightedColorCircle2D,
    kMultiColorCircle2D,
    kBuildingWieghted2D,
    kWeightedPointSizeCircle2D,
    kCursorInter
};
//using PlanNodeType = Layer::Type;

class PlanNode;

using PlanNodePtr = std::shared_ptr<PlanNode>;
using ValueType = zilliz::lib::ValueType;

class PlanNode {
 public:
    class Visitor;

 public:
    PlanNode() = default;
    PlanNode(const PlanNode &) = default;
    PlanNode(PlanNode &&) = default;
    virtual ~PlanNode() = default;

    virtual void
    Apply(Visitor &visitor) = 0;

 public:
    void
    set_plan_node_type(PlanNodeType plan_node_type) { plan_node_type_ = plan_node_type; }

    PlanNodeType
    plan_node_type() const { return plan_node_type_; }

    const PlanNodePtr &
    child() const { return child_; }

    void
    set_child(PlanNodePtr child) { child_ = child; }

    const std::vector<ColumnID> &
    data_params() const { return data_params_; }

    const std::vector<ValueType> &
    data_param_type() const { return data_params_type_; }

    std::vector<ColumnID> &
    mutable_data_params() { return data_params_; }

    std::vector<ValueType> &
    mutable_data_params_type() { return data_params_type_; }

    const TableID&
    output_id() { return output_id_; }

    void
    set_output_id(ColumnID output_id) { output_id_ = output_id; }

    const int&
    dev_id() const { return dev_id_; }

    void
    set_dev_id(int dev_id) { dev_id_ = dev_id;}

 private:
    PlanNodeType plan_node_type_;
    PlanNodePtr child_;

    std::vector<ColumnID> data_params_;
    std::vector<ValueType> data_params_type_;
    TableID output_id_;
    int dev_id_ = 0;
};


class PlanNodeIcon2D;
class PlanNodeCircle2D;
class PlanNodeMultiColorCircle2D;
class PlanNodeWeightedColorCircle2D;
class PlanNodeBuildingWeighted2D;
class PlanNodeWeightedPointSizeCircle2D;
class PlanNodeWeightedCircle2D;
class PlanNodeHeatMap2D;
class PlanNodeCursorInter;

class PlanNode::Visitor {
 public:
    virtual void
    Visit(PlanNodeCircle2D &plan_node) = 0;

    virtual void
    Visit(PlanNodeWeightedColorCircle2D &plan_node) = 0;

    virtual void
    Visit(PlanNodeWeightedPointSizeCircle2D &plan_node) = 0;

    virtual void
    Visit(PlanNodeWeightedCircle2D &plan_node) = 0;

    virtual void
    Visit(PlanNodeMultiColorCircle2D &plan_node) = 0;

    virtual void
    Visit(PlanNodeIcon2D &plan_node) = 0;

    virtual void
    Visit(PlanNodeBuildingWeighted2D &plan_node) = 0;

    virtual void
    Visit(PlanNodeHeatMap2D &plan_node) = 0;

    virtual void
    Visit(PlanNodeCursorInter &plan_node) = 0;

    //virtual void
    //Visit(MorePlanNodeType &plan_node) = 0;
};


} // namespace engine
} // namespace render
} // namespace zilliz
