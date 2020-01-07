#include "zlibrary/type/value.h"

#include "zcommon/plan/plan/fused_node.h"
#include "zcommon/plan/plan/plan_node.h"
#include "zcommon/plan/plan/expr.h"
#include "zcommon/plan/node/render/render_node.h"
#include "zcommon/plan/expr/column_expr.h"
#include "zcommon/plan/expr/string_column_expr.h"
#include "zcommon/plan/expr/func_expr.h"
#include "zcommon/plan/expr/const_expr.h"

#include "render/utils/vega/vega_parser.h"
#include "render/engine/common/error.h"
#include "render/engine/common/log.h"
#include "render/engine/operator/operator_builder.h"
#include "render/engine/operator/op_icon_2d.h"
#include "render/engine/operator/op_circles_2d.h"
#include "render/engine/operator/op_multi_color_circles_2d.h"
#include "render/engine/operator/op_weighted_circles_2d.h"
#include "render/engine/operator/op_weighted_color_circles_2d.h"
#include "render/engine/operator/op_building_weighted_2d.h"
#include "render/engine/operator/op_weighted_pointsize_circles_2d.h"
#include "render/engine/operator/op_heatmap_2d.h"
#include "render/engine/operator/op_cursor_inter.h"

#include "render/engine/plan/node/plan_node_circles_2d.h"
#include "render/engine/plan/node/plan_node_weighted_color_circles_2d.h"
#include "render/engine/plan/node/plan_node_weighted_pointsize_circles_2d.h"
#include "render/engine/plan/node/plan_node_weighted_circles_2d.h"
#include "render/engine/plan/node/plan_node_multi_color_circles_2d.h"
#include "render/engine/plan/node/plan_node_building_weighted_2d.h"
#include "render/engine/common/dyn_cast.h"
#include "render/engine/image/loader/image_loader.h"


namespace zilliz {
namespace render {
namespace engine {


OperatorPtr
OperatorBuilder::GetOperator(const zilliz::plan::FusedNodePtr &fused_node) {

    arg_offset_ = 0;

    // we have only one target entry, which is the result image
    auto expr_ptr = fused_node->root_plan_node()->target_entries().front();
    auto target_entry_ptr = dyn_cast<zilliz::plan::TargetEntry>(expr_ptr);
    render_func_expr_ = dyn_cast<zilliz::plan::FuncExpr>(target_entry_ptr->expr());

    // by convention, any render function's last argument should be the render plan described by vega json
    auto const_expr_ptr = dyn_cast<zilliz::plan::ConstExpr>(render_func_expr_->args().back());
    auto vega_json_query_ptr = dyn_cast<zilliz::lib::StringValue>(const_expr_ptr->value());

    VegaParser vega_parser;
    auto plan_ptr = vega_parser.GetPlan(vega_json_query_ptr->value());

    auto plan_node_ptr = plan_ptr->root_plan_node();

    while (plan_node_ptr != nullptr) {
        RENDER_ENGINE_LOG_TRACE << "set plan_node";
        plan_node_ptr->Apply(*this);
        plan_node_ptr->set_output_id(target_entry_ptr->id());
        plan_node_ptr = plan_node_ptr->child();
    }

    if (IsOpCircles2D(plan_ptr)) {
        auto op_circles_2d = std::make_shared<OpCircles2D>();
        op_circles_2d->set_plan(plan_ptr);
        return op_circles_2d;
    } else if (IsOpIcon2D(plan_ptr)) {
        auto op_icon_2d = std::make_shared<OpIcon2D>();
        op_icon_2d->set_plan(plan_ptr);
        return op_icon_2d;
    } else if (IsOpMultiColorCircles2D(plan_ptr)) {
        auto op_circles_multi_color_2d = std::make_shared<OpMultiColorCircles2D>();
        op_circles_multi_color_2d->set_plan(plan_ptr);
        return op_circles_multi_color_2d;
    } else if (IsOpWeightedColorCircles2D(plan_ptr)) {
        auto op_circles_weighted_color_2d = std::make_shared<OpWeightedColorCircles2D>();
        op_circles_weighted_color_2d->set_plan(plan_ptr);
        return op_circles_weighted_color_2d;
    } else if (IsOpBuildingWeighted2D(plan_ptr)) {
        auto op_building_weighted_2d = std::make_shared<OpBuildingWeighted2D>();
        op_building_weighted_2d->set_plan(plan_ptr);
        return op_building_weighted_2d;
    } else if (IsOpWeightedPointSizeCircles2D(plan_ptr)) {
        auto op_circles_weighted_pointsize_2d = std::make_shared<OpWeightedPointSizeCircles2D>();
        op_circles_weighted_pointsize_2d->set_plan(plan_ptr);
        return op_circles_weighted_pointsize_2d;
    } else if (IsOpWeightedCircles2D(plan_ptr)) {
        auto op_circles_weighted_2d = std::make_shared<OpWeightedCircles2D>();
        op_circles_weighted_2d->set_plan(plan_ptr);
        return op_circles_weighted_2d;
    } else if (IsOpHeatMap2D(plan_ptr)) {
        auto op_heatmap_2d = std::make_shared<OpHeatMap2D>();
        op_heatmap_2d->set_plan(plan_ptr);
        return op_heatmap_2d;
    } else if (IsOpCursorInter(plan_ptr)) {
        auto op_cursor_inter = std::make_shared<OpCursorInter>();
        op_cursor_inter->set_plan(plan_ptr);
        return op_cursor_inter;
    } else {
        THROW_RENDER_ENGINE_ERROR(UNKNOWN_PLAN_TYPE, "unknown render plan type.")
    }
}


OperatorPtr
OperatorBuilder::GetOperator(const zilliz::plan::FusedNodePtr &fused_node, int dev_id) {

    arg_offset_ = 0;

    // we have only one target entry, which is the result image
    auto expr_ptr = fused_node->root_plan_node()->target_entries().front();
    auto target_entry_ptr = dyn_cast<zilliz::plan::TargetEntry>(expr_ptr);
    render_func_expr_ = dyn_cast<zilliz::plan::FuncExpr>(target_entry_ptr->expr());

    // by convention, any render function's last argument should be the render plan described by vega json
    auto const_expr_ptr = dyn_cast<zilliz::plan::ConstExpr>(render_func_expr_->args().back());
    auto vega_json_query_ptr = dyn_cast<zilliz::lib::StringValue>(const_expr_ptr->value());

    VegaParser vega_parser;
    auto plan_ptr = vega_parser.GetPlan(vega_json_query_ptr->value());

    auto plan_node_ptr = plan_ptr->root_plan_node();

    while (plan_node_ptr != nullptr) {
        RENDER_ENGINE_LOG_TRACE << "set plan_node";
        plan_node_ptr->Apply(*this);
        plan_node_ptr->set_output_id(target_entry_ptr->id());
        plan_node_ptr->set_dev_id(dev_id);
        plan_node_ptr = plan_node_ptr->child();
    }

    if (IsOpCircles2D(plan_ptr)) {
        auto op_circles_2d = std::make_shared<OpCircles2D>();
        op_circles_2d->set_plan(plan_ptr);
        return op_circles_2d;
    } else if (IsOpIcon2D(plan_ptr)) {
        auto op_icon_2d = std::make_shared<OpIcon2D>();
        op_icon_2d->set_plan(plan_ptr);
        return op_icon_2d;
    } else if (IsOpMultiColorCircles2D(plan_ptr)) {
        auto op_circles_multi_color_2d = std::make_shared<OpMultiColorCircles2D>();
        op_circles_multi_color_2d->set_plan(plan_ptr);
        return op_circles_multi_color_2d;
    } else if (IsOpWeightedColorCircles2D(plan_ptr)) {
        auto op_circles_weighted_color_2d = std::make_shared<OpWeightedColorCircles2D>();
        op_circles_weighted_color_2d->set_plan(plan_ptr);
        return op_circles_weighted_color_2d;
    } else if (IsOpBuildingWeighted2D(plan_ptr)) {
        auto op_building_weighted_2d = std::make_shared<OpBuildingWeighted2D>();
        op_building_weighted_2d->set_plan(plan_ptr);
        return op_building_weighted_2d;
    } else if (IsOpWeightedPointSizeCircles2D(plan_ptr)) {
        auto op_circles_weighted_pointsize_2d = std::make_shared<OpWeightedPointSizeCircles2D>();
        op_circles_weighted_pointsize_2d->set_plan(plan_ptr);
        return op_circles_weighted_pointsize_2d;
    } else if (IsOpWeightedCircles2D(plan_ptr)) {
        auto op_circles_weighted_2d = std::make_shared<OpWeightedCircles2D>();
        op_circles_weighted_2d->set_plan(plan_ptr);
        return op_circles_weighted_2d;
    } else if (IsOpHeatMap2D(plan_ptr)) {
        auto op_heatmap_2d = std::make_shared<OpHeatMap2D>();
        op_heatmap_2d->set_plan(plan_ptr);
        return op_heatmap_2d;
    } else if (IsOpCursorInter(plan_ptr)) {
        auto op_cursor_inter = std::make_shared<OpCursorInter>();
        op_cursor_inter->set_plan(plan_ptr);
        return op_cursor_inter;
    } else {
        THROW_RENDER_ENGINE_ERROR(UNKNOWN_PLAN_TYPE, "unknown render plan type.")
    }
}

void
OperatorBuilder::Visit(PlanNodeIcon2D &plan_node) {

    auto x_id = render_func_expr_->args()[arg_offset_++]->id();
    auto y_id = render_func_expr_->args()[arg_offset_++]->id();

    plan_node.mutable_data_params().emplace_back(x_id);
    plan_node.mutable_data_params().emplace_back(y_id);
}


void
OperatorBuilder::Visit(PlanNodeCircle2D &plan_node) {

    auto x_id = render_func_expr_->args()[arg_offset_++]->id();
    auto y_id = render_func_expr_->args()[arg_offset_++]->id();

    plan_node.mutable_data_params().emplace_back(x_id);
    plan_node.mutable_data_params().emplace_back(y_id);
}

void
OperatorBuilder::Visit(PlanNodeWeightedCircle2D &plan_node) {

    auto x_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto x_id = render_func_expr_->args()[arg_offset_++]->id();

    auto y_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto y_id = render_func_expr_->args()[arg_offset_++]->id();

    auto count_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto count_id = render_func_expr_->args()[arg_offset_++]->id();

    auto pointsize_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto pointsize_id = render_func_expr_->args()[arg_offset_++]->id();

    plan_node.mutable_data_params().emplace_back(x_id);
    plan_node.mutable_data_params().emplace_back(y_id);
    plan_node.mutable_data_params().emplace_back(count_id);
    plan_node.mutable_data_params().emplace_back(pointsize_id);

    plan_node.mutable_data_params_type().emplace_back(x_value_type);
    plan_node.mutable_data_params_type().emplace_back(y_value_type);
    plan_node.mutable_data_params_type().emplace_back(count_value_type);
    plan_node.mutable_data_params_type().emplace_back(pointsize_value_type);
}

void
OperatorBuilder::Visit(PlanNodeWeightedColorCircle2D &plan_node) {

    auto x_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto x_id = render_func_expr_->args()[arg_offset_++]->id();

    auto y_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto y_id = render_func_expr_->args()[arg_offset_++]->id();

    auto count_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto count_id = render_func_expr_->args()[arg_offset_++]->id();

    plan_node.mutable_data_params().emplace_back(x_id);
    plan_node.mutable_data_params().emplace_back(y_id);
    plan_node.mutable_data_params().emplace_back(count_id);

    plan_node.mutable_data_params_type().emplace_back(x_value_type);
    plan_node.mutable_data_params_type().emplace_back(y_value_type);
    plan_node.mutable_data_params_type().emplace_back(count_value_type);
}

void
OperatorBuilder::Visit(PlanNodeWeightedPointSizeCircle2D &plan_node) {

    auto x_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto x_id = render_func_expr_->args()[arg_offset_++]->id();

    auto y_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto y_id = render_func_expr_->args()[arg_offset_++]->id();

    auto pointsize_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto pointsize_id = render_func_expr_->args()[arg_offset_++]->id();

    plan_node.mutable_data_params().emplace_back(x_id);
    plan_node.mutable_data_params().emplace_back(y_id);
    plan_node.mutable_data_params().emplace_back(pointsize_id);

    plan_node.mutable_data_params_type().emplace_back(x_value_type);
    plan_node.mutable_data_params_type().emplace_back(y_value_type);
    plan_node.mutable_data_params_type().emplace_back(pointsize_value_type);
}

void
OperatorBuilder::Visit(PlanNodeMultiColorCircle2D &plan_node) {

    auto x_id = render_func_expr_->args()[arg_offset_++]->id();
    auto y_id = render_func_expr_->args()[arg_offset_++]->id();
    auto label_id = render_func_expr_->args()[arg_offset_]->id();

    auto string_column_expr =
        std::dynamic_pointer_cast<zilliz::plan::StringColumnExpr>(render_func_expr_->args()[arg_offset_++]);
    if (string_column_expr) {
        plan_node.mutable_old_column_id() = string_column_expr->old_column_id();
    } else {
        std::string msg = "Type of label column must be text.";
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_COLUMN_TYPE, msg)
    }

    plan_node.mutable_data_params().emplace_back(x_id);
    plan_node.mutable_data_params().emplace_back(y_id);
    plan_node.mutable_data_params().emplace_back(label_id);
}

void
OperatorBuilder::Visit(PlanNodeHeatMap2D &plan_node) {

    auto x_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto x_id = render_func_expr_->args()[arg_offset_++]->id();

    auto y_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto y_id = render_func_expr_->args()[arg_offset_++]->id();

    auto count_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto count_id = render_func_expr_->args()[arg_offset_++]->id();

    plan_node.mutable_data_params().emplace_back(x_id);
    plan_node.mutable_data_params().emplace_back(y_id);
    plan_node.mutable_data_params().emplace_back(count_id);

    plan_node.mutable_data_params_type().emplace_back(x_value_type);
    plan_node.mutable_data_params_type().emplace_back(y_value_type);
    plan_node.mutable_data_params_type().emplace_back(count_value_type);
}

void
OperatorBuilder::Visit(PlanNodeBuildingWeighted2D &plan_node) {

    auto building_list_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto building_list_id = render_func_expr_->args()[arg_offset_]->id();

    auto string_column_expr =
        std::dynamic_pointer_cast<zilliz::plan::StringColumnExpr>(render_func_expr_->args()[arg_offset_++]);
    if (string_column_expr) {
        plan_node.set_old_column_id(string_column_expr->old_column_id());
    } else {
        std::string msg = "Building list column must be text.";
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_COLUMN_TYPE, msg)
    }

    auto count_value_type = render_func_expr_->args()[arg_offset_]->result_type();
    auto count_id = render_func_expr_->args()[arg_offset_++]->id();

    plan_node.mutable_data_params().emplace_back(building_list_id);
    plan_node.mutable_data_params().emplace_back(count_id);

    plan_node.mutable_data_params_type().emplace_back(building_list_value_type);
    plan_node.mutable_data_params_type().emplace_back(count_value_type);
}


void
OperatorBuilder::Visit(PlanNodeCursorInter &plan_node) {
    // We don't need to do anything else here.
}


bool
OperatorBuilder::IsOpIcon2D(const RenderPlanPtr &render_plan) {
    if (render_plan->plan_type() == RenderPlan::Type::k2D) {
        auto root_plan_node = render_plan->root_plan_node();
        if (root_plan_node != nullptr
            && root_plan_node->child() == nullptr
            && root_plan_node->plan_node_type() == PlanNodeType::kIcon2D) {
            return true;
        }
    }

    return false;
}

bool
OperatorBuilder::IsOpCircles2D(const RenderPlanPtr &render_plan) {
    if (render_plan->plan_type() == RenderPlan::Type::k2D) {
        auto root_plan_node = render_plan->root_plan_node();
        if (root_plan_node != nullptr
            && root_plan_node->child() == nullptr
            && root_plan_node->plan_node_type() == PlanNodeType::kCircle2D) {
            return true;
        }
    }

    return false;
}

bool
OperatorBuilder::IsOpMultiColorCircles2D(const RenderPlanPtr &render_plan) {
    if (render_plan->plan_type() == RenderPlan::Type::k2D) {
        auto root_plan_node = render_plan->root_plan_node();
        if (root_plan_node != nullptr
            && root_plan_node->child() == nullptr
            && root_plan_node->plan_node_type() == PlanNodeType::kMultiColorCircle2D) {
            return true;
        }
    }
    return false;
}

bool
OperatorBuilder::IsOpWeightedColorCircles2D(const RenderPlanPtr &render_plan) {
    if (render_plan->plan_type() == RenderPlan::Type::k2D) {
        auto root_plan_node = render_plan->root_plan_node();
        if (root_plan_node != nullptr
            && root_plan_node->child() == nullptr
            && root_plan_node->plan_node_type() == PlanNodeType::kWeightedColorCircle2D) {
            return true;
        }
    }
    return false;
}

bool
OperatorBuilder::IsOpBuildingWeighted2D(const RenderPlanPtr &render_plan) {
    if (render_plan->plan_type() == RenderPlan::Type::k2D) {
        auto root_plan_node = render_plan->root_plan_node();
        if (root_plan_node != nullptr
            && root_plan_node->child() == nullptr
            && root_plan_node->plan_node_type() == PlanNodeType::kBuildingWieghted2D) {
            return true;
        }
    }
    return false;
}

bool
OperatorBuilder::IsOpWeightedPointSizeCircles2D(const RenderPlanPtr &render_plan) {
    if (render_plan->plan_type() == RenderPlan::Type::k2D) {
        auto root_plan_node = render_plan->root_plan_node();
        if (root_plan_node != nullptr
            && root_plan_node->child() == nullptr
            && root_plan_node->plan_node_type() == PlanNodeType::kWeightedPointSizeCircle2D) {
            return true;
        }
    }
    return false;
}

bool
OperatorBuilder::IsOpWeightedCircles2D(const RenderPlanPtr &render_plan) {
    if (render_plan->plan_type() == RenderPlan::Type::k2D) {
        auto root_plan_node = render_plan->root_plan_node();
        if (root_plan_node != nullptr
            && root_plan_node->child() == nullptr
            && root_plan_node->plan_node_type() == PlanNodeType::kWeightedCircle2D) {
            return true;
        }
    }
    return false;
}

bool
OperatorBuilder::IsOpHeatMap2D(const RenderPlanPtr &render_plan) {
    if (render_plan->plan_type() == RenderPlan::Type::k2D) {
        auto root_plan_node = render_plan->root_plan_node();
        if (root_plan_node != nullptr
            && root_plan_node->child() == nullptr
            && root_plan_node->plan_node_type() == PlanNodeType::kHeatMap) {
            return true;
        }
    }
    return false;
}


bool
OperatorBuilder::IsOpCursorInter(const RenderPlanPtr &render_plan) {
    if (render_plan->plan_type() == RenderPlan::Type::k2D) {
        auto root_plan_node = render_plan->root_plan_node();
        if (root_plan_node != nullptr
            && root_plan_node->child() == nullptr
            && root_plan_node->plan_node_type() == PlanNodeType::kCursorInter) {
            return true;
        }
    }
    return false;
}


} // namespace engine
} // namespace render
} // namespace zilliz
