#pragma once

#include "zcommon/plan/plan/expr.h"
#include "zcommon/plan/plan/target_entry_table.h"
#include "zcommon/plan/expr/func_expr.h"
#include "zcommon/plan/expr/const_expr.h"

#include "zlibrary/type/value.h"

#include "render/engine/plan/node/plan_node_building_weighted_2d.h"
#include "render/engine/plan/node/plan_node_cursor_inter.h"
#include "render/engine/plan/plan/plan_node.h"
#include "render/engine/common/error.h"
#include "render/engine/operator/dataset.h"


namespace zilliz {
namespace render {
namespace engine {

template<typename T>
std::shared_ptr<
    typename std::enable_if<(!std::is_same<zilliz::plan::Expr, T>::value)
                                && (std::is_base_of<zilliz::plan::Expr, T>::value), T>::type>
dyn_cast(std::shared_ptr<zilliz::plan::Expr> &ptr) {
    if (ptr == nullptr) {
        THROW_RENDER_ENGINE_ERROR(NULL_PTR, "ExprPtr is null.")
    }
    switch (ptr->expr_type()) {
        case zilliz::plan::ExprType::kTargetEntry: {
            if (std::is_same<T, zilliz::plan::TargetEntry>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case zilliz::plan::ExprType::kFuncExpr: {
            if (std::is_same<T, zilliz::plan::FuncExpr>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case zilliz::plan::ExprType::kConstExpr: {
            if (std::is_same<T, zilliz::plan::ConstExpr>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        default: {
            THROW_RENDER_ENGINE_ERROR(CAST_FAILED, "Cannot cast ExprPtr to derived ptr.")
        }
    }
}


template<typename T>
std::shared_ptr<
    typename std::enable_if<(!std::is_same<PlanNode, T>::value)
                                && (std::is_base_of<PlanNode, T>::value), T>::type>
dyn_cast(std::shared_ptr<PlanNode> &ptr) {
    if (ptr == nullptr) {
        THROW_RENDER_ENGINE_ERROR(NULL_PTR, "PlanNodePtr is null.")
    }
    switch (ptr->plan_node_type()) {
        case PlanNodeType::kIcon2D : {
            if (std::is_same<T, PlanNodeIcon2D>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case PlanNodeType::kCircle2D : {
            if (std::is_same<T, PlanNodeCircle2D>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case PlanNodeType::kMultiColorCircle2D : {
            if (std::is_same<T, PlanNodeMultiColorCircle2D>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case PlanNodeType::kWeightedColorCircle2D : {
            if (std::is_same<T, PlanNodeWeightedColorCircle2D>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case PlanNodeType::kBuildingWieghted2D : {
            if (std::is_same<T, PlanNodeBuildingWeighted2D>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case PlanNodeType::kWeightedPointSizeCircle2D : {
            if (std::is_same<T, PlanNodeWeightedPointSizeCircle2D>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case PlanNodeType::kWeightedCircle2D : {
            if (std::is_same<T, PlanNodeWeightedCircle2D>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case PlanNodeType::kHeatMap : {
            if (std::is_same<T, PlanNodeHeatMap2D>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        case PlanNodeType::kCursorInter : {
            if (std::is_same<T, PlanNodeCursorInter>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        default: {
            THROW_RENDER_ENGINE_ERROR(CAST_FAILED, "Cannot cast PlanNodePtr to derived ptr.")
        }
    }
}


template<typename T>
std::shared_ptr<
    typename std::enable_if<(!std::is_same<zilliz::lib::ValueBase, T>::value)
                                && (std::is_base_of<zilliz::lib::ValueBase, T>::value), T>::type>
dyn_cast(std::shared_ptr<zilliz::lib::ValueBase> &ptr) {
    if (ptr == nullptr) {
        THROW_RENDER_ENGINE_ERROR(NULL_PTR, "ValueBasePtr is null.")
    }
    switch (ptr->value_type()) {
        case zilliz::lib::ValueType::kValText : {
            if (std::is_same<T, zilliz::lib::StringValue>::value) return std::static_pointer_cast<T>(ptr);
            else return nullptr;
        }
        default: {
            THROW_RENDER_ENGINE_ERROR(CAST_FAILED, "Cannot cast ValueBasePtr to derived ptr.")
        }
    }
}


template<typename T>
std::shared_ptr<
    typename std::enable_if<(!std::is_same<Dataset::Meta, T>::value)
                                && (std::is_base_of<Dataset::Meta, T>::value), T>::type>
dyn_cast(std::shared_ptr<Dataset::Meta> &ptr) {
    if (ptr == nullptr) {
        THROW_RENDER_ENGINE_ERROR(NULL_PTR, "AccessorPtr is null.")
    }
    auto derived_ptr = std::static_pointer_cast<FragmentMeta>(ptr);
    if (derived_ptr != nullptr) {
        return derived_ptr;
    } else {
        THROW_RENDER_ENGINE_ERROR(CAST_FAILED, "Cannot cast AccessorPtr to FragmentBoardPtr.")
    }
}


} // namespace engine
} // namespace render
} // namespace zilliz