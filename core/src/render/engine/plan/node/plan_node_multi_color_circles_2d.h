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


class PlanNodeMultiColorCircle2D : public PlanNode {
 public:
    void
    Apply(Visitor &visitor) override { visitor.Visit(*this); }

    const std::unordered_map<int64_t, CircleParams2D> &
    circle_params() const { return circle_params_; }

    const std::unordered_map<std::string, CircleParams2D> &
    string_circle_params() const { return string_circle_params_; }

    const ColumnID &
    old_column_id() const { return old_column_id_; }

    std::unordered_map<int64_t, CircleParams2D> &
    mutable_circle_params() { return circle_params_; }

    std::unordered_map<std::string, CircleParams2D> &
    mutable_string_circle_params() { return string_circle_params_; }

    ColumnID &
    mutable_old_column_id() { return old_column_id_; }

    void
    set_circle_params(std::unordered_map<int64_t, CircleParams2D> circle_params) { circle_params_ = circle_params; }

 private:
    std::unordered_map<std::string, CircleParams2D> string_circle_params_;
    std::unordered_map<int64_t, CircleParams2D> circle_params_;
    ColumnID old_column_id_;
};

using PlanNodeMultiColorCircle2DPtr = std::shared_ptr<PlanNodeMultiColorCircle2D>;

} // namespace engine
} // namespace render
} // namespace zilliz
