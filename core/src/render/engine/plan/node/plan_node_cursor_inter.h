#pragma once

#include "render/engine/plan/plan/plan_node.h"
#include "render/utils/geo/geo_abstractor.h"

namespace zilliz {
namespace render {
namespace engine {


class PlanNodeCursorInter : public PlanNode {
 public:
    void
    Apply(Visitor &visitor) override { visitor.Visit(*this); }

    const std::pair<double, double> &
    cursor_position() const { return cursor_position_; }

    void
    set_cursor_position(std::pair<double, double> &cursor_position) { cursor_position_ = cursor_position; }

 private:
    std::pair<double, double> cursor_position_;
};

using PlanNodeCursorInterPtr = std::shared_ptr<PlanNodeCursorInter>;

} // namespace engine
} // namespace render
} // namespace zilliz