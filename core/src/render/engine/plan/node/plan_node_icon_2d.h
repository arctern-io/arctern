#pragma once

#include "render/engine/plan/plan/plan_node.h"
#include "chewie/cache/object/cache_hint.h"


namespace zilliz {
namespace render {
namespace engine {

using CacheHint = zilliz::chewie::CacheHint;


class PlanNodeIcon2D : public PlanNode {
 public:
    void
    Apply(Visitor &visitor) override { visitor.Visit(*this); }

    const std::string &
    icon_name() const { return icon_name_; }

    void
    set_icon_name(std::string icon_name) { icon_name_ = icon_name; }

 private:
    std::string icon_name_;
};

using PlanNodeIcon2DPtr = std::shared_ptr<PlanNodeIcon2D>;

} // namespace engine
} // namespace render
} // namespace zilliz

