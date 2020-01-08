#pragma once

#include "render/engine/image/format/format.h"
#include "render/engine/plan/plan/plan_node.h"
#include "render/engine/common/window_params.h"


namespace zilliz {
namespace render {
namespace engine {

class RenderPlan {
 public:
    enum class Type { kUnknown = 0, k2D };

 public:
    const WindowParams&
    window_params() const { return window_params_; }

    const PlanNodePtr
    root_plan_node() const { return root_plan_node_; }

    PlanNodePtr &
    mutable_root_plan_node() { return root_plan_node_; }

    const ImageFormatPtr
    image_format() const { return image_format_; }
    
    Type
    plan_type() const { return plan_type_; }

    void
    set_root_plan_node(PlanNodePtr plan_node) { root_plan_node_ = plan_node; }

    void
    set_window_params(WindowParams window_params) { window_params_ = window_params; }

    void
    set_image_format(ImageFormatPtr image_format) { image_format_ = image_format; }

    void
    set_plan_type(Type plan_type) { plan_type_ = plan_type; }

 private:
    Type plan_type_;
    PlanNodePtr root_plan_node_;
    WindowParams window_params_;
    ImageFormatPtr image_format_;
};

using RenderPlanPtr = std::shared_ptr<RenderPlan>;

} // namespace engine
} // namespace render
} // namespace zilliz