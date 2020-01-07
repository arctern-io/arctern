#pragma once

#include <memory>
#include "render/engine/operator/dataset.h"
#include "render/engine/plan/prim/circle.h"
#include "render/engine/plan/node/plan_node_multi_color_circles_2d.h"
#include "render/engine/layer/layer.h"
#include "render/engine/common/dyn_cast.h"


namespace zilliz {
namespace render {
namespace engine {


class LayerMultiColorCircles2D : public Layer {
 public:

    void
    Init() final;

    void
    Render() final;

    void
    Shader() final ;

    const PlanNodeMultiColorCircle2DPtr &
    plan_node() const { return plan_node_; }

    void
    set_plan_node(PlanNodePtr plan_node) override {
        plan_node_ = dyn_cast<PlanNodeMultiColorCircle2D>(plan_node);
    }

    void
    set_window_params(WindowParams window_params) { window_params_ = window_params; }

 private:
    void
    SetVerticesAndColors();

    void
    CircleParamsTrans();

 private:
    PlanNodeMultiColorCircle2DPtr plan_node_;
    std::unordered_map<uint64_t, std::pair<std::atomic<int64_t>, std::vector<std::pair<uint32_t, uint32_t>>>> records_;
    WindowParams window_params_;
    // opengl shader members
    unsigned int VAO;
    unsigned int VBO;
    int shader_program_;
};

using LayerMultiColorCircles2DPtr = std::shared_ptr<LayerMultiColorCircles2D>;

} // namespace engine
} // namespace render
} // namespace zilliz
