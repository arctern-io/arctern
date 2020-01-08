#pragma once

#include <memory>

#include "render/engine/operator/dataset.h"
#include "render/engine/plan/prim/circle.h"
#include "render/engine/plan/node/plan_node_circles_2d.h"
#include "render/engine/layer/layer.h"
#include "render/engine/common/dyn_cast.h"


namespace zilliz {
namespace render {
namespace engine {


class LayerCircles2D : public Layer {
 public:
    LayerCircles2D();

 public:
    void
    Render() final ;

    void
    Init() final ;

    void
    Shader() final;

    const PlanNodeCircle2DPtr &
    plan_node() const { return plan_node_; }

    void
    set_plan_node(PlanNodePtr plan_node) override {
        plan_node_ = dyn_cast<PlanNodeCircle2D>(plan_node);
    }

    void
    set_window_params(WindowParams window_params) { window_params_ = window_params; }

    const size_t&
    num_vertices() { return num_vertices_; }

 private:
    void
    SetVertices();

    void
    SetColors();

 private:
    PlanNodeCircle2DPtr plan_node_;
    unsigned int VAO;
    unsigned int VBO[2];
    std::shared_ptr<uint32_t> vertices_x_;
    std::shared_ptr<uint32_t> vertices_y_;
    CircleParams2D::Color colors_;
    size_t num_vertices_;
    WindowParams window_params_;
};

using LayerCircles2DPtr = std::shared_ptr<LayerCircles2D>;

} // namespace engine
} // namespace render
} // namespace zilliz
