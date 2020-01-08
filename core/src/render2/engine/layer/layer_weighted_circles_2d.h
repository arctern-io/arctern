#pragma once

#include <memory>
#include "render/engine/operator/dataset.h"
#include "render/engine/plan/prim/circle.h"
#include "render/engine/plan/node/plan_node_weighted_circles_2d.h"
#include "render/engine/layer/layer.h"
#include "render/engine/common/dyn_cast.h"


namespace zilliz {
namespace render {
namespace engine {


class LayerWeightedCircles2D : public Layer {
 public:
    LayerWeightedCircles2D();

    ~LayerWeightedCircles2D();

    void
    Init() final;

    void
    Render() final;

    void
    Shader() final;

 public:
    void
    set_plan_node(PlanNodePtr plan_node) override {
        plan_node_ = dyn_cast<PlanNodeWeightedCircle2D>(plan_node);
    }

    const PlanNodeWeightedCircle2DPtr &
    plan_node() const { return plan_node_; }


    void
    set_window_params(WindowParams window_params) {
        window_params_ = window_params;
    }

 private:
    template<typename T>
    void
    set_vertices_colors();

    const size_t &
    num_vertices() { return num_vertices_; }

 private:
    PlanNodeWeightedCircle2DPtr plan_node_;

    std::shared_ptr<uint32_t > vertices_x_;
    std::shared_ptr<uint32_t > vertices_y_;
    float *colors_;
    float *pointsize_;
    size_t num_vertices_;
    unsigned int VAO;
    unsigned int VBO[4];
    WindowParams window_params_;
};

using LayerWeightedCircles2DPtr = std::shared_ptr<LayerWeightedCircles2D>;

} // namespace engine
} // namespace render
} // namespace zilliz
