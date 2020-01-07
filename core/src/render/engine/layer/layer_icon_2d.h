#pragma once

#include <memory>

#include "render/engine/operator/dataset.h"
#include "render/engine/plan/node/plan_node_icon_2d.h"
#include "render/engine/layer/layer.h"
#include "render/engine/common/dyn_cast.h"


namespace zilliz {
namespace render {
namespace engine {


class LayerIcon2D : public Layer {
 public:
    LayerIcon2D();

    ~LayerIcon2D();

 public:
    void
    Render() final;

    void
    Shader() final;

    void
    Init() final;

    const PlanNodeIcon2DPtr &
    plan_node() const { return plan_node_; }

    void
    set_plan_node(PlanNodePtr plan_node) override {
        plan_node_ = dyn_cast<PlanNodeIcon2D>(plan_node);
    }

    void
    set_window_params(WindowParams window_params) { window_params_ = window_params; }

    std::vector<unsigned char> &
    mutable_pixel_vec() { return pixel_vec_; }

    const std::vector<unsigned char> &
    pixel_vec() const { return pixel_vec_; }

    const uint32_t *
    vertices() const { return vertices_; }

 private:
    void
    SetVertices();

 private:
    PlanNodeIcon2DPtr plan_node_;
    uint32_t *vertices_;
    int num_vertices_;
    std::vector<unsigned char> pixel_vec_;
    WindowParams window_params_;
};

using LayerIcon2DPtr = std::shared_ptr<LayerIcon2D>;

} // namespace engine
} // namespace render
} // namespace zilliz