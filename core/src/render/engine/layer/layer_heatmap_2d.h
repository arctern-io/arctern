#pragma once

#include <memory>
#include <vector>
#include <render/engine/lib/cuda_check.h>
#include "layer.h"
#include "render/engine/operator/dataset.h"
#include "render/engine/plan/prim/circle.h"
#include "render/engine/plan/node/plan_node_heatmap_2d.h"
#include "render/engine/common/dyn_cast.h"
#include "render/utils/color/color_gradient.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"


namespace zilliz {
namespace render {
namespace engine {

class LayerHeatMap2D : public Layer {
 public:
    LayerHeatMap2D();
    ~LayerHeatMap2D();

    void
    Init() final;

    void
    Render() final;

    void
    Shader() final;

 public:
    void
    set_plan_node(PlanNodePtr plan_node) override {
        plan_node_ = dyn_cast<PlanNodeHeatMap2D>(plan_node);
    }

    const PlanNodeHeatMap2DPtr &
    plan_node() const { return plan_node_; }

    void
    set_window_params(WindowParams window_params) {
        window_params_ = window_params;
    }

    unsigned char*
    colors() { return colors_;}
 private:
    template<typename T>
    void
    Interop();

    inline static unsigned int
    iDivUp( const unsigned int &a, const unsigned int &b ) { return (a+b-1)/b; }

 private:
    PlanNodeHeatMap2DPtr plan_node_;
    ColorGradient color_gradient_;

    uint32_t *vertices_x_;
    uint32_t *vertices_y_;
    unsigned char *colors_;
    int64_t num_vertices_;
    WindowParams window_params_;
    cudaError cuda_state;
};

using LayerHeatMap2DPtr = std::shared_ptr<LayerHeatMap2D>;
using ErrorCode = lib::ErrorCode;
} // namespace engine
} // namespace render
} // namespace zilliz
