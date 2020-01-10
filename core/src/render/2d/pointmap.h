#pragma once

#include "render/2d/input.h"
#include "render/2d/general_2d.h"

namespace zilliz {
namespace render {

class PointMap : public General2D {
 public:
    PointMap();

    void
    DataInit() final;

    std::shared_ptr<arrow::Array>
    Render() final;

    void
    Shader() final;

    void
    Draw() final;

 public:
    void
    set_vega(Vega vega) { vega_ = vega; }

    void
    set_window_params(WindowParams window_params) { window_params_ = window_params; }

 private:
    unsigned int VAO_;
    unsigned int VBO_[2];
    std::shared_ptr<uint32_t > vertices_x_;
    std::shared_ptr<uint32_t > vertices_y_;
    Vega vega_;
    size_t num_vertices_;
    WindowParams window_params_;
};

} //namespace render
} //namespace zilliz

