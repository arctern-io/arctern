#pragma once

#include "render/utils/vega/vega_scatter_plot/vega_circle2d.h"
#include "render/2d/input.h"
#include "render/2d/general_2d.h"

namespace zilliz {
namespace render {

class PointMap : public General2D {
 public:
    PointMap();

    PointMap(uint32_t* input_x, uint32_t* input_y, int64_t num_vertices);

    void
    DataInit() final;

    std::shared_ptr<uint8_t >
    Render() final;

    void
    Shader() final;

    void
    Draw() final;

    void
    InputInit() final;

 public:
    uint32_t*
    mutable_vertices_x() { return vertices_x_; }

    uint32_t*
    mutable_vertices_y() { return vertices_y_; }

    VegaCircle2d&
    mutable_point_vega() {return point_vega_; }

    const size_t
    num_vertices() const { return num_vertices_; }

 private:
    unsigned int VAO_;
    unsigned int VBO_[2];
    uint32_t* vertices_x_;
    uint32_t* vertices_y_;
    size_t num_vertices_;
    VegaCircle2d point_vega_;

};

} //namespace render
} //namespace zilliz

