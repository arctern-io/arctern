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

 private:
    unsigned int VAO_;
    unsigned int VBO_[2];
    std::shared_ptr<uint32_t > vertices_x_;
    std::shared_ptr<uint32_t > vertices_y_;
    size_t num_vertices_;

};

} //namespace render
} //namespace zilliz

