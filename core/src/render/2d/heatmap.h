#pragma once

#include "render/2d/general_2d.h"
#include "render/2d/set_color.h"

namespace zilliz {
namespace render {

template<typename T>
class HeatMap : public General2D {
 public:
    HeatMap();

    HeatMap(std::shared_ptr<uint32_t> input_x,
            std::shared_ptr<uint32_t> input_y,
            std::shared_ptr<T> count,
            int64_t num_vertices);

    void
    DataInit() final;

    std::shared_ptr<uint8_t>
    Render() final;

    void
    Shader() final;

    void
    Draw() final;

    void
    InputInit() final;

 public:
    VegaHeatMap&
    mutable_heatmap_vega() {return heatmap_vega_; }

 private:
    unsigned int VAO_;
    unsigned int VBO_[3];
    std::shared_ptr<uint32_t> vertices_x_;
    std::shared_ptr<uint32_t> vertices_y_;
    std::shared_ptr<T> count_;
    float *colors_;
    int64_t num_vertices_;
    VegaHeatMap heatmap_vega_;

};

} // namespace render
} // namespace zilliz
