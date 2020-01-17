#pragma once


#include "render/utils/vega/vega_heatmap/vega_heatmap.h"
#include "general_2d.h"

namespace zilliz {
namespace render {

template<typename T>
class HeatMap : public General2D {
 public:
    HeatMap();

    HeatMap(std::shared_ptr<uint32_t> input_x,
            std::shared_ptr<uint32_t> input_y,
            std::shared_ptr<uint32_t> count,
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

 private:
    inline void
    set_colors();

    void
    set_colors_cpu();

    void
    set_colors_gpu();

    inline static unsigned int
    iDivUp( const unsigned int &a, const unsigned int &b ) { return (a+b-1)/b; }

 private:
    unsigned int VAO_;
    unsigned int VBO_[2];
    std::shared_ptr<uint32_t> vertices_x_;
    std::shared_ptr<uint32_t> vertices_y_;
    std::shared_ptr<T> count_;
    float *colors_;
    int64_t num_vertices_;
    VegaHeatMap heatmap_vega_;

};

template <typename T>
inline void HeatMap<T>::set_colors() {
#ifdef CPU_ONLY
    set_colors_cpu();
#else
    set_colors_gpu();
#endif
}

void guassiankernel(float *kernel, int size, float sigma);

void matproduct(float a[], float b[], float c[], int m, int n, int p);

void guassiankernel2d(float *kernel, int sizeX, int sizeY, float sigmaX, float sigmaY);

} // namespace render
} // namespace zilliz
