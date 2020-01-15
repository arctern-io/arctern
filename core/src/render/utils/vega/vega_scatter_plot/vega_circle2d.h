#pragma once

#include <string>
#include "render/utils/vega/vega_scatter_plot/vega_scatter_plot.h"

namespace zilliz {
namespace render {


class VegaCircle2d: public VegaScatterPlot {
 public:
    VegaCircle2d() = default;

    VegaCircle2d(const std::string& json);

    std::string
    Build() final;

    const CircleParams
    circle_params() const {return circle_params_; }

 private:
    // vega json to vega struct
    void
    Parse(const std::string& json) final;

 private:
    VegaScatterPlot::CircleParams circle_params_;
};


} // namespace render
} // namespace zilliz