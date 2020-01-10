#pragma once

#include "../vega.h"
#include "../color_parser/color.h"

namespace zilliz {
namespace render {


class VegaScatterPlot: public Vega {
 public:
    struct CircleParams {
        int radius;
        Color color;
    };

    virtual std::string
    Build() = 0;

 protected:
    // vega json to vega struct
    virtual void
    Parse(const std::string& json) = 0;
};


} // namespace render
} // namespace zilliz