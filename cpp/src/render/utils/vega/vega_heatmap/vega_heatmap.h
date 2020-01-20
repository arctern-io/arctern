#pragma once

#include "render/utils/vega/vega.h"
#include "render/utils/color/color.h"

namespace zilliz {
namespace render {


class VegaHeatMap: public Vega {
 public:
    VegaHeatMap() = default;

    VegaHeatMap(const std::string& json);

    std::string
    Build() final;

 public:
    const double&
    map_scale() const { return map_scale_; }

 protected:
    // vega json to vega struct
    void
    Parse(const std::string& json) final;

 private:
    double map_scale_;
};


} // namespace render
} // namespace zilliz
