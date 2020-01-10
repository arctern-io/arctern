#pragma once

#include "render/window/window_params.h"

namespace zilliz {
namespace render {

struct CircleParams2D {
    struct Color {
        float r;
        float g;
        float b;
        float a;
    };

    float radius;
    Color color;
};

/***
 * TODO: add comments
 */
struct Vega {
 public:
    Vega(const std::string& json);

    std::string ToString();

    const CircleParams2D&
    point_format() const { return point_format_; }

    const WindowParams&
    window_params() const { return window_params_; }

 private:
    // vega json to vega struct
    void
    Parse(const std::string& json);

 private:
    CircleParams2D point_format_;
    WindowParams window_params_;
};


} // namespace render
} // namespace zilliz
