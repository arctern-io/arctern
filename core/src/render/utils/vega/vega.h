#pragma once

#include "rapidjson/document.h"
#include "render/window/window_params.h"

namespace zilliz {
namespace render {


/***
 * TODO: add comments
 */
class Vega {
 public:
    struct WindowParams {
        int width;
        int height;
    };

 public:
    std::string ToString();

    virtual std::string
    Build() = 0;
    const CircleParams2D&
    point_format() const { return point_format_; }

    const WindowParams&
    window_params() const { return window_params_; }

 protected:
    // vega json to vega struct
    virtual void
    Parse(const std::string& json) = 0;


 protected:
    CircleParams2D point_format_;
    WindowParams window_params_;
};


} // namespace render
} // namespace zilliz
