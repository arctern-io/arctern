#pragma once

#include "rapidjson/document.h"

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

 protected:
    // vega json to vega struct
    virtual void
    Parse(const std::string& json) = 0;

 protected:
    WindowParams window_params_;
};


} // namespace render
} // namespace zilliz
