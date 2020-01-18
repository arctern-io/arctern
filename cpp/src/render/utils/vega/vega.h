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
//    std::string ToString();

    virtual std::string
    Build() = 0;

    const WindowParams&
    window_params() const { return window_params_; }

 protected:
    // vega json to vega struct
    virtual void
    Parse(const std::string& json) = 0;

    bool
    JsonLabelCheck(rapidjson::Value &value, const std::string &label);

    bool
    JsonSizeCheck(rapidjson::Value &value, const std::string &label, size_t size);

    bool
    JsonTypeCheck(rapidjson::Value &value, rapidjson::Type type);

    bool
    JsonNullCheck(rapidjson::Value &value);

 protected:
    WindowParams window_params_;
};


} // namespace render
} // namespace zilliz
