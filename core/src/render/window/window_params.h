#pragma once

#include <memory>

namespace zilliz {
namespace render {


class WindowParams {
 public:

    const float&
    width() const { return width_;}

    const float&
    height() const { return height_;}

    float&
    mutable_width() { return width_;}

    float&
    mutable_height() { return height_;}

    void
    set_width(float w) { width_ = w;}

    void
    set_height(float h) {height_ = h; }

    void operator=(WindowParams& windowParams){
        width_ = windowParams.width();
        height_ = windowParams.height();
    }
 private:
    float width_;
    float height_;
};

using WindowParamsPtr = std::shared_ptr<WindowParams>;

} // namespace render
} // namespace zilliz