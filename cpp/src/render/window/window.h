#pragma once

#include "render/window/window_params.h"

namespace zilliz {
namespace render {

enum class WindowType {
    kUnknown = 0,
    k2D
};

class Window {
 public:
    virtual void
    Init() = 0;

    virtual void
    Terminate() = 0;

 public:
    const WindowParams
    window_params() const { return window_params_; }

    void
    set_window_params(WindowParams window_params) { window_params_ = window_params; }

    void
    set_window_type(WindowType window_type) { window_type_ = window_type; }

 protected:
    WindowType window_type_;
    WindowParams window_params_;
};

using WindowPtr = std::shared_ptr<Window>;

} // namespace render
} // namespace zilliz
