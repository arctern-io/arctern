#pragma once

#define MESA_EGL_NO_X11_HEADERS 1
#include <EGL/egl.h>
#include <GL/gl.h>


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

    EGLDisplay &
    mutable_egl_dpy() { return egl_dpy_; }

    EGLDisplay &
    mutable_egl_surf() { return egl_surf_; }

    EGLContext &
    mutable_egl_context() { return egl_context_; }

 protected:
    EGLDisplay egl_dpy_;
    EGLSurface egl_surf_;
    EGLContext egl_context_;
    WindowType window_type_;
    WindowParams window_params_;
};

using WindowPtr = std::shared_ptr<Window>;

} // namespace render
} // namespace zilliz
