#pragma once

#include "render/window/window.h"

#define MESA_EGL_NO_X11_HEADERS 1
#include <EGL/egl.h>
#include <GL/gl.h>

namespace zilliz {
namespace render {

class WindowGPU : public Window {
 public:
    virtual void
    Init() = 0;

    virtual void
    Terminate() = 0;

 public:

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
};

using WindowGPUPtr = std::shared_ptr<WindowGPU>;

} // namespace render
} // namespace zilliz
