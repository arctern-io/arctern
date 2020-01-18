#pragma once

#include "GL/osmesa.h"

#include "render/window/window.h"

namespace zilliz {
namespace render {

class WindowCPU : public Window {
 public:
    virtual void
    Init() = 0;

    virtual void
    Terminate() = 0;

    GLubyte *
    mutable_buffer() { return buffer_; }

 protected:
    GLubyte *buffer_;
    OSMesaContext context_;
};

using WindowCPUPtr = std::shared_ptr<WindowCPU>;

} // namespace render
} // namespace zilliz
