#pragma once

#include "window_cpu.h"


namespace zilliz {
namespace render {

class WindowCPU2D : public WindowCPU {
 public:
    void
    Init() final;

    void
    Terminate() final;

};

using WindowCPU2DPtr = std::shared_ptr<WindowCPU2D>;

} // namespace render
} // namespace zilliz