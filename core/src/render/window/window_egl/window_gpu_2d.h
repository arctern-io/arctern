#pragma once

#include "window_gpu.h"


namespace zilliz {
namespace render {

class WindowGPU2D : public WindowGPU {
 public:
    void
    Init() final;

    void
    Terminate() final;

};

using WindowGPU2DPtr = std::shared_ptr<WindowGPU2D>;

} // namespace render
} // namespace zilliz