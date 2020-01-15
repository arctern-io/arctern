#pragma once

#include "render/window/window.h"


namespace zilliz {
namespace render {

class Window2D : public Window {
 public:
    void
    Init() override;

    void
    Terminate() override;

};

using Window2DPtr = std::shared_ptr<Window2D>;

} // namespace render
} // namespace zilliz