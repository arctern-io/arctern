#pragma once

#include "window.h"


namespace zilliz {
namespace render {
namespace engine {

class Window2D : public Window {
 public:
    void
    Init() override;

    void
    Terminate() override;

};


} // namespace engine
} // namespace render
} // namespace zilliz