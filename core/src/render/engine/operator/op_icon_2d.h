#pragma once

#include "render/engine/operator/op_general_2d.h"


namespace zilliz {
namespace render {
namespace engine {

class OpIcon2D : public OpGeneral2D {
 public:
    DatasetPtr
    Render() override;
};

using OpIcon2DPtr = std::shared_ptr<OpIcon2D>;

} // namespace engine
} // namespace render
} // namespace zilliz