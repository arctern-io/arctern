#pragma once

#include "render/engine/image/format/format.h"


namespace zilliz {
namespace render {
namespace engine {


struct PNGFormat : public ImageFormat {
    PNGFormat() {
        type = ImageFormat::Type::kPNG;
    }
};


} // namespace engine
} // namespace render
} // namespace zilliz
