#pragma once

#include <memory>


namespace zilliz {
namespace render {
namespace engine {


struct ImageFormat {
    enum Type {
        kUnknown = 0,
        kPNG,
        KBMP,
        KTGA,
        KHDR,
    };
    Type type;
};


using ImageFormatPtr = std::shared_ptr<ImageFormat>;

} // namespace engine
} // namespace render
} // namespace zilliz
