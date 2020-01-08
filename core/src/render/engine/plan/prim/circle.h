#pragma once


namespace zilliz {
namespace render {
namespace engine {


struct CircleParams2D {
    struct Color {
        float r;
        float g;
        float b;
        float a;
    };

    float radius;
    Color color;
};


} // namespace engine
} // namespace render
} // namespace zilliz
