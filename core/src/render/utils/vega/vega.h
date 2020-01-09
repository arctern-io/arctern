#pragma once


namespace zilliz {
namespace render {

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

/***
 * TODO: add comments
 */
struct Vega {
 public:
    Vega(const std::string& json);

    std::string ToString();

    const CircleParams2D
    point_format() {return point_format_; }

 private:
    // vega json to vega struct
    void
    Parse(const std::string& json);

 private:
    CircleParams2D point_format_;
};


} // namespace render
} // namespace zilliz
