#pragma once

#include <string>


namespace zilliz {
namespace render {

struct Color {
    float r;
    float g;
    float b;
    float a;
};

class ColorParser {
 public:
    ColorParser(const std::string& css_color_string);

    const Color&
    color() const { return color_; }

 private:
    void
    ParseHEX();

    //TODO: add ParseRGBA(), ParseHSL(), ParseHSV(), ParseHWB(), ParseCMYK()

 private:
    Color color_;
    std::string css_color_string_;
};


} // namespace render
} // namespace zilliz

