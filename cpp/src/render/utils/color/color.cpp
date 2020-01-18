#include "color.h"

#include <regex>
#include <iostream>


namespace zilliz {
namespace render {


ColorParser::ColorParser(const std::string &css_color_string) {
    // for now we only support css HEX color
    // TODO: extends css color: RGBA, HSL, HSV, HWB, CMYK

    css_color_string_ = css_color_string;
    ParseHEX();
}


void
ColorParser::ParseHEX() {
    std::regex pattern_rgb("#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})");

    std::smatch match;
    if (std::regex_match(css_color_string_, match, pattern_rgb)) {

        color_.r = std::stoul(match[1].str(), nullptr, 16);
        color_.g = std::stoul(match[2].str(), nullptr, 16);
        color_.b = std::stoul(match[3].str(), nullptr, 16);
    } else {
        std::cout << css_color_string_ << " is an invalid rgb color\n";
        return;
    }
}


} //namespace render
} //namespace zilliz
