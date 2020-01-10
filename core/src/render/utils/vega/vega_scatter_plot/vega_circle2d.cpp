#include "vega_circle2d.h"
#include "../color_parser/color.h"

namespace zilliz {
namespace render {

VegaCircle2d::VegaCircle2d(const std::string &json) {
    Parse(json);
}

std::string
VegaCircle2d::Build() {
    // TODO: add Build() api to build a vega json string.
    return "";
}


void
VegaCircle2d::Parse(const std::string &json) {

    rapidjson::Document document;
    rapidjson::ParseResult parse_result = document.Parse(json.c_str());

    if (parse_result.IsError()) {
        printf("illegal");
    }

    window_params_.width = document["width"].GetInt();
    window_params_.height = document["height"].GetInt();

    rapidjson::Value mark_enter;
    mark_enter = document["marks"][0]["encode"]["enter"];

    circle_params_.radius = mark_enter["strokeWidth"]["value"].GetInt();
    circle_params_.color = ColorParser(mark_enter["stroke"]["value"].GetString()).color();
    circle_params_.color.a = mark_enter["opacity"]["value"].GetFloat();
}


} //namespace render
} //namespace zilliz

