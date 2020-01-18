
#include "vega_circle2d.h"

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
    document.Parse(json.c_str());

    if (document.Parse(json.c_str()).HasParseError()) {
        printf("json format error\n");
        return;
    }


    if (!JsonLabelCheck(document, "width") ||
        !JsonLabelCheck(document, "height") ||
        !JsonTypeCheck(document["width"], rapidjson::Type::kNumberType) ||
        !JsonTypeCheck(document["height"], rapidjson::Type::kNumberType)) {
        return;
    }
    window_params_.mutable_width() = document["width"].GetInt();
    window_params_.mutable_height() = document["height"].GetInt();


    if (!JsonLabelCheck(document, "marks") ||
        !JsonTypeCheck(document["marks"], rapidjson::Type::kArrayType) ||
        !JsonSizeCheck(document["marks"], "marks", 1) ||
        !JsonLabelCheck(document["marks"][0], "encode") ||
        !JsonLabelCheck(document["marks"][0]["encode"], "enter")) {
        return;
    }
    rapidjson::Value mark_enter;
    mark_enter = document["marks"][0]["encode"]["enter"];

    if (!JsonLabelCheck(mark_enter, "strokeWidth") ||
        !JsonLabelCheck(mark_enter, "stroke") ||
        !JsonLabelCheck(mark_enter, "opacity") ||
        !JsonLabelCheck(mark_enter["strokeWidth"], "value") ||
        !JsonLabelCheck(mark_enter["stroke"], "value") ||
        !JsonLabelCheck(mark_enter["opacity"], "value") ||
        !JsonTypeCheck(mark_enter["strokeWidth"]["value"], rapidjson::Type::kNumberType) ||
        !JsonTypeCheck(mark_enter["stroke"]["value"], rapidjson::Type::kStringType) ||
        !JsonTypeCheck(mark_enter["opacity"]["value"], rapidjson::Type::kNumberType)) {
        return;
    }
    circle_params_.radius = mark_enter["strokeWidth"]["value"].GetInt();
    circle_params_.color = ColorParser(mark_enter["stroke"]["value"].GetString()).color();
    circle_params_.color.a = mark_enter["opacity"]["value"].GetDouble();
}


} //namespace render
} //namespace zilliz

