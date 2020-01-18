#include "render/utils/vega/vega_heatmap/vega_heatmap.h"

namespace zilliz {
namespace render {

VegaHeatMap::VegaHeatMap(const std::string &json) {
    Parse(json);
}

std::string
VegaHeatMap::Build() {
    // TODO: add Build() api to build a vega json string.
    return "";
}

void
VegaHeatMap::Parse(const std::string &json) {

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

    if (!JsonLabelCheck(mark_enter, "map_scale") ||
        !JsonTypeCheck(mark_enter["map_scale"]["value"], rapidjson::Type::kNumberType)) {
        return;
    }
    map_scale_ = mark_enter["map_scale"]["value"].GetDouble();
}

} //namespace render
} //namespace zilliz
