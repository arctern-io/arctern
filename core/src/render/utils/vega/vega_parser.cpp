#include "render/engine/plan/node/plan_node_building_weighted_2d.h"
#include "render/engine/plan/node/plan_node_cursor_inter.h"
#include "render/engine/common/error.h"
#include "render/engine/common/log.h"
#include "render/engine/image/format/png_format.h"
#include "render/engine/plan/plan/render_plan.h"
#include "render/utils/vega/vega_parser.h"


namespace zilliz {
namespace render {
namespace engine {


bool
VegaParser::JsonLabelCheck(Value &value, const std::string &label) {

    if (!value.HasMember(label.c_str())) {
        RENDER_ENGINE_LOG_ERROR << "Cannot find label [" << label << "] !";
        return false;
    }
    return true;
}


bool
VegaParser::JsonSizeCheck(Value &value, const std::string &label, size_t size) {

    if (value.Size() != size) {
        RENDER_ENGINE_LOG_ERROR << "Member [" << label << "].size should be " << size;
        return false;
    }
    return true;
}


bool
VegaParser::WindowParamsCheck() {

    return JsonLabelCheck(document_, "data") &&
        JsonLabelCheck(document_, "height") &&
        JsonLabelCheck(document_, "width");
}


bool
VegaParser::RenderTypeCheck(int i) {

    if (!JsonSizeCheck(data_[i]["values"], "render_type", 1)) {
        return false;
    }
    if (!data_[i]["values"][0].IsString()) {
        return false;
    }
    if (std::string(data_[i]["values"][0].GetString()) != "circles_2d" &&
        std::string(data_[i]["values"][0].GetString()) != "icon_2d" &&
        std::string(data_[i]["values"][0].GetString()) != "multi_color_circles_2d" &&
        std::string(data_[i]["values"][0].GetString()) != "weighted_color_circles_2d" &&
        std::string(data_[i]["values"][0].GetString()) != "building_weighted_2d" &&
        std::string(data_[i]["values"][0].GetString()) != "weighted_pointsize_circles_2d" &&
        std::string(data_[i]["values"][0].GetString()) != "weighted_circles_2d" &&
        std::string(data_[i]["values"][0].GetString()) != "heatmap_2d" &&
        std::string(data_[i]["values"][0].GetString()) != "get_building_shape") {
        RENDER_ENGINE_LOG_ERROR << "illegal render type.";
        return false;
    }
    return true;
}


bool
VegaParser::RadiusCheck(int i) {

    int radius_size = data_[i]["values"].Size();

    for (auto j = 0; j < radius_size; j++) {
        if (!data_[i]["values"][j].IsNumber()) {
            RENDER_ENGINE_LOG_ERROR << "illegal radius.";
            return false;
        }
    }
    return true;
}


bool
VegaParser::ImageFormatCheck(int i) {

    if (!JsonSizeCheck(data_[i]["values"], "image_format", 1)) {
        return false;
    }
    if (!data_[i]["values"][0].IsString()) {
        return false;
    }
    if (std::string(data_[i]["values"][0].GetString()) != "png" &&
        std::string(data_[i]["values"][0].GetString()) != "bmp" &&
        std::string(data_[i]["values"][0].GetString()) != "tga" &&
        std::string(data_[i]["values"][0].GetString()) != "hdr") {
        RENDER_ENGINE_LOG_ERROR << "illegal image format.";
        return false;
    }
    return true;
}



bool
VegaParser::SingleColorCheck() {

    if (!JsonSizeCheck(colors_, "colors", 1)) {
        return false;
    }
    if (!JsonLabelCheck(colors_[0], "color_r") ||
        !JsonLabelCheck(colors_[0], "color_g") ||
        !JsonLabelCheck(colors_[0], "color_b") ||
        !JsonLabelCheck(colors_[0], "color_a")) {
        RENDER_ENGINE_LOG_ERROR << "illegal single colors.";
        return false;
    }
    return true;
}


bool
VegaParser::MultiColorCheck() {

    int num_colors = colors_.Size();
    for (auto j = 0; j < num_colors; j++) {
        if (!JsonLabelCheck(colors_[j], "label") ||
            !JsonLabelCheck(colors_[j], "color_r") ||
            !JsonLabelCheck(colors_[j], "color_g") ||
            !JsonLabelCheck(colors_[j], "color_b") ||
            !JsonLabelCheck(colors_[j], "color_a")) {
            RENDER_ENGINE_LOG_ERROR << "illegal multi colors.";
            return false;
        }
    }
    return true;
}


bool
VegaParser::WeightedColorStyleCheck() {

    if (!JsonSizeCheck(color_style_, "color_style", 1)) {
        return false;
    }
    if (!JsonLabelCheck(color_style_[0], "ruler")) {
        return false;
    }
    if (!JsonLabelCheck(color_style_[0], "style")) {
        return false;
    }
    if (std::string(color_style_[0]["style"].GetString()) != "blue_to_red" &&
        std::string(color_style_[0]["style"].GetString()) != "skyblue_to_white" &&
        std::string(color_style_[0]["style"].GetString()) != "purple_to_yellow" &&
        std::string(color_style_[0]["style"].GetString()) != "red_transparency" &&
        std::string(color_style_[0]["style"].GetString()) != "blue_transparency" &&
        std::string(color_style_[0]["style"].GetString()) != "blue_green_yellow" &&
        std::string(color_style_[0]["style"].GetString()) != "white_blue" &&
        std::string(color_style_[0]["style"].GetString()) != "blue_white_red" &&
        std::string(color_style_[0]["style"].GetString()) != "green_yellow_red") {
        RENDER_ENGINE_LOG_ERROR << "illegal color style.";
        return false;
    }
    if (!JsonSizeCheck(color_style_[0]["ruler"], "ruler", 2)) {
        return false;
    }
    for (int i = 0; i < 2; i++) {
        if (!color_style_[0]["ruler"][i].IsNumber()) {
            return false;
        }
    }

    return true;
}

bool
VegaParser::BuildingColorStyleCheck() {

    if (!JsonLabelCheck(color_style_[0], "style")) {
        return false;
    }
    if (std::string(color_style_[0]["style"].GetString()) != "blue_to_red" &&
        std::string(color_style_[0]["style"].GetString()) != "skyblue_to_white" &&
        std::string(color_style_[0]["style"].GetString()) != "purple_to_yellow" &&
        std::string(color_style_[0]["style"].GetString()) != "red_transparency" &&
        std::string(color_style_[0]["style"].GetString()) != "blue_transparency" &&
        std::string(color_style_[0]["style"].GetString()) != "blue_green_yellow" &&
        std::string(color_style_[0]["style"].GetString()) != "white_blue" &&
        std::string(color_style_[0]["style"].GetString()) != "blue_white_red" &&
        std::string(color_style_[0]["style"].GetString()) != "green_yellow_red") {
        RENDER_ENGINE_LOG_ERROR << "illegal color style.";
        return false;
    }

    return true;
}

bool
VegaParser::WeightedPointSizeStyleCheck() {
    if (!JsonSizeCheck(colors_, "colors", 1)) {
        return false;
    }
    if (!JsonLabelCheck(colors_[0], "color_r") ||
        !JsonLabelCheck(colors_[0], "color_g") ||
        !JsonLabelCheck(colors_[0], "color_b") ||
        !JsonLabelCheck(colors_[0], "color_a")) {
        RENDER_ENGINE_LOG_ERROR << "illegal weighted_pointsize colors.";
        return false;
    }
    return true;
}

bool
VegaParser::HeatmapStyleCheck() {

    if (!JsonSizeCheck(map_scale_ratio_, "map_scale_ratio", 1)) {
        return false;
    }

    if (!map_scale_ratio_[0].IsNumber()) {
        RENDER_ENGINE_LOG_ERROR << "illegal map_scale_ratio.";
        return false;
    }

    return true;
}


bool
VegaParser::HasColorCheck() {

    for (size_t i = 0; i < data_.Size(); i++) {
        if (std::string(data_[i]["name"].GetString()) == "color_style") {
            return true;
        }
        if (std::string(data_[i]["name"].GetString()) == "colors") {
            return true;
        }
        if (std::string(data_[i]["name"].GetString()) == "map_scale_ratio") {
            return true;
        }
    }
    return false;
}


bool
VegaParser::BoundBoxCheck() {

    if (!JsonSizeCheck(bound_box_, "bound_box", 4)) {
        return false;
    }

    for (int i = 0; i < 4; i++) {
        if (!bound_box_[i].IsNumber()) {
            return false;
        }
    }

    return true;
}


bool
VegaParser::VegaJsonFormatCheck(const std::string &query) {

    data_ = document_["data"];

    bool has_render_type = false;

    for (size_t i = 0; i < data_.Size(); i++) {
        if (!JsonLabelCheck(data_[i], "name") ||
            !JsonLabelCheck(data_[i], "values")) {
            return false;
        }
        if (std::string(data_[i]["name"].GetString()) == "render_type") {
            if (!RenderTypeCheck(i)) {
                return false;
            }
            has_render_type = true;
        }
    }

    if (!has_render_type) {
        RENDER_ENGINE_LOG_ERROR << "Cannot find label render_type.";
        return false;
    }
    return true;
}


void
VegaParser::SetCommonValueFromJson() {

    for (size_t i = 0; i < data_.Size(); i++) {
        std::string data_name(data_[i]["name"].GetString());
        Value &common_value = data_[i]["values"];
        if (data_name == "image_format") {
            image_format_ = common_value;
        }
        if (data_name == "render_type") {
            render_layers_ = common_value;
        }
        if (data_name == "radius") {
            radius_ = common_value;
        }
    }
}


void
VegaParser::SetColorValueFromJson() {

    for (size_t i = 0; i < data_.Size(); i++) {
        std::string data_name(data_[i]["name"].GetString());
        Value &color_value = data_[i]["values"];
        if (data_name == "colors") {
            colors_ = color_value;
            break;
        }
        if (data_name == "color_style") {
            color_style_ = color_value;
            break;
        }
        if (data_name == "map_scale_ratio") {
            map_scale_ratio_ = color_value;
            break;
        }
    }
}

void VegaParser::SetBoundBoxFromJson() {

    for (size_t i = 0; i < data_.Size(); i++) {
        std::string data_name(data_[i]["name"].GetString());
        Value &bound_box = data_[i]["values"];
        if (data_name == "bound_box") {
            bound_box_ = bound_box;
        }
    }
}

void VegaParser::SetGeoTypeFromJson() {

    for (size_t i = 0; i < data_.Size(); i++) {
        std::string data_name(data_[i]["name"].GetString());
        Value &geo_type = data_[i]["values"];
        if (data_name == "geo_type") {
            geo_type_ = geo_type;
        }
    }
}


WindowParams
VegaParser::GetWindowParamsFromJson() {

    WindowParams window_params{};

    if (document_["height"].IsNull() || document_["width"].IsNull()) {
        RENDER_ENGINE_LOG_WARNING << "no window params, make sure the render output type is not image.";
        return window_params;
    }

    window_params.set_height(document_["height"].GetFloat());
    window_params.set_width(document_["width"].GetFloat());
    return window_params;
}


ImageFormatPtr
VegaParser::GetImageFormatFromJson() {

    if( image_format_.IsNull()) {
        RENDER_ENGINE_LOG_WARNING << "no image format, make sure the render output type is not image.";
        return nullptr;
    }

    auto image_format_string = std::string(image_format_[0].GetString());

    if (image_format_string == "png") {
        auto image_format = std::make_shared<PNGFormat>();
        return image_format;
    } else {
        RENDER_ENGINE_LOG_WARNING << "Return unknown image format";
        auto image_format = std::make_shared<ImageFormat>();
        image_format->type = ImageFormat::kUnknown;
        return image_format;
    }
}


PlanNodePtr
VegaParser::GetPlanNodesFromJson() {

    PlanNodePtr root_plan_node = nullptr;

    int num_render_layer = render_layers_.Size();
    PlanNodePtr prev_plan_node;

    for (auto j = 0; j < num_render_layer; j++) {
        PlanNodePtr cur_plan_node;
        auto render_type = std::string(render_layers_[j].GetString());

        if (render_type == "circles_2d") {
            cur_plan_node = std::make_shared<PlanNodeCircle2D>();
            cur_plan_node->set_plan_node_type(PlanNodeType::kCircle2D);
        } else if (render_type == "icon_2d") {
            cur_plan_node = std::make_shared<PlanNodeIcon2D>();
            cur_plan_node->set_plan_node_type(PlanNodeType::kIcon2D);
        } else if (render_type == "multi_color_circles_2d") {
            cur_plan_node = std::make_shared<PlanNodeMultiColorCircle2D>();
            cur_plan_node->set_plan_node_type(PlanNodeType::kMultiColorCircle2D);
        } else if (render_type == "weighted_color_circles_2d") {
            cur_plan_node = std::make_shared<PlanNodeWeightedColorCircle2D>();
            cur_plan_node->set_plan_node_type(PlanNodeType::kWeightedColorCircle2D);
        } else if (render_type == "building_weighted_2d") {
            cur_plan_node = std::make_shared<PlanNodeBuildingWeighted2D>();
            cur_plan_node->set_plan_node_type(PlanNodeType::kBuildingWieghted2D);
        } else if (render_type == "weighted_pointsize_circles_2d") {
            cur_plan_node = std::make_shared<PlanNodeWeightedPointSizeCircle2D>();
            cur_plan_node->set_plan_node_type(PlanNodeType::kWeightedPointSizeCircle2D);
        } else if (render_type == "weighted_circles_2d") {
            cur_plan_node = std::make_shared<PlanNodeWeightedCircle2D>();
            cur_plan_node->set_plan_node_type(PlanNodeType::kWeightedCircle2D);
        } else if (render_type == "heatmap_2d") {
            cur_plan_node = std::make_shared<PlanNodeHeatMap2D>();
            cur_plan_node->set_plan_node_type(PlanNodeType::kHeatMap);
        } else if (render_type == "get_building_shape") {
            cur_plan_node = std::make_shared<PlanNodeCursorInter>();
            cur_plan_node->set_plan_node_type(PlanNodeType::kCursorInter);
        } else {
            std::string msg = "unrecognized render type '" + render_type + "'";
            THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, msg)
        }

        if (root_plan_node == nullptr) {
            root_plan_node = cur_plan_node;
        } else {
            prev_plan_node->set_child(cur_plan_node);
        }
        prev_plan_node = cur_plan_node;
    }

    return root_plan_node;
}


CircleParams2D
VegaParser::GetCircleParams2DFromJson(int i) {

    CircleParams2D circle_params_2d{};

    circle_params_2d.color.r = colors_[i]["color_r"].GetFloat();
    circle_params_2d.color.g = colors_[i]["color_g"].GetFloat();
    circle_params_2d.color.b = colors_[i]["color_b"].GetFloat();
    circle_params_2d.color.a = colors_[i]["color_a"].GetFloat();
    return circle_params_2d;
}


void
VegaParser::Visit(PlanNodeIcon2D &plan_node) {
    for (size_t i = 0; i < data_.Size(); i++) {
        std::string data_name(data_[i]["name"].GetString());
        if (data_name == "icon_name") {
            auto icon_name = std::string(data_[i]["values"][0].GetString());
            plan_node.set_icon_name(icon_name);
            break;
        }
    }
}


void
VegaParser::Visit(PlanNodeCircle2D &plan_node) {

    if (!HasColorCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Cannot find color json.")
    }

    SetColorValueFromJson();

    if (!SingleColorCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid color json.")
    }

    if (!JsonSizeCheck(radius_, "radius", 1)) {
        RENDER_ENGINE_LOG_WARNING << "Size of radius should be 1.";
    }

    auto circle_params = GetCircleParams2DFromJson(0);
    circle_params.radius = radius_[0].GetFloat();

    plan_node.set_circle_params(circle_params);
}


void
VegaParser::Visit(PlanNodeMultiColorCircle2D &plan_node) {

    if (!HasColorCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Cannot find color json.")
    }

    SetColorValueFromJson();

    if (!MultiColorCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid color json.")
    }

    if (!JsonSizeCheck(radius_, "radius", colors_.Size())) {
        RENDER_ENGINE_LOG_WARNING << "Size of radius should be " << colors_.Size();
    }

    auto &param = plan_node.mutable_string_circle_params();
    int num_colors = colors_.Size();

    for (auto i = 0; i < num_colors; i++) {
        auto circle_params = GetCircleParams2DFromJson(i);
        circle_params.radius = radius_[i].GetFloat();

        std::string label_str = colors_[i]["label"].GetString();
        param.emplace(label_str, circle_params);
    }
}


void
VegaParser::Visit(PlanNodeWeightedColorCircle2D &plan_node) {

    if (!HasColorCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Cannot find color_style json.")
    }

    SetColorValueFromJson();

    if (!WeightedColorStyleCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid color_style json.")
    }

    auto &color_style = plan_node.mutable_color_style();
    auto &count_start = plan_node.mutable_count_start();
    auto &count_end = plan_node.mutable_count_end();

    auto color_style_string = std::string(color_style_[0]["style"].GetString());
    auto &count_range = color_style_[0]["ruler"];
    count_start = count_range[0].GetDouble();
    count_end = count_range[1].GetDouble();

    if (count_end <= count_start) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid Ruler json.")
    }

    if (color_style_string == "blue_to_red") {
        color_style = ColorStyle::kBlueToRed;
    } else if (color_style_string == "skyblue_to_white") {
        color_style = ColorStyle::kSkyBlueToWhite;
    } else if (color_style_string == "purple_to_yellow") {
        color_style = ColorStyle::kPurpleToYellow;
    } else if (color_style_string == "red_transparency") {
        color_style = ColorStyle::kRedTransParency;
    } else if (color_style_string == "blue_transparency") {
        color_style = ColorStyle::kBlueTransParency;
    } else if (color_style_string == "blue_green_yellow") {
        color_style = ColorStyle::kBlueGreenYellow;
    } else if (color_style_string == "white_blue") {
        color_style = ColorStyle::kWhiteToBlue;
    } else if (color_style_string == "blue_white_red") {
        color_style = ColorStyle::kBlueWhiteRed;
    } else if (color_style_string == "green_yellow_red") {
        color_style = ColorStyle::kGreenYellowRed;
    } else {
        std::string msg = "unsupported color style '" + color_style_string + "'.";
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, msg)
    }
    plan_node.set_color_style(color_style);
    plan_node.set_radius(radius_[0].GetFloat());
}

void
VegaParser::Visit(PlanNodeWeightedPointSizeCircle2D &plan_node) {

    if (!HasColorCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Cannot find color json.")
    }

    SetColorValueFromJson();

    if (!WeightedPointSizeStyleCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid color json.")
    }

    auto circle_params = GetCircleParams2DFromJson(0);

    plan_node.set_circle_params(circle_params);
}

void
VegaParser::Visit(PlanNodeWeightedCircle2D &plan_node) {

    if (!HasColorCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Cannot find color_style json.")
    }

    SetColorValueFromJson();

    if (!WeightedColorStyleCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid color_style json.")
    }

    auto &color_style = plan_node.mutable_color_style();
    auto &count_start = plan_node.mutable_count_start();
    auto &count_end = plan_node.mutable_count_end();

    auto color_style_string = std::string(color_style_[0]["style"].GetString());
    auto &count_range = color_style_[0]["ruler"];
    count_start = count_range[0].GetDouble();
    count_end = count_range[1].GetDouble();

    if (count_end <= count_start) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid Ruler json.")
    }

    if (color_style_string == "blue_to_red") {
        color_style = ColorStyle::kBlueToRed;
    } else if (color_style_string == "skyblue_to_white") {
        color_style = ColorStyle::kSkyBlueToWhite;
    } else if (color_style_string == "purple_to_yellow") {
        color_style = ColorStyle::kPurpleToYellow;
    } else if (color_style_string == "red_transparency") {
        color_style = ColorStyle::kRedTransParency;
    } else if (color_style_string == "blue_transparency") {
        color_style = ColorStyle::kBlueTransParency;
    } else {
        std::string msg = "unsupported color style '" + color_style_string + "'.";
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, msg)
    }
    plan_node.set_color_style(color_style);
}

void
VegaParser::Visit(PlanNodeHeatMap2D &plan_node) {

    if (!HasColorCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Cannot find color_style json.")
    }

    SetColorValueFromJson();

    if (!HeatmapStyleCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid color_style json.")
    }

    plan_node.set_map_scale_ratio(map_scale_ratio_[0].GetDouble());
}

void
VegaParser::Visit(PlanNodeBuildingWeighted2D &plan_node) {

    if (!HasColorCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Cannot find color_style json.")
    }

    SetColorValueFromJson();

    if (!BuildingColorStyleCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid color_style json.")
    }

    auto &color_style = plan_node.mutable_color_style();
    auto color_style_string = std::string(color_style_[0]["style"].GetString());

    auto &count_range = color_style_[0]["ruler"];
    auto count_start = count_range[0].GetDouble();
    auto count_end = count_range[1].GetDouble();
    plan_node.set_count_start(count_start);
    plan_node.set_count_end(count_end);

    if (color_style_string == "blue_to_red") {
        color_style = ColorStyle::kBlueToRed;
    } else if (color_style_string == "skyblue_to_white") {
        color_style = ColorStyle::kSkyBlueToWhite;
    } else if (color_style_string == "purple_to_yellow") {
        color_style = ColorStyle::kPurpleToYellow;
    } else if (color_style_string == "red_transparency") {
        color_style = ColorStyle::kRedTransParency;
    } else if (color_style_string == "blue_transparency") {
        color_style = ColorStyle::kBlueTransParency;
    } else if (color_style_string == "blue_green_yellow") {
        color_style = ColorStyle::kBlueGreenYellow;
    } else if (color_style_string == "white_blue") {
        color_style = ColorStyle::kWhiteToBlue;
    } else if (color_style_string == "blue_white_red") {
        color_style = ColorStyle::kBlueWhiteRed;
    } else if (color_style_string == "green_yellow_red") {
        color_style = ColorStyle::kGreenYellowRed;
    } else {
        std::string msg = "unsupported color style '" + color_style_string + "'.";
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, msg)
    }
    plan_node.set_color_style(color_style);

    SetBoundBoxFromJson();

    if (!BoundBoxCheck()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid bound box.")
    }

    plan_node.mutable_bounding_box().longitude_left = bound_box_[0].GetDouble();
    plan_node.mutable_bounding_box().latitude_left = bound_box_[1].GetDouble();
    plan_node.mutable_bounding_box().longitude_right = bound_box_[2].GetDouble();
    plan_node.mutable_bounding_box().latitude_right = bound_box_[3].GetDouble();

    SetGeoTypeFromJson();

    auto geo_type_string = std::string(geo_type_[0].GetString());
    GeoType geo_type;
    if (geo_type_string == "building") {
        geo_type = GeoType::kBuilding;
    } else if (geo_type_string == "block") {
        geo_type = GeoType::kBlock;
    } else if (geo_type_string == "district") {
        geo_type = GeoType::kDistrict;
    } else {
        RENDER_ENGINE_LOG_ERROR << "Unknown geo type.";
        geo_type = GeoType::kUnknown;
    }

    plan_node.set_geo_type(geo_type);
}


void
VegaParser::Visit(PlanNodeCursorInter &plan_node) {

    std::pair<double, double> cursor_position{0, 0};

    for (size_t i = 0; i < data_.Size(); i++) {
        std::string data_name(data_[i]["name"].GetString());
        Value &vega_cursor_position = data_[i]["values"];
        if (data_name == "cursor_position") {
            cursor_position.first = vega_cursor_position[0].GetDouble();
            cursor_position.second = vega_cursor_position[1].GetDouble();
            break;
        }
    }

    plan_node.set_cursor_position(cursor_position);
}


RenderPlanPtr
VegaParser::GetPlan(const std::string &query) {

    rapidjson::ParseResult parse_result = document_.Parse(query.c_str());

    if (parse_result.IsError()) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Invalid json format.")
    }

    if (!VegaJsonFormatCheck(query)) {
        THROW_RENDER_ENGINE_ERROR(ILLEGAL_VEGA_FORMAT, "Vega format check failed.")
    }

    SetCommonValueFromJson();

    auto plan = std::make_shared<RenderPlan>();
    //TODO::hard code 2D
    plan->set_plan_type(RenderPlan::Type::k2D);
    plan->set_window_params(GetWindowParamsFromJson());
    plan->set_image_format(GetImageFormatFromJson());

    plan->set_root_plan_node(GetPlanNodesFromJson());
    auto plan_node_ptr = plan->root_plan_node();

    while (plan_node_ptr != nullptr) {
        plan_node_ptr->Apply(*this);
        plan_node_ptr = plan_node_ptr->child();
    }

    return plan;
}


} // namespace engine
} // namespace render
} // namespace zilliz
