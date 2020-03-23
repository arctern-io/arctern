/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <string>
#include <utility>

#include "render/utils/vega/vega_scatter_plot/vega_weighted_pointmap.h"

namespace arctern {
namespace render {

VegaWeightedPointmap::VegaWeightedPointmap(const std::string& json) { Parse(json); }

void VegaWeightedPointmap::Parse(const std::string& json) {
  rapidjson::Document document;
  document.Parse(json.c_str());

  if (document.Parse(json.c_str()).HasParseError()) {
    // TODO: add log here
    printf("json format error\n");
    is_valid_ = false;
    return;
  }

  // 1. parse image width and height
  if (!JsonLabelCheck(document, "width") || !JsonLabelCheck(document, "height") ||
      !JsonNullCheck(document["width"]) || !JsonNullCheck(document["height"]) ||
      !JsonTypeCheck(document["width"], rapidjson::Type::kNumberType) ||
      !JsonTypeCheck(document["height"], rapidjson::Type::kNumberType)) {
    return;
  }
  window_params_.mutable_width() = document["width"].GetInt();
  window_params_.mutable_height() = document["height"].GetInt();

  // 2. parse marks root
  if (!JsonLabelCheck(document, "marks") ||
      !JsonTypeCheck(document["marks"], rapidjson::Type::kArrayType) ||
      !JsonSizeCheck(document["marks"], "marks", 1) ||
      !JsonLabelCheck(document["marks"][0], "encode") ||
      !JsonLabelCheck(document["marks"][0]["encode"], "enter")) {
    return;
  }
  rapidjson::Value mark_enter;
  mark_enter = document["marks"][0]["encode"]["enter"];

  // 3. parse point map color
  if (!JsonLabelCheck(mark_enter, "color") ||
      !JsonLabelCheck(mark_enter["color"], "value") ||
      !JsonTypeCheck(mark_enter["color"]["value"], rapidjson::Type::kStringType)) {
    return;
  }
  auto color_string = std::string(mark_enter["color"]["value"].GetString());
  ColorParser color_parser(color_string);
  if (color_parser.is_css_hex_color()) {
    is_multiple_color_ = false;
    circle_params_.color = color_parser.color();
  } else if (color_string == "blue_to_red") {
    is_multiple_color_ = true;
    color_style_ = ColorStyle::kBlueToRed;
  } else if (color_string == "skyblue_to_white") {
    is_multiple_color_ = true;
    color_style_ = ColorStyle::kSkyBlueToWhite;
  } else if (color_string == "purple_to_yellow") {
    is_multiple_color_ = true;
    color_style_ = ColorStyle::kPurpleToYellow;
  } else if (color_string == "red_transparency") {
    is_multiple_color_ = true;
    color_style_ = ColorStyle::kRedTransParency;
  } else if (color_string == "blue_transparency") {
    is_multiple_color_ = true;
    color_style_ = ColorStyle::kBlueTransParency;
  } else if (color_string == "blue_green_yellow") {
    is_multiple_color_ = true;
    color_style_ = ColorStyle::kBlueGreenYellow;
  } else if (color_string == "white_blue") {
    is_multiple_color_ = true;
    color_style_ = ColorStyle::kWhiteToBlue;
  } else if (color_string == "blue_white_red") {
    is_multiple_color_ = true;
    color_style_ = ColorStyle::kBlueWhiteRed;
  } else if (color_string == "green_yellow_red") {
    is_multiple_color_ = true;
    color_style_ = ColorStyle::kGreenYellowRed;
  } else {
    is_valid_ = false;
    std::string msg = "unsupported color '" + color_string + "'.";
    // TODO: add log here
  }

  // 4. parse color_ruler
  if (!JsonLabelCheck(mark_enter, "color_ruler") ||
      !JsonLabelCheck(mark_enter["color_ruler"], "value") ||
      !JsonTypeCheck(mark_enter["color_ruler"]["value"], rapidjson::Type::kArrayType)) {
    return;
  }
  auto color_ruler_size = mark_enter["color_ruler"]["value"].Size();
  if (color_ruler_size == 2 &&
      JsonTypeCheck(mark_enter["color_ruler"]["value"][0],
                    rapidjson::Type::kNumberType) &&
      JsonTypeCheck(mark_enter["color_ruler"]["value"][1],
                    rapidjson::Type::kNumberType)) {
    color_ruler_ = std::make_pair(mark_enter["color_ruler"]["value"][0].GetDouble(),
                                  mark_enter["color_ruler"]["value"][1].GetDouble());
  } else {
    // TODO: add log here
    std::string msg = "unsupported color ruler.";
    return;
  }

  // 5. parse stroke_ruler
  if (!JsonLabelCheck(mark_enter, "stroke_ruler") ||
      !JsonLabelCheck(mark_enter["stroke_ruler"], "value") ||
      !JsonTypeCheck(mark_enter["stroke_ruler"]["value"], rapidjson::Type::kArrayType)) {
    return;
  }
  auto stroke_ruler_size = mark_enter["stroke_ruler"]["value"].Size();
  if (stroke_ruler_size == 1 && JsonTypeCheck(mark_enter["stroke_ruler"]["value"][0],
                                              rapidjson::Type::kNumberType)) {
    is_multiple_point_size_ = false;
    circle_params_.radius = mark_enter["stroke_ruler"]["value"][0].GetDouble();
  } else if (stroke_ruler_size == 2 &&
             JsonTypeCheck(mark_enter["stroke_ruler"]["value"][0],
                           rapidjson::Type::kNumberType) &&
             JsonTypeCheck(mark_enter["stroke_ruler"]["value"][1],
                           rapidjson::Type::kNumberType)) {
    is_multiple_point_size_ = true;
    stroke_ruler_ = std::make_pair(mark_enter["stroke_ruler"]["value"][0].GetDouble(),
                                   mark_enter["stroke_ruler"]["value"][1].GetDouble());
  } else {
    is_valid_ = false;
    // TODO: add log here
    std::string msg = "unsupported color ruler.";
    return;
  }

  // 6. parse opacity
  if (!JsonLabelCheck(mark_enter, "opacity") ||
      !JsonLabelCheck(mark_enter["opacity"], "value") ||
      !JsonTypeCheck(mark_enter["opacity"]["value"], rapidjson::Type::kNumberType)) {
    return;
  }
  circle_params_.color.a = mark_enter["opacity"]["value"].GetDouble();
}

}  // namespace render
}  // namespace arctern
