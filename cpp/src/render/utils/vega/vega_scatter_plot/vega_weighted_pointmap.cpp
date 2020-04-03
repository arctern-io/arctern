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
    is_valid_ = false;
    std::string err_msg = "json format error";
    throw std::runtime_error(err_msg);
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
  if (!JsonLabelCheck(mark_enter, "color_gradient") ||
      !JsonLabelCheck(mark_enter["color_gradient"], "value") ||
      !JsonTypeCheck(mark_enter["color_gradient"]["value"],
                     rapidjson::Type::kStringType)) {
    return;
  }
  auto color_string = std::string(mark_enter["color_gradient"]["value"].GetString());
  ColorParser color_parser(color_string);
  if (color_parser.is_css_hex_color()) {
    is_multiple_color_ = false;
    point_params_.color = color_parser.color();
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
    std::string err_msg = "unsupported color '" + color_string + "'.";
    throw std::runtime_error(err_msg);
  }

  // 4. parse color_ruler
  if (!JsonLabelCheck(mark_enter, "color_bound") ||
      !JsonLabelCheck(mark_enter["color_bound"], "value") ||
      !JsonTypeCheck(mark_enter["color_bound"]["value"], rapidjson::Type::kArrayType)) {
    return;
  }
  auto color_ruler_size = mark_enter["color_bound"]["value"].Size();
  if (color_ruler_size == 2 &&
      JsonTypeCheck(mark_enter["color_bound"]["value"][0],
                    rapidjson::Type::kNumberType) &&
      JsonTypeCheck(mark_enter["color_bound"]["value"][1],
                    rapidjson::Type::kNumberType)) {
    color_bound_ = std::make_pair(mark_enter["color_bound"]["value"][0].GetDouble(),
                                  mark_enter["color_bound"]["value"][1].GetDouble());
  } else {
    is_valid_ = false;
    std::string err_msg = "unsupported color bound";
    throw std::runtime_error(err_msg);
  }

  // 5. parse size_bound
  if (!JsonLabelCheck(mark_enter, "size_bound") ||
      !JsonLabelCheck(mark_enter["size_bound"], "value") ||
      !JsonTypeCheck(mark_enter["size_bound"]["value"], rapidjson::Type::kArrayType)) {
    return;
  }
  auto size_bound = mark_enter["size_bound"]["value"].Size();
  if (size_bound == 1 &&
      JsonTypeCheck(mark_enter["size_bound"]["value"][0], rapidjson::Type::kNumberType)) {
    is_multiple_point_size_ = false;
    point_params_.point_size = mark_enter["size_bound"]["value"][0].GetDouble();
  } else if (size_bound == 2 &&
             JsonTypeCheck(mark_enter["size_bound"]["value"][0],
                           rapidjson::Type::kNumberType) &&
             JsonTypeCheck(mark_enter["size_bound"]["value"][1],
                           rapidjson::Type::kNumberType)) {
    is_multiple_point_size_ = true;
    size_bound_ = std::make_pair(mark_enter["size_bound"]["value"][0].GetDouble(),
                                 mark_enter["size_bound"]["value"][1].GetDouble());
  } else {
    is_valid_ = false;
    std::string err_msg = "unsupported size bound";
    throw std::runtime_error(err_msg);
  }

  // 6. parse opacity
  if (!JsonLabelCheck(mark_enter, "opacity") ||
      !JsonLabelCheck(mark_enter["opacity"], "value") ||
      !JsonTypeCheck(mark_enter["opacity"]["value"], rapidjson::Type::kNumberType)) {
    return;
  }
  point_params_.color.a = mark_enter["opacity"]["value"].GetDouble();
}

}  // namespace render
}  // namespace arctern
