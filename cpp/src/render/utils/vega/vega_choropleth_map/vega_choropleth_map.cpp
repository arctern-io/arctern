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

#include "render/utils/vega/vega_choropleth_map/vega_choropleth_map.h"

namespace arctern {
namespace render {

VegaChoroplethMap::VegaChoroplethMap(const std::string& json) { Parse(json); }

std::string VegaChoroplethMap::Build() {
  // TODO: add Build() api to build a vega json string.
  return "";
}

void VegaChoroplethMap::Parse(const std::string& json) {
  rapidjson::Document document;
  document.Parse(json.c_str());

  if (document.Parse(json.c_str()).HasParseError()) {
    printf("json format error\n");
    return;
  }

  if (!JsonLabelCheck(document, "width") || !JsonLabelCheck(document, "height") ||
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

  // parse bounding box
  if (!JsonLabelCheck(mark_enter, "bounding_box") ||
      !JsonLabelCheck(mark_enter["bounding_box"], "value") ||
      !JsonTypeCheck(mark_enter["bounding_box"]["value"], rapidjson::Type::kArrayType) ||
      !JsonSizeCheck(mark_enter["bounding_box"]["value"], "bounding_box.value", 4)) {
    return;
  }
  for (int i = 0; i < 4; i++) {
    if (!JsonTypeCheck(mark_enter["bounding_box"]["value"][i],
                       rapidjson::Type::kNumberType)) {
      return;
    }
  }
  bounding_box_.longitude_left = mark_enter["bounding_box"]["value"][0].GetDouble();
  bounding_box_.latitude_left = mark_enter["bounding_box"]["value"][1].GetDouble();
  bounding_box_.longitude_right = mark_enter["bounding_box"]["value"][2].GetDouble();
  bounding_box_.latitude_right = mark_enter["bounding_box"]["value"][3].GetDouble();

  // parse color style
  if (!JsonLabelCheck(mark_enter, "color_style") ||
      !JsonLabelCheck(mark_enter["color_style"], "value") ||
      !JsonTypeCheck(mark_enter["color_style"]["value"], rapidjson::Type::kStringType)) {
    return;
  }
  auto color_style_string = std::string(mark_enter["color_style"]["value"].GetString());
  if (color_style_string == "blue_to_red") {
    color_style_ = ColorStyle::kBlueToRed;
  } else if (color_style_string == "skyblue_to_white") {
    color_style_ = ColorStyle::kSkyBlueToWhite;
  } else if (color_style_string == "purple_to_yellow") {
    color_style_ = ColorStyle::kPurpleToYellow;
  } else if (color_style_string == "red_transparency") {
    color_style_ = ColorStyle::kRedTransParency;
  } else if (color_style_string == "blue_transparency") {
    color_style_ = ColorStyle::kBlueTransParency;
  } else if (color_style_string == "blue_green_yellow") {
    color_style_ = ColorStyle::kBlueGreenYellow;
  } else if (color_style_string == "white_blue") {
    color_style_ = ColorStyle::kWhiteToBlue;
  } else if (color_style_string == "blue_white_red") {
    color_style_ = ColorStyle::kBlueWhiteRed;
  } else if (color_style_string == "green_yellow_red") {
    color_style_ = ColorStyle::kGreenYellowRed;
  } else {
    std::string msg = "unsupported color style '" + color_style_string + "'.";
    // TODO: add log here
  }

  // parse ruler
  if (!JsonLabelCheck(mark_enter, "ruler") ||
      !JsonLabelCheck(mark_enter["ruler"], "value") ||
      !JsonTypeCheck(mark_enter["ruler"]["value"], rapidjson::Type::kArrayType) ||
      !JsonSizeCheck(mark_enter["ruler"]["value"], "ruler.value", 2)) {
    return;
  }
  for (int i = 0; i < 2; i++) {
    if (!JsonTypeCheck(mark_enter["ruler"]["value"][i], rapidjson::Type::kNumberType)) {
      return;
    }
  }
  ruler_ = std::make_pair(mark_enter["ruler"]["value"][0].GetDouble(),
                          mark_enter["ruler"]["value"][1].GetDouble());

  // parse opacity
  if (!JsonLabelCheck(mark_enter, "opacity") ||
      !JsonLabelCheck(mark_enter["opacity"], "value") ||
      !JsonTypeCheck(mark_enter["opacity"]["value"], rapidjson::Type::kNumberType)) {
    return;
  }
  opacity_ = mark_enter["opacity"]["value"].GetDouble();
}

}  // namespace render
}  // namespace arctern
