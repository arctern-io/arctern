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

#include "render/utils/vega/vega_fishnet_map/vega_fishnet_map.h"

namespace arctern {
namespace render {

VegaFishNetMap::VegaFishNetMap(const std::string& json) { Parse(json); }

void VegaFishNetMap::Parse(const std::string& json) {
  rapidjson::Document document;
  document.Parse(json.c_str());

  if (document.Parse(json.c_str()).HasParseError()) {
    is_valid_ = false;
    std::string err_msg = "json format error";
    throw std::runtime_error(err_msg);
  }

  if (!JsonLabelCheck(document, "width") || !JsonLabelCheck(document, "height") ||
      !JsonNullCheck(document["width"]) || !JsonNullCheck(document["height"]) ||
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

  // parse color gradient
  if (!JsonLabelCheck(mark_enter, "color_gradient") ||
      !JsonLabelCheck(mark_enter["color_gradient"], "value") ||
      !JsonTypeCheck(mark_enter["color_gradient"]["value"],
                     rapidjson::Type::kArrayType)) {
    return;
  }

  auto color_gradient_size = mark_enter["color_gradient"]["value"].Size();
  if (color_gradient_size == 1 && JsonTypeCheck(mark_enter["color_gradient"]["value"][0],
                                                rapidjson::Type::kStringType)) {
    auto color =
        ColorParser(mark_enter["color_gradient"]["value"][0].GetString()).color();
    color.a = mark_enter["opacity"]["value"].GetDouble();
    color_gradient_.emplace_back(color);
  } else if (color_gradient_size == 2 &&
             JsonTypeCheck(mark_enter["color_gradient"]["value"][0],
                           rapidjson::Type::kStringType) &&
             JsonTypeCheck(mark_enter["color_gradient"]["value"][1],
                           rapidjson::Type::kStringType)) {
    auto color_start =
        ColorParser(mark_enter["color_gradient"]["value"][0].GetString()).color();
    auto color_end =
        ColorParser(mark_enter["color_gradient"]["value"][1].GetString()).color();
    auto opacity = mark_enter["opacity"]["value"].GetDouble();
    color_start.a = opacity;
    color_end.a = opacity;
    color_gradient_.emplace_back(color_start);
    color_gradient_.emplace_back(color_end);
  } else {
    std::string err_msg = "unsupported color gradient";
    throw std::runtime_error(err_msg);
  }

  if (!JsonLabelCheck(mark_enter, "cell_size") ||
      !JsonLabelCheck(mark_enter["cell_size"], "value") ||
      !JsonTypeCheck(mark_enter["cell_size"]["value"], rapidjson::Type::kNumberType)) {
    return;
  }
  cell_size_ = mark_enter["cell_size"]["value"].GetDouble();

  if (!JsonLabelCheck(mark_enter, "cell_spacing") ||
      !JsonLabelCheck(mark_enter["cell_spacing"], "value") ||
      !JsonTypeCheck(mark_enter["cell_spacing"]["value"], rapidjson::Type::kNumberType)) {
    return;
  }
  cell_spacing_ = mark_enter["cell_spacing"]["value"].GetDouble();

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
