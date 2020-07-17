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
#include <unordered_map>
#include <vector>

#include "render/utils/vega/vega_unique_value_map/vega_unique_value_map.h"

namespace arctern {
namespace render {

VegaUniqueValueMap::VegaUniqueValueMap(const std::string& json) { Parse(json); }

void VegaUniqueValueMap::Parse(const std::string& json) {
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

  if (!JsonLabelCheck(mark_enter, "unique_value_infos") ||
      !JsonLabelCheck(mark_enter, "opacity") ||
      !JsonLabelCheck(mark_enter["opacity"], "value") ||
      !JsonLabelCheck(mark_enter["unique_value_infos"], "value") ||
      !JsonTypeCheck(mark_enter["unique_value_infos"]["value"],
                     rapidjson::Type::kArrayType) ||
      !JsonTypeCheck(mark_enter["opacity"]["value"], rapidjson::Type::kNumberType)) {
    return;
  }

  auto opacity = mark_enter["opacity"]["value"].GetDouble();

  auto unique_value_infos_size = mark_enter["unique_value_infos"]["value"].Size();
  std::vector<rapidjson::Type> unique_value_infos_key_types;
  unique_value_infos_key_types.resize(unique_value_infos_size);

  for (int i = 0; i < unique_value_infos_size; i++) {
    if (!JsonLabelCheck(mark_enter["unique_value_infos"]["value"][i], "label") ||
        !JsonLabelCheck(mark_enter["unique_value_infos"]["value"][i], "color")) {
      return;
    }
    unique_value_infos_key_types[i] =
        mark_enter["unique_value_infos"]["value"][i]["label"].GetType();
  }

  for (int i = 0; i < unique_value_infos_key_types.size() - 1; i++) {
    if (unique_value_infos_key_types[i] != rapidjson::Type::kStringType &&
        unique_value_infos_key_types[i] != rapidjson::Type::kNumberType) {
      std::string err_msg = "Invalid unique value infos label type";
      throw std::runtime_error(err_msg);
    }
    if (unique_value_infos_key_types[i] != unique_value_infos_key_types[i + 1]) {
      std::string err_msg = "Unique value infos label must have the same type";
      throw std::runtime_error(err_msg);
    }
  }

  for (int i = 0; i < unique_value_infos_size; i++) {
    auto color_string = mark_enter["unique_value_infos"]["value"][i]["color"].GetString();

    auto color_parser = ColorParser(color_string);
    if (!color_parser.is_css_hex_color()) {
      std::string err_msg =
          color_parser.css_color_string() + " is not a valid hex rgb color";
      throw std::runtime_error(err_msg);
    }

    auto color = color_parser.color();
    color.a = opacity;

    const auto& type = unique_value_infos_key_types[i];
    if (type == rapidjson::Type::kStringType) {
      auto label_value =
          mark_enter["unique_value_infos"]["value"][i]["label"].GetString();
      unique_value_infos_string_map_[label_value] = color;
    } else {
      auto label_value =
          mark_enter["unique_value_infos"]["value"][i]["label"].GetDouble();
      unique_value_infos_numeric_map_[label_value] = color;
    }
  }
}

}  // namespace render
}  // namespace arctern
