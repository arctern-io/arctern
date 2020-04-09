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
#include "render/utils/vega/vega_scatter_plot/vega_pointmap.h"

namespace arctern {
namespace render {

VegaPointmap::VegaPointmap(const std::string& json) { Parse(json); }

void VegaPointmap::Parse(const std::string& json) {
  rapidjson::Document doc;
  doc.Parse(json.c_str());

  if (doc.Parse(json.c_str()).HasParseError()) {
    is_valid_ = false;
    std::string err_msg = "json format error";
    throw std::runtime_error(err_msg);
  }

  if (!JsonLabelCheck(doc, "width") || !JsonLabelCheck(doc, "height") ||
      !JsonNullCheck(doc["width"]) || !JsonNullCheck(doc["height"]) ||
      !JsonTypeCheck(doc["width"], rapidjson::Type::kNumberType) ||
      !JsonTypeCheck(doc["height"], rapidjson::Type::kNumberType)) {
    return;
  }
  window_params_.mutable_width() = doc["width"].GetInt();
  window_params_.mutable_height() = doc["height"].GetInt();

  if (!JsonLabelCheck(doc, "marks") ||
      !JsonTypeCheck(doc["marks"], rapidjson::Type::kArrayType) ||
      !JsonSizeCheck(doc["marks"], "marks", 1) ||
      !JsonLabelCheck(doc["marks"][0], "encode") ||
      !JsonLabelCheck(doc["marks"][0]["encode"], "enter")) {
    return;
  }
  rapidjson::Value mark_enter;
  mark_enter = doc["marks"][0]["encode"]["enter"];

  if (!JsonLabelCheck(mark_enter, "point_size") ||
      !JsonLabelCheck(mark_enter, "point_color") ||
      !JsonLabelCheck(mark_enter, "opacity") ||
      !JsonLabelCheck(mark_enter["point_size"], "value") ||
      !JsonLabelCheck(mark_enter["point_color"], "value") ||
      !JsonLabelCheck(mark_enter["opacity"], "value") ||
      !JsonTypeCheck(mark_enter["point_size"]["value"], rapidjson::Type::kNumberType) ||
      !JsonTypeCheck(mark_enter["point_color"]["value"], rapidjson::Type::kStringType) ||
      !JsonTypeCheck(mark_enter["opacity"]["value"], rapidjson::Type::kNumberType)) {
    return;
  }
  point_params_.point_size = mark_enter["point_size"]["value"].GetDouble();
  point_params_.color =
      ColorParser(mark_enter["point_color"]["value"].GetString()).color();
  point_params_.color.a = mark_enter["opacity"]["value"].GetDouble();
}

}  // namespace render
}  // namespace arctern
