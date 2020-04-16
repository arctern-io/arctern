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

#include "render/utils/vega/vega_icon/vega_icon.h"

namespace arctern {
namespace render {

VegaIcon::VegaIcon(const std::string& json) { Parse(json); }

void VegaIcon::Parse(const std::string& json) {
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

  if (!JsonLabelCheck(mark_enter, "icon_path") ||
      !JsonTypeCheck(mark_enter["icon_path"]["value"], rapidjson::Type::kStringType)) {
    return;
  }
  icon_path_ = mark_enter["icon_path"]["value"].GetString();
}

}  // namespace render
}  // namespace arctern
