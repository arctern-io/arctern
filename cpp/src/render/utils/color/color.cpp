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
#include <iostream>
#include <regex>

#include "render/utils/color/color.h"

namespace arctern {
namespace render {

ColorParser::ColorParser(const std::string& css_color_string) {
  // for now we only support css HEX color
  // TODO: extends css color: RGBA, HSL, HSV, HWB, CMYK

  css_color_string_ = css_color_string;
  ParseHEX();
}

void ColorParser::ParseHEX() {
  std::regex pattern_rgb("#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})");

  std::smatch match;
  if (std::regex_match(css_color_string_, match, pattern_rgb)) {
    color_.r = std::stoul(match[1].str(), nullptr, 16);
    color_.g = std::stoul(match[2].str(), nullptr, 16);
    color_.b = std::stoul(match[3].str(), nullptr, 16);
  } else {
    // TODO: add log here
    std::cout << css_color_string_ << " is an invalid rgb color\n";
    return;
  }
}

}  // namespace render
}  // namespace arctern
