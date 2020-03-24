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
#pragma once

#include <string>

namespace arctern {
namespace render {

struct Color {
  float r;
  float g;
  float b;
  float a;
};

enum class ColorStyle : int {
  kUnknown,
  kBlueToRed,
  kSkyBlueToWhite,
  kPurpleToYellow,
  kRedTransParency,
  kBlueTransParency,
  kBlueGreenYellow,
  kWhiteToBlue,
  kBlueWhiteRed,
  kGreenYellowRed
};

class ColorParser {
 public:
  explicit ColorParser(const std::string& css_color_string);

  const Color& color() const { return color_; }

  const bool& is_css_hex_color() const { return is_css_hex_color_; }

 private:
  void ParseHEX();

  // TODO: add ParseRGBA(), ParseHSL(), ParseHSV(), ParseHWB(), ParseCMYK()

 private:
  Color color_;
  bool is_css_hex_color_;
  std::string css_color_string_;
};

}  // namespace render
}  // namespace arctern
