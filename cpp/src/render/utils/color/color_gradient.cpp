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

#include "render/utils/color/color_gradient.h"

#define TRANSPARENCY 0.5f

namespace arctern {
namespace render {

void ColorGradient::createDefaultHeatMapGradient() {
  color.clear();
  color.push_back(Color(0, 0, 1, 0.0f));   // Blue.
  color.push_back(Color(0, 1, 1, 0.25f));  // Cyan.
  color.push_back(Color(0, 1, 0, 0.5f));   // Green.
  color.push_back(Color(1, 1, 0, 0.75f));  // Yellow.
  color.push_back(Color(1, 0, 0, 1.0f));   // Red.
}

void ColorGradient::createSquareMapGradient(std::vector<Color> color) {
  color.clear();

  color.push_back(Color(0, 1, 0, 0.0f));  // Green.
  color.push_back(Color(1, 1, 0, 0.5f));  // Yellow.
  color.push_back(Color(1, 0, 0, 1.0f));  // Red.
}

void ColorGradient::getColorAtValue(const float value, float& red, float& green,
                                    float& blue) {
  if (color.size() == 0) return;

  for (unsigned int i = 0; i < color.size(); i++) {
    Color& curr_color = color[i];
    if (value < curr_color.a) {
      int index = (i - 1) > 0 ? i - 1 : 0;
      Color& prev_color = color[index];
      float value_diff = (prev_color.a - curr_color.a);
      float fract_between = (value_diff == 0) ? 0 : (value - curr_color.a) / value_diff;
      red = (prev_color.r - curr_color.r) * fract_between + curr_color.r;
      green = (prev_color.g - curr_color.g) * fract_between + curr_color.g;
      blue = (prev_color.b - curr_color.b) * fract_between + curr_color.b;
      return;
    }
  }
  red = color.back().r;
  green = color.back().g;
  blue = color.back().b;
  return;
}

Color ColorGradient::GetColor(Color color_start, Color color_end, double ratio) {
  Color color;
  if (color_start == color_end) {
    color.r = color_start.r;
    color.g = color_start.g;
    color.b = color_start.b;
    color.a = ratio;
  } else {
    color.r = color_start.r + (color_end.r - color_start.r) * ratio;
    color.g = color_start.g + (color_end.g - color_start.g) * ratio;
    color.b = color_start.b + (color_end.b - color_start.b) * ratio;
    color.a = color_start.a;
  }

  return color;
}

}  // namespace render
}  // namespace arctern
