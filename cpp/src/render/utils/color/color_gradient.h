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

#include <vector>

namespace zilliz {
namespace render {

class ColorGradient {
 private:
  struct ColorPoint {
    float r, g, b;
    float val;
    ColorPoint(float red, float green, float blue, float value)
        : r(red), g(green), b(blue), val(value) {}
  };
  std::vector<ColorPoint> color;

 public:
  ColorGradient() { createDefaultHeatMapGradient(); }

  void createDefaultHeatMapGradient();

  void getColorAtValue(const float value, float& red, float& green, float& blue);
};

}  // namespace render
}  // namespace zilliz
