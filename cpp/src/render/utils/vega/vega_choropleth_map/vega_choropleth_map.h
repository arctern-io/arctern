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
#include <utility>

#include "render/utils/color/color.h"
#include "render/utils/vega/vega.h"

namespace arctern {
namespace render {

class VegaChoroplethMap : public Vega {
 public:
  struct BoundingBox {
    double longitude_left;
    double latitude_left;
    double longitude_right;
    double latitude_right;
  };

 public:
  VegaChoroplethMap() = default;

  explicit VegaChoroplethMap(const std::string& json);

  std::string Build() final;

  const BoundingBox& bounding_box() const { return bounding_box_; }

  const std::pair<double, double>& ruler() const { return ruler_; }

  const ColorStyle& color_style() const { return color_style_; }

  const double& opacity() const { return opacity_; }

 private:
  // vega json to vega struct
  void Parse(const std::string& json) final;

 private:
  BoundingBox bounding_box_;
  std::pair<double, double> ruler_;
  ColorStyle color_style_;
  double opacity_;
};

}  // namespace render
}  // namespace arctern
