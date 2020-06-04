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

#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <string>
#include <vector>

#include "render/2d/general_2d.h"
#include "render/utils/vega/vega_choropleth_map/vega_choropleth_map.h"

namespace arctern {
namespace render {

template <typename T>
class ChoroplethMap : public General2D {
 public:
  ChoroplethMap() = delete;

  ChoroplethMap(std::vector<OGRGeometry*> choropleth_wkb, T* count, int64_t num_vertices);

  std::vector<uint8_t> Render() final;

  void Draw() final;

  VegaChoroplethMap& mutable_choroplethmap_vega() { return choropleth_vega_; }

 private:
  void Transform();

  void SetColor();

 private:
  std::vector<OGRGeometry*> choropleth_wkb_;
  T* count_;
  int64_t num_buildings_;
  VegaChoroplethMap choropleth_vega_;

  std::vector<std::vector<int>> buildings_x_;
  std::vector<std::vector<int>> buildings_y_;
  std::vector<float> colors_;
};

}  // namespace render
}  // namespace arctern
