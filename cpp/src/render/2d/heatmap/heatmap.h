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

#include "render/2d/general_2d.h"
#include "render/2d/heatmap/set_color.h"

namespace arctern {
namespace render {

template <typename T>
class HeatMap : public General2D {
 public:
  HeatMap();

  HeatMap(uint32_t* input_x, uint32_t* input_y, T* count, int64_t num_vertices);

  ~HeatMap();

  void DataInit() final;

  uint8_t* Render() final;

  void Shader();

  void Draw() final;

  void InputInit() final;

 public:
  VegaHeatMap& mutable_heatmap_vega() { return heatmap_vega_; }

 private:
  unsigned int VAO_;
  unsigned int VBO_[3];
  uint32_t* vertices_x_;
  uint32_t* vertices_y_;
  T* count_;
  float* colors_;
  int64_t num_vertices_;
  VegaHeatMap heatmap_vega_;
};

}  // namespace render
}  // namespace arctern
