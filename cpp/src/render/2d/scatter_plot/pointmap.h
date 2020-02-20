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
#include "render/2d/input.h"
#include "render/utils/vega/vega_scatter_plot/vega_circle2d.h"

namespace zilliz {
namespace render {

class PointMap : public General2D {
 public:
  PointMap();

  PointMap(uint32_t* input_x, uint32_t* input_y, int64_t num_vertices);

  void DataInit() final;

  uint8_t* Render() final;

  void Shader();

  void Draw() final;

  void InputInit() final;

 public:
  uint32_t* mutable_vertices_x() { return vertices_x_; }

  uint32_t* mutable_vertices_y() { return vertices_y_; }

  VegaCircle2d& mutable_point_vega() { return point_vega_; }

  const size_t num_vertices() const { return num_vertices_; }

 private:
  unsigned int VAO_;
  unsigned int VBO_[2];
  uint32_t* vertices_x_;
  uint32_t* vertices_y_;
  size_t num_vertices_;
  VegaCircle2d point_vega_;
};

}  // namespace render
}  // namespace zilliz
