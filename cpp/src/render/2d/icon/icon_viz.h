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
#include "render/utils/vega/vega_icon/vega_icon.h"

namespace arctern {
namespace render {

class IconViz : public General2D {
 public:
  IconViz() = delete;

  IconViz(uint32_t* input_x, uint32_t* input_y, int64_t num_icons);

  std::vector<uint8_t> Render() final;

  void Draw() final;

  VegaIcon& mutable_icon_vega() { return icon_vega_; }

 private:
  uint32_t* vertices_x_;
  uint32_t* vertices_y_;
  size_t num_icons_;
  VegaIcon icon_vega_;
};

}  // namespace render
}  // namespace arctern
