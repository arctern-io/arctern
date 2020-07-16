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
#include <type_traits>
#include <vector>

#include "render/2d/general_2d.h"
#include "render/utils/vega/vega_unique_value_map/vega_unique_value_map.h"

namespace arctern {
namespace render {

template <typename T>
class UniqueValueMap : public General2D {
 public:
  UniqueValueMap() = delete;

  UniqueValueMap(std::vector<OGRGeometryUniquePtr>&& geometries, std::vector<T> values,
                 int64_t num_geometries);

  std::vector<uint8_t> Render() final;

  void Draw() final;

  VegaUniqueValueMap& mutable_vega_unique_value_map() { return vega_unique_value_map_; }

 private:
  void SetColor();

 private:
  std::vector<OGRGeometryUniquePtr> geometries_;
  std::vector<T> values_;
  std::vector<Color> colors_;
  int64_t num_geometries_;
  VegaUniqueValueMap vega_unique_value_map_;
};

}  // namespace render
}  // namespace arctern
