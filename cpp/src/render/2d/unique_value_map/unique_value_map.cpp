#include <utility>

#include <utility>

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
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "render/2d/unique_value_map/unique_value_map.h"

namespace arctern {
namespace render {

template class UniqueValueMap<int8_t>;

template class UniqueValueMap<int16_t>;

template class UniqueValueMap<int32_t>;

template class UniqueValueMap<int64_t>;

template class UniqueValueMap<uint8_t>;

template class UniqueValueMap<uint16_t>;

template class UniqueValueMap<uint32_t>;

template class UniqueValueMap<uint64_t>;

template class UniqueValueMap<float>;

template class UniqueValueMap<double>;

template class UniqueValueMap<std::string>;

template <typename T>
UniqueValueMap<T>::UniqueValueMap(std::vector<OGRGeometryUniquePtr>&& geometries,
                                  std::vector<T> values, int64_t num_geometries)
    : geometries_(std::move(geometries)),
      values_(values),
      num_geometries_(num_geometries) {}

template <typename T>
void UniqueValueMap<T>::Draw() {
#ifndef USE_GPU
  glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1,
          1);
#endif

  glClear(GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ZERO);

  const auto& opacity = vega_unique_value_map_.opacity();

  for (int i = 0; i < geometries_.size(); i++) {
    glColor4f(colors_[i].r, colors_[i].g, colors_[i].b, opacity);

    auto& geometry = geometries_[i];
    auto type = geometry->getGeometryType();

    if (type == OGRwkbGeometryType::wkbPolygon) {
      auto ring = geometry->toPolygon()->getExteriorRing();
      auto ring_size = ring->getNumPoints();

      glBegin(GL_POLYGON);

      for (int j = 0; j < ring_size; j++) {
        glVertex2d(ring->getX(j), ring->getY(j));
      }

      glEnd();
    } else if (type == OGRwkbGeometryType::wkbMultiPolygon) {
      auto polygons = geometry->toGeometryCollection();
      auto polygon_size = polygons->getNumGeometries();

      for (int j = 0; j < polygon_size; j++) {
        auto polygon = polygons->getGeometryRef(j)->toPolygon();
        auto ring = polygon->getExteriorRing();
        auto ring_size = ring->getNumPoints();

        glBegin(GL_POLYGON);

        for (int k = 0; k < ring_size; k++) {
          glVertex2d(ring->getX(k), ring->getY(k));
        }

        glEnd();
      }
    } else {
      std::string err_msg = "Unknown geometry type";
      throw std::runtime_error(err_msg);
    }
  }

  glFinish();
}

// SetColor for unique_value_infos_numeric_map
template <typename T>
void UniqueValueMap<T>::SetColor() {
  colors_.resize(num_geometries_);
  const auto& unique_value_infos_numeric_map =
      vega_unique_value_map_.unique_value_infos_numeric_map();

  if (unique_value_infos_numeric_map.empty()) {
    std::string err_msg = "Unique value infos map is empty";
    throw std::runtime_error(err_msg);
  }

  for (auto i = 0; i < num_geometries_; i++) {
    auto color = unique_value_infos_numeric_map.at(values_[i]);
    colors_[i] = color;
  }
}

// SetColor for unique_value_infos_string_map
template <>
void UniqueValueMap<std::string>::SetColor() {
  colors_.resize(num_geometries_);
  const auto& unique_value_infos_string_map =
      vega_unique_value_map_.unique_value_infos_string_map();

  if (unique_value_infos_string_map.empty()) {
    std::string err_msg = "Unique value infos map is empty";
    throw std::runtime_error(err_msg);
  }

  for (auto i = 0; i < num_geometries_; i++) {
    auto color = unique_value_infos_string_map.at(values_[i]);
    colors_[i] = color;
  }
}

template <typename T>
std::vector<uint8_t> UniqueValueMap<T>::Render() {
  WindowsInit(vega_unique_value_map_.window_params());
  SetColor();
  Draw();
  Finalize();
  return Output();
}

}  // namespace render
}  // namespace arctern
