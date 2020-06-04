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
#include <utility>
#include <vector>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <gdal_utils.h>
#include <ogrsf_frmts.h>

#include "render/2d/choropleth_map/choropleth_map.h"
#include "render/utils/color/color_gradient.h"

namespace arctern {
namespace render {

template class ChoroplethMap<int8_t>;

template class ChoroplethMap<int16_t>;

template class ChoroplethMap<int32_t>;

template class ChoroplethMap<int64_t>;

template class ChoroplethMap<uint8_t>;

template class ChoroplethMap<uint16_t>;

template class ChoroplethMap<uint32_t>;

template class ChoroplethMap<uint64_t>;

template class ChoroplethMap<float>;

template class ChoroplethMap<double>;

template <typename T>
ChoroplethMap<T>::ChoroplethMap(std::vector<OGRGeometry*> choropleth_wkb, T* count,
                                int64_t num_buildings)
    : choropleth_wkb_(std::move(choropleth_wkb)),
      count_(count),
      num_buildings_(num_buildings) {}

template <typename T>
void ChoroplethMap<T>::Draw() {
#ifndef USE_GPU
  glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1,
          1);
#endif

  glClear(GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ZERO);

  for (int i = 0; i < num_buildings_; i++) {
    glColor4f(colors_[i * 4], colors_[i * 4 + 1], colors_[i * 4 + 2], colors_[i * 4 + 3]);
    glBegin(GL_POLYGON);
    for (int j = 0; j < buildings_x_[i].size(); j++) {
      glVertex2d(buildings_x_[i][j], buildings_y_[i][j]);
    }
    glEnd();
  }

  glFinish();
}

template <typename T>
void ChoroplethMap<T>::Transform() {
  buildings_x_.resize(num_buildings_);
  buildings_y_.resize(num_buildings_);

  for (int i = 0; i < num_buildings_; i++) {
    OGRGeometry* geometry = choropleth_wkb_[i];
    auto type = geometry->getGeometryType();
    if (type == OGRwkbGeometryType::wkbPolygon) {
      auto ring = geometry->toPolygon()->getExteriorRing();
      auto ring_size = ring->getNumPoints();
      buildings_x_[i].resize(ring_size);
      buildings_y_[i].resize(ring_size);
      for (int j = 0; j < ring_size; j++) {
        buildings_x_[i][j] = ring->getX(j);
        buildings_y_[i][j] = ring->getY(j);
      }
    } else {
      std::string err_msg = "Unknown geometry type";
      throw std::runtime_error(err_msg);
    }
    OGRGeometryFactory::destroyGeometry(geometry);
  }
}

template <typename T>
void ChoroplethMap<T>::SetColor() {
  colors_.resize(num_buildings_ * 4);

  auto count_start = choropleth_vega_.color_bound().first;
  auto count_end = choropleth_vega_.color_bound().second;
  auto count_range = count_end - count_start;

  size_t c_offset = 0;
  for (auto i = 0; i < num_buildings_; i++) {
    auto color_gradient = choropleth_vega_.color_gradient();
    if (color_gradient.size() == 1) {
      auto color = color_gradient[0];
      colors_[c_offset++] = color.r;
      colors_[c_offset++] = color.g;
      colors_[c_offset++] = color.b;
      colors_[c_offset++] = color.a;
    } else {
      auto count = count_[i] >= count_start ? count_[i] : count_start;
      count = count_[i] <= count_end ? count : count_end;
      auto ratio = (count - count_start) / count_range;
      auto color_start = color_gradient[0];
      auto color_end = color_gradient[1];
      auto color = ColorGradient::GetColor(color_start, color_end, ratio);
      colors_[c_offset++] = color.r;
      colors_[c_offset++] = color.g;
      colors_[c_offset++] = color.b;
      colors_[c_offset++] = color.a;
    }
  }
}

template <typename T>
std::vector<uint8_t> ChoroplethMap<T>::Render() {
  WindowsInit(choropleth_vega_.window_params());
  SetColor();
  Transform();
  Draw();
  Finalize();
  return Output();
}

}  // namespace render
}  // namespace arctern
