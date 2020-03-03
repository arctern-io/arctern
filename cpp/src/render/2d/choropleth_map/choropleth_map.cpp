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

namespace zilliz {
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
ChoroplethMap<T>::ChoroplethMap()
    : choropleth_wkt_(), count_(nullptr), num_buildings_(0) {}

template <typename T>
ChoroplethMap<T>::ChoroplethMap(std::vector<std::string> choropleth_wkt, T* count,
                                int64_t num_buildings)
    : choropleth_wkt_(std::move(choropleth_wkt)),
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

  auto bounding_box = choropleth_vega_.bounding_box();

  auto x_left = bounding_box.longitude_left * 111319.490778;
  auto x_right = bounding_box.longitude_right * 111319.490778;

  auto y_left =
      6378136.99911 * log(tan(.00872664626 * bounding_box.latitude_left + .785398163397));
  auto y_right = 6378136.99911 *
                 log(tan(.00872664626 * bounding_box.latitude_right + .785398163397));

  auto width = window_->window_params().width();
  auto height = window_->window_params().height();

  for (int i = 0; i < num_buildings_; i++) {
    OGRGeometry* geometry;
    OGRGeometryFactory::createFromWkt(choropleth_wkt_[i].c_str(), nullptr, &geometry);

    auto type = geometry->getGeometryType();

    if (type == OGRwkbGeometryType::wkbPolygon) {
      auto ring = geometry->toPolygon()->getExteriorRing();

      auto ring_size = ring->getNumPoints();
      buildings_x_[i].resize(ring_size);
      buildings_y_[i].resize(ring_size);
      for (int j = 0; j < ring_size; j++) {
        double x_pos = ring->getX(j) * 111319.490778;
        int ret_x = (int)(((x_pos - x_left) / (x_right - x_left)) * width - 1E-9);
        buildings_x_[i][j] = ret_x;

        double y_pos =
            6378136.99911 * log(tan(.00872664626 * ring->getY(j) + .785398163397));
        int ret_y = (int)(((y_pos - y_left) / (y_right - y_left)) * height - 1E-9);
        buildings_y_[i][j] = ret_y;
      }

    } else {
      // TODO: add log here
      std::cout << "Unknown geometry type." << std::endl;
    }

    OGRGeometryFactory::destroyGeometry(geometry);
  }
}

template <typename T>
void ChoroplethMap<T>::SetColor() {
  colors_.resize(num_buildings_ * 4);

  auto count_start = choropleth_vega_.ruler().first;
  auto count_end = choropleth_vega_.ruler().second;
  auto count_range = count_end - count_start;

  size_t c_offset = 0;
  for (auto i = 0; i < num_buildings_; i++) {
    auto count = count_[i] >= count_start ? count_[i] : count_start;
    count = count_[i] <= count_end ? count : count_end;
    auto ratio = (count - count_start) / count_range;
    auto circle_params_2d =
        ColorGradient::GetCircleParams(choropleth_vega_.color_style(), ratio);
    colors_[c_offset++] = circle_params_2d.color.r;
    colors_[c_offset++] = circle_params_2d.color.g;
    colors_[c_offset++] = circle_params_2d.color.b;
    colors_[c_offset++] = circle_params_2d.color.a;
  }
}

template <typename T>
uint8_t* ChoroplethMap<T>::Render() {
  WindowsInit(choropleth_vega_.window_params());
  SetColor();
  Transform();
  Draw();
  Finalize();
  return Output();
}

}  // namespace render
}  // namespace zilliz
