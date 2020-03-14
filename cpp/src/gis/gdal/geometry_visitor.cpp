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

#include "gis/gdal/geometry_visitor.h"

namespace arctern {
namespace gis {
namespace gdal {

void HasCircularVisitor::visit(const OGRCircularString* geo) {
  if (geo->IsEmpty()) {
    return;
  }
  has_circular_ = true;
}

void NPointsVisitor::visit(const OGRPoint* geo) {
  if (geo->IsEmpty()) return;
  npoints_++;
}

double PrecisionReduceVisitor::coordinate_precision_reduce(double coordinate) {
  int32_t sign_flag = 0;

  if (coordinate < 0) {
    sign_flag = 1;
    coordinate = -coordinate;
  }

  std::string coordinate_string = std::to_string(coordinate);

  if (int64_t(coordinate_string.find(".")) != -1) {
    if (coordinate_string.length() <= (precision_ + 1)) {
    } else {
      if (coordinate_string.find(".") > precision_) {
        double carry_value = 1;
        for (int32_t i = 0; i < (coordinate_string.find(".") - precision_); i++) {
          carry_value *= 10;
        }
        coordinate /= carry_value;
        if (int32_t(coordinate_string[precision_] - 48) < 5) {
          coordinate = int64_t(coordinate) * carry_value;
        } else {
          coordinate = int64_t(coordinate + 1) * carry_value;
        }
      } else {
        double carry_value = 1;
        for (int32_t i = 0; i < (precision_ - coordinate_string.find(".")); i++) {
          carry_value *= 10;
        }
        coordinate *= carry_value;
        if (int32_t(coordinate_string[precision_ + 1] - 48) < 5) {
          coordinate = int64_t(coordinate) / carry_value;
        } else {
          coordinate = int64_t(coordinate + 1) / carry_value;
        }
      }
    }
  } else {
    if (coordinate_string.length() < precision_) {
    } else {
      int32_t carry_value = 1;
      for (int32_t i = 0; i < (coordinate_string.length() - precision_); i++) {
        carry_value *= 10;
      }
      coordinate /= carry_value;
      if (coordinate_string[precision_] < 5) {
        coordinate = int64_t(coordinate) * carry_value;
      } else {
        coordinate = int64_t(coordinate + 1) * carry_value;
      }
    }
  }

  if (sign_flag == 1) {
    coordinate = -coordinate;
  }

  return coordinate;
}

void PrecisionReduceVisitor::visit(OGRPoint* geo) {
  double coordinate_x = geo->getX();
  double coordinate_y = geo->getY();
  geo->setX(coordinate_precision_reduce(coordinate_x));
  geo->setY(coordinate_precision_reduce(coordinate_y));
}

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
