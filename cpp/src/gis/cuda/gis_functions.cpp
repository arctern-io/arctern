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

#include "gis/cuda/gis_functions.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "common/version.h"
#include "gis/cuda/conversion/conversions.h"
#include "gis/cuda/functor/st_area.h"
#include "gis/cuda/functor/st_crosses.h"
#include "gis/cuda/functor/st_distance.h"
#include "gis/cuda/functor/st_envelope.h"
#include "gis/cuda/functor/st_equals.h"
#include "gis/cuda/functor/st_intersects.h"
#include "gis/cuda/functor/st_length.h"
#include "gis/cuda/functor/st_overlaps.h"
#include "gis/cuda/functor/st_point.h"
#include "gis/cuda/functor/st_touches.h"
#include "gis/cuda/functor/st_within.h"
#include "gis/gdal/format_conversion.h"
#include "utils/check_status.h"

namespace arctern {
namespace gis {
namespace cuda {

WkbArrayPtr ST_Point(const DoubleArrayPtr& x_values, const DoubleArrayPtr& y_values) {
  assert(x_values->length() == y_values->length());
  auto len = x_values->length();
  auto xs = std::static_pointer_cast<arrow::DoubleArray>(x_values);
  auto ys = std::static_pointer_cast<arrow::DoubleArray>(y_values);

  GeometryVector geo_vector;
  ST_Point(xs->raw_values(), ys->raw_values(), len, geo_vector);
  auto wkb_points = GeometryVectorToArrowWkb(geo_vector);

  return wkb_points;
}

WkbArrayPtr ST_Envelope(const WkbArrayPtr& input_geo) {
  auto input_geo_vec = ArrowWkbToGeometryVector(input_geo);

  GeometryVector geo_vec_envelopes;
  ST_Envelope(input_geo_vec, geo_vec_envelopes);
  auto wkb_envelopes = GeometryVectorToArrowWkb(geo_vec_envelopes);

  return wkb_envelopes;
}

DoubleArrayPtr ST_Area(const WkbArrayPtr& input_geo) {
  auto len = input_geo->length();
  auto input_geo_vec = ArrowWkbToGeometryVector(input_geo);

  auto raw_area = std::make_unique<double[]>(len);
  ST_Area(input_geo_vec, raw_area.get());

  arrow::DoubleBuilder builder;
  CHECK_ARROW(builder.AppendValues(raw_area.get(), len));
  DoubleArrayPtr area;
  CHECK_ARROW(builder.Finish(&area));

  return area;
}

DoubleArrayPtr ST_Length(const WkbArrayPtr& input_geo) {
  auto len = input_geo->length();
  auto input_geo_vector = ArrowWkbToGeometryVector(input_geo);

  auto raw_length = std::make_unique<double[]>(len);
  ST_Length(input_geo_vector, raw_length.get());

  arrow::DoubleBuilder builder;
  CHECK_ARROW(builder.AppendValues(raw_length.get(), len));
  DoubleArrayPtr length;
  CHECK_ARROW(builder.Finish(&length));

  return length;
}

DoubleArrayPtr ST_Distance(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo) {
  auto len = lhs_geo->length();
  auto lhs_geo_vec = ArrowWkbToGeometryVector(lhs_geo);
  auto rhs_geo_vec = ArrowWkbToGeometryVector(rhs_geo);
  auto raw_distance = std::make_unique<double[]>(len);
  ST_Distance(lhs_geo_vec, rhs_geo_vec, raw_distance.get());

  arrow::DoubleBuilder builder;
  CHECK_ARROW(builder.AppendValues(raw_distance.get(), len));
  DoubleArrayPtr distance;
  CHECK_ARROW(builder.Finish(&distance));

  return distance;
}

using RelateFunc = void (*)(const GeometryVector&, const GeometryVector&, bool*);
static BooleanArrayPtr RelateTemplate(RelateFunc func, const WkbArrayPtr& lhs_geo,
                                      const WkbArrayPtr& rhs_geo) {
  auto len = lhs_geo->length();
  auto lhs_geo_vec = ArrowWkbToGeometryVector(lhs_geo);
  auto rhs_geo_vec = ArrowWkbToGeometryVector(rhs_geo);
  auto raw_info = std::make_unique<bool[]>(len);
  func(lhs_geo_vec, rhs_geo_vec, raw_info.get());
  arrow::BooleanBuilder builder;
  CHECK_ARROW(builder.AppendValues((uint8_t*)raw_info.get(), len));
  BooleanArrayPtr within;
  CHECK_ARROW(builder.Finish(&within));
  return within;
}

BooleanArrayPtr ST_Equals(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo) {
  return RelateTemplate(ST_Equals, lhs_geo, rhs_geo);
}

BooleanArrayPtr ST_Touches(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo) {
  return RelateTemplate(ST_Touches, lhs_geo, rhs_geo);
}

BooleanArrayPtr ST_Overlaps(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo) {
  return RelateTemplate(ST_Overlaps, lhs_geo, rhs_geo);
}

BooleanArrayPtr ST_Crosses(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo) {
  return RelateTemplate(ST_Crosses, lhs_geo, rhs_geo);
}

BooleanArrayPtr ST_Contains(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo) {
  // equivalent to ST_Within
  return RelateTemplate(ST_Within, rhs_geo, lhs_geo);
}

BooleanArrayPtr ST_Intersects(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo) {
  return RelateTemplate(ST_Intersects, lhs_geo, rhs_geo);
}

BooleanArrayPtr ST_Within(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo) {
  return RelateTemplate(ST_Within, lhs_geo, rhs_geo);
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
