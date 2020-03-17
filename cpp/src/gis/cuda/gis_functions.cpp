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

#include "gis_functions.h"

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
#include "gis/cuda/functor/st_distance.h"
#include "gis/cuda/functor/st_envelope.h"
#include "gis/cuda/functor/st_length.h"
#include "gis/cuda/functor/st_point.h"
#include "gis/cuda/functor/st_within.h"
#include "gis/gdal/format_conversion.h"
#include "utils/check_status.h"

namespace arctern {
namespace gis {
namespace cuda {

std::shared_ptr<arrow::Array> ST_Point(const std::shared_ptr<arrow::Array>& x_values,
                                       const std::shared_ptr<arrow::Array>& y_values) {
  assert(x_values->length() == y_values->length());
  auto len = x_values->length();
  auto xs = std::static_pointer_cast<arrow::DoubleArray>(x_values);
  auto ys = std::static_pointer_cast<arrow::DoubleArray>(y_values);

  GeometryVector geo_vector;
  ST_Point(xs->raw_values(), ys->raw_values(), len, geo_vector);
  auto wkb_points = GeometryVectorToArrowWkb(geo_vector);

  return gdal::WkbToWkt(wkb_points);
}

std::shared_ptr<arrow::Array> ST_Envelope(const std::shared_ptr<arrow::Array>& wkt) {
  auto input_wkb = gdal::WktToWkb(wkt);
  auto input_geo_vec = ArrowWkbToGeometryVector(input_wkb);

  GeometryVector geo_vec_envelopes;
  ST_Envelope(input_geo_vec, geo_vec_envelopes);
  auto wkb_envelopes = GeometryVectorToArrowWkb(geo_vec_envelopes);

  return gdal::WkbToWkt(wkb_envelopes);
}

std::shared_ptr<arrow::Array> ST_Area(const std::shared_ptr<arrow::Array>& wkt) {
  auto len = wkt->length();
  auto input_wkb = gdal::WktToWkb(wkt);
  auto input_geo_vec = ArrowWkbToGeometryVector(input_wkb);

  auto raw_area = std::make_unique<double[]>(len);
  ST_Area(input_geo_vec, raw_area.get());

  arrow::DoubleBuilder builder;
  CHECK_ARROW(builder.AppendValues(raw_area.get(), len));
  std::shared_ptr<arrow::Array> area;
  CHECK_ARROW(builder.Finish(&area));

  return area;
}

std::shared_ptr<arrow::Array> ST_Length(const std::shared_ptr<arrow::Array>& wkt) {
  auto len = wkt->length();
  auto input_wkb = gdal::WktToWkb(wkt);
  auto input_geo_vector = ArrowWkbToGeometryVector(input_wkb);

  auto raw_length = std::make_unique<double[]>(len);
  ST_Length(input_geo_vector, raw_length.get());

  arrow::DoubleBuilder builder;
  CHECK_ARROW(builder.AppendValues(raw_length.get(), len));
  std::shared_ptr<arrow::Array> length;
  CHECK_ARROW(builder.Finish(&length));

  return length;
}

std::shared_ptr<arrow::Array> ST_Distance(const std::shared_ptr<arrow::Array>& lhs_geo,
                                          const std::shared_ptr<arrow::Array>& rhs_geo) {
  auto len = lhs_geo->length();
  auto lhs_wkb = gdal::WktToWkb(lhs_geo);
  auto rhs_wkb = gdal::WktToWkb(rhs_geo);
  auto lhs_geo_vec = ArrowWkbToGeometryVector(lhs_wkb);
  auto rhs_geo_vec = ArrowWkbToGeometryVector(rhs_wkb);
  auto raw_distance = std::make_unique<double[]>(len);
  ST_Distance(lhs_geo_vec, rhs_geo_vec, raw_distance.get());

  arrow::DoubleBuilder builder;
  CHECK_ARROW(builder.AppendValues(raw_distance.get(), len));
  std::shared_ptr<arrow::Array> distance;
  CHECK_ARROW(builder.Finish(&distance));

  return distance;
}

std::shared_ptr<arrow::Array> ST_Within(const std::shared_ptr<arrow::Array>& lhs_geo,
                                        const std::shared_ptr<arrow::Array>& rhs_geo) {
  auto len = lhs_geo->length();
  auto lhs_wkb = gdal::WktToWkb(lhs_geo);
  auto rhs_wkb = gdal::WktToWkb(rhs_geo);
  auto lhs_geo_vec = ArrowWkbToGeometryVector(lhs_wkb);
  auto rhs_geo_vec = ArrowWkbToGeometryVector(rhs_wkb);

  auto raw_within = std::make_unique<bool[]>(len);
  ST_Within(lhs_geo_vec, rhs_geo_vec, raw_within.get());

  arrow::BooleanBuilder builder;
  CHECK_ARROW(builder.AppendValues((uint8_t*)raw_within.get(), len));
  std::shared_ptr<arrow::Array> within;
  CHECK_ARROW(builder.Finish(&within));

  return within;
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
