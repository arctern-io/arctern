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

#include <stdint.h>

#include <memory>
#include <string>

#include "arrow/api.h"
#include "arrow/array.h"
#include "utils/arrow_alias.h"

namespace arctern {
namespace gis {
namespace cuda {

/**************************** GEOMETRY CONSTRUCTOR ***************************/

WkbArrayPtr ST_Point(const DoubleArrayPtr& x_values, const DoubleArrayPtr& y_values);

// std::shared_ptr<arrow::Array> ST_PolygonFromEnvelope(
//    const std::shared_ptr<arrow::Array>& min_x_values,
//    const std::shared_ptr<arrow::Array>& min_y_values,
//    const std::shared_ptr<arrow::Array>& max_x_values,
//    const std::shared_ptr<arrow::Array>& max_y_values);
//
// std::shared_ptr<arrow::Array> ST_GeomFromGeoJSON(
//    const std::shared_ptr<arrow::Array>& json);

/***************************** GEOMETRY ACCESSOR *****************************/

// std::shared_ptr<arrow::Array> ST_IsValid(const std::shared_ptr<arrow::Array>&
// geometries);
//
// std::shared_ptr<arrow::Array> ST_IsSimple(
//    const std::shared_ptr<arrow::Array>& geometries);
//
// std::shared_ptr<arrow::Array> ST_GeometryType(
//    const std::shared_ptr<arrow::Array>& geometries);
//
// std::shared_ptr<arrow::Array> ST_NPoints(const std::shared_ptr<arrow::Array>&
// geometries);

WkbArrayPtr ST_Envelope(const WkbArrayPtr& input_geo);

/**************************** GEOMETRY PROCESSING ****************************/

// std::shared_ptr<arrow::Array> ST_Buffer(const std::shared_ptr<arrow::Array>&
// geometries,
//                                        double buffer_distance,
//                                        int n_quadrant_segments = 30);

// std::shared_ptr<arrow::Array>
// ST_PrecisionReduce(const std::shared_ptr<arrow::Array> &geometries,
//                    int32_t precision);

// std::shared_ptr<arrow::Array> ST_Intersection(
//    const std::shared_ptr<arrow::Array>& geometries_1,
//    const std::shared_ptr<arrow::Array>& geometries_2);
//
// std::shared_ptr<arrow::Array> ST_MakeValid(
//    const std::shared_ptr<arrow::Array>& geometries);
//
// std::shared_ptr<arrow::Array> ST_SimplifyPreserveTopology(
//    const std::shared_ptr<arrow::Array>& geometries, double distance_tolerance);
//
// std::shared_ptr<arrow::Array> ST_Centroid(
//    const std::shared_ptr<arrow::Array>& geometries);
//
// std::shared_ptr<arrow::Array> ST_ConvexHull(
//    const std::shared_ptr<arrow::Array>& geometries);
//
// std::shared_ptr<arrow::Array> ST_Transform(const std::shared_ptr<arrow::Array>& geos,
//                                           const std::string& src_rs,
//                                           const std::string& dst_rs);

/*************************** MEASUREMENT FUNCTIONS ***************************/

DoubleArrayPtr ST_Distance(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo);

DoubleArrayPtr ST_Area(const WkbArrayPtr& input_geo);

DoubleArrayPtr ST_Length(const WkbArrayPtr& input_geo);

// std::shared_ptr<arrow::Array> ST_HausdorffDistance(
//    const std::shared_ptr<arrow::Array>& geo1, const std::shared_ptr<arrow::Array>&
//    geo2);

/**************************** SPATIAL RELATIONSHIP ***************************/

BooleanArrayPtr ST_Equals(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo);

BooleanArrayPtr ST_Touches(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo);

BooleanArrayPtr ST_Overlaps(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo);

BooleanArrayPtr ST_Crosses(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo);

BooleanArrayPtr ST_Contains(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo);

BooleanArrayPtr ST_Intersects(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo);

BooleanArrayPtr ST_Within(const WkbArrayPtr& lhs_geo, const WkbArrayPtr& rhs_geo);

/*************************** AGGREGATE FUNCTIONS ***************************/

// std::shared_ptr<arrow::Array> ST_Union_Aggr(
//    const std::shared_ptr<arrow::Array>& geometries);
//
// std::shared_ptr<arrow::Array> ST_Envelope_Aggr(
//    const std::shared_ptr<arrow::Array>& geometries);

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
