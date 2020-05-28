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
#include <vector>

#include "arrow/api.h"
#include "arrow/array.h"

namespace arctern {
namespace gis {
namespace gdal {

/**************************** GEOMETRY CONSTRUCTOR ***************************/

std::vector<std::shared_ptr<arrow::Array>> ST_Point(
    const std::vector<std::shared_ptr<arrow::Array>>& x_values_raw,
    const std::vector<std::shared_ptr<arrow::Array>>& y_values_raw);

std::vector<std::shared_ptr<arrow::Array>> ST_PolygonFromEnvelope(
    const std::vector<std::shared_ptr<arrow::Array>>& min_x_values,
    const std::vector<std::shared_ptr<arrow::Array>>& min_y_values,
    const std::vector<std::shared_ptr<arrow::Array>>& max_x_values,
    const std::vector<std::shared_ptr<arrow::Array>>& max_y_values);

std::vector<std::shared_ptr<arrow::Array>> ST_GeomFromGeoJSON(
    const std::shared_ptr<arrow::Array>& json);

std::vector<std::shared_ptr<arrow::Array>> ST_GeomFromText(
    const std::shared_ptr<arrow::Array>& text);

std::vector<std::shared_ptr<arrow::Array>> ST_AsText(
    const std::shared_ptr<arrow::Array>& wkb);

std::vector<std::shared_ptr<arrow::Array>> ST_AsGeoJSON(
    const std::shared_ptr<arrow::Array>& wkb);

/***************************** GEOMETRY ACCESSOR *****************************/

std::shared_ptr<arrow::Array> ST_IsValid(const std::shared_ptr<arrow::Array>& geometries);

std::shared_ptr<arrow::Array> ST_IsSimple(
    const std::shared_ptr<arrow::Array>& geometries);

std::shared_ptr<arrow::Array> ST_GeometryType(
    const std::shared_ptr<arrow::Array>& geometries);

std::shared_ptr<arrow::Array> ST_NPoints(const std::shared_ptr<arrow::Array>& geometries);

std::shared_ptr<arrow::Array> ST_Envelope(
    const std::shared_ptr<arrow::Array>& geometries);

/**************************** GEOMETRY PROCESSING ****************************/

std::vector<std::shared_ptr<arrow::Array>> ST_Buffer(
    const std::shared_ptr<arrow::Array>& geometries, double buffer_distance,
    int n_quadrant_segments = 30);

std::shared_ptr<arrow::Array> ST_PrecisionReduce(
    const std::shared_ptr<arrow::Array>& geometries, int32_t precision);

std::vector<std::shared_ptr<arrow::Array>> ST_Intersection(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_2);

std::shared_ptr<arrow::Array> ST_MakeValid(
    const std::shared_ptr<arrow::Array>& geometries);

std::shared_ptr<arrow::Array> ST_SimplifyPreserveTopology(
    const std::shared_ptr<arrow::Array>& geometries, double distance_tolerance);

std::shared_ptr<arrow::Array> ST_Centroid(
    const std::shared_ptr<arrow::Array>& geometries);

std::shared_ptr<arrow::Array> ST_ConvexHull(
    const std::shared_ptr<arrow::Array>& geometries);

std::shared_ptr<arrow::Array> ST_Transform(const std::shared_ptr<arrow::Array>& geos,
                                           const std::string& src_rs,
                                           const std::string& dst_rs);

std::vector<std::shared_ptr<arrow::Array>> ST_CurveToLine(
    const std::shared_ptr<arrow::Array>& geometries);

/*************************** MEASUREMENT FUNCTIONS ***************************/

std::vector<std::shared_ptr<arrow::Array>> ST_DistanceSphere(
    const std::vector<std::shared_ptr<arrow::Array>>& point_left,
    const std::vector<std::shared_ptr<arrow::Array>>& point_right);

std::vector<std::shared_ptr<arrow::Array>> ST_Distance(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_2);

std::shared_ptr<arrow::Array> ST_Area(const std::shared_ptr<arrow::Array>& geometries);

std::shared_ptr<arrow::Array> ST_Length(const std::shared_ptr<arrow::Array>& geometries);

std::vector<std::shared_ptr<arrow::Array>> ST_HausdorffDistance(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2);

/**************************** SPATIAL RELATIONSHIP ***************************/

std::vector<std::shared_ptr<arrow::Array>> ST_Equals(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_2);

std::vector<std::shared_ptr<arrow::Array>> ST_Touches(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_2);

std::vector<std::shared_ptr<arrow::Array>> ST_Overlaps(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_2);

std::vector<std::shared_ptr<arrow::Array>> ST_Crosses(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_2);

std::vector<std::shared_ptr<arrow::Array>> ST_Contains(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_2);

std::vector<std::shared_ptr<arrow::Array>> ST_Intersects(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_2);

std::vector<std::shared_ptr<arrow::Array>> ST_Within(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_2);

std::vector<std::shared_ptr<arrow::Array>> ST_WithinOpt(
    const std::vector<std::shared_ptr<arrow::Array>>& geometries_1,
    const std::string& geometry_2);

/*************************** AGGREGATE FUNCTIONS ***************************/

std::shared_ptr<arrow::Array> ST_Union_Aggr(
    const std::shared_ptr<arrow::Array>& geometries);

std::shared_ptr<arrow::Array> ST_Envelope_Aggr(
    const std::shared_ptr<arrow::Array>& geometries);

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
