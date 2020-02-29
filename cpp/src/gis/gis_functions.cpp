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

#include "gis/gdal/gis_functions.h"
#include "common/version.h"
#include "gis/api.h"
#include "utils/check_status.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace zilliz {
namespace gis {

/**************************** GEOMETRY CONSTRUCTOR ***************************/

std::shared_ptr<arrow::Array> ST_Point(const std::shared_ptr<arrow::Array>& x_values,
                                       const std::shared_ptr<arrow::Array>& y_values) {
#ifdef ENABLE_GPU
  // todo: add cuda pass
#else
  return gdal::ST_Point(x_values, y_values);
#endif
}

std::shared_ptr<arrow::Array> ST_PolygonFromEnvelope(
    const std::shared_ptr<arrow::Array>& min_x_values,
    const std::shared_ptr<arrow::Array>& min_y_values,
    const std::shared_ptr<arrow::Array>& max_x_values,
    const std::shared_ptr<arrow::Array>& max_y_values) {
  return gdal::ST_PolygonFromEnvelope(min_x_values, min_y_values, max_x_values,
                                      max_y_values);
}

std::shared_ptr<arrow::Array> ST_GeomFromGeoJSON(
    const std::shared_ptr<arrow::Array>& json){
  return gdal::ST_GeomFromGeoJSON(json);
}

/***************************** GEOMETRY ACCESSOR *****************************/

std::shared_ptr<arrow::Array> ST_IsValid(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_IsValid(geometries);
}

std::shared_ptr<arrow::Array> ST_IsSimple(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_IsSimple(geometries);
}

std::shared_ptr<arrow::Array> ST_GeometryType(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_GeometryType(geometries);
}

std::shared_ptr<arrow::Array> ST_NPoints(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_NPoints(geometries);
}

std::shared_ptr<arrow::Array> ST_Envelope(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_Envelope(geometries);
}

/**************************** GEOMETRY PROCESSING ****************************/

std::shared_ptr<arrow::Array> ST_Buffer(const std::shared_ptr<arrow::Array>& geometries,
                                        double buffer_distance, int n_quadrant_segments) {
  return gdal::ST_Buffer(geometries, buffer_distance, n_quadrant_segments);
}

// std::shared_ptr<arrow::Array>
// ST_PrecisionReduce(const std::shared_ptr<arrow::Array> &geometries,
//                    int32_t precision);

std::shared_ptr<arrow::Array> ST_Intersection(
    const std::shared_ptr<arrow::Array>& geometries_1,
    const std::shared_ptr<arrow::Array>& geometries_2) {
  return gdal::ST_Intersection(geometries_1, geometries_2);
}

std::shared_ptr<arrow::Array> ST_MakeValid(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_MakeValid(geometries);
}

std::shared_ptr<arrow::Array> ST_SimplifyPreserveTopology(
    const std::shared_ptr<arrow::Array>& geometries, double distance_tolerance) {
  return gdal::ST_SimplifyPreserveTopology(geometries, distance_tolerance);
}

std::shared_ptr<arrow::Array> ST_Centroid(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_Centroid(geometries);
}

std::shared_ptr<arrow::Array> ST_ConvexHull(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_ConvexHull(geometries);
}

std::shared_ptr<arrow::Array> ST_Transform(
    const std::shared_ptr<arrow::Array>& geometries, const std::string& src_rs,
    const std::string& dst_rs) {
  return gdal::ST_Transform(geometries, src_rs, dst_rs);
}

/*************************** MEASUREMENT FUNCTIONS ***************************/

std::shared_ptr<arrow::Array> ST_Distance(
    const std::shared_ptr<arrow::Array>& geometries_1,
    const std::shared_ptr<arrow::Array>& geometries_2) {
  return gdal::ST_Distance(geometries_1, geometries_2);
}

std::shared_ptr<arrow::Array> ST_Area(const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_Area(geometries);
}

std::shared_ptr<arrow::Array> ST_Length(const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_Length(geometries);
}

/**************************** SPATIAL RELATIONSHIP ***************************/

std::shared_ptr<arrow::Array> ST_Equals(
    const std::shared_ptr<arrow::Array>& geometries_1,
    const std::shared_ptr<arrow::Array>& geometries_2) {
  return gdal::ST_Equals(geometries_1, geometries_2);
}

std::shared_ptr<arrow::Array> ST_Touches(
    const std::shared_ptr<arrow::Array>& geometries_1,
    const std::shared_ptr<arrow::Array>& geometries_2) {
  return gdal::ST_Touches(geometries_1, geometries_2);
}

std::shared_ptr<arrow::Array> ST_Overlaps(
    const std::shared_ptr<arrow::Array>& geometries_1,
    const std::shared_ptr<arrow::Array>& geometries_2) {
  return gdal::ST_Overlaps(geometries_1, geometries_2);
}

std::shared_ptr<arrow::Array> ST_Crosses(
    const std::shared_ptr<arrow::Array>& geometries_1,
    const std::shared_ptr<arrow::Array>& geometries_2) {
  return gdal::ST_Crosses(geometries_1, geometries_2);
}

std::shared_ptr<arrow::Array> ST_Contains(
    const std::shared_ptr<arrow::Array>& geometries_1,
    const std::shared_ptr<arrow::Array>& geometries_2) {
  return gdal::ST_Contains(geometries_1, geometries_2);
}

std::shared_ptr<arrow::Array> ST_Intersects(
    const std::shared_ptr<arrow::Array>& geometries_1,
    const std::shared_ptr<arrow::Array>& geometries_2) {
  return gdal::ST_Intersects(geometries_1, geometries_2);
}

std::shared_ptr<arrow::Array> ST_Within(
    const std::shared_ptr<arrow::Array>& geometries_1,
    const std::shared_ptr<arrow::Array>& geometries_2) {
  return gdal::ST_Within(geometries_1, geometries_2);
}

/*************************** AGGREGATE FUNCTIONS ***************************/

std::shared_ptr<arrow::Array> ST_Union_Aggr(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_Union_Aggr(geometries);
}

std::shared_ptr<arrow::Array> ST_Envelope_Aggr(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_Envelope_Aggr(geometries);
}

/*************************** AGGREGATE FUNCTIONS ***************************/

std::shared_ptr<std::string> GIS_Version() {
  const std::string info = "gis version : " + std::string(LIB_VERSION) + "\n" +
                           "build tyoe : " + CMAKE_BUILD_TYPE + "/" + CPU_OR_GPU + "\n" +
                           "build time : " + BUILD_TIME + "\n" +
                           "commit id : " + LAST_COMMIT_ID + "\n";
  return std::make_shared<std::string>(info);
}

}  // namespace gis
}  // namespace zilliz
