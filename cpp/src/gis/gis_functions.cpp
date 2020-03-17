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

#include "gis/gdal/type_scan.h"
#include "gis/wkb_types.h"
#ifdef USE_GPU
#include "gis/cuda/gis_functions.h"
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/version.h"
#include "gis/api.h"
#include "gis/gdal/gis_functions.h"
#include "utils/check_status.h"

namespace arctern {
namespace gis {

/**************************** GEOMETRY CONSTRUCTOR ***************************/

std::shared_ptr<arrow::Array> ST_Point(const std::shared_ptr<arrow::Array>& x_values,
                                       const std::shared_ptr<arrow::Array>& y_values) {
  // #if defined(USE_GPU)
  // return cuda::ST_Point(x_values, y_values);
  // #else
  return gdal::ST_Point(x_values, y_values);
  // #endif
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
    const std::shared_ptr<arrow::Array>& json) {
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
  // #if defined(USE_GPU)
  //   // currently support ST_Point, ST_LineString, ST_Polygon
  //   gdal::TypeScannerForWkt scanner(geometries);
  //   GroupedWkbTypes supported_types = {WkbTypes::kPoint, WkbTypes::kLineString,
  //                                      WkbTypes::kPolygon};
  //   scanner.mutable_types().push_back(supported_types);
  //   auto type_masks = scanner.Scan();
  //   if (type_masks->is_unique_group && (type_masks->unique_group == supported_types)) {
  //     return cuda::ST_Envelope(geometries);
  //   } else {
  //     return gdal::ST_Envelope(geometries);
  //   }
  // #else
  return gdal::ST_Envelope(geometries);
  // #endif
}

/**************************** GEOMETRY PROCESSING ****************************/

std::shared_ptr<arrow::Array> ST_Buffer(const std::shared_ptr<arrow::Array>& geometries,
                                        double buffer_distance, int n_quadrant_segments) {
  return gdal::ST_Buffer(geometries, buffer_distance, n_quadrant_segments);
}

std::shared_ptr<arrow::Array> ST_PrecisionReduce(
    const std::shared_ptr<arrow::Array>& geometries, int32_t precision) {
  return gdal::ST_PrecisionReduce(geometries, precision);
}

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

std::shared_ptr<arrow::Array> ST_CurveToLine(
    const std::shared_ptr<arrow::Array>& geometries) {
  return gdal::ST_CurveToLine(geometries);
}

/*************************** MEASUREMENT FUNCTIONS ***************************/

std::shared_ptr<arrow::Array> ST_Distance(
    const std::shared_ptr<arrow::Array>& geo_left,
    const std::shared_ptr<arrow::Array>& geo_right) {
#if defined(USE_GPU) && false
  // TODO: NOT ENABLED
  auto groupped_type = std::set<WkbTypes>({WkbTypes::kPoint});
  auto scan = [&groupped_type](const std::shared_ptr<arrow::Array>& geometries) {
    gdal::TypeScannerForWkt scanner(geometries);
    scanner.mutable_types().emplace_back(groupped_type);
    return scanner.Scan();
  };
  auto lhs_record = scan(geo_left);
  auto rhs_record = scan(geo_right);
  enum class Stategy { GDAL_ONLY = 0, MIXED = 1, CUDA_ONLY = 2 };
  auto get_stategy = [&groupped_type](const GeometryTypeMasks& record) {
    if (record.is_unique_type) {
      if (record.unique_type == groupped_type) {
        return Stategy::CUDA_ONLY;
      } else {
        return Stategy::GDAL_ONLY;
      }
    } else {
      return Stategy::MIXED;
    }
  };
  auto lhs_stategy = get_stategy(*lhs_record);
  auto rhs_stategy = get_stategy(*rhs_record);
  auto stategy = (Stategy)std::min((int)lhs_stategy, (int)rhs_stategy);
  if (stategy == Stategy::CUDA_ONLY) {
    return cuda::ST_Distance(geo_left, geo_right);
  } else if (stategy == Stategy::GDAL_ONLY) {
    return gdal::ST_Distance(geo_left, geo_right);
  }
  assert(stategy == Stategy::MIXED);
  // TODO: MIXED DISPATCH
  return gdal::ST_Distance(geo_left, geo_right);
  //   // currently support ST_Point
  //   bool lhs_ok = false;
  //   bool rhs_ok = false;
  //   //  {
  //   //    gdal::TypeScannerForWkt lhs_scanner(geometries_1);
  //   //    GroupedWkbTypes lhs_supported_types = {WkbTypes::kPoint};
  //   //    lhs_scanner.mutable_types().push_back(lhs_supported_types);
  //   //    auto lhs_type_masks = lhs_scanner.Scan();
  //   //    lhs_ok = lhs_type_masks->is_unique_group &&
  //   //             (lhs_type_masks->unique_group == lhs_supported_types);
  //   //  }
  //   //  {
  //   //    gdal::TypeScannerForWkt rhs_scanner(geometries_2);
  //   //    GroupedWkbTypes rhs_supported_types = {WkbTypes::kPoint};
  //   //    rhs_scanner.mutable_types().push_back(rhs_supported_types);
  //   //    auto rhs_type_masks = rhs_scanner.Scan();
  //   //    rhs_ok = rhs_type_masks->is_unique_group &&
  //   //             (rhs_type_masks->unique_group == rhs_supported_types);
  //   //  }

#else
  return gdal::ST_Distance(geo_left, geo_right);
#endif
}

std::shared_ptr<arrow::Array> ST_Area(const std::shared_ptr<arrow::Array>& geometries) {
  //   #if defined(USE_GPU)
  //     // currently support ST_Polygon
  //     gdal::TypeScannerForWkt scanner(geometries);
  //     GroupedWkbTypes supported_types = {WkbTypes::kPolygon};
  //     scanner.mutable_types().push_back(supported_types);
  //     auto type_masks = scanner.Scan();
  //     if (type_masks->is_unique_group && (type_masks->unique_group == supported_types))
  //     {
  //       return cuda::ST_Area(geometries);
  //     } else {
  //       return gdal::ST_Area(geometries);
  //     }
  //   #else
  return gdal::ST_Area(geometries);
  // #endif
}

std::shared_ptr<arrow::Array> ST_Length(const std::shared_ptr<arrow::Array>& geometries) {
#if defined(USE_GPU) && false
  // currently support ST_LineString
  gdal::TypeScannerForWkt scanner(geometries);
  GroupedWkbTypes supported_types = {WkbTypes::kLineString};
  scanner.mutable_types().push_back(supported_types);
  auto type_masks = scanner.Scan();
  if (type_masks->is_unique_type) {
    // UNIQUE METHOD
    if (type_masks->unique_type == supported_types) {
      return cuda::ST_Length(geometries);
    } else {
      return gdal::ST_Length(geometries);
    }
  }
  // MIXED METHOD
  auto mask = type_masks->get_masks(supported_types);
  auto split_inputs = gdal::WktArraySplit(geometries, mask);
  assert(split_inputs[1]->null_count() == 0);
  auto gdal_output = gdal::ST_Length(split_inputs[0]);
  auto cuda_output = cuda::ST_Length(split_inputs[1]);
  return gdal::DoubleArrayMerge({gdal_output, cuda_output}, mask);
#else
  return gdal::ST_Length(geometries);
#endif
}

std::shared_ptr<arrow::Array> ST_HausdorffDistance(
    const std::shared_ptr<arrow::Array>& geo1,
    const std::shared_ptr<arrow::Array>& geo2) {
  return gdal::ST_HausdorffDistance(geo1, geo2);
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
  // #if defined(USE_GPU)
  //   // currently support ST_Point within ST_Polygon
  //   gdal::TypeScannerForWkt lhs_scanner(geometries_1);
  //   GroupedWkbTypes lhs_supported_types = {WkbTypes::kPoint};
  //   lhs_scanner.mutable_types().push_back(lhs_supported_types);
  //   auto lhs_type_masks = lhs_scanner.Scan();

  //   gdal::TypeScannerForWkt rhs_scanner(geometries_2);
  //   GroupedWkbTypes rhs_supported_types = {WkbTypes::kPolygon};
  //   rhs_scanner.mutable_types().push_back(rhs_supported_types);
  //   auto rhs_type_masks = rhs_scanner.Scan();

  //   if (lhs_type_masks->is_unique_group &&
  //       (lhs_type_masks->unique_group == lhs_supported_types) &&
  //       rhs_type_masks->is_unique_group &&
  //       (rhs_type_masks->unique_group == rhs_supported_types)) {
  //     return cuda::ST_Within(geometries_1, geometries_2);
  //   } else {
  //     return gdal::ST_Within(geometries_1, geometries_2);
  //   }
  // #else
  return gdal::ST_Within(geometries_1, geometries_2);
  // #endif
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
#ifdef USE_GPU
                           "build type : " + CMAKE_BUILD_TYPE + "/GPU \n" +
#else
                           "build type : " + CMAKE_BUILD_TYPE + "/CPU \n" +
#endif
                           "build time : " + BUILD_TIME + "\n" +
                           "commit id : " + LAST_COMMIT_ID + "\n";
  return std::make_shared<std::string>(info);
}

}  // namespace gis
}  // namespace arctern
