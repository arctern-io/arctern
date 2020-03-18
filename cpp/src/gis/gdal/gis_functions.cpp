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
#include "gis/gdal/arctern_geos.h"
#include "gis/gdal/geometry_visitor.h"
#include "gis/parser.h"
#include "utils/check_status.h"

#include <assert.h>
#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace arctern {
namespace gis {
namespace gdal {

#define UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T1(RESULT_BUILDER_TYPE, GEO_VAR,              \
                                              APPEND_RESULT)                             \
  do {                                                                                   \
    auto len = geometries->length();                                                     \
    auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);      \
    RESULT_BUILDER_TYPE builder;                                                         \
    void* GEO_VAR;                                                                       \
    for (int32_t i = 0; i < len; i++) {                                                  \
      CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt_geometries->GetString(i).c_str(), \
                                                   nullptr, (OGRGeometry**)(&GEO_VAR))); \
      CHECK_ARROW(builder.Append(APPEND_RESULT));                                        \
      OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR);                        \
    }                                                                                    \
    std::shared_ptr<arrow::Array> results;                                               \
    CHECK_ARROW(builder.Finish(&results));                                               \
    return results;                                                                      \
  } while (0)

#define UNARY_WKT_FUNC_WITH_GDAL_IMPL_T1(FUNC_NAME, RESULT_BUILDER_TYPE, GEO_VAR,       \
                                         APPEND_RESULT)                                 \
  std::shared_ptr<arrow::Array> FUNC_NAME(                                              \
      const std::shared_ptr<arrow::Array>& geometries) {                                \
    UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T1(RESULT_BUILDER_TYPE, GEO_VAR, APPEND_RESULT); \
  }

#define UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(RESULT_BUILDER_TYPE, GEO_VAR,              \
                                              APPEND_RESULT)                             \
  do {                                                                                   \
    auto len = geometries->length();                                                     \
    auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);      \
    RESULT_BUILDER_TYPE builder;                                                         \
    void *GEO_VAR, *geo_tmp;                                                             \
    char* wkt_tmp;                                                                       \
    for (int32_t i = 0; i < len; i++) {                                                  \
      CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt_geometries->GetString(i).c_str(), \
                                                   nullptr, (OGRGeometry**)(&GEO_VAR))); \
      geo_tmp = APPEND_RESULT;                                                           \
      CHECK_GDAL(OGR_G_ExportToWkt(geo_tmp, &wkt_tmp));                                  \
      CHECK_ARROW(builder.Append(wkt_tmp));                                              \
      OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR);                        \
      OGRGeometryFactory::destroyGeometry((OGRGeometry*)geo_tmp);                        \
      CPLFree(wkt_tmp);                                                                  \
    }                                                                                    \
    std::shared_ptr<arrow::Array> results;                                               \
    CHECK_ARROW(builder.Finish(&results));                                               \
    return results;                                                                      \
  } while (0)

#define UNARY_WKT_FUNC_WITH_GDAL_IMPL_T2(FUNC_NAME, RESULT_BUILDER_TYPE, GEO_VAR,       \
                                         APPEND_RESULT)                                 \
  std::shared_ptr<arrow::Array> FUNC_NAME(                                              \
      const std::shared_ptr<arrow::Array>& geometries) {                                \
    UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(RESULT_BUILDER_TYPE, GEO_VAR, APPEND_RESULT); \
  }

#define BINARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T1(RESULT_BUILDER_TYPE, GEO_VAR_1,          \
                                               GEO_VAR_2, APPEND_RESULT)                \
  do {                                                                                  \
    assert(geometries_1->length() == geometries_2->length());                           \
    auto len = geometries_1->length();                                                  \
    auto wkt_geometries_1 = std::static_pointer_cast<arrow::StringArray>(geometries_1); \
    auto wkt_geometries_2 = std::static_pointer_cast<arrow::StringArray>(geometries_2); \
    RESULT_BUILDER_TYPE builder;                                                        \
    void *GEO_VAR_1, *GEO_VAR_2;                                                        \
    for (int32_t i = 0; i < len; i++) {                                                 \
      CHECK_GDAL(                                                                       \
          OGRGeometryFactory::createFromWkt(wkt_geometries_1->GetString(i).c_str(),     \
                                            nullptr, (OGRGeometry**)(&GEO_VAR_1)));     \
      CHECK_GDAL(                                                                       \
          OGRGeometryFactory::createFromWkt(wkt_geometries_2->GetString(i).c_str(),     \
                                            nullptr, (OGRGeometry**)(&GEO_VAR_2)));     \
      CHECK_ARROW(builder.Append(APPEND_RESULT));                                       \
      OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR_1);                     \
      OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR_2);                     \
    }                                                                                   \
    std::shared_ptr<arrow::Array> results;                                              \
    CHECK_ARROW(builder.Finish(&results));                                              \
    return results;                                                                     \
  } while (0)

#define UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T3(RESULT_BUILDER_TYPE, GEO_WKT_VAR,     \
                                              APPEND_RESULT)                        \
  do {                                                                              \
    auto len = geometries->length();                                                \
    auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries); \
    RESULT_BUILDER_TYPE builder;                                                    \
    const char* GEO_WKT_VAR;                                                        \
    for (int32_t i = 0; i < len; i++) {                                             \
      auto geo_wkt_str = wkt_geometries->GetString(i);                              \
      GEO_WKT_VAR = geo_wkt_str.c_str();                                            \
      CHECK_ARROW(builder.Append(APPEND_RESULT));                                   \
    }                                                                               \
    std::shared_ptr<arrow::Array> results;                                          \
    CHECK_ARROW(builder.Finish(&results));                                          \
    return results;                                                                 \
  } while (0)

#define UNARY_WKT_FUNC_WITH_GDAL_IMPL_T3(FUNC_NAME, RESULT_BUILDER_TYPE, GEO_WKT_VAR, \
                                         APPEND_RESULT)                               \
  std::shared_ptr<arrow::Array> FUNC_NAME(                                            \
      const std::shared_ptr<arrow::Array>& geometries) {                              \
    UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T3(RESULT_BUILDER_TYPE, GEO_WKT_VAR,           \
                                          APPEND_RESULT);                             \
  }

#define BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(FUNC_NAME, RESULT_BUILDER_TYPE, GEO_VAR_1,  \
                                          GEO_VAR_2, APPEND_RESULT)                   \
  std::shared_ptr<arrow::Array> FUNC_NAME(                                            \
      const std::shared_ptr<arrow::Array>& geometries_1,                              \
      const std::shared_ptr<arrow::Array>& geometries_2) {                            \
    BINARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T1(RESULT_BUILDER_TYPE, GEO_VAR_1, GEO_VAR_2, \
                                           APPEND_RESULT);                            \
  }

#define BINARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(RESULT_BUILDER_TYPE, GEO_VAR_1,          \
                                               GEO_VAR_2, APPEND_RESULT)                \
  do {                                                                                  \
    assert(geometries_1->length() == geometries_2->length());                           \
    auto len = geometries_1->length();                                                  \
    auto wkt_geometries_1 = std::static_pointer_cast<arrow::StringArray>(geometries_1); \
    auto wkt_geometries_2 = std::static_pointer_cast<arrow::StringArray>(geometries_2); \
    RESULT_BUILDER_TYPE builder;                                                        \
    void *GEO_VAR_1, *GEO_VAR_2, *geo_tmp;                                              \
    char* wkt_tmp;                                                                      \
    for (int32_t i = 0; i < len; i++) {                                                 \
      CHECK_GDAL(                                                                       \
          OGRGeometryFactory::createFromWkt(wkt_geometries_1->GetString(i).c_str(),     \
                                            nullptr, (OGRGeometry**)(&GEO_VAR_1)));     \
      CHECK_GDAL(                                                                       \
          OGRGeometryFactory::createFromWkt(wkt_geometries_2->GetString(i).c_str(),     \
                                            nullptr, (OGRGeometry**)(&GEO_VAR_2)));     \
      geo_tmp = APPEND_RESULT;                                                          \
      CHECK_GDAL(OGR_G_ExportToWkt(geo_tmp, &wkt_tmp));                                 \
      CHECK_ARROW(builder.Append(wkt_tmp));                                             \
      OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR_1);                     \
      OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR_2);                     \
      OGRGeometryFactory::destroyGeometry((OGRGeometry*)geo_tmp);                       \
      CPLFree(wkt_tmp);                                                                 \
    }                                                                                   \
    std::shared_ptr<arrow::Array> results;                                              \
    CHECK_ARROW(builder.Finish(&results));                                              \
    return results;                                                                     \
  } while (0)

#define BINARY_WKT_FUNC_WITH_GDAL_IMPL_T2(FUNC_NAME, RESULT_BUILDER_TYPE, GEO_VAR_1,  \
                                          GEO_VAR_2, APPEND_RESULT)                   \
  std::shared_ptr<arrow::Array> FUNC_NAME(                                            \
      const std::shared_ptr<arrow::Array>& geometries_1,                              \
      const std::shared_ptr<arrow::Array>& geometries_2) {                            \
    BINARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(RESULT_BUILDER_TYPE, GEO_VAR_1, GEO_VAR_2, \
                                           APPEND_RESULT);                            \
  }

inline void* Wrapper_OGR_G_Centroid(void* geo) {
  void* centroid = new OGRPoint();
  OGR_G_Centroid(geo, centroid);
  return centroid;
}

inline bool Wrapper_OGR_G_IsValid(const char* geo_wkt) {
  void* geo = nullptr;
  if (parser::IsValidWkt(geo_wkt) == false) return false;
  bool is_valid = false;
  auto err_code =
      OGRGeometryFactory::createFromWkt(geo_wkt, nullptr, (OGRGeometry**)(&geo));
  if (err_code == OGRERR_NONE) {
    is_valid = OGR_G_IsValid(geo) != 0;
  }
  if (geo) {
    OGRGeometryFactory::destroyGeometry((OGRGeometry*)geo);
  }
  return is_valid;
}

inline OGRGeometry* Wrapper_createFromWkt(
    const std::shared_ptr<arrow::StringArray>& array, int idx) {
  if (array->IsNull(idx)) return nullptr;
  // if (parser::IsValidWkt(array->GetString(idx).c_str()) == false) return nullptr;
  OGRGeometry* geo = nullptr;
  auto err_code =
      OGRGeometryFactory::createFromWkt(array->GetString(idx).c_str(), nullptr, &geo);
  if (err_code) return nullptr;
  return geo;
}

inline OGRGeometry* Wrapper_createFromWkt(const char* geo_wkt) {
  OGRGeometry* geo = nullptr;
  auto err_code = OGRGeometryFactory::createFromWkt(geo_wkt, nullptr, &geo);
  if (err_code) {
    std::string err_msg =
        "failed to create geometry from \"" + std::string(geo_wkt) + "\"";
    throw std::runtime_error(err_msg);
  }
  return geo;
}

inline OGRGeometry* Wrapper_CurveToLine(OGRGeometry* geo, HasCurveVisitor* has_curve) {
  if (geo != nullptr) {
    has_curve->reset();
    geo->accept(has_curve);
    if (has_curve->has_curve()) {
      auto linear = geo->getLinearGeometry();
      OGRGeometryFactory::destroyGeometry(geo);
      return linear;
    }
  }
  return geo;
}

inline char* Wrapper_OGR_G_ExportToWkt(OGRGeometry* geo) {
  char* str;
  auto err_code = OGR_G_ExportToWkt(geo, &str);
  if (err_code != OGRERR_NONE) {
    std::string err_msg =
        "failed to export to wkt, error code = " + std::to_string(err_code);
    throw std::runtime_error(err_msg);
  }
  return str;
}

inline std::string Wrapper_OGR_G_GetGeometryName(void* geo) {
  auto ogr_geometry_name = OGR_G_GetGeometryName(geo);
  std::string adjust_geometry_name = "ST_" + std::string(ogr_geometry_name);
  return adjust_geometry_name;
}

/************************ GEOMETRY CONSTRUCTOR ************************/

std::shared_ptr<arrow::Array> ST_Point(const std::shared_ptr<arrow::Array>& x_values,
                                       const std::shared_ptr<arrow::Array>& y_values) {
  assert(x_values->length() == y_values->length());
  auto len = x_values->length();
  auto x_double_values = std::static_pointer_cast<arrow::DoubleArray>(x_values);
  auto y_double_values = std::static_pointer_cast<arrow::DoubleArray>(y_values);
  OGRPoint point;
  char* wkt = nullptr;
  arrow::StringBuilder builder;

  for (int32_t i = 0; i < len; i++) {
    point.setX(x_double_values->Value(i));
    point.setY(y_double_values->Value(i));
    CHECK_GDAL(OGR_G_ExportToWkt(&point, &wkt));
    CHECK_ARROW(builder.Append(wkt));
    CPLFree(wkt);
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_PolygonFromEnvelope(
    const std::shared_ptr<arrow::Array>& min_x_values,
    const std::shared_ptr<arrow::Array>& min_y_values,
    const std::shared_ptr<arrow::Array>& max_x_values,
    const std::shared_ptr<arrow::Array>& max_y_values) {
  assert(min_x_values->length() == max_x_values->length());
  assert(min_y_values->length() == max_y_values->length());
  assert(min_x_values->length() == min_y_values->length());
  auto len = min_x_values->length();

  auto min_x_double_values =
      std::static_pointer_cast<const arrow::DoubleArray>(min_x_values);
  auto min_y_double_values =
      std::static_pointer_cast<const arrow::DoubleArray>(min_y_values);
  auto max_x_double_values =
      std::static_pointer_cast<const arrow::DoubleArray>(max_x_values);
  auto max_y_double_values =
      std::static_pointer_cast<const arrow::DoubleArray>(max_y_values);

  arrow::StringBuilder builder;

  for (int32_t i = 0; i < len; i++) {
    if ((min_x_double_values->Value(i) > max_x_double_values->Value(i)) ||
        (min_y_double_values->Value(i) > max_y_double_values->Value(i))) {
      CHECK_ARROW(builder.Append("POLYGON EMPTY"));
    } else {
      OGRLinearRing ring;
      ring.addPoint(min_x_double_values->Value(i), min_y_double_values->Value(i));
      ring.addPoint(min_x_double_values->Value(i), max_y_double_values->Value(i));
      ring.addPoint(max_x_double_values->Value(i), max_y_double_values->Value(i));
      ring.addPoint(max_x_double_values->Value(i), min_y_double_values->Value(i));
      ring.addPoint(min_x_double_values->Value(i), min_y_double_values->Value(i));
      ring.closeRings();
      OGRPolygon polygon;
      polygon.addRing(&ring);
      char* wkt = nullptr;
      CHECK_GDAL(OGR_G_ExportToWkt(&polygon, &wkt));
      CHECK_ARROW(builder.Append(wkt));
      CPLFree(wkt);
    }
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_GeomFromGeoJSON(
    const std::shared_ptr<arrow::Array>& json) {
  auto json_geo = std::static_pointer_cast<arrow::StringArray>(json);
  int len = json_geo->length();
  arrow::StringBuilder builder;
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkt(json_geo, i);
    if (geo != nullptr) {
      char* wkt = Wrapper_OGR_G_ExportToWkt(geo);
      CHECK_ARROW(builder.Append(wkt));
      CPLFree(wkt);
      OGRGeometryFactory::destroyGeometry(geo);
    } else {
      builder.AppendNull();
    }
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_GeomFromText(const std::shared_ptr<arrow::Array>& text) {
  auto geo = std::static_pointer_cast<arrow::StringArray>(text);
  int len = geo->length();
  arrow::StringBuilder builder;
  for (int i = 0; i < len; ++i) {
    auto ogr = Wrapper_createFromWkt(geo, i);
    if (ogr == nullptr) {
      builder.AppendNull();
    } else {
      if (parser::IsValidWkt(geo->GetString(i).c_str()) == false) {
        builder.AppendNull();
      } else {
        char* wkt = Wrapper_OGR_G_ExportToWkt(ogr);
        CHECK_ARROW(builder.Append(wkt));
        CPLFree(wkt);
      }
    }
    OGRGeometryFactory::destroyGeometry(ogr);
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

/************************* GEOMETRY ACCESSOR **************************/

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T3(ST_IsValid, arrow::BooleanBuilder, geo_wkt,
                                 Wrapper_OGR_G_IsValid(geo_wkt));

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T1(ST_GeometryType, arrow::StringBuilder, geo,
                                 Wrapper_OGR_G_GetGeometryName(geo));

std::shared_ptr<arrow::Array> ST_IsSimple(const std::shared_ptr<arrow::Array>& geo) {
  auto wkt = std::static_pointer_cast<arrow::StringArray>(geo);
  auto len = geo->length();
  arrow::BooleanBuilder builder;
  auto has_circular = new HasCircularVisitor;
  const char* papszOptions[] = {(const char*)"ADD_INTERMEDIATE_POINT=YES", nullptr};
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkt(wkt, i);
    if (geo == nullptr) {
      builder.AppendNull();
    } else {
      has_circular->reset();
      geo->accept(has_circular);
      if (has_circular->has_circular()) {
        auto linear = geo->getLinearGeometry(0, papszOptions);
        builder.Append(linear->IsSimple() != 0);
        OGRGeometryFactory::destroyGeometry(linear);
      } else {
        builder.Append(geo->IsSimple() != 0);
      }
    }
    OGRGeometryFactory::destroyGeometry(geo);
  }
  delete has_circular;
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_NPoints(const std::shared_ptr<arrow::Array>& geo) {
  auto wkt = std::static_pointer_cast<arrow::StringArray>(geo);
  auto len = geo->length();
  arrow::Int64Builder builder;
  auto npoints = new NPointsVisitor;
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkt(wkt, i);
    if (geo == nullptr) {
      builder.AppendNull();
    } else {
      npoints->reset();
      geo->accept(npoints);
      builder.Append(npoints->npoints());
    }
    OGRGeometryFactory::destroyGeometry(geo);
  }
  delete npoints;
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_Envelope(
    const std::shared_ptr<arrow::Array>& geometries) {
  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);
  auto len = geometries->length();
  arrow::StringBuilder builder;
  OGREnvelope env;
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkt(wkt_geometries, i);
    if (geo == nullptr) {
      builder.AppendNull();
    } else if (geo->IsEmpty()) {
      CHECK_ARROW(builder.Append(wkt_geometries->GetString(i)));
    } else {
      OGR_G_GetEnvelope(geo, &env);
      char* wkt = nullptr;
      if (env.MinX == env.MaxX) {    // vertical line or Point
        if (env.MinY == env.MaxY) {  // point
          OGRPoint point(env.MinX, env.MinY);
          wkt = Wrapper_OGR_G_ExportToWkt(&point);
        } else {  // line
          OGRLineString line;
          line.addPoint(env.MinX, env.MinY);
          line.addPoint(env.MinX, env.MaxY);
          wkt = Wrapper_OGR_G_ExportToWkt(&line);
        }
      } else {
        if (env.MinY == env.MaxY) {  // horizontal line
          OGRLineString line;
          line.addPoint(env.MinX, env.MinY);
          line.addPoint(env.MaxX, env.MinY);
          wkt = Wrapper_OGR_G_ExportToWkt(&line);
        } else {  // polygon
          OGRLinearRing ring;
          ring.addPoint(env.MinX, env.MinY);
          ring.addPoint(env.MinX, env.MaxY);
          ring.addPoint(env.MaxX, env.MaxY);
          ring.addPoint(env.MaxX, env.MinY);
          ring.addPoint(env.MinX, env.MinY);
          OGRPolygon polygon;
          polygon.addRing(&ring);
          wkt = Wrapper_OGR_G_ExportToWkt(&polygon);
        }
      }
      CHECK_ARROW(builder.Append(wkt));
      OGRGeometryFactory::destroyGeometry(geo);
      CPLFree(wkt);
    }
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

/************************ GEOMETRY PROCESSING ************************/

std::shared_ptr<arrow::Array> ST_Buffer(const std::shared_ptr<arrow::Array>& geometries,
                                        double buffer_distance, int n_quadrant_segments) {
  UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(
      arrow::StringBuilder, geo, OGR_G_Buffer(geo, buffer_distance, n_quadrant_segments));
}

std::shared_ptr<arrow::Array> ST_PrecisionReduce(
    const std::shared_ptr<arrow::Array>& geometries, int32_t precision) {
  auto precision_reduce_visitor = new PrecisionReduceVisitor(precision);
  auto len = geometries->length();
  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);
  arrow::StringBuilder builder;

  for (int32_t i = 0; i < len; i++) {
    auto geo = Wrapper_createFromWkt(wkt_geometries, i);
    if (geo == nullptr) {
      CHECK_ARROW(builder.AppendNull());
    } else {
      geo->accept(precision_reduce_visitor);
      auto wkt_tmp = Wrapper_OGR_G_ExportToWkt(geo);
      CHECK_ARROW(builder.Append(wkt_tmp));
      OGRGeometryFactory::destroyGeometry((OGRGeometry*)geo);
      CPLFree(wkt_tmp);
    }
  }

  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  delete precision_reduce_visitor;
  return results;
}

std::shared_ptr<arrow::Array> ST_Intersection(const std::shared_ptr<arrow::Array>& geo1,
                                              const std::shared_ptr<arrow::Array>& geo2) {
  auto wkt1 = std::static_pointer_cast<arrow::StringArray>(geo1);
  auto wkt2 = std::static_pointer_cast<arrow::StringArray>(geo2);
  auto len = wkt1->length();
  arrow::StringBuilder builder;
  auto has_curve = new HasCurveVisitor;

  for (int i = 0; i < len; ++i) {
    auto ogr1 = Wrapper_createFromWkt(wkt1, i);
    auto ogr2 = Wrapper_createFromWkt(wkt2, i);
    ogr1 = Wrapper_CurveToLine(ogr1, has_curve);
    ogr2 = Wrapper_CurveToLine(ogr2, has_curve);

    if ((ogr1 == nullptr) && (ogr2 == nullptr)) {
      builder.AppendNull();
    } else if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
      builder.Append("GEOMETRYCOLLECTION EMPTY");
    } else {
      auto rst = ogr1->Intersection(ogr2);
      if (rst == nullptr) {
        builder.AppendNull();
      } else if (rst->IsEmpty()) {
        builder.Append("GEOMETRYCOLLECTION EMPTY");
      } else {
        char* wkt = Wrapper_OGR_G_ExportToWkt(rst);
        builder.Append(std::string(wkt));
        CPLFree(wkt);
      }
      OGRGeometryFactory::destroyGeometry(rst);
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }

  delete has_curve;

  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));

  return results;
}

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T2(ST_MakeValid, arrow::StringBuilder, geo,
                                 OGR_G_MakeValid(geo));

std::shared_ptr<arrow::Array> ST_SimplifyPreserveTopology(
    const std::shared_ptr<arrow::Array>& geometries, double distance_tolerance) {
  UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(
      arrow::StringBuilder, geo, OGR_G_SimplifyPreserveTopology(geo, distance_tolerance));
}

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T2(ST_Centroid, arrow::StringBuilder, geo,
                                 Wrapper_OGR_G_Centroid(geo));

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T2(ST_ConvexHull, arrow::StringBuilder, geo,
                                 OGR_G_ConvexHull(geo));

/*
 * The detailed EPSG information can be found at EPSG.io [https://epsg.io/]
 */
std::shared_ptr<arrow::Array> ST_Transform(const std::shared_ptr<arrow::Array>& geos,
                                           const std::string& src_rs,
                                           const std::string& dst_rs) {
  OGRSpatialReference oSrcSRS;
  oSrcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  if (oSrcSRS.SetFromUserInput(src_rs.c_str()) != OGRERR_NONE) {
    std::string err_msg = "faild to tranform with sourceCRS = " + src_rs;
    throw std::runtime_error(err_msg);
  }

  OGRSpatialReference oDstS;
  oDstS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  if (oDstS.SetFromUserInput(dst_rs.c_str()) != OGRERR_NONE) {
    std::string err_msg = "faild to tranform with targetCRS = " + dst_rs;
    throw std::runtime_error(err_msg);
  }

  void* poCT = OCTNewCoordinateTransformation(&oSrcSRS, &oDstS);
  arrow::StringBuilder builder;

  auto len = geos->length();
  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geos);

  for (int32_t i = 0; i < len; i++) {
    auto geo = Wrapper_createFromWkt(wkt_geometries, i);
    if (geo == nullptr) {
      CHECK_ARROW(builder.AppendNull());
    } else {
      CHECK_GDAL(OGR_G_Transform(geo, (OGRCoordinateTransformation*)poCT))
      auto wkt = Wrapper_OGR_G_ExportToWkt(geo);
      CHECK_ARROW(builder.Append(wkt));
      OGRGeometryFactory::destroyGeometry(geo);
      CPLFree(wkt);
    }
  }

  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  OCTDestroyCoordinateTransformation(poCT);

  return results;
}

std::shared_ptr<arrow::Array> ST_CurveToLine(const std::shared_ptr<arrow::Array>& geos) {
  auto len = geos->length();
  auto wkt = std::static_pointer_cast<arrow::StringArray>(geos);
  arrow::StringBuilder builder;
  for (int32_t i = 0; i < len; i++) {
    auto ogr = Wrapper_createFromWkt(wkt, i);
    if (ogr == nullptr) {
      CHECK_ARROW(builder.AppendNull());
    } else {
      auto line = ogr->getLinearGeometry();
      auto line_wkt = Wrapper_OGR_G_ExportToWkt(line);
      CHECK_ARROW(builder.Append(line_wkt));
      CPLFree(line_wkt);
      OGRGeometryFactory::destroyGeometry(line);
      OGRGeometryFactory::destroyGeometry(ogr);
    }
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));

  return results;
}

/************************ MEASUREMENT FUNCTIONS ************************/

std::shared_ptr<arrow::Array> ST_Area(const std::shared_ptr<arrow::Array>& geometries) {
  auto len = geometries->length();
  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);
  arrow::DoubleBuilder builder;
  auto* area = new AreaVisitor;
  for (int32_t i = 0; i < len; i++) {
    auto ogr = Wrapper_createFromWkt(wkt_geometries, i);
    if (ogr == nullptr) {
      builder.AppendNull();
    } else {
      area->reset();
      ogr->accept(area);
      builder.Append(area->area());
    }
    OGRGeometryFactory::destroyGeometry(ogr);
  }
  delete area;
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_Length(const std::shared_ptr<arrow::Array>& geometries) {
  auto len = geometries->length();
  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);
  arrow::DoubleBuilder builder;
  auto* len_sum = new LengthVisitor;
  for (int i = 0; i < len; i++) {
    auto ogr = Wrapper_createFromWkt(wkt_geometries, i);
    if (ogr == nullptr) {
      builder.AppendNull();
    } else {
      len_sum->reset();
      ogr->accept(len_sum);
      builder.Append(len_sum->length());
    }
    OGRGeometryFactory::destroyGeometry(ogr);
  }
  delete len_sum;
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_HausdorffDistance(
    const std::shared_ptr<arrow::Array>& geo1,
    const std::shared_ptr<arrow::Array>& geo2) {
  auto len = geo1->length();
  auto wkt1 = std::static_pointer_cast<arrow::StringArray>(geo1);
  auto wkt2 = std::static_pointer_cast<arrow::StringArray>(geo2);
  arrow::DoubleBuilder builder;
  auto geos_ctx = OGRGeometry::createGEOSContext();
  for (int32_t i = 0; i < len; ++i) {
    auto ogr1 = Wrapper_createFromWkt(wkt1, i);
    auto ogr2 = Wrapper_createFromWkt(wkt2, i);
    if ((ogr1 == nullptr) || (ogr1->IsEmpty()) || (ogr2 == nullptr) ||
        (ogr2->IsEmpty())) {
      CHECK_ARROW(builder.AppendNull());
    } else {
      auto geos1 = ogr1->exportToGEOS(geos_ctx);
      auto geos2 = ogr2->exportToGEOS(geos_ctx);
      double dist;
      int geos_err = GEOSHausdorffDistance_r(geos_ctx, geos1, geos2, &dist);
      if (geos_err == 0) {  // geos error
        dist = -1;
      }
      GEOSGeom_destroy_r(geos_ctx, geos1);
      GEOSGeom_destroy_r(geos_ctx, geos2);
      CHECK_ARROW(builder.Append(dist));
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }
  OGRGeometry::freeGEOSContext(geos_ctx);
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_Distance(const std::shared_ptr<arrow::Array>& geo1,
                                          const std::shared_ptr<arrow::Array>& geo2) {
  auto len = geo1->length();
  auto wkt1 = std::static_pointer_cast<arrow::StringArray>(geo1);
  auto wkt2 = std::static_pointer_cast<arrow::StringArray>(geo2);
  arrow::DoubleBuilder builder;

  for (int i = 0; i < len; ++i) {
    auto ogr1 = Wrapper_createFromWkt(wkt1, i);
    auto ogr2 = Wrapper_createFromWkt(wkt2, i);
    if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
      builder.AppendNull();
    } else if (ogr1->IsEmpty() || ogr2->IsEmpty()) {
      builder.AppendNull();
    } else {
      double dist = OGR_G_Distance(ogr1, ogr2);
      if (dist < 0) {
        builder.AppendNull();
      } else {
        builder.Append(dist);
      }
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

/************************ SPATIAL RELATIONSHIP ************************/

/*************************************************
 * https://postgis.net/docs/ST_Equals.html
 * Returns TRUE if the given Geometries are "spatially equal".
 * Use this for a 'better' answer than '='.
 * Note by spatially equal we mean ST_Within(A,B) = true and ST_Within(B,A) = true and
 * also mean ordering of points can be different but represent the same geometry
 * structure. To verify the order of points is consistent, use ST_OrderingEquals (it must
 * be noted ST_OrderingEquals is a little more stringent than simply verifying order of
 * points are the same).
 * ***********************************************/
std::shared_ptr<arrow::Array> ST_Equals(const std::shared_ptr<arrow::Array>& geo1,
                                        const std::shared_ptr<arrow::Array>& geo2) {
  auto len = geo1->length();
  auto wkt1 = std::static_pointer_cast<arrow::StringArray>(geo1);
  auto wkt2 = std::static_pointer_cast<arrow::StringArray>(geo2);
  arrow::BooleanBuilder builder;
  for (int32_t i = 0; i < len; ++i) {
    auto ogr1 = Wrapper_createFromWkt(wkt1, i);
    auto ogr2 = Wrapper_createFromWkt(wkt2, i);
    if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
      builder.AppendNull();
    } else if (ogr1->IsEmpty() && ogr2->IsEmpty()) {
      builder.Append(true);
    } else if (ogr1->Within(ogr2) && ogr2->Within(ogr1)) {
      builder.Append(true);
    } else {
      builder.Append(false);
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(ST_Touches, arrow::BooleanBuilder, geo_1, geo_2,
                                  OGR_G_Touches(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(ST_Overlaps, arrow::BooleanBuilder, geo_1, geo_2,
                                  OGR_G_Overlaps(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(ST_Crosses, arrow::BooleanBuilder, geo_1, geo_2,
                                  OGR_G_Crosses(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(ST_Contains, arrow::BooleanBuilder, geo_1, geo_2,
                                  OGR_G_Contains(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(ST_Intersects, arrow::BooleanBuilder, geo_1, geo_2,
                                  OGR_G_Intersects(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(ST_Within, arrow::BooleanBuilder, geo_1, geo_2,
                                  OGR_G_Within(geo_1, geo_2) != 0);

/*********************** AGGREGATE FUNCTIONS ***************************/

std::shared_ptr<arrow::Array> ST_Union_Aggr(const std::shared_ptr<arrow::Array>& geo) {
  auto len = geo->length();
  auto wkt = std::static_pointer_cast<arrow::StringArray>(geo);
  std::vector<OGRGeometry*> union_agg;
  OGRPolygon empty_polygon;
  OGRGeometry *g0, *g1;
  OGRGeometry *u0, *u1;
  auto has_curve = new HasCurveVisitor;
  for (int i = 0; i <= len / 2; i++) {
    if ((i * 2) < len) {
      g0 = Wrapper_createFromWkt(wkt, 2 * i);
      g0 = Wrapper_CurveToLine(g0, has_curve);
    } else {
      g0 = nullptr;
    }

    if ((i * 2 + 1) < len) {
      g1 = Wrapper_createFromWkt(wkt, 2 * i + 1);
      g1 = Wrapper_CurveToLine(g1, has_curve);
    } else {
      g1 = nullptr;
    }

    if (g0 != nullptr) {
      auto type = wkbFlatten(g0->getGeometryType());
      if (type == wkbMultiPolygon) {
        u0 = g0->UnionCascaded();
        OGRGeometryFactory::destroyGeometry(g0);
      } else {
        u0 = g0;
      }
    } else {
      u0 = nullptr;
    }

    if (g1 != nullptr) {
      auto type = wkbFlatten(g1->getGeometryType());
      if (type == wkbMultiPolygon) {
        u1 = g1->UnionCascaded();
        OGRGeometryFactory::destroyGeometry(g1);
      } else {
        u1 = g1;
      }
    } else {
      u1 = nullptr;
    }

    if ((u0 != nullptr) && (u1 != nullptr)) {
      OGRGeometry* ua = u0->Union(u1);
      union_agg.push_back(ua);
      OGRGeometryFactory::destroyGeometry(u0);
      OGRGeometryFactory::destroyGeometry(u1);
    } else if ((u0 != nullptr) && (u1 == nullptr)) {
      union_agg.push_back(u0);
    } else if ((u0 == nullptr) && (u1 != nullptr)) {
      union_agg.push_back(u1);
    }
  }
  len = union_agg.size();
  while (len > 1) {
    std::vector<OGRGeometry*> union_tmp;
    for (int i = 0; i <= len / 2; ++i) {
      if (i * 2 < len) {
        u0 = union_agg[i * 2];
      } else {
        u0 = nullptr;
      }

      if (i * 2 + 1 < len) {
        u1 = union_agg[i * 2 + 1];
      } else {
        u1 = nullptr;
      }

      if ((u0 != nullptr) && (u1 != nullptr)) {
        OGRGeometry* ua = u0->Union(u1);
        union_tmp.push_back(ua);
        OGRGeometryFactory::destroyGeometry(u0);
        OGRGeometryFactory::destroyGeometry(u1);
      } else if ((u0 != nullptr) && (u1 == nullptr)) {
        union_tmp.push_back(u0);
      } else if ((u0 == nullptr) && (u1 != nullptr)) {
        union_tmp.push_back(u1);
      }
    }
    union_agg = std::move(union_tmp);
    len = union_agg.size();
  }
  arrow::StringBuilder builder;
  if (union_agg.empty()) {
    builder.AppendNull();
  } else {
    char* wkt = Wrapper_OGR_G_ExportToWkt(union_agg[0]);
    CHECK_ARROW(builder.Append(wkt));
    CPLFree(wkt);
    OGRGeometryFactory::destroyGeometry(union_agg[0]);
  }
  delete has_curve;
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_Envelope_Aggr(
    const std::shared_ptr<arrow::Array>& geometries) {
  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);
  auto len = geometries->length();
  double inf = std::numeric_limits<double>::infinity();
  double xmin = inf;
  double xmax = -inf;
  double ymin = inf;
  double ymax = -inf;

  OGREnvelope env;
  bool set_env = false;
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkt(wkt_geometries, i);
    if (geo == nullptr) continue;
    if (geo->IsEmpty()) continue;
    set_env = true;
    OGR_G_GetEnvelope(geo, &env);
    if (env.MinX < xmin) xmin = env.MinX;
    if (env.MaxX > xmax) xmax = env.MaxX;
    if (env.MinY < ymin) ymin = env.MinY;
    if (env.MaxY > ymax) ymax = env.MaxY;
    OGRGeometryFactory::destroyGeometry(geo);
  }
  arrow::StringBuilder builder;
  if (set_env) {
    OGRLinearRing ring;
    ring.addPoint(xmin, ymin);
    ring.addPoint(xmin, ymax);
    ring.addPoint(xmax, ymax);
    ring.addPoint(xmax, ymin);
    ring.addPoint(xmin, ymin);
    OGRPolygon polygon;
    polygon.addRing(&ring);
    char* wkt = nullptr;
    wkt = Wrapper_OGR_G_ExportToWkt(&polygon);
    CHECK_ARROW(builder.Append(wkt));
    CPLFree(wkt);
  } else {
    CHECK_ARROW(builder.Append("POLYGON EMPTY"));
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
