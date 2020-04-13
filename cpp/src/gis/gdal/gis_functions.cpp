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
#include <functional>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace arctern {
namespace gis {
namespace gdal {

// inline void* Wrapper_OGR_G_Centroid(void* geo) {
//   void* centroid = new OGRPoint();
//   OGR_G_Centroid(geo, centroid);
//   return centroid;
// }

inline OGRGeometry* Wrapper_createFromWkt(
    const std::shared_ptr<arrow::StringArray>& array, int idx) {
  if (array->IsNull(idx)) return nullptr;
  auto wkb_str = array->GetString(idx);

  if (parser::IsValidWkt(wkb_str.c_str()) == false) return nullptr;
  OGRGeometry* geo = nullptr;
  auto err_code = OGRGeometryFactory::createFromWkt(wkb_str.c_str(), nullptr, &geo);
  if (err_code != OGRERR_NONE) return nullptr;
  return geo;
}

inline OGRGeometry* Wrapper_createFromWkb(
    const std::shared_ptr<arrow::BinaryArray>& array, int idx) {
  if (array->IsNull(idx)) return nullptr;
  arrow::BinaryArray::offset_type offset;
  auto data_ptr = array->GetValue(idx, &offset);
  if (offset <= 0) return nullptr;

  OGRGeometry* geo = nullptr;
  auto err_code = OGRGeometryFactory::createFromWkb(data_ptr, nullptr, &geo, offset);
  if (err_code != OGRERR_NONE) return nullptr;
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

inline void AppendWkbNDR(arrow::BinaryBuilder& builder, const OGRGeometry* geo) {
  if (geo == nullptr) {
    builder.AppendNull();
  } else if (geo->IsEmpty() && (geo->getGeometryType() == wkbPoint)) {
    builder.AppendNull();
  } else {
    auto wkb_size = geo->WkbSize();
    auto wkb = static_cast<unsigned char*>(CPLMalloc(wkb_size));
    auto err_code = geo->exportToWkb(OGRwkbByteOrder::wkbNDR, wkb);
    if (err_code != OGRERR_NONE) {
      builder.AppendNull();
      // std::string err_msg =
      //     "failed to export to wkb, error code = " + std::to_string(err_code);
      // throw std::runtime_error(err_msg);
    } else {
      CHECK_ARROW(builder.Append(wkb, wkb_size));
    }
    CPLFree(wkb);
  }
}

template <typename T>
typename std::enable_if<std::is_base_of<arrow::ArrayBuilder, T>::value,
                        std::shared_ptr<typename arrow::Array>>::type
UnaryOp(const std::shared_ptr<arrow::Array>& array,
        std::function<void(T&, OGRGeometry*)> op) {
  auto wkb = std::static_pointer_cast<arrow::BinaryArray>(array);
  auto len = array->length();
  T builder;
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkb(wkb, i);
    if (geo == nullptr) {
      builder.AppendNull();
    } else {
      op(builder, geo);
    }
    OGRGeometryFactory::destroyGeometry(geo);
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

template <typename T>
typename std::enable_if<std::is_base_of<arrow::ArrayBuilder, T>::value,
                        std::shared_ptr<typename arrow::Array>>::type
BinaryOp(const std::shared_ptr<arrow::Array>& geo1,
         const std::shared_ptr<arrow::Array>& geo2,
         std::function<void(T&, OGRGeometry*, OGRGeometry*)> op,
         std::function<void(T&, OGRGeometry*, OGRGeometry*)> null_op = nullptr) {
  auto len = geo1->length();
  auto wkt1 = std::static_pointer_cast<arrow::BinaryArray>(geo1);
  auto wkt2 = std::static_pointer_cast<arrow::BinaryArray>(geo2);
  T builder;
  for (int i = 0; i < len; ++i) {
    auto ogr1 = Wrapper_createFromWkb(wkt1, i);
    auto ogr2 = Wrapper_createFromWkb(wkt2, i);
    if ((ogr1 == nullptr) && (ogr2 == nullptr)) {
      builder.AppendNull();
    } else if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
      if (null_op == nullptr) {
        builder.AppendNull();
      } else {
        null_op(builder, ogr1, ogr2);
      }
    } else {
      op(builder, ogr1, ogr2);
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
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
  arrow::BinaryBuilder builder;

  for (int32_t i = 0; i < len; i++) {
    point.setX(x_double_values->Value(i));
    point.setY(y_double_values->Value(i));
    AppendWkbNDR(builder, &point);
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

  arrow::BinaryBuilder builder;
  OGRPolygon empty;
  auto empty_size = empty.WkbSize();
  auto empty_wkb = static_cast<unsigned char*>(CPLMalloc(empty_size));
  empty.exportToWkb(OGRwkbByteOrder::wkbNDR, empty_wkb);

  for (int32_t i = 0; i < len; i++) {
    if ((min_x_double_values->Value(i) > max_x_double_values->Value(i)) ||
        (min_y_double_values->Value(i) > max_y_double_values->Value(i))) {
      builder.Append(empty_wkb, empty_size);
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
      AppendWkbNDR(builder, &polygon);
    }
  }
  CPLFree(empty_wkb);
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_GeomFromGeoJSON(
    const std::shared_ptr<arrow::Array>& json) {
  auto json_geo = std::static_pointer_cast<arrow::StringArray>(json);
  int len = json_geo->length();
  arrow::BinaryBuilder builder;
  for (int i = 0; i < len; ++i) {
    if (json_geo->IsNull(i)) {
      builder.AppendNull();
    } else {
      auto str = json_geo->GetString(i);
      auto geo = (OGRGeometry*)OGR_G_CreateGeometryFromJson(str.c_str());
      if (geo != nullptr) {
        AppendWkbNDR(builder, geo);
        OGRGeometryFactory::destroyGeometry(geo);
      } else {
        builder.AppendNull();
      }
    }
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_GeomFromText(const std::shared_ptr<arrow::Array>& text) {
  auto geo = std::static_pointer_cast<arrow::StringArray>(text);
  auto len = geo->length();
  arrow::BinaryBuilder builder;
  for (int i = 0; i < len; ++i) {
    auto ogr = Wrapper_createFromWkt(geo, i);
    if (ogr == nullptr) {
      builder.AppendNull();
    } else {
      AppendWkbNDR(builder, ogr);
    }
    OGRGeometryFactory::destroyGeometry(ogr);
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_AsText(const std::shared_ptr<arrow::Array>& wkb) {
  auto op = [](arrow::StringBuilder& builder, OGRGeometry* geo) {
    char* str;
    auto err_code = geo->exportToWkt(&str);
    if (err_code != OGRERR_NONE) {
      builder.AppendNull();
    } else {
      builder.Append(std::string(str));
    }
    CPLFree(str);
  };
  return UnaryOp<arrow::StringBuilder>(wkb, op);
}

std::shared_ptr<arrow::Array> ST_AsGeoJSON(const std::shared_ptr<arrow::Array>& wkb) {
  auto op = [](arrow::StringBuilder& builder, OGRGeometry* geo) {
    char* str = geo->exportToJson();
    if (str == nullptr) {
      builder.AppendNull();
    } else {
      builder.Append(std::string(str));
      CPLFree(str);
    }
  };
  return UnaryOp<arrow::StringBuilder>(wkb, op);
}

/************************* GEOMETRY ACCESSOR **************************/
std::shared_ptr<arrow::Array> ST_IsValid(const std::shared_ptr<arrow::Array>& array) {
  auto op = [](arrow::BooleanBuilder& builder, OGRGeometry* geo) {
    builder.Append(geo->IsValid() != 0);
  };
  return UnaryOp<arrow::BooleanBuilder>(array, op);
}

std::shared_ptr<arrow::Array> ST_GeometryType(
    const std::shared_ptr<arrow::Array>& array) {
  auto op = [](arrow::StringBuilder& builder, OGRGeometry* geo) {
    std::string name = std::string("ST_") + geo->getGeometryName();
    builder.Append(name);
  };
  return UnaryOp<arrow::StringBuilder>(array, op);
}

std::shared_ptr<arrow::Array> ST_IsSimple(const std::shared_ptr<arrow::Array>& array) {
  auto has_circular = new HasCircularVisitor;
  const char* papszOptions[] = {(const char*)"ADD_INTERMEDIATE_POINT=YES", nullptr};
  auto op = [&has_circular, &papszOptions](arrow::BooleanBuilder& builder,
                                           OGRGeometry* geo) {
    has_circular->reset();
    geo->accept(has_circular);
    if (has_circular->has_circular()) {
      auto linear = geo->getLinearGeometry(0, papszOptions);
      builder.Append(linear->IsSimple() != 0);
      OGRGeometryFactory::destroyGeometry(linear);
    } else {
      builder.Append(geo->IsSimple() != 0);
    }
  };
  auto results = UnaryOp<arrow::BooleanBuilder>(array, op);
  delete has_circular;
  return results;
}

std::shared_ptr<arrow::Array> ST_NPoints(const std::shared_ptr<arrow::Array>& array) {
  auto npoints = new NPointsVisitor;
  auto op = [&npoints](arrow::Int64Builder& builder, OGRGeometry* geo) {
    npoints->reset();
    geo->accept(npoints);
    builder.Append(npoints->npoints());
  };
  auto results = UnaryOp<arrow::Int64Builder>(array, op);
  delete npoints;
  return results;
}

std::shared_ptr<arrow::Array> ST_Envelope(const std::shared_ptr<arrow::Array>& array) {
  OGREnvelope env;
  auto op = [&env](arrow::BinaryBuilder& builder, OGRGeometry* geo) {
    if (geo->IsEmpty()) {
      AppendWkbNDR(builder, geo);
    } else {
      OGR_G_GetEnvelope(geo, &env);
      if (env.MinX == env.MaxX) {    // vertical line or Point
        if (env.MinY == env.MaxY) {  // point
          OGRPoint point(env.MinX, env.MinY);
          AppendWkbNDR(builder, &point);
        } else {  // line
          OGRLineString line;
          line.addPoint(env.MinX, env.MinY);
          line.addPoint(env.MinX, env.MaxY);
          AppendWkbNDR(builder, &line);
        }
      } else {
        if (env.MinY == env.MaxY) {  // horizontal line
          OGRLineString line;
          line.addPoint(env.MinX, env.MinY);
          line.addPoint(env.MaxX, env.MinY);
          AppendWkbNDR(builder, &line);
        } else {  // polygon
          OGRLinearRing ring;
          ring.addPoint(env.MinX, env.MinY);
          ring.addPoint(env.MinX, env.MaxY);
          ring.addPoint(env.MaxX, env.MaxY);
          ring.addPoint(env.MaxX, env.MinY);
          ring.addPoint(env.MinX, env.MinY);
          OGRPolygon polygon;
          polygon.addRing(&ring);
          AppendWkbNDR(builder, &polygon);
        }
      }
    }
  };

  return UnaryOp<arrow::BinaryBuilder>(array, op);
}

/************************ GEOMETRY PROCESSING ************************/
std::shared_ptr<arrow::Array> ST_Buffer(const std::shared_ptr<arrow::Array>& array,
                                        double buffer_distance, int n_quadrant_segments) {
  auto op = [&buffer_distance, &n_quadrant_segments](arrow::BinaryBuilder& builder,
                                                     OGRGeometry* geo) {
    auto buffer = geo->Buffer(buffer_distance, n_quadrant_segments);
    AppendWkbNDR(builder, buffer);
    OGRGeometryFactory::destroyGeometry(buffer);
  };
  return UnaryOp<arrow::BinaryBuilder>(array, op);
}

std::shared_ptr<arrow::Array> ST_PrecisionReduce(
    const std::shared_ptr<arrow::Array>& geometries, int32_t precision) {
  auto precision_reduce_visitor = new PrecisionReduceVisitor(precision);
  auto op = [&precision_reduce_visitor](arrow::BinaryBuilder& builder, OGRGeometry* geo) {
    geo->accept(precision_reduce_visitor);
    AppendWkbNDR(builder, geo);
  };

  auto results = UnaryOp<arrow::BinaryBuilder>(geometries, op);
  delete precision_reduce_visitor;
  return results;
}

std::shared_ptr<arrow::Array> ST_Intersection(const std::shared_ptr<arrow::Array>& geo1,
                                              const std::shared_ptr<arrow::Array>& geo2) {
  auto wkt1 = std::static_pointer_cast<arrow::BinaryArray>(geo1);
  auto wkt2 = std::static_pointer_cast<arrow::BinaryArray>(geo2);
  auto len = wkt1->length();
  arrow::BinaryBuilder builder;
  auto has_curve = new HasCurveVisitor;

  OGRGeometryCollection empty;
  auto empty_size = empty.WkbSize();
  auto empty_wkb = static_cast<unsigned char*>(CPLMalloc(empty_size));
  empty.exportToWkb(OGRwkbByteOrder::wkbNDR, empty_wkb);

  for (int i = 0; i < len; ++i) {
    auto ogr1 = Wrapper_createFromWkb(wkt1, i);
    auto ogr2 = Wrapper_createFromWkb(wkt2, i);
    ogr1 = Wrapper_CurveToLine(ogr1, has_curve);
    ogr2 = Wrapper_CurveToLine(ogr2, has_curve);

    if ((ogr1 == nullptr) && (ogr2 == nullptr)) {
      builder.AppendNull();
    } else if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
      builder.Append(empty_wkb, empty_size);
    } else {
      auto rst = ogr1->Intersection(ogr2);
      if (rst == nullptr) {
        builder.AppendNull();
      } else if (rst->IsEmpty()) {
        builder.Append(empty_wkb, empty_size);
      } else {
        AppendWkbNDR(builder, rst);
      }
      OGRGeometryFactory::destroyGeometry(rst);
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }

  delete has_curve;
  CPLFree(empty_wkb);

  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_MakeValid(const std::shared_ptr<arrow::Array>& array) {
  auto wkb = std::static_pointer_cast<arrow::BinaryArray>(array);
  int len = wkb->length();
  arrow::BinaryBuilder builder;
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkb(wkb, i);
    if (geo == nullptr) {
      builder.AppendNull();
    } else {
      if (geo->IsValid()) {
        arrow::BinaryArray::offset_type offset;
        auto data_ptr = wkb->GetValue(i, &offset);
        builder.Append(data_ptr, offset);
      } else {
        auto make_valid = geo->MakeValid();
        AppendWkbNDR(builder, make_valid);
        OGRGeometryFactory::destroyGeometry(make_valid);
      }
    }
    OGRGeometryFactory::destroyGeometry(geo);
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_SimplifyPreserveTopology(
    const std::shared_ptr<arrow::Array>& array, double distance_tolerance) {
  auto op = [&distance_tolerance](arrow::BinaryBuilder& builder, OGRGeometry* geo) {
    auto simple = geo->SimplifyPreserveTopology(distance_tolerance);
    AppendWkbNDR(builder, simple);
    OGRGeometryFactory::destroyGeometry(simple);
  };
  return UnaryOp<arrow::BinaryBuilder>(array, op);
}

std::shared_ptr<arrow::Array> ST_Centroid(const std::shared_ptr<arrow::Array>& array) {
  OGRPoint centro_point;
  auto op = [&centro_point](arrow::BinaryBuilder& builder, OGRGeometry* geo) {
    auto err_code = geo->Centroid(&centro_point);
    if (err_code == OGRERR_NONE) {
      AppendWkbNDR(builder, &centro_point);
    } else {
      builder.AppendNull();
    }
  };
  return UnaryOp<arrow::BinaryBuilder>(array, op);
}

std::shared_ptr<arrow::Array> ST_ConvexHull(const std::shared_ptr<arrow::Array>& array) {
  auto op = [](arrow::BinaryBuilder& builder, OGRGeometry* geo) {
    auto cvx = geo->ConvexHull();
    AppendWkbNDR(builder, cvx);
    OGRGeometryFactory::destroyGeometry(cvx);
  };
  return UnaryOp<arrow::BinaryBuilder>(array, op);
}

/*
 * The detailed EPSG information can be found at EPSG.io [https://epsg.io/]
 */
std::shared_ptr<arrow::Array> ST_Transform(
    const std::shared_ptr<arrow::Array>& geometries, const std::string& src_rs,
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

  auto op = [&poCT](arrow::BinaryBuilder& builder, OGRGeometry* geo) {
    auto err_code = geo->transform((OGRCoordinateTransformation*)poCT);
    if (err_code == OGRERR_NONE) {
      AppendWkbNDR(builder, geo);
    } else {
      builder.AppendNull();
    }
  };
  auto results = UnaryOp<arrow::BinaryBuilder>(geometries, op);
  OCTDestroyCoordinateTransformation(poCT);
  return results;
}

std::shared_ptr<arrow::Array> ST_CurveToLine(
    const std::shared_ptr<arrow::Array>& geometries) {
  auto op = [](arrow::BinaryBuilder& builder, OGRGeometry* geo) {
    auto line = geo->getLinearGeometry();
    AppendWkbNDR(builder, line);
    OGRGeometryFactory::destroyGeometry(line);
  };
  return UnaryOp<arrow::BinaryBuilder>(geometries, op);
}

/************************ MEASUREMENT FUNCTIONS ************************/

std::shared_ptr<arrow::Array> ST_Area(const std::shared_ptr<arrow::Array>& geometries) {
  auto* area = new AreaVisitor;
  auto op = [&area](arrow::DoubleBuilder& builder, OGRGeometry* geo) {
    area->reset();
    geo->accept(area);
    builder.Append(area->area());
  };
  auto results = UnaryOp<arrow::DoubleBuilder>(geometries, op);
  delete area;
  return results;
}

std::shared_ptr<arrow::Array> ST_Length(const std::shared_ptr<arrow::Array>& geometries) {
  auto* len_sum = new LengthVisitor;
  auto op = [&len_sum](arrow::DoubleBuilder& builder, OGRGeometry* geo) {
    len_sum->reset();
    geo->accept(len_sum);
    builder.Append(len_sum->length());
  };
  auto results = UnaryOp<arrow::DoubleBuilder>(geometries, op);
  delete len_sum;
  return results;
}

std::shared_ptr<arrow::Array> ST_HausdorffDistance(
    const std::shared_ptr<arrow::Array>& geo1,
    const std::shared_ptr<arrow::Array>& geo2) {
  auto geos_ctx = OGRGeometry::createGEOSContext();
  auto op = [&geos_ctx](arrow::DoubleBuilder& builder, OGRGeometry* ogr1,
                        OGRGeometry* ogr2) {
    if (ogr1->IsEmpty() || ogr2->IsEmpty()) {
      builder.AppendNull();
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
      builder.Append(dist);
    }
  };
  auto results = BinaryOp<arrow::DoubleBuilder>(geo1, geo2, op);
  OGRGeometry::freeGEOSContext(geos_ctx);
  return results;
}

std::shared_ptr<arrow::Array> ST_Distance(const std::shared_ptr<arrow::Array>& geo1,
                                          const std::shared_ptr<arrow::Array>& geo2) {
  auto op = [](arrow::DoubleBuilder& builder, OGRGeometry* ogr1, OGRGeometry* ogr2) {
    if (ogr1->IsEmpty() || ogr2->IsEmpty()) {
      builder.AppendNull();
    } else {
      auto dist = ogr1->Distance(ogr2);
      if (dist < 0) {
        builder.AppendNull();
      } else {
        builder.Append(dist);
      }
    }
  };
  return BinaryOp<arrow::DoubleBuilder>(geo1, geo2, op);
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
  auto op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1, OGRGeometry* ogr2) {
    if (ogr1->IsEmpty() && ogr2->IsEmpty()) {
      builder.Append(true);
    } else if (ogr1->Within(ogr2) && ogr2->Within(ogr1)) {
      builder.Append(true);
    } else {
      builder.Append(false);
    }
  };
  auto null_op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { builder.Append(false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::shared_ptr<arrow::Array> ST_Touches(const std::shared_ptr<arrow::Array>& geo1,
                                         const std::shared_ptr<arrow::Array>& geo2) {
  auto op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1, OGRGeometry* ogr2) {
    builder.Append(ogr1->Touches(ogr2) != 0);
  };
  auto null_op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { builder.Append(false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::shared_ptr<arrow::Array> ST_Overlaps(const std::shared_ptr<arrow::Array>& geo1,
                                          const std::shared_ptr<arrow::Array>& geo2) {
  auto op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1, OGRGeometry* ogr2) {
    builder.Append(ogr1->Overlaps(ogr2) != 0);
  };
  auto null_op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { builder.Append(false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::shared_ptr<arrow::Array> ST_Crosses(const std::shared_ptr<arrow::Array>& geo1,
                                         const std::shared_ptr<arrow::Array>& geo2) {
  auto op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1, OGRGeometry* ogr2) {
    builder.Append(ogr1->Crosses(ogr2) != 0);
  };
  auto null_op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { builder.Append(false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::shared_ptr<arrow::Array> ST_Contains(const std::shared_ptr<arrow::Array>& geo1,
                                          const std::shared_ptr<arrow::Array>& geo2) {
  auto op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1, OGRGeometry* ogr2) {
    builder.Append(ogr1->Contains(ogr2) != 0);
  };
  auto null_op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { builder.Append(false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::shared_ptr<arrow::Array> ST_Intersects(const std::shared_ptr<arrow::Array>& geo1,
                                            const std::shared_ptr<arrow::Array>& geo2) {
  auto op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1, OGRGeometry* ogr2) {
    builder.Append(ogr1->Intersects(ogr2) != 0);
  };
  auto null_op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { builder.Append(false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::shared_ptr<arrow::Array> ST_Within(const std::shared_ptr<arrow::Array>& geo1,
                                        const std::shared_ptr<arrow::Array>& geo2) {
  auto op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1, OGRGeometry* ogr2) {
    bool flag = true;
    do {
      /*
       * speed up for point within circle
       * point pattern : 'POINT ( x y )'
       * circle pattern : 'CurvePolygon ( CircularString ( x1 y1, x2 y2, x1 y2 ) )'
       *                   if the circularstring has 3 points and closed,
       *                   it becomes a circle,
       *                   the centre is (x1+x2)/2, (y1+y2)/2
       *                   the radius is sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y2-y2))/2
       */
      auto type1 = ogr1->getGeometryType();
      if (type1 != wkbPoint) break;
      auto point = reinterpret_cast<OGRPoint*>(ogr1);

      auto type2 = ogr2->getGeometryType();
      if (type2 != wkbCurvePolygon) break;
      auto curve_poly = reinterpret_cast<OGRCurvePolygon*>(ogr2);

      auto curve_it = curve_poly->begin();
      if (curve_it == curve_poly->end()) break;
      auto curve = *curve_it;
      ++curve_it;
      if (curve_it != curve_poly->end()) break;

      auto curve_type = curve->getGeometryType();
      if (curve_type != wkbCircularString) break;
      auto circular_string = reinterpret_cast<OGRCircularString*>(curve);
      if (circular_string->getNumPoints() != 3) break;
      if (!circular_string->get_IsClosed()) break;

      auto circular_point_it = circular_string->begin();
      auto circular_point = &(*circular_point_it);
      if (circular_point->getGeometryType() != wkbPoint) break;
      auto p0_x = circular_point->getX();
      auto p0_y = circular_point->getY();

      ++circular_point_it;
      circular_point = &(*circular_point_it);
      if (circular_point->getGeometryType() != wkbPoint) break;
      auto p1_x = circular_point->getX();
      auto p1_y = circular_point->getY();

      auto d_x = (p0_x + p1_x) / 2 - point->getX();
      auto d_y = (p0_y + p1_y) / 2 - point->getY();
      auto dd = 4 * (d_x * d_x + d_y * d_y);
      auto l_x = p0_x - p1_x;
      auto l_y = p0_y - p1_y;
      auto ll = l_x * l_x + l_y * l_y;
      builder.Append(dd <= ll);

      flag = false;
    } while (0);
    if (flag) builder.Append(ogr1->Within(ogr2) != 0);
  };
  auto null_op = [](arrow::BooleanBuilder& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { builder.Append(false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

/*********************** AGGREGATE FUNCTIONS ***************************/

std::shared_ptr<arrow::Array> ST_Union_Aggr(const std::shared_ptr<arrow::Array>& geo) {
  auto len = geo->length();
  auto wkt = std::static_pointer_cast<arrow::BinaryArray>(geo);
  std::vector<OGRGeometry*> union_agg;
  OGRPolygon empty_polygon;
  OGRGeometry *g0, *g1;
  OGRGeometry *u0, *u1;
  auto has_curve = new HasCurveVisitor;
  for (int i = 0; i <= len / 2; i++) {
    if ((i * 2) < len) {
      g0 = Wrapper_createFromWkb(wkt, 2 * i);
      g0 = Wrapper_CurveToLine(g0, has_curve);
    } else {
      g0 = nullptr;
    }

    if ((i * 2 + 1) < len) {
      g1 = Wrapper_createFromWkb(wkt, 2 * i + 1);
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
  arrow::BinaryBuilder builder;
  if (union_agg.empty()) {
    builder.AppendNull();
  } else {
    AppendWkbNDR(builder, union_agg[0]);
    OGRGeometryFactory::destroyGeometry(union_agg[0]);
  }
  delete has_curve;
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_Envelope_Aggr(
    const std::shared_ptr<arrow::Array>& geometries) {
  auto wkt_geometries = std::static_pointer_cast<arrow::BinaryArray>(geometries);
  auto len = geometries->length();
  double inf = std::numeric_limits<double>::infinity();
  double xmin = inf;
  double xmax = -inf;
  double ymin = inf;
  double ymax = -inf;

  OGREnvelope env;
  bool set_env = false;
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkb(wkt_geometries, i);
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
  arrow::BinaryBuilder builder;
  OGRPolygon polygon;
  if (set_env) {
    OGRLinearRing ring;
    ring.addPoint(xmin, ymin);
    ring.addPoint(xmin, ymax);
    ring.addPoint(xmax, ymax);
    ring.addPoint(xmax, ymin);
    ring.addPoint(xmin, ymin);
    polygon.addRing(&ring);
  }
  AppendWkbNDR(builder, &polygon);
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
