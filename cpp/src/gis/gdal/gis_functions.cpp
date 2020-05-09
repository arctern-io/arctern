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

#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <functional>
#include <iomanip>
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

template <typename T>
struct ChunkArrayIdx {
  int chunk_idx = 0;
  int array_idx = 0;
  bool is_null = false;
  T item_value;
};

struct WkbItem {
  const void* data_ptr;
  int wkb_size;
  OGRGeometry* ToGeometry() {
    if (data_ptr == nullptr) return nullptr;
    if (wkb_size <= 0) return nullptr;
    OGRGeometry* geo = nullptr;
    auto err_code = OGRGeometryFactory::createFromWkb(data_ptr, nullptr, &geo, wkb_size);
    if (err_code != OGRERR_NONE) return nullptr;
    return geo;
  }
};

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
  arrow::BinaryArray::offset_type wkb_size;
  auto data_ptr = array->GetValue(idx, &wkb_size);
  if (wkb_size <= 0) return nullptr;

  OGRGeometry* geo = nullptr;
  auto err_code = OGRGeometryFactory::createFromWkb(data_ptr, nullptr, &geo, wkb_size);
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

// inline char* Wrapper_OGR_G_ExportToWkt(OGRGeometry* geo) {
//   char* str;
//   auto err_code = OGR_G_ExportToWkt(geo, &str);
//   if (err_code != OGRERR_NONE) {
//     std::string err_msg =
//         "failed to export to wkt, error code = " + std::to_string(err_code);
//     throw std::runtime_error(err_msg);
//   }
//   return str;
// }

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

template <typename T, typename Enable = void>
struct ChunkArrayBuilder {
  static constexpr int64_t CAPACITY = 1024 * 1024 * 1024;
};

template <typename T>
struct ChunkArrayBuilder<
    T, typename std::enable_if<std::is_base_of<arrow::ArrayBuilder, T>::value>::type> {
  T array_builder;
  int64_t array_size = 0;
};

inline std::shared_ptr<arrow::Array> AppendBoolean(
    ChunkArrayBuilder<arrow::BooleanBuilder>& builder, bool val) {
  std::shared_ptr<arrow::Array> array_ptr = nullptr;
  if (builder.array_size / 8 >= ChunkArrayBuilder<void>::CAPACITY) {
    CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
    builder.array_size = 0;
  }
  builder.array_builder.Append(val);
  ++builder.array_size;
  return array_ptr;
}

inline std::shared_ptr<arrow::Array> AppendDouble(
    ChunkArrayBuilder<arrow::DoubleBuilder>& builder, double val) {
  std::shared_ptr<arrow::Array> array_ptr = nullptr;
  if (builder.array_size + sizeof(val) > ChunkArrayBuilder<void>::CAPACITY) {
    CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
    builder.array_size = 0;
  }
  builder.array_builder.Append(val);
  builder.array_size += sizeof(val);
  return array_ptr;
}

inline std::shared_ptr<arrow::Array> AppendString(
    ChunkArrayBuilder<arrow::StringBuilder>& builder, std::string&& str_val) {
  std::shared_ptr<arrow::Array> array_ptr = nullptr;
  if (builder.array_size + str_val.size() > ChunkArrayBuilder<void>::CAPACITY) {
    CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
    builder.array_size = 0;
  }
  builder.array_size += str_val.size();
  CHECK_ARROW(builder.array_builder.Append(std::move(str_val)));
  return array_ptr;
}

inline std::shared_ptr<arrow::Array> AppendString(
    ChunkArrayBuilder<arrow::StringBuilder>& builder, const char* val) {
  if (val == nullptr) {
    builder.array_builder.AppendNull();
    return nullptr;
  } else {
    auto str_val = std::string(val);
    return AppendString(builder, std::move(str_val));
  }
}

inline std::shared_ptr<arrow::Array> AppendWkb(
    ChunkArrayBuilder<arrow::BinaryBuilder>& builder, const OGRGeometry* geo) {
  std::shared_ptr<arrow::Array> array_ptr = nullptr;
  if (geo == nullptr) {
    builder.array_builder.AppendNull();
  } else if (geo->IsEmpty() && (geo->getGeometryType() == wkbPoint)) {
    builder.array_builder.AppendNull();
  } else {
    auto wkb_size = geo->WkbSize();
    auto wkb = static_cast<unsigned char*>(CPLMalloc(wkb_size));
    auto err_code = geo->exportToWkb(OGRwkbByteOrder::wkbNDR, wkb);
    if (err_code != OGRERR_NONE) {
      builder.array_builder.AppendNull();
    } else {
      if (builder.array_size + wkb_size > ChunkArrayBuilder<void>::CAPACITY) {
        CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
        builder.array_size = 0;
      }
      CHECK_ARROW(builder.array_builder.Append(wkb, wkb_size));
      builder.array_size += wkb_size;
    }
    CPLFree(wkb);
  }
  return array_ptr;
}

bool GetNextValue(const std::vector<std::shared_ptr<arrow::Array>>& chunk_array,
                  ChunkArrayIdx<WkbItem>& idx) {
  if (idx.chunk_idx >= (int)chunk_array.size()) return false;
  int len = chunk_array[idx.chunk_idx]->length();
  if (idx.array_idx >= len) {
    idx.chunk_idx++;
    idx.array_idx = 0;
    return GetNextValue(chunk_array, idx);
  }
  if (chunk_array[idx.chunk_idx]->IsNull(idx.array_idx)) {
    idx.array_idx++;
    idx.is_null = true;
    idx.item_value.data_ptr = nullptr;
    idx.item_value.wkb_size = 0;
    return true;
  }
  auto binary_array =
      std::static_pointer_cast<arrow::BinaryArray>(chunk_array[idx.chunk_idx]);
  arrow::BinaryArray::offset_type wkb_size;
  auto data_ptr = binary_array->GetValue(idx.array_idx, &wkb_size);
  idx.item_value.data_ptr = data_ptr;
  idx.item_value.wkb_size = wkb_size;
  idx.array_idx++;
  idx.is_null = (idx.item_value.wkb_size > 0);
  return true;
}

bool GetNextValue(const std::vector<std::shared_ptr<arrow::Array>>& chunk_array,
                  ChunkArrayIdx<double>& idx) {
  if (idx.chunk_idx >= (int)chunk_array.size()) return false;
  int len = chunk_array[idx.chunk_idx]->length();
  if (idx.array_idx >= len) {
    idx.chunk_idx++;
    idx.array_idx = 0;
    return GetNextValue(chunk_array, idx);
  }
  if (chunk_array[idx.chunk_idx]->IsNull(idx.array_idx)) {
    idx.array_idx++;
    idx.is_null = true;
    return true;
  }
  auto double_array =
      std::static_pointer_cast<arrow::DoubleArray>(chunk_array[idx.chunk_idx]);
  idx.item_value = double_array->Value(idx.array_idx);
  idx.array_idx++;
  idx.is_null = false;
  return true;
}

template <typename T>
bool GetNextValue(std::vector<std::vector<std::shared_ptr<arrow::Array>>>& array_list,
                  std::vector<ChunkArrayIdx<T>>& idx_list, bool& is_null) {
  auto ret_val = GetNextValue(array_list[0], idx_list[0]);
  is_null = idx_list[0].is_null;

  for (int i = 1; i < array_list.size(); ++i) {
    auto cur_val = GetNextValue(array_list[i], idx_list[i]);
    if (cur_val != ret_val) {
      throw std::runtime_error("incorrect input data");
    }
    is_null |= idx_list[i].is_null;
  }
  return ret_val;
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
                        std::vector<std::shared_ptr<typename arrow::Array>>>::type
UnaryOp(const std::shared_ptr<arrow::Array>& array,
        std::function<std::shared_ptr<typename arrow::Array>(ChunkArrayBuilder<T>&,
                                                             OGRGeometry*)>
            op) {
  auto wkb = std::static_pointer_cast<arrow::BinaryArray>(array);
  auto len = array->length();
  ChunkArrayBuilder<T> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;

  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkb(wkb, i);
    if (geo == nullptr) {
      builder.array_builder.AppendNull();
    } else {
      auto array_ptr = op(builder, geo);
      if (array_ptr != nullptr) result_array.push_back(array_ptr);
    }
    OGRGeometryFactory::destroyGeometry(geo);
  }
  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

// template <typename T>
// typename std::enable_if<std::is_base_of<arrow::ArrayBuilder, T>::value,
//                         std::shared_ptr<typename arrow::Array>>::type
// BinaryOp(const std::shared_ptr<arrow::Array>& geo1,
//          const std::shared_ptr<arrow::Array>& geo2,
//          std::function<void(T&, OGRGeometry*, OGRGeometry*)> op,
//          std::function<void(T&, OGRGeometry*, OGRGeometry*)> null_op = nullptr) {
//   auto len = geo1->length();
//   auto wkt1 = std::static_pointer_cast<arrow::BinaryArray>(geo1);
//   auto wkt2 = std::static_pointer_cast<arrow::BinaryArray>(geo2);
//   T builder;
//   for (int i = 0; i < len; ++i) {
//     auto ogr1 = Wrapper_createFromWkb(wkt1, i);
//     auto ogr2 = Wrapper_createFromWkb(wkt2, i);
//     if ((ogr1 == nullptr) && (ogr2 == nullptr)) {
//       builder.AppendNull();
//     } else if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
//       if (null_op == nullptr) {
//         builder.AppendNull();
//       } else {
//         null_op(builder, ogr1, ogr2);
//       }
//     } else {
//       op(builder, ogr1, ogr2);
//     }
//     OGRGeometryFactory::destroyGeometry(ogr1);
//     OGRGeometryFactory::destroyGeometry(ogr2);
//   }
//   std::shared_ptr<arrow::Array> results;
//   CHECK_ARROW(builder.Finish(&results));
//   return results;
// }

template <typename T>
typename std::enable_if<std::is_base_of<arrow::ArrayBuilder, T>::value,
                        std::vector<std::shared_ptr<typename arrow::Array>>>::type
BinaryOp(const std::vector<std::shared_ptr<typename arrow::Array>>& geo1,
         const std::vector<std::shared_ptr<typename arrow::Array>>& geo2,
         std::function<std::shared_ptr<typename arrow::Array>(ChunkArrayBuilder<T>&,
                                                              OGRGeometry*, OGRGeometry*)>
             op,
         std::function<std::shared_ptr<typename arrow::Array>(ChunkArrayBuilder<T>&,
                                                              OGRGeometry*, OGRGeometry*)>
             null_op = nullptr) {
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> array_list{geo1, geo2};
  std::vector<ChunkArrayIdx<WkbItem>> idx_list(2);
  ChunkArrayBuilder<T> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;
  bool is_null;

  while (GetNextValue(array_list, idx_list, is_null)) {
    auto ogr1 = idx_list[0].item_value.ToGeometry();
    auto ogr2 = idx_list[1].item_value.ToGeometry();
    if ((ogr1 == nullptr) && (ogr2 == nullptr)) {
      builder.array_builder.AppendNull();
    } else if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
      if (null_op == nullptr) {
        builder.array_builder.AppendNull();
      } else {
        auto array_ptr = null_op(builder, ogr1, ogr2);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
      }
    } else {
      auto array_ptr = op(builder, ogr1, ogr2);
      if (array_ptr != nullptr) result_array.push_back(array_ptr);
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }

  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

/************************ GEOMETRY CONSTRUCTOR ************************/

std::vector<std::shared_ptr<arrow::Array>> ST_Point(
    const std::vector<std::shared_ptr<arrow::Array>>& x_values_raw,
    const std::vector<std::shared_ptr<arrow::Array>>& y_values_raw) {
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> array_list{x_values_raw,
                                                                     y_values_raw};
  std::vector<ChunkArrayIdx<double>> idx_list(2);

  OGRPoint point;
  ChunkArrayBuilder<arrow::BinaryBuilder> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;
  bool is_null;

  while (GetNextValue(array_list, idx_list, is_null)) {
    if (is_null) {
      builder.array_builder.AppendNull();
    } else {
      point.setX(idx_list[0].item_value);
      point.setY(idx_list[1].item_value);
      auto array_ptr = AppendWkb(builder, &point);
      if (array_ptr != nullptr) result_array.push_back(array_ptr);
    }
  }

  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

std::vector<std::shared_ptr<arrow::Array>> ST_PolygonFromEnvelope(
    const std::vector<std::shared_ptr<arrow::Array>>& min_x_values,
    const std::vector<std::shared_ptr<arrow::Array>>& min_y_values,
    const std::vector<std::shared_ptr<arrow::Array>>& max_x_values,
    const std::vector<std::shared_ptr<arrow::Array>>& max_y_values) {
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> array_list{
      min_x_values, min_y_values, max_x_values, max_y_values};
  std::vector<ChunkArrayIdx<double>> idx_list(4);
  ChunkArrayBuilder<arrow::BinaryBuilder> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;
  bool is_null;
  OGRPolygon empty;

  while (GetNextValue(array_list, idx_list, is_null)) {
    if (is_null) {
      builder.array_builder.AppendNull();
    } else {
      if ((idx_list[0].item_value > idx_list[2].item_value) ||
          (idx_list[1].item_value > idx_list[3].item_value)) {
        auto array_ptr = AppendWkb(builder, &empty);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
      } else {
        OGRLinearRing ring;
        ring.addPoint(idx_list[0].item_value, idx_list[1].item_value);
        ring.addPoint(idx_list[0].item_value, idx_list[3].item_value);
        ring.addPoint(idx_list[2].item_value, idx_list[3].item_value);
        ring.addPoint(idx_list[2].item_value, idx_list[1].item_value);
        ring.addPoint(idx_list[0].item_value, idx_list[1].item_value);
        ring.closeRings();
        OGRPolygon polygon;
        polygon.addRing(&ring);
        auto array_ptr = AppendWkb(builder, &polygon);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
      }
    }
  }
  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

std::vector<std::shared_ptr<arrow::Array>> ST_GeomFromGeoJSON(
    const std::shared_ptr<arrow::Array>& json) {
  auto json_geo = std::static_pointer_cast<arrow::StringArray>(json);
  int len = json_geo->length();
  ChunkArrayBuilder<arrow::BinaryBuilder> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;

  for (int i = 0; i < len; ++i) {
    if (json_geo->IsNull(i)) {
      builder.array_builder.AppendNull();
    } else {
      auto str = json_geo->GetString(i);
      auto geo = (OGRGeometry*)OGR_G_CreateGeometryFromJson(str.c_str());
      if (geo != nullptr) {
        auto array_ptr = AppendWkb(builder, geo);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
        OGRGeometryFactory::destroyGeometry(geo);
      } else {
        builder.array_builder.AppendNull();
      }
    }
  }
  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

std::vector<std::shared_ptr<arrow::Array>> ST_GeomFromText(
    const std::shared_ptr<arrow::Array>& text) {
  auto geo = std::static_pointer_cast<arrow::StringArray>(text);
  auto len = geo->length();
  ChunkArrayBuilder<arrow::BinaryBuilder> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;

  for (int i = 0; i < len; ++i) {
    auto ogr = Wrapper_createFromWkt(geo, i);
    if (ogr == nullptr) {
      builder.array_builder.AppendNull();
    } else {
      auto array_ptr = AppendWkb(builder, ogr);
      if (array_ptr != nullptr) result_array.push_back(array_ptr);
    }
    OGRGeometryFactory::destroyGeometry(ogr);
  }
  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

std::vector<std::shared_ptr<arrow::Array>> ST_AsText(
    const std::shared_ptr<arrow::Array>& wkb) {
  auto op = [](ChunkArrayBuilder<arrow::StringBuilder>& builder, OGRGeometry* geo) {
    char* str;
    auto err_code = geo->exportToWkt(&str);
    std::shared_ptr<arrow::Array> array_ptr = nullptr;
    if (err_code != OGRERR_NONE) {
      builder.array_builder.AppendNull();
    } else {
      array_ptr = AppendString(builder, str);
    }
    CPLFree(str);
    return array_ptr;
  };
  return UnaryOp<arrow::StringBuilder>(wkb, op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_AsGeoJSON(
    const std::shared_ptr<arrow::Array>& wkb) {
  auto op = [](ChunkArrayBuilder<arrow::StringBuilder>& builder, OGRGeometry* geo) {
    char* str = geo->exportToJson();
    std::shared_ptr<arrow::Array> array_ptr = nullptr;
    if (str == nullptr) {
      builder.array_builder.AppendNull();
    } else {
      array_ptr = AppendString(builder, str);
    }
    CPLFree(str);
    return array_ptr;
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
std::vector<std::shared_ptr<arrow::Array>> ST_Buffer(
    const std::shared_ptr<arrow::Array>& array, double buffer_distance,
    int n_quadrant_segments) {
  auto op = [&buffer_distance, &n_quadrant_segments](
                ChunkArrayBuilder<arrow::BinaryBuilder>& builder, OGRGeometry* geo) {
    auto buffer = geo->Buffer(buffer_distance, n_quadrant_segments);
    auto array_ptr = AppendWkb(builder, buffer);
    OGRGeometryFactory::destroyGeometry(buffer);
    return array_ptr;
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

std::vector<std::shared_ptr<arrow::Array>> ST_Intersection(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> array_list{geo1, geo2};
  std::vector<ChunkArrayIdx<WkbItem>> idx_list(2);
  ChunkArrayBuilder<arrow::BinaryBuilder> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;
  bool is_null;
  auto has_curve = new HasCurveVisitor;
  OGRGeometryCollection empty;

  while (GetNextValue(array_list, idx_list, is_null)) {
    auto ogr1 = idx_list[0].item_value.ToGeometry();
    auto ogr2 = idx_list[1].item_value.ToGeometry();

    ogr1 = Wrapper_CurveToLine(ogr1, has_curve);
    ogr2 = Wrapper_CurveToLine(ogr2, has_curve);

    if ((ogr1 == nullptr) && (ogr2 == nullptr)) {
      builder.array_builder.AppendNull();
    } else if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
      auto array_ptr = AppendWkb(builder, &empty);
      if (array_ptr != nullptr) result_array.push_back(array_ptr);
    } else {
      auto rst = ogr1->Intersection(ogr2);
      if (rst == nullptr) {
        builder.array_builder.AppendNull();
      } else if (rst->IsEmpty()) {
        auto array_ptr = AppendWkb(builder, &empty);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
      } else {
        auto array_ptr = AppendWkb(builder, rst);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
      }
      OGRGeometryFactory::destroyGeometry(rst);
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }

  delete has_curve;

  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
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

std::vector<std::shared_ptr<arrow::Array>> ST_CurveToLine(
    const std::shared_ptr<arrow::Array>& geometries) {
  auto op = [](ChunkArrayBuilder<arrow::BinaryBuilder>& builder, OGRGeometry* geo) {
    auto line = geo->getLinearGeometry();
    auto array_ptr = AppendWkb(builder, line);
    OGRGeometryFactory::destroyGeometry(line);
    return array_ptr;
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

std::vector<std::shared_ptr<arrow::Array>> ST_HausdorffDistance(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  auto geos_ctx = OGRGeometry::createGEOSContext();
  auto op = [&geos_ctx](ChunkArrayBuilder<arrow::DoubleBuilder>& builder,
                        OGRGeometry* ogr1, OGRGeometry* ogr2) {
    std::shared_ptr<arrow::Array> array_ptr = nullptr;
    if (ogr1->IsEmpty() || ogr2->IsEmpty()) {
      builder.array_builder.AppendNull();
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
      array_ptr = AppendDouble(builder, dist);
    }
    return array_ptr;
  };
  auto results = BinaryOp<arrow::DoubleBuilder>(geo1, geo2, op);
  OGRGeometry::freeGEOSContext(geos_ctx);
  return results;
}

std::vector<std::shared_ptr<arrow::Array>> ST_DistanceSphere(
    const std::vector<std::shared_ptr<arrow::Array>>& point_left,
    const std::vector<std::shared_ptr<arrow::Array>>& point_right) {
  auto distance = [](double fromlon, double fromlat, double tolon, double tolat) {
    double latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886;
    double longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886;
    double latitudeH = sin(latitudeArc * 0.5);
    latitudeH *= latitudeH;
    double lontitudeH = sin(longitudeArc * 0.5);
    lontitudeH *= lontitudeH;
    double tmp = cos(fromlat * 0.017453292519943295769236907684886) *
                 cos(tolat * 0.017453292519943295769236907684886);
    return 6372797.560856 * (2.0 * asin(sqrt(latitudeH + tmp * lontitudeH)));
  };

  auto op = [&distance](ChunkArrayBuilder<arrow::DoubleBuilder>& builder, OGRGeometry* g1,
                        OGRGeometry* g2) {
    std::shared_ptr<arrow::Array> array_ptr = nullptr;
    if ((g1->getGeometryType() != wkbPoint) || (g2->getGeometryType() != wkbPoint)) {
      builder.array_builder.AppendNull();
    } else {
      auto p1 = reinterpret_cast<OGRPoint*>(g1);
      auto p2 = reinterpret_cast<OGRPoint*>(g2);
      double fromlat = p1->getX();
      double fromlon = p1->getY();
      double tolat = p2->getX();
      double tolon = p2->getY();
      if ((fromlat > 180) || (fromlat < -180) || (fromlon > 90) || (fromlon < -90) ||
          (tolat > 180) || (tolat < -180) || (tolon > 90) || (tolon < -90)) {
        builder.array_builder.AppendNull();
      } else {
        array_ptr = AppendDouble(builder, distance(fromlat, fromlon, tolat, tolon));
      }
    }
    return array_ptr;
  };
  return BinaryOp<arrow::DoubleBuilder>(point_left, point_right, op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Distance(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  auto op = [](ChunkArrayBuilder<arrow::DoubleBuilder>& builder, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    std::shared_ptr<arrow::Array> array_ptr = nullptr;
    if (ogr1->IsEmpty() || ogr2->IsEmpty()) {
      builder.array_builder.AppendNull();
    } else {
      auto dist = ogr1->Distance(ogr2);
      if (dist < 0) {
        builder.array_builder.AppendNull();
      } else {
        array_ptr = AppendDouble(builder, dist);
      }
    }
    return array_ptr;
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

std::vector<std::shared_ptr<arrow::Array>> ST_Equals(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  auto op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    if (ogr1->IsEmpty() && ogr2->IsEmpty()) {
      return AppendBoolean(builder, true);
    } else if (ogr1->Within(ogr2) && ogr2->Within(ogr1)) {
      return AppendBoolean(builder, true);
    } else {
      return AppendBoolean(builder, false);
    }
  };
  auto null_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { return AppendBoolean(builder, false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Touches(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  auto op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    return AppendBoolean(builder, ogr1->Touches(ogr2) != 0);
  };
  auto null_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { return AppendBoolean(builder, false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Overlaps(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  auto op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    return AppendBoolean(builder, ogr1->Overlaps(ogr2) != 0);
  };
  auto null_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { return AppendBoolean(builder, false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Crosses(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  auto op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    return AppendBoolean(builder, ogr1->Crosses(ogr2) != 0);
  };
  auto null_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { return AppendBoolean(builder, false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Contains(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  auto op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    return AppendBoolean(builder, ogr1->Contains(ogr2) != 0);
  };
  auto null_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { return AppendBoolean(builder, false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Intersects(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  auto op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    return AppendBoolean(builder, ogr1->Intersects(ogr2) != 0);
  };
  auto null_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { return AppendBoolean(builder, false); };
  return BinaryOp<arrow::BooleanBuilder>(geo1, geo2, op, null_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Within(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  auto op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    bool flag = true;
    std::shared_ptr<arrow::Array> ret_ptr = nullptr;
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
      ret_ptr = AppendBoolean(builder, dd <= ll);

      flag = false;
    } while (0);
    if (flag) ret_ptr = AppendBoolean(builder, ogr1->Within(ogr2) != 0);
    return ret_ptr;
  };
  auto null_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { return AppendBoolean(builder, false); };
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
