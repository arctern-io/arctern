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
#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "render/render_builder.h"

#include "arrow/render_api.h"

namespace arctern {
namespace render {

std::shared_ptr<arrow::Array> out_pic(std::pair<uint8_t*, int64_t> output) {
  if (output.first == nullptr || output.second < 0) {
    std::string err_msg =
        "Null image buffer, in most cases, it was caused by incorrect vega json";
    throw std::runtime_error(err_msg);
  }

  auto output_length = output.second;
  auto output_data = output.first;
  auto bit_map = (uint8_t*)malloc(output_length);
  memset(bit_map, 0xff, output_length);

  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, output_length);
  auto buffer1 = std::make_shared<arrow::Buffer>(output_data, output_length);
  auto buffers = std::vector<std::shared_ptr<arrow::Buffer>>();
  buffers.emplace_back(buffer0);
  buffers.emplace_back(buffer1);

  auto data_type = arrow::uint8();
  auto array_data = arrow::ArrayData::Make(data_type, output_length, buffers);
  auto array = arrow::MakeArray(array_data);
  return array;
}

std::shared_ptr<arrow::Array> WktToWkb(const std::shared_ptr<arrow::Array>& arr_wkt) {
  auto wkts = std::static_pointer_cast<arrow::StringArray>(arr_wkt);
  auto wkt_size = arr_wkt->length();
  auto wkt_type = arr_wkt->type_id();
  assert(wkt_type == arrow::Type::STRING);

  arrow::BinaryBuilder builder;
  for (int i = 0; i < wkt_size; i++) {
    auto wkt = wkts->GetString(i);
    OGRGeometry* geo = nullptr;
    CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt.c_str(), nullptr, &geo));
    auto sz = geo->WkbSize();
    std::vector<char> wkb(sz);
    CHECK_GDAL(geo->exportToWkb(OGRwkbByteOrder::wkbNDR, (uint8_t*)wkb.data()));
    OGRGeometryFactory::destroyGeometry(geo);
    auto st = builder.Append(wkb.data(), wkb.size());
    assert(st.ok());
  }
  std::shared_ptr<arrow::Array> result;
  auto st = builder.Finish(&result);
  assert(st.ok());
  return result;
}

std::shared_ptr<arrow::Array> WkbToWkt(const std::shared_ptr<arrow::Array>& arr_wkb) {
  auto wkbs = std::static_pointer_cast<arrow::BinaryArray>(arr_wkb);
  auto wkb_size = arr_wkb->length();
  auto wkb_type = arr_wkb->type_id();
  assert(wkb_type == arrow::Type::BINARY);

  arrow::StringBuilder builder;
  for (int i = 0; i < wkb_size; i++) {
    auto wkb = wkbs->GetString(i);
    OGRGeometry* geo = nullptr;
    CHECK_GDAL(OGRGeometryFactory::createFromWkb(wkb.c_str(), nullptr, &geo));
    char* str;
    CHECK_GDAL(geo->exportToWkt(&str));
    OGRGeometryFactory::destroyGeometry(geo);
    auto st = builder.Append(std::string(str));
    assert(st.ok());
    free(str);
  }
  std::shared_ptr<arrow::Array> result;
  auto st = builder.Finish(&result);
  assert(st.ok());
  return result;
}

template <typename T>
std::pair<uint8_t*, int64_t> render_weighted_pointmap(
    const std::shared_ptr<arrow::Array>& points, const std::shared_ptr<arrow::Array>& arr,
    const std::string& conf) {
  auto data = weight_agg<T>(points, arr);
  auto num_point = data.first.size();

  std::vector<uint32_t> input_x(num_point);
  std::vector<uint32_t> input_y(num_point);
  std::vector<T> input_c(num_point);

  rapidjson::Document document;
  document.Parse(conf.c_str());
  rapidjson::Value mark_enter;
  mark_enter = document["marks"][0]["encode"]["enter"];
  auto agg = mark_enter["aggregation_type"]["value"].GetString();
  AggType type_agg = agg_type(agg);

  const auto& result_wkb = data.first;
  const auto& result_weight = data.second;

  switch (type_agg) {
    case AggType::MAX: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *max_element(result_weight[i].begin(), result_weight[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::MIN: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *min_element(result_weight[i].begin(), result_weight[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::COUNT: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = result_weight[i].size();
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::SUM: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = accumulate(result_weight[i].begin(), result_weight[i].end(), 0);
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::STDDEV: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        T sum = accumulate(result_weight[i].begin(), result_weight[i].end(), 0);
        T mean = sum / result_weight[i].size();
        T accum = 0;
        std::for_each(std::begin(result_weight[i]), std::end(result_weight[i]),
                      [&](const T d) { accum += (d - mean) * (d - mean); });
        input_c[i] = sqrt(accum / result_weight[i].size());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::AVG: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        T sum_data = accumulate(result_weight[i].begin(), result_weight[i].end(), 0);
        input_c[i] = sum_data / result_weight[i].size();
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
  }

  return weighted_pointmap<T>(&input_x[0], &input_y[0], &input_c[0], num_point, conf);
}

template <typename T>
std::pair<uint8_t*, int64_t> render_weighted_pointmap(
    const std::shared_ptr<arrow::Array>& points,
    const std::shared_ptr<arrow::Array>& arr_c,
    const std::shared_ptr<arrow::Array>& arr_s, const std::string& conf) {
  auto agg_res = weight_agg_multiple_column<T>(points, arr_c, arr_s);
  auto num_point = std::get<0>(agg_res).size();

  std::vector<uint32_t> input_x(num_point);
  std::vector<uint32_t> input_y(num_point);
  std::vector<T> input_c(num_point);
  std::vector<T> input_s(num_point);

  rapidjson::Document document;
  document.Parse(conf.c_str());
  rapidjson::Value mark_enter;
  mark_enter = document["marks"][0]["encode"]["enter"];
  auto agg = mark_enter["aggregation_type"]["value"].GetString();
  AggType type_agg = agg_type(agg);

  const auto& result_wkb = std::get<0>(agg_res);
  const auto& result_c = std::get<1>(agg_res);
  const auto& result_s = std::get<2>(agg_res);

  switch (type_agg) {
    case AggType::MAX: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *max_element(result_c[i].begin(), result_c[i].end());
        input_s[i] = *max_element(result_s[i].begin(), result_s[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::MIN: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *min_element(result_c[i].begin(), result_c[i].end());
        input_s[i] = *min_element(result_c[i].begin(), result_c[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::COUNT: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = result_c[i].size();
        input_s[i] = result_c[i].size();
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::SUM: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = accumulate(result_c[i].begin(), result_c[i].end(), 0);
        input_s[i] = accumulate(result_c[i].begin(), result_c[i].end(), 0);
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::STDDEV: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();

        T sum_c = accumulate(result_c[i].begin(), result_c[i].end(), 0);
        T mean_c = sum_c / result_c[i].size();
        T accum_c = 0;
        std::for_each(std::begin(result_c[i]), std::end(result_c[i]),
                      [&](const T d) { accum_c += (d - mean_c) * (d - mean_c); });
        input_c[i] = sqrt(accum_c / result_c[i].size());

        T sum_s = accumulate(result_c[i].begin(), result_c[i].end(), 0);
        T mean_s = sum_s / result_c[i].size();
        T accum_s = 0;
        std::for_each(std::begin(result_c[i]), std::end(result_c[i]),
                      [&](const T d) { accum_s += (d - mean_s) * (d - mean_s); });
        input_s[i] = sqrt(accum_s / result_c[i].size());

        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::AVG: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        T sum_data_c = accumulate(result_c[i].begin(), result_c[i].end(), 0);
        T sum_data_s = accumulate(result_c[i].begin(), result_c[i].end(), 0);
        input_c[i] = sum_data_c / result_c[i].size();
        input_s[i] = sum_data_s / result_c[i].size();
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
  }

  return weighted_pointmap<T>(&input_x[0], &input_y[0], &input_c[0], &input_s[0],
                              num_point, conf);
}

template <typename T>
std::pair<uint8_t*, int64_t> render_heatmap(const std::shared_ptr<arrow::Array>& points,
                                            const std::shared_ptr<arrow::Array>& arr_c,
                                            const std::string& conf) {
  auto data = weight_agg<T>(points, arr_c);
  auto num_point = data.first.size();

  std::vector<uint32_t> input_x(num_point);
  std::vector<uint32_t> input_y(num_point);
  std::vector<T> input_c(num_point);

  rapidjson::Document document;
  document.Parse(conf.c_str());
  rapidjson::Value mark_enter;
  mark_enter = document["marks"][0]["encode"]["enter"];
  auto agg = mark_enter["aggregation_type"]["value"].GetString();
  AggType type_agg = agg_type(agg);

  const auto& result_wkb = data.first;
  const auto& result_weight = data.second;

  switch (type_agg) {
    case AggType::MAX: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *max_element(result_weight[i].begin(), result_weight[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::MIN: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *min_element(result_weight[i].begin(), result_weight[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::COUNT: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = result_weight[i].size();
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::SUM: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = accumulate(result_weight[i].begin(), result_weight[i].end(), 0);
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::STDDEV: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        T sum = accumulate(result_weight[i].begin(), result_weight[i].end(), 0);
        T mean = sum / result_weight[i].size();
        T accum = 0;
        std::for_each(std::begin(result_weight[i]), std::end(result_weight[i]),
                      [&](const T d) { accum += (d - mean) * (d - mean); });
        input_c[i] = sqrt(accum / result_weight[i].size());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggType::AVG: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        T sum_data = accumulate(result_weight[i].begin(), result_weight[i].end(), 0);
        input_c[i] = sum_data / result_weight[i].size();
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
  }

  return heatmap<T>(&input_x[0], &input_y[0], &input_c[0], num_point, conf);
}

template <typename T>
std::pair<uint8_t*, int64_t> render_choroplethmap(
    const std::shared_ptr<arrow::Array>& arr_wkb,
    const std::shared_ptr<arrow::Array>& arr_c, const std::string& conf) {
  auto data = weight_agg<T>(arr_wkb, arr_c);
  auto num_geo = data.second.size();

  std::vector<T> input_c(num_geo);

  rapidjson::Document document;
  document.Parse(conf.c_str());
  rapidjson::Value mark_enter;
  mark_enter = document["marks"][0]["encode"]["enter"];
  auto agg = mark_enter["aggregation_type"]["value"].GetString();
  AggType type_agg = agg_type(agg);

  const auto& result_weight = data.second;
  std::size_t i = 0;

  switch (type_agg) {
    case AggType::MAX: {
      for (auto& item : result_weight) {
        input_c[i++] = *max_element(item.begin(), item.end());
      }
      break;
    }
    case AggType::MIN: {
      for (auto& item : result_weight) {
        input_c[i++] = *min_element(item.begin(), item.end());
      }
      break;
    }
    case AggType::COUNT: {
      for (auto& item : result_weight) {
        input_c[i++] = item.size();
      }
      break;
    }
    case AggType::SUM: {
      for (auto& item : result_weight) {
        input_c[i++] = accumulate(item.begin(), item.end(), 0);
      }
      break;
    }
    case AggType::STDDEV: {
      for (auto& item : result_weight) {
        T sum = accumulate(item.begin(), item.end(), 0);
        T mean = sum / item.size();
        T accum = 0;
        std::for_each(std::begin(item), std::end(item),
                      [&](const T d) { accum += (d - mean) * (d - mean); });
        input_c[i++] = sqrt(accum / item.size());
      }
      break;
    }
    case AggType::AVG: {
      for (auto& item : result_weight) {
        T sum_data = accumulate(item.begin(), item.end(), 0);
        input_c[i++] = sum_data / item.size();
      }
      break;
    }
  }

  return choroplethmap<T>(data.first, &input_c[0], num_geo, conf);
}

std::shared_ptr<arrow::Array> projection(const std::shared_ptr<arrow::Array>& geos,
                                         const std::string& bottom_right,
                                         const std::string& top_left, const int& height,
                                         const int& width) {
  return Projection(geos, bottom_right, top_left, height, width);
}

std::shared_ptr<arrow::Array> transform_and_projection(
    const std::shared_ptr<arrow::Array>& geos, const std::string& src_rs,
    const std::string& dst_rs, const std::string& bottom_right,
    const std::string& top_left, const int& height, const int& width) {
  return TransformAndProjection(geos, src_rs, dst_rs, bottom_right, top_left, height,
                                width);
}

std::shared_ptr<arrow::Array> point_map(const std::shared_ptr<arrow::Array>& points,
                                        const std::string& conf) {
  auto point_arr = std::static_pointer_cast<arrow::BinaryArray>(points);
  auto num_point = points->length();
  auto wkb_type = points->type_id();
  assert(wkb_type == arrow::Type::BINARY);

  std::vector<uint32_t> input_x(num_point);
  std::vector<uint32_t> input_y(num_point);
  OGRGeometry* res_geo = nullptr;
  for (size_t i = 0; i < num_point; i++) {
    std::string geo_wkb = point_arr->GetString(i);
    CHECK_GDAL(OGRGeometryFactory::createFromWkb(geo_wkb.c_str(), nullptr, &res_geo));
    auto rs_pointer = reinterpret_cast<OGRPoint*>(res_geo);
    input_x[i] = rs_pointer->getX();
    input_y[i] = rs_pointer->getY();
  }
  auto result = pointmap(&input_x[0], &input_y[0], num_point, conf);

  return out_pic(result);
}

std::shared_ptr<arrow::Array> point_map(const std::shared_ptr<arrow::Array>& arr_x,
                                        const std::shared_ptr<arrow::Array>& arr_y,
                                        const std::string& conf) {
  auto x_length = arr_x->length();
  auto y_length = arr_y->length();
  auto x_type = arr_x->type_id();
  auto y_type = arr_y->type_id();
  assert(x_length == y_length);
  assert(x_type == arrow::Type::UINT32);
  assert(y_type == arrow::Type::UINT32);

  auto input_x = (uint32_t*)arr_x->data()->GetValues<uint8_t>(1);
  auto input_y = (uint32_t*)arr_y->data()->GetValues<uint8_t>(1);

  return out_pic(pointmap(input_x, input_y, x_length, conf));
}

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::shared_ptr<arrow::Array>& arr1, const std::string& conf) {
  auto wkt_type = arr1->type_id();
  assert(wkt_type == arrow::Type::BINARY);

  auto point_arr = std::static_pointer_cast<arrow::BinaryArray>(arr1);
  auto num_point = arr1->length();

  std::vector<uint32_t> input_x(num_point);
  std::vector<uint32_t> input_y(num_point);

  OGRGeometry* res_geo = nullptr;
  for (size_t i = 0; i < num_point; i++) {
    std::string geo_wkb = point_arr->GetString(i);
    CHECK_GDAL(OGRGeometryFactory::createFromWkb(geo_wkb.c_str(), nullptr, &res_geo));
    auto rst_pointer = reinterpret_cast<OGRPoint*>(res_geo);
    input_x[i] = (uint32_t)rst_pointer->getX();
    input_y[i] = (uint32_t)rst_pointer->getY();
  }

  auto result = weighted_pointmap<int8_t>(&input_x[0], &input_y[0], num_point, conf);

  return out_pic(result);
}

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::shared_ptr<arrow::Array>& arr1, const std::shared_ptr<arrow::Array>& arr2,
    const std::string& conf) {
  auto type1 = arr1->type_id();
  auto type2 = arr2->type_id();
  auto length1 = arr1->length();
  auto length2 = arr2->length();

  assert(length1 == length2);

  if (type1 == arrow::Type::UINT32 && type2 == arrow::Type::UINT32) {
    auto input_x = (uint32_t*)arr1->data()->GetValues<uint8_t>(1);
    auto input_y = (uint32_t*)arr2->data()->GetValues<uint8_t>(1);

    return out_pic(weighted_pointmap<int8_t>(input_x, input_y, length1, conf));
  } else if (type1 == arrow::Type::BINARY) {
    std::pair<uint8_t*, int64_t> result;
    switch (type2) {
      case arrow::Type::INT8: {
        result = render_weighted_pointmap<int8_t>(arr1, arr2, conf);
        break;
      }
      case arrow::Type::INT16: {
        result = render_weighted_pointmap<int16_t>(arr1, arr2, conf);
        break;
      }
      case arrow::Type::INT32: {
        result = render_weighted_pointmap<int32_t>(arr1, arr2, conf);
        break;
      }
      case arrow::Type::INT64: {
        result = render_weighted_pointmap<int64_t>(arr1, arr2, conf);
        break;
      }
      case arrow::Type::UINT8: {
        result = render_weighted_pointmap<uint8_t>(arr1, arr2, conf);
        break;
      }
      case arrow::Type::UINT16: {
        result = render_weighted_pointmap<uint16_t>(arr1, arr2, conf);
        break;
      }
      case arrow::Type::UINT32: {
        result = render_weighted_pointmap<uint32_t>(arr1, arr2, conf);
        break;
      }
      case arrow::Type::UINT64: {
        result = render_weighted_pointmap<uint64_t>(arr1, arr2, conf);
        break;
      }
      case arrow::Type::FLOAT: {
        result = render_weighted_pointmap<float>(arr1, arr2, conf);
        break;
      }
      case arrow::Type::DOUBLE: {
        result = render_weighted_pointmap<double>(arr1, arr2, conf);
        break;
      }
      default:
        std::string err_msg =
            "type error of count while running weighted_point map, type = " +
            std::to_string(type2);
        throw std::runtime_error(err_msg);
    }

    return out_pic(result);

  } else {
    std::string err_msg =
        "type error of arrow::Array while running weighted_point map, type = " +
        std::to_string(type1);
    throw std::runtime_error(err_msg);
  }
}

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::shared_ptr<arrow::Array>& arr1, const std::shared_ptr<arrow::Array>& arr2,
    const std::shared_ptr<arrow::Array>& arr3, const std::string& conf) {
  auto length1 = arr1->length();
  auto length2 = arr2->length();
  auto length3 = arr3->length();

  auto type1 = arr1->type_id();
  auto type2 = arr2->type_id();
  auto type3 = arr3->type_id();

  assert(length1 == length2);
  assert(length2 == length3);

  if (type1 == arrow::Type::UINT32 && type2 == arrow::Type::UINT32) {
    auto input_x = (uint32_t*)arr1->data()->GetValues<uint8_t>(1);
    auto input_y = (uint32_t*)arr2->data()->GetValues<uint8_t>(1);

    switch (type3) {
      case arrow::Type::INT8: {
        auto input = (int8_t*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(weighted_pointmap<int8_t>(input_x, input_y, input, length1, conf));
      }
      case arrow::Type::INT16: {
        auto input = (int16_t*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(
            weighted_pointmap<int16_t>(input_x, input_y, input, length1, conf));
      }
      case arrow::Type::INT32: {
        auto input = (int32_t*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(
            weighted_pointmap<int32_t>(input_x, input_y, input, length1, conf));
      }
      case arrow::Type::INT64: {
        auto input = (int64_t*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(
            weighted_pointmap<int64_t>(input_x, input_y, input, length1, conf));
      }
      case arrow::Type::UINT8: {
        auto input = (uint8_t*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(
            weighted_pointmap<uint8_t>(input_x, input_y, input, length1, conf));
      }
      case arrow::Type::UINT16: {
        auto input = (uint16_t*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(
            weighted_pointmap<uint16_t>(input_x, input_y, input, length1, conf));
      }
      case arrow::Type::UINT32: {
        auto input = (uint32_t*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(
            weighted_pointmap<uint32_t>(input_x, input_y, input, length1, conf));
      }
      case arrow::Type::UINT64: {
        auto input = (uint64_t*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(
            weighted_pointmap<uint64_t>(input_x, input_y, input, length1, conf));
      }
      case arrow::Type::FLOAT: {
        auto input = (float*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(weighted_pointmap<float>(input_x, input_y, input, length1, conf));
      }
      case arrow::Type::DOUBLE: {
        auto input = (double*)arr3->data()->GetValues<uint8_t>(1);
        return out_pic(weighted_pointmap<double>(input_x, input_y, input, length1, conf));
      }
      default:
        std::string err_msg =
            "type error of count while running weighted_point map, type = " +
            std::to_string(type3);
        throw std::runtime_error(err_msg);
    }
  } else if (type1 == arrow::Type::BINARY && type2 == type3) {
    std::pair<uint8_t*, int64_t> result;

    switch (type3) {
      case arrow::Type::INT8: {
        result = render_weighted_pointmap<int8_t>(arr1, arr2, arr3, conf);
        break;
      }
      case arrow::Type::INT16: {
        result = render_weighted_pointmap<int16_t>(arr1, arr2, arr3, conf);
        break;
      }
      case arrow::Type::INT32: {
        result = render_weighted_pointmap<int32_t>(arr1, arr2, arr3, conf);
        break;
      }
      case arrow::Type::INT64: {
        result = render_weighted_pointmap<int64_t>(arr1, arr2, arr3, conf);
        break;
      }
      case arrow::Type::UINT8: {
        result = render_weighted_pointmap<uint8_t>(arr1, arr2, arr3, conf);
        break;
      }
      case arrow::Type::UINT16: {
        result = render_weighted_pointmap<uint16_t>(arr1, arr2, arr3, conf);
        break;
      }
      case arrow::Type::UINT32: {
        result = render_weighted_pointmap<uint32_t>(arr1, arr2, arr3, conf);
        break;
      }
      case arrow::Type::UINT64: {
        result = render_weighted_pointmap<uint64_t>(arr1, arr2, arr3, conf);
        break;
      }
      case arrow::Type::FLOAT: {
        result = render_weighted_pointmap<float>(arr1, arr2, arr3, conf);
        break;
      }
      case arrow::Type::DOUBLE: {
        result = render_weighted_pointmap<double>(arr1, arr2, arr3, conf);
        break;
      }
      default:
        std::string err_msg =
            "type error of count while running weighted_point map, type = " +
            std::to_string(type3);
        throw std::runtime_error(err_msg);
    }
    return out_pic(result);
  } else {
    std::string err_msg =
        "type error of arrow::Array while running weighted_point map, type = " +
        std::to_string(type1);
    throw std::runtime_error(err_msg);
  }
}

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::shared_ptr<arrow::Array>& arr_x,
    const std::shared_ptr<arrow::Array>& arr_y,
    const std::shared_ptr<arrow::Array>& arr_c,
    const std::shared_ptr<arrow::Array>& arr_s, const std::string& conf) {
  auto x_length = arr_x->length();
  auto y_length = arr_y->length();
  auto c_length = arr_c->length();
  auto s_length = arr_c->length();

  auto x_type = arr_x->type_id();
  auto y_type = arr_y->type_id();
  auto c_type = arr_c->type_id();
  auto s_type = arr_c->type_id();

  assert(x_length == y_length);
  assert(x_length == c_length);
  assert(c_length == s_length);
  assert(x_type == arrow::Type::UINT32);
  assert(y_type == arrow::Type::UINT32);
  assert(c_type == s_type);

  auto input_x = (uint32_t*)arr_x->data()->GetValues<uint8_t>(1);
  auto input_y = (uint32_t*)arr_y->data()->GetValues<uint8_t>(1);

  switch (c_type) {
    case arrow::Type::INT8: {
      auto input_c = (int8_t*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (int8_t*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(
          weighted_pointmap<int8_t>(input_x, input_y, input_c, input_s, x_length, conf));
    }
    case arrow::Type::INT16: {
      auto input_c = (int16_t*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (int16_t*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(
          weighted_pointmap<int16_t>(input_x, input_y, input_c, input_s, x_length, conf));
    }
    case arrow::Type::INT32: {
      auto input_c = (int32_t*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (int32_t*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(
          weighted_pointmap<int32_t>(input_x, input_y, input_c, input_s, x_length, conf));
    }
    case arrow::Type::INT64: {
      auto input_c = (int64_t*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (int64_t*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(
          weighted_pointmap<int64_t>(input_x, input_y, input_c, input_s, x_length, conf));
    }
    case arrow::Type::UINT8: {
      auto input_c = (uint8_t*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (uint8_t*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(
          weighted_pointmap<uint8_t>(input_x, input_y, input_c, input_s, x_length, conf));
    }
    case arrow::Type::UINT16: {
      auto input_c = (uint16_t*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (uint16_t*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(weighted_pointmap<uint16_t>(input_x, input_y, input_c, input_s,
                                                 x_length, conf));
    }
    case arrow::Type::UINT32: {
      auto input_c = (uint32_t*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (uint32_t*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(weighted_pointmap<uint32_t>(input_x, input_y, input_c, input_s,
                                                 x_length, conf));
    }
    case arrow::Type::UINT64: {
      auto input_c = (uint64_t*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (uint64_t*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(weighted_pointmap<uint64_t>(input_x, input_y, input_c, input_s,
                                                 x_length, conf));
    }
    case arrow::Type::FLOAT: {
      auto input_c = (float*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (float*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(
          weighted_pointmap<float>(input_x, input_y, input_c, input_s, x_length, conf));
    }
    case arrow::Type::DOUBLE: {
      auto input_c = (double*)arr_c->data()->GetValues<uint8_t>(1);
      auto input_s = (double*)arr_s->data()->GetValues<uint8_t>(1);
      return out_pic(
          weighted_pointmap<double>(input_x, input_y, input_c, input_s, x_length, conf));
    }
    default:
      std::string err_msg =
          "type error of count while running weighted_point map, type = " +
          std::to_string(c_type);
      throw std::runtime_error(err_msg);
  }
}

std::shared_ptr<arrow::Array> heat_map(const std::shared_ptr<arrow::Array>& points,
                                       const std::shared_ptr<arrow::Array>& arr_c,
                                       const std::string& conf) {
  auto points_arr = std::static_pointer_cast<arrow::BinaryArray>(points);
  auto wkb_type = points->type_id();
  assert(wkb_type == arrow::Type::BINARY);

  std::pair<uint8_t*, int64_t> result;
  auto c_type = arr_c->type_id();
  switch (c_type) {
    case arrow::Type::INT8: {
      result = render_heatmap<int8_t>(points, arr_c, conf);
      break;
    }
    case arrow::Type::INT16: {
      result = render_heatmap<int16_t>(points, arr_c, conf);
      break;
    }
    case arrow::Type::INT32: {
      result = render_heatmap<int32_t>(points, arr_c, conf);
      break;
    }
    case arrow::Type::INT64: {
      result = render_heatmap<int64_t>(points, arr_c, conf);
      break;
    }
    case arrow::Type::UINT8: {
      result = render_heatmap<uint8_t>(points, arr_c, conf);
      break;
    }
    case arrow::Type::UINT16: {
      result = render_heatmap<uint16_t>(points, arr_c, conf);
      break;
    }
    case arrow::Type::UINT32: {
      result = render_heatmap<uint32_t>(points, arr_c, conf);
      break;
    }
    case arrow::Type::UINT64: {
      result = render_heatmap<uint64_t>(points, arr_c, conf);
      break;
    }
    case arrow::Type::FLOAT: {
      result = render_heatmap<float>(points, arr_c, conf);
      break;
    }
    case arrow::Type::DOUBLE: {
      result = render_heatmap<double>(points, arr_c, conf);
      break;
    }
    default:
      std::string err_msg =
          "type error of count while running heat map, type = " + std::to_string(c_type);
      throw std::runtime_error(err_msg);
  }

  return out_pic(result);
}

std::shared_ptr<arrow::Array> heat_map(const std::shared_ptr<arrow::Array>& arr_x,
                                       const std::shared_ptr<arrow::Array>& arr_y,
                                       const std::shared_ptr<arrow::Array>& arr_c,
                                       const std::string& conf) {
  auto x_length = arr_x->length();
  auto y_length = arr_y->length();
  auto c_length = arr_c->length();
  auto x_type = arr_x->type_id();
  auto y_type = arr_y->type_id();
  auto c_type = arr_c->type_id();
  assert(x_length == y_length);
  assert(x_length == c_length);
  assert(x_type == arrow::Type::UINT32);
  assert(y_type == arrow::Type::UINT32);

  auto input_x = (uint32_t*)arr_x->data()->GetValues<uint8_t>(1);
  auto input_y = (uint32_t*)arr_y->data()->GetValues<uint8_t>(1);

  switch (c_type) {
    case arrow::Type::INT8: {
      auto input_c_int8 = (int8_t*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<int8_t>(input_x, input_y, input_c_int8, x_length, conf));
    }
    case arrow::Type::INT16: {
      auto input_c_int16 = (int16_t*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<int16_t>(input_x, input_y, input_c_int16, x_length, conf));
    }
    case arrow::Type::INT32: {
      auto input_c_int32 = (int32_t*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<int32_t>(input_x, input_y, input_c_int32, x_length, conf));
    }
    case arrow::Type::INT64: {
      auto input_c_int64 = (int64_t*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<int64_t>(input_x, input_y, input_c_int64, x_length, conf));
    }
    case arrow::Type::UINT8: {
      auto input_c_uint8 = (uint8_t*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<uint8_t>(input_x, input_y, input_c_uint8, x_length, conf));
    }
    case arrow::Type::UINT16: {
      auto input_c_uint16 = (uint16_t*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<uint16_t>(input_x, input_y, input_c_uint16, x_length, conf));
    }
    case arrow::Type::UINT32: {
      auto input_c_uint32 = (uint32_t*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<uint32_t>(input_x, input_y, input_c_uint32, x_length, conf));
    }
    case arrow::Type::UINT64: {
      auto input_c_uint64 = (uint64_t*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<uint64_t>(input_x, input_y, input_c_uint64, x_length, conf));
    }
    case arrow::Type::FLOAT: {
      auto input_c_float = (float*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<float>(input_x, input_y, input_c_float, x_length, conf));
    }
    case arrow::Type::DOUBLE: {
      auto input_c_double = (double*)arr_c->data()->GetValues<uint8_t>(1);
      return out_pic(heatmap<double>(input_x, input_y, input_c_double, x_length, conf));
    }
    default:
      std::string err_msg =
          "type error of count while running heat map, type = " + std::to_string(c_type);
      throw std::runtime_error(err_msg);
  }
}

std::shared_ptr<arrow::Array> choropleth_map(const std::shared_ptr<arrow::Array>& arr_wkb,
                                             const std::shared_ptr<arrow::Array>& arr_c,
                                             const std::string& conf) {
  auto geo_arr = std::static_pointer_cast<arrow::BinaryArray>(arr_wkb);
  auto geo_size = arr_wkb->length();
  auto wkb_type = arr_wkb->type_id();
  assert(wkb_type == arrow::Type::BINARY);

  std::pair<uint8_t*, int64_t> result;
  auto c_size = arr_c->length();
  auto c_type = arr_c->type_id();
  assert(geo_size == c_size);
  switch (c_type) {
    case arrow::Type::INT8: {
      result = render_choroplethmap<int8_t>(arr_wkb, arr_c, conf);
      break;
    }
    case arrow::Type::INT16: {
      result = render_choroplethmap<int16_t>(arr_wkb, arr_c, conf);
      break;
    }
    case arrow::Type::INT32: {
      result = render_choroplethmap<int32_t>(arr_wkb, arr_c, conf);
      break;
    }
    case arrow::Type::INT64: {
      result = render_choroplethmap<int64_t>(arr_wkb, arr_c, conf);
      break;
    }
    case arrow::Type::UINT8: {
      result = render_choroplethmap<uint8_t>(arr_wkb, arr_c, conf);
      break;
    }
    case arrow::Type::UINT16: {
      result = render_choroplethmap<uint16_t>(arr_wkb, arr_c, conf);
      break;
    }
    case arrow::Type::UINT32: {
      result = render_choroplethmap<uint32_t>(arr_wkb, arr_c, conf);
      break;
    }
    case arrow::Type::UINT64: {
      result = render_choroplethmap<uint64_t>(arr_wkb, arr_c, conf);
      break;
    }
    case arrow::Type::FLOAT: {
      result = render_choroplethmap<float>(arr_wkb, arr_c, conf);
      break;
    }
    case arrow::Type::DOUBLE: {
      result = render_choroplethmap<double>(arr_wkb, arr_c, conf);
      break;
    }
    default:
      std::string err_msg = "type error of count while running choropleth map, type = " +
                            std::to_string(c_type);
      throw std::runtime_error(err_msg);
  }
  return out_pic(result);
}

std::shared_ptr<arrow::Array> icon_viz(const std::shared_ptr<arrow::Array>& points,
                                       const std::string& conf) {
  auto point_arr = std::static_pointer_cast<arrow::BinaryArray>(points);
  auto num_icons = points->length();
  auto wkb_type = points->type_id();
  assert(wkb_type == arrow::Type::BINARY);

  std::vector<uint32_t> input_x(num_icons);
  std::vector<uint32_t> input_y(num_icons);

  OGRGeometry* res_geo = nullptr;
  for (size_t i = 0; i < num_icons; i++) {
    std::string geo_wkb = point_arr->GetString(i);
    CHECK_GDAL(OGRGeometryFactory::createFromWkb(geo_wkb.c_str(), nullptr, &res_geo));
    auto rs_pointer = reinterpret_cast<OGRPoint*>(res_geo);
    input_x[i] = rs_pointer->getX();
    input_y[i] = rs_pointer->getY();
  }

  auto result = iconviz(&input_x[0], &input_y[0], num_icons, conf);

  return out_pic(result);
}

std::shared_ptr<arrow::Array> icon_viz(const std::shared_ptr<arrow::Array>& arr_x,
                                       const std::shared_ptr<arrow::Array>& arr_y,
                                       const std::string& conf) {
  auto x_length = arr_x->length();
  auto y_length = arr_y->length();
  auto x_type = arr_x->type_id();
  auto y_type = arr_y->type_id();

  assert(x_length == y_length);
  assert(x_type == arrow::Type::UINT32);
  assert(y_type == arrow::Type::UINT32);

  auto input_x = (uint32_t*)arr_x->data()->GetValues<uint8_t>(1);
  auto input_y = (uint32_t*)arr_y->data()->GetValues<uint8_t>(1);

  return out_pic(iconviz(input_x, input_y, x_length, conf));
}

}  // namespace render
}  // namespace arctern
