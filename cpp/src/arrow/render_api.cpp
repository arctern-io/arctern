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
#include "arrow/render_api.h"

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
#include "render/utils/agg/agg_handler.h"
#include "render/utils/render_utils.h"

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
    const std::vector<std::string>& points, const std::vector<T>& weights,
    const std::string& conf) {
  auto data = AggHandler::weight_agg<T>(points, weights);
  auto num_point = data.first.size();

  std::vector<uint32_t> input_x(num_point);
  std::vector<uint32_t> input_y(num_point);
  std::vector<T> input_c(num_point);

  rapidjson::Document document;
  document.Parse(conf.c_str());
  rapidjson::Value mark_enter;
  mark_enter = document["marks"][0]["encode"]["enter"];
  auto agg = mark_enter["aggregation_type"]["value"].GetString();
  AggHandler::AggType type_agg = AggHandler::agg_type(agg);

  const auto& result_wkb = data.first;
  const auto& result_weight = data.second;

  switch (type_agg) {
    case AggHandler::AggType::MAX: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *max_element(result_weight[i].begin(), result_weight[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::MIN: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *min_element(result_weight[i].begin(), result_weight[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::COUNT: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = result_weight[i].size();
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::SUM: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = accumulate(result_weight[i].begin(), result_weight[i].end(), 0);
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::STDDEV: {
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
    case AggHandler::AggType::AVG: {
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
    const std::vector<std::string>& points, const std::vector<T>& arr_c,
    const std::vector<T>& arr_s, const std::string& conf) {
  auto agg_res = AggHandler::weight_agg_multiple_column<T>(points, arr_c, arr_s);
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
  AggHandler::AggType type_agg = AggHandler::agg_type(agg);

  const auto& result_wkb = std::get<0>(agg_res);
  const auto& result_c = std::get<1>(agg_res);
  const auto& result_s = std::get<2>(agg_res);

  switch (type_agg) {
    case AggHandler::AggType::MAX: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *max_element(result_c[i].begin(), result_c[i].end());
        input_s[i] = *max_element(result_s[i].begin(), result_s[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::MIN: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *min_element(result_c[i].begin(), result_c[i].end());
        input_s[i] = *min_element(result_c[i].begin(), result_c[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::COUNT: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = result_c[i].size();
        input_s[i] = result_c[i].size();
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::SUM: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = accumulate(result_c[i].begin(), result_c[i].end(), 0);
        input_s[i] = accumulate(result_c[i].begin(), result_c[i].end(), 0);
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::STDDEV: {
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
    case AggHandler::AggType::AVG: {
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
std::pair<uint8_t*, int64_t> render_heatmap(const std::vector<std::string>& points,
                                            const std::vector<T>& arr_c,
                                            const std::string& conf) {
  auto data = AggHandler::weight_agg<T>(points, arr_c);
  auto num_point = data.first.size();

  std::vector<uint32_t> input_x(num_point);
  std::vector<uint32_t> input_y(num_point);
  std::vector<T> input_c(num_point);

  rapidjson::Document document;
  document.Parse(conf.c_str());
  rapidjson::Value mark_enter;
  mark_enter = document["marks"][0]["encode"]["enter"];
  auto agg = mark_enter["aggregation_type"]["value"].GetString();
  AggHandler::AggType type_agg = AggHandler::agg_type(agg);

  const auto& result_wkb = data.first;
  const auto& result_weight = data.second;

  switch (type_agg) {
    case AggHandler::AggType::MAX: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *max_element(result_weight[i].begin(), result_weight[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::MIN: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = *min_element(result_weight[i].begin(), result_weight[i].end());
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::COUNT: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = result_weight[i].size();
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::SUM: {
      for (int i = 0; i < num_point; i++) {
        input_x[i] = result_wkb[i]->toPoint()->getX();
        input_y[i] = result_wkb[i]->toPoint()->getY();
        input_c[i] = accumulate(result_weight[i].begin(), result_weight[i].end(), 0);
        OGRGeometryFactory::destroyGeometry(result_wkb[i]);
      }
      break;
    }
    case AggHandler::AggType::STDDEV: {
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
    case AggHandler::AggType::AVG: {
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
std::pair<uint8_t*, int64_t> render_choroplethmap(const std::vector<std::string>& arr_wkb,
                                                  const std::vector<T>& arr_c,
                                                  const std::string& conf) {
  auto data = AggHandler::weight_agg<T>(arr_wkb, arr_c);
  auto num_geo = data.second.size();

  std::vector<T> input_c(num_geo);

  rapidjson::Document document;
  document.Parse(conf.c_str());
  rapidjson::Value mark_enter;
  mark_enter = document["marks"][0]["encode"]["enter"];
  auto agg = mark_enter["aggregation_type"]["value"].GetString();
  AggHandler::AggType type_agg = AggHandler::agg_type(agg);

  const auto& result_weight = data.second;
  std::size_t i = 0;

  switch (type_agg) {
    case AggHandler::AggType::MAX: {
      for (auto& item : result_weight) {
        input_c[i++] = *max_element(item.begin(), item.end());
      }
      break;
    }
    case AggHandler::AggType::MIN: {
      for (auto& item : result_weight) {
        input_c[i++] = *min_element(item.begin(), item.end());
      }
      break;
    }
    case AggHandler::AggType::COUNT: {
      for (auto& item : result_weight) {
        input_c[i++] = item.size();
      }
      break;
    }
    case AggHandler::AggType::SUM: {
      for (auto& item : result_weight) {
        input_c[i++] = accumulate(item.begin(), item.end(), 0);
      }
      break;
    }
    case AggHandler::AggType::STDDEV: {
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
    case AggHandler::AggType::AVG: {
      for (auto& item : result_weight) {
        T sum_data = accumulate(item.begin(), item.end(), 0);
        input_c[i++] = sum_data / item.size();
      }
      break;
    }
  }

  return choroplethmap<T>(data.first, &input_c[0], num_geo, conf);
}

template <typename T>
std::pair<uint8_t*, int64_t> render_fishnetmap(const std::vector<std::string>& points,
                                               const std::vector<T>& arr_c,
                                               const std::string& conf) {
  rapidjson::Document document;
  document.Parse(conf.c_str());
  rapidjson::Value mark_enter;
  mark_enter = document["marks"][0]["encode"]["enter"];
  auto agg = mark_enter["aggregation_type"]["value"].GetString();
  int cell_size = mark_enter["cell_size"]["value"].GetDouble();
  int cell_spacing = mark_enter["cell_spacing"]["value"].GetDouble();
  auto region_size = cell_size + cell_spacing;
  AggHandler::AggType type_agg = AggHandler::agg_type(agg);

  auto data = AggHandler::region_agg<T>(points, arr_c, region_size);
  auto num_point = data.size();

  std::vector<uint32_t> input_x(num_point);
  std::vector<uint32_t> input_y(num_point);
  std::vector<T> input_c(num_point);
  int i = 0;

  switch (type_agg) {
    case AggHandler::AggType::MAX: {
      for (auto iter = data.begin(); iter != data.end(); iter++) {
        auto result_point = iter->first;
        auto result_weight = iter->second;
        input_x[i] = result_point.first;
        input_y[i] = result_point.second;
        input_c[i] = *max_element(result_weight.begin(), result_weight.end());
        i++;
      }
      break;
    }
    case AggHandler::AggType::MIN: {
      for (auto iter = data.begin(); iter != data.end(); iter++) {
        auto result_point = iter->first;
        auto result_weight = iter->second;
        input_x[i] = result_point.first;
        input_y[i] = result_point.second;
        input_c[i] = *min_element(result_weight.begin(), result_weight.end());
        i++;
      }
      break;
    }
    case AggHandler::AggType::COUNT: {
      for (auto iter = data.begin(); iter != data.end(); iter++) {
        auto result_point = iter->first;
        auto result_weight = iter->second;
        input_x[i] = result_point.first;
        input_y[i] = result_point.second;
        input_c[i] = result_weight.size();
        i++;
      }
      break;
    }
    case AggHandler::AggType::SUM: {
      for (auto iter = data.begin(); iter != data.end(); iter++) {
        auto result_point = iter->first;
        auto result_weight = iter->second;
        input_x[i] = result_point.first;
        input_y[i] = result_point.second;
        input_c[i] = accumulate(result_weight.begin(), result_weight.end(), 0);
        i++;
      }
      break;
    }
    case AggHandler::AggType::STDDEV: {
      for (auto iter = data.begin(); iter != data.end(); iter++) {
        auto result_point = iter->first;
        auto result_weight = iter->second;
        input_x[i] = result_point.first;
        input_y[i] = result_point.second;
        T sum = accumulate(result_weight.begin(), result_weight.end(), 0);
        T mean = sum / result_weight.size();
        T accum = 0;
        std::for_each(std::begin(result_weight), std::end(result_weight),
                      [&](const T d) { accum += (d - mean) * (d - mean); });
        input_c[i] = sqrt(accum / result_weight.size());
        i++;
      }
      break;
    }
    case AggHandler::AggType::AVG: {
      for (auto iter = data.begin(); iter != data.end(); iter++) {
        auto result_point = iter->first;
        auto result_weight = iter->second;
        input_x[i] = result_point.first;
        input_y[i] = result_point.second;
        T sum_data = accumulate(result_weight.begin(), result_weight.end(), 0);
        input_c[i] = sum_data / result_weight.size();
        i++;
      }
      break;
    }
  }

  return fishnetmap<T>(&input_x[0], &input_y[0], &input_c[0], num_point, conf);
}

const std::vector<std::shared_ptr<arrow::Array>> projection(
    const std::vector<std::shared_ptr<arrow::Array>>& geos,
    const std::string& bottom_right, const std::string& top_left, const int& height,
    const int& width) {
  const auto& geo_vec = GeometryExtraction(geos);
  Projection(geo_vec, bottom_right, top_left, height, width);
  const auto& res = GeometryExport(geo_vec, geos.size());
  return res;
}

const std::vector<std::shared_ptr<arrow::Array>> transform_and_projection(
    const std::vector<std::shared_ptr<arrow::Array>>& geos, const std::string& src_rs,
    const std::string& dst_rs, const std::string& bottom_right,
    const std::string& top_left, const int& height, const int& width) {
  const auto& geo_vec = GeometryExtraction(geos);
  TransformAndProjection(geo_vec, src_rs, dst_rs, bottom_right, top_left, height, width);
  const auto& res = GeometryExport(geo_vec, geos.size());
  return res;
}

std::shared_ptr<arrow::Array> point_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::string& conf) {
  const auto& wkb_vec = WkbExtraction(points_vector);
  auto num_of_point = wkb_vec.size();

  std::vector<uint32_t> input_x(num_of_point);
  std::vector<uint32_t> input_y(num_of_point);

  OGRGeometry* res_geo = nullptr;
  for (size_t i = 0; i < num_of_point; i++) {
    std::string geo_wkb = wkb_vec[i];
    CHECK_GDAL(OGRGeometryFactory::createFromWkb(geo_wkb.c_str(), nullptr, &res_geo));
    auto rs_pointer = reinterpret_cast<OGRPoint*>(res_geo);
    input_x[i] = rs_pointer->getX();
    input_y[i] = rs_pointer->getY();
  }
  auto result = pointmap(&input_x[0], &input_y[0], num_of_point, conf);

  return out_pic(result);
}

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::string& conf) {
  const auto& wkb_vec = WkbExtraction(points_vector);
  auto num_of_point = wkb_vec.size();

  std::vector<uint32_t> input_x(num_of_point);
  std::vector<uint32_t> input_y(num_of_point);

  OGRGeometry* res_geo = nullptr;
  for (size_t i = 0; i < num_of_point; i++) {
    std::string geo_wkb = wkb_vec[i];
    CHECK_GDAL(OGRGeometryFactory::createFromWkb(geo_wkb.c_str(), nullptr, &res_geo));
    auto rst_pointer = reinterpret_cast<OGRPoint*>(res_geo);
    input_x[i] = (uint32_t)rst_pointer->getX();
    input_y[i] = (uint32_t)rst_pointer->getY();
  }

  auto result = weighted_pointmap<int8_t>(&input_x[0], &input_y[0], num_of_point, conf);
  return out_pic(result);
}

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& weights_vector,
    const std::string& conf) {
  const auto& wkb_vec = WkbExtraction(points_vector);

  auto weight_data_type = weights_vector[0]->type_id();

  switch (weight_data_type) {
    case arrow::Type::INT8: {
      const auto& arr_c = WeightExtraction<int8_t>(weights_vector);
      return out_pic(render_weighted_pointmap<int8_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT16: {
      const auto& arr_c = WeightExtraction<int16_t>(weights_vector);
      return out_pic(render_weighted_pointmap<int16_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT32: {
      const auto& arr_c = WeightExtraction<int32_t>(weights_vector);
      return out_pic(render_weighted_pointmap<int32_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT64: {
      const auto& arr_c = WeightExtraction<int64_t>(weights_vector);
      return out_pic(render_weighted_pointmap<int64_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT8: {
      const auto& arr_c = WeightExtraction<uint8_t>(weights_vector);
      return out_pic(render_weighted_pointmap<uint8_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT16: {
      const auto& arr_c = WeightExtraction<uint16_t>(weights_vector);
      return out_pic(render_weighted_pointmap<uint16_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT32: {
      const auto& arr_c = WeightExtraction<uint32_t>(weights_vector);
      return out_pic(render_weighted_pointmap<uint32_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT64: {
      const auto& arr_c = WeightExtraction<uint64_t>(weights_vector);
      return out_pic(render_weighted_pointmap<uint64_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::FLOAT: {
      const auto& arr_c = WeightExtraction<float>(weights_vector);
      return out_pic(render_weighted_pointmap<float>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::DOUBLE: {
      const auto& arr_c = WeightExtraction<double>(weights_vector);
      return out_pic(render_weighted_pointmap<double>(wkb_vec, arr_c, conf));
    }
    default:
      std::string err_msg =
          "type error of count while running weighted_pointmap, type = " +
          std::to_string(weight_data_type);
      throw std::runtime_error(err_msg);
  }
}

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& color_weights_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& size_weights_vector,
    const std::string& conf) {
  const auto& wkb_vec = WkbExtraction(points_vector);

  auto weight_data_type = color_weights_vector[0]->type_id();

  switch (weight_data_type) {
    case arrow::Type::INT8: {
      const auto& arr_c = WeightExtraction<int8_t>(color_weights_vector);
      const auto& arr_s = WeightExtraction<int8_t>(size_weights_vector);
      return out_pic(render_weighted_pointmap<int8_t>(wkb_vec, arr_c, arr_s, conf));
    }
    case arrow::Type::INT16: {
      const auto& arr_c = WeightExtraction<int16_t>(color_weights_vector);
      const auto& arr_s = WeightExtraction<int16_t>(size_weights_vector);
      return out_pic(render_weighted_pointmap<int16_t>(wkb_vec, arr_c, arr_s, conf));
    }
    case arrow::Type::INT32: {
      const auto& arr_c = WeightExtraction<int32_t>(color_weights_vector);
      const auto& arr_s = WeightExtraction<int32_t>(size_weights_vector);
      return out_pic(render_weighted_pointmap<int32_t>(wkb_vec, arr_c, arr_s, conf));
    }
    case arrow::Type::INT64: {
      const auto& arr_c = WeightExtraction<int64_t>(color_weights_vector);
      const auto& arr_s = WeightExtraction<int64_t>(size_weights_vector);
      return out_pic(render_weighted_pointmap<int64_t>(wkb_vec, arr_c, arr_s, conf));
    }
    case arrow::Type::UINT8: {
      const auto& arr_c = WeightExtraction<uint8_t>(color_weights_vector);
      const auto& arr_s = WeightExtraction<uint8_t>(size_weights_vector);
      return out_pic(render_weighted_pointmap<uint8_t>(wkb_vec, arr_c, arr_s, conf));
    }
    case arrow::Type::UINT16: {
      const auto& arr_c = WeightExtraction<uint16_t>(color_weights_vector);
      const auto& arr_s = WeightExtraction<uint16_t>(size_weights_vector);
      return out_pic(render_weighted_pointmap<uint16_t>(wkb_vec, arr_c, arr_s, conf));
    }
    case arrow::Type::UINT32: {
      const auto& arr_c = WeightExtraction<uint32_t>(color_weights_vector);
      const auto& arr_s = WeightExtraction<uint32_t>(size_weights_vector);
      return out_pic(render_weighted_pointmap<uint32_t>(wkb_vec, arr_c, arr_s, conf));
    }
    case arrow::Type::UINT64: {
      const auto& arr_c = WeightExtraction<uint64_t>(color_weights_vector);
      const auto& arr_s = WeightExtraction<uint64_t>(size_weights_vector);
      return out_pic(render_weighted_pointmap<uint64_t>(wkb_vec, arr_c, arr_s, conf));
    }
    case arrow::Type::FLOAT: {
      const auto& arr_c = WeightExtraction<float>(color_weights_vector);
      const auto& arr_s = WeightExtraction<float>(size_weights_vector);
      return out_pic(render_weighted_pointmap<float>(wkb_vec, arr_c, arr_s, conf));
    }
    case arrow::Type::DOUBLE: {
      const auto& arr_c = WeightExtraction<double>(color_weights_vector);
      const auto& arr_s = WeightExtraction<double>(size_weights_vector);
      return out_pic(render_weighted_pointmap<double>(wkb_vec, arr_c, arr_s, conf));
    }
    default:
      std::string err_msg =
          "type error of count while running weighted_pointmap, type = " +
          std::to_string(weight_data_type);
      throw std::runtime_error(err_msg);
  }
}

std::shared_ptr<arrow::Array> heat_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& weights_vector,
    const std::string& conf) {
  const auto& wkb_vec = WkbExtraction(points_vector);

  auto weight_data_type = weights_vector[0]->type_id();

  switch (weight_data_type) {
    case arrow::Type::INT8: {
      const auto& arr_c = WeightExtraction<int8_t>(weights_vector);
      return out_pic(render_heatmap<int8_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT16: {
      const auto& arr_c = WeightExtraction<int16_t>(weights_vector);
      return out_pic(render_heatmap<int16_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT32: {
      const auto& arr_c = WeightExtraction<int32_t>(weights_vector);
      return out_pic(render_heatmap<int32_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT64: {
      const auto& arr_c = WeightExtraction<int64_t>(weights_vector);
      return out_pic(render_heatmap<int64_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT8: {
      const auto& arr_c = WeightExtraction<uint8_t>(weights_vector);
      return out_pic(render_heatmap<uint8_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT16: {
      const auto& arr_c = WeightExtraction<uint16_t>(weights_vector);
      return out_pic(render_heatmap<uint16_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT32: {
      const auto& arr_c = WeightExtraction<uint32_t>(weights_vector);
      return out_pic(render_heatmap<uint32_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT64: {
      const auto& arr_c = WeightExtraction<uint64_t>(weights_vector);
      return out_pic(render_heatmap<uint64_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::FLOAT: {
      const auto& arr_c = WeightExtraction<float>(weights_vector);
      return out_pic(render_heatmap<float>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::DOUBLE: {
      const auto& arr_c = WeightExtraction<double>(weights_vector);
      return out_pic(render_heatmap<double>(wkb_vec, arr_c, conf));
    }
    default:
      std::string err_msg = "type error of count while running heatmap, type = " +
                            std::to_string(weight_data_type);
      throw std::runtime_error(err_msg);
  }
}

std::shared_ptr<arrow::Array> choropleth_map(
    const std::vector<std::shared_ptr<arrow::Array>>& polygons_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& weights_vector,
    const std::string& conf) {
  const auto& wkb_vec = WkbExtraction(polygons_vector);
  auto weight_data_type = weights_vector[0]->type_id();

  switch (weight_data_type) {
    case arrow::Type::INT8: {
      const auto& arr_c = WeightExtraction<int8_t>(weights_vector);
      return out_pic(render_choroplethmap<int8_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT16: {
      const auto& arr_c = WeightExtraction<int16_t>(weights_vector);
      return out_pic(render_choroplethmap<int16_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT32: {
      const auto& arr_c = WeightExtraction<int32_t>(weights_vector);
      return out_pic(render_choroplethmap<int32_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT64: {
      const auto& arr_c = WeightExtraction<int64_t>(weights_vector);
      return out_pic(render_choroplethmap<int64_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT8: {
      const auto& arr_c = WeightExtraction<uint8_t>(weights_vector);
      return out_pic(render_choroplethmap<uint8_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT16: {
      const auto& arr_c = WeightExtraction<uint16_t>(weights_vector);
      return out_pic(render_choroplethmap<uint16_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT32: {
      const auto& arr_c = WeightExtraction<uint32_t>(weights_vector);
      return out_pic(render_choroplethmap<uint32_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT64: {
      const auto& arr_c = WeightExtraction<uint64_t>(weights_vector);
      return out_pic(render_choroplethmap<uint64_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::FLOAT: {
      const auto& arr_c = WeightExtraction<float>(weights_vector);
      return out_pic(render_choroplethmap<float>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::DOUBLE: {
      const auto& arr_c = WeightExtraction<double>(weights_vector);
      return out_pic(render_choroplethmap<double>(wkb_vec, arr_c, conf));
    }
    default:
      std::string err_msg = "type error of count while running choroplethmap, type = " +
                            std::to_string(weight_data_type);
      throw std::runtime_error(err_msg);
  }
}

std::shared_ptr<arrow::Array> icon_viz(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::string& conf) {
  const auto& wkb_vec = WkbExtraction(points_vector);
  auto num_of_point = wkb_vec.size();

  std::vector<uint32_t> input_x(num_of_point);
  std::vector<uint32_t> input_y(num_of_point);

  OGRGeometry* res_geo = nullptr;
  for (size_t i = 0; i < num_of_point; i++) {
    std::string geo_wkb = wkb_vec[i];
    CHECK_GDAL(OGRGeometryFactory::createFromWkb(geo_wkb.c_str(), nullptr, &res_geo));
    auto rs_pointer = reinterpret_cast<OGRPoint*>(res_geo);
    input_x[i] = rs_pointer->getX();
    input_y[i] = rs_pointer->getY();
  }

  auto result = iconviz(&input_x[0], &input_y[0], num_of_point, conf);

  return out_pic(result);
}

std::shared_ptr<arrow::Array> fishnet_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& weights_vector,
    const std::string& conf) {
  const auto& wkb_vec = WkbExtraction(points_vector);

  auto weight_data_type = weights_vector[0]->type_id();

  switch (weight_data_type) {
    case arrow::Type::INT8: {
      const auto& arr_c = WeightExtraction<int8_t>(weights_vector);
      return out_pic(render_fishnetmap<int8_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT16: {
      const auto& arr_c = WeightExtraction<int16_t>(weights_vector);
      return out_pic(render_fishnetmap<int16_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT32: {
      const auto& arr_c = WeightExtraction<int32_t>(weights_vector);
      return out_pic(render_fishnetmap<int32_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::INT64: {
      const auto& arr_c = WeightExtraction<int64_t>(weights_vector);
      return out_pic(render_fishnetmap<int64_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT8: {
      const auto& arr_c = WeightExtraction<uint8_t>(weights_vector);
      return out_pic(render_fishnetmap<uint8_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT16: {
      const auto& arr_c = WeightExtraction<uint16_t>(weights_vector);
      return out_pic(render_fishnetmap<uint16_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT32: {
      const auto& arr_c = WeightExtraction<uint32_t>(weights_vector);
      return out_pic(render_fishnetmap<uint32_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::UINT64: {
      const auto& arr_c = WeightExtraction<uint64_t>(weights_vector);
      return out_pic(render_fishnetmap<uint64_t>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::FLOAT: {
      const auto& arr_c = WeightExtraction<float>(weights_vector);
      return out_pic(render_fishnetmap<float>(wkb_vec, arr_c, conf));
    }
    case arrow::Type::DOUBLE: {
      const auto& arr_c = WeightExtraction<double>(weights_vector);
      return out_pic(render_fishnetmap<double>(wkb_vec, arr_c, conf));
    }
    default:
      std::string err_msg = "type error of count while running square_map, type = " +
                            std::to_string(weight_data_type);
      throw std::runtime_error(err_msg);
  }
}

}  // namespace render
}  // namespace arctern
