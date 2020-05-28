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

#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "utils/check_status.h"

namespace arctern {
namespace render {

class AggHandler {
 public:
  enum AggType {
    SUM = 0,
    MIN,
    MAX,
    COUNT,
    STDDEV,
    AVG,
  };

  static AggType agg_type(std::string type);

  struct hash_pair {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& point) const {
      auto hash_x = std::hash<T1>{}(point.first);
      auto hash_y = std::hash<T2>{}(point.second);
      return hash_x ^ hash_y;
    }
  };

  template <typename T>
  static std::unordered_map<std::pair<uint32_t, uint32_t>, std::vector<T>, hash_pair>
  region_agg(const std::vector<std::string>& wkb_arr, const std::vector<T>& arr_c,
             int region_size) {
    assert(wkb_arr.size() == arr_c.size());

    std::unordered_map<std::pair<uint32_t, uint32_t>, std::vector<T>, hash_pair> result;
    for (size_t i = 0; i < wkb_arr.size(); i++) {
      std::string wkb = wkb_arr[i];
      OGRGeometry* geo;
      CHECK_GDAL(OGRGeometryFactory::createFromWkb(wkb.c_str(), nullptr, &geo));
      uint32_t x = geo->toPoint()->getX() / region_size;
      uint32_t y = geo->toPoint()->getY() / region_size;
      OGRGeometryFactory::destroyGeometry(geo);
      auto point = std::make_pair(x, y);
      if (result.find(point) == result.end()) {
        std::vector<T> weight;
        weight.emplace_back(arr_c[i]);
        result[point] = weight;
      } else {
        auto& weight = result[point];
        weight.emplace_back(arr_c[i]);
      }
    }

    return result;
  }

  template <typename T>
  static std::pair<std::vector<OGRGeometry*>, std::vector<std::vector<T>>> weight_agg(
      const std::vector<std::string>& wkb_arr, const std::vector<T>& arr_c) {
    assert(wkb_arr.size() == arr_c.size());

    std::unordered_map<std::string, std::vector<T>> wkb_map;
    for (size_t i = 0; i < wkb_arr.size(); i++) {
      std::string wkb = wkb_arr[i];
      if (wkb_map.find(wkb) == wkb_map.end()) {
        std::vector<T> weight;
        weight.emplace_back(arr_c[i]);
        wkb_map[wkb] = weight;
      } else {
        auto& weight = wkb_map[wkb];
        weight.emplace_back(arr_c[i]);
      }
    }

    std::vector<OGRGeometry*> results_wkb(wkb_map.size());
    std::vector<std::vector<T>> results_weight(wkb_map.size());
    int i = 0;
    for (auto iter = wkb_map.begin(); iter != wkb_map.end(); iter++) {
      OGRGeometry* res_geo;
      CHECK_GDAL(
          OGRGeometryFactory::createFromWkb(iter->first.c_str(), nullptr, &res_geo));
      results_wkb[i] = res_geo;
      results_weight[i] = iter->second;
      i++;
    }

    return std::make_pair(results_wkb, results_weight);
  }

  template <typename T>
  std::tuple<std::vector<OGRGeometry*>, std::vector<std::vector<T>>,
             std::vector<std::vector<
                 T>>> static weight_agg_multiple_column(const std::vector<std::string>&
                                                            geos,
                                                        const std::vector<T>& arr_c,
                                                        const std::vector<T>& arr_s) {
    assert(geos.size() == arr_c.size());
    assert(arr_c.size() == arr_s.size());

    using vector_pair = std::pair<std::vector<T>, std::vector<T>>;
    std::unordered_map<std::string, vector_pair> wkb_map;

    for (size_t i = 0; i < geos.size(); i++) {
      std::string geo_wkb = geos[i];
      if (wkb_map.find(geo_wkb) == wkb_map.end()) {
        std::vector<T> weight_c;
        std::vector<T> weight_s;
        weight_c.emplace_back(arr_c[i]);
        weight_s.emplace_back(arr_s[i]);
        wkb_map[geo_wkb] = std::make_pair(weight_c, weight_s);
      } else {
        auto& weight_c = wkb_map[geo_wkb].first;
        auto& weight_s = wkb_map[geo_wkb].second;
        weight_c.emplace_back(arr_c[i]);
        weight_s.emplace_back(arr_s[i]);
      }
    }

    std::vector<OGRGeometry*> results_wkb(wkb_map.size());
    std::vector<std::vector<T>> results_weight_c(wkb_map.size());
    std::vector<std::vector<T>> results_weight_s(wkb_map.size());

    int i = 0;
    for (auto iter = wkb_map.begin(); iter != wkb_map.end(); iter++) {
      OGRGeometry* res_geo;
      CHECK_GDAL(
          OGRGeometryFactory::createFromWkb(iter->first.c_str(), nullptr, &res_geo));
      results_wkb[i] = res_geo;
      results_weight_c[i] = iter->second.first;
      results_weight_s[i] = iter->second.second;
      i++;
    }

    return std::make_tuple(results_wkb, results_weight_c, results_weight_s);
  }
};

}  // namespace render
}  // namespace arctern
