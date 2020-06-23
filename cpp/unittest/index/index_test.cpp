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

#include "index/index.h"

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

using IndexTree = arctern::geo_indexing::IndexTree;

std::vector<std::shared_ptr<arrow::Array>> wkb(const std::vector<std::string>& wkt) {
  std::vector<std::shared_ptr<arrow::Array>> wkb_array_vec;
  arrow::BinaryBuilder builder;
  for (int32_t i = 0; i < wkt.size(); i++) {
    OGRGeometry* geo = nullptr;
    auto error = OGRGeometryFactory::createFromWkt(wkt[i].c_str(), nullptr, &geo);

    if (error) {
      builder.AppendNull();
    } else {
      auto wkb_size = geo->WkbSize();
      auto wkb = static_cast<unsigned char*>(CPLMalloc(wkb_size));
      OGR_G_ExportToWkb(geo, OGRwkbByteOrder::wkbNDR, wkb);
      builder.Append(wkb, wkb_size);
      free(wkb);
    }
    OGRGeometryFactory::destroyGeometry(geo);
  }
  std::shared_ptr<arrow::Array> wkb_array;
  builder.Finish(&wkb_array);
  wkb_array_vec.push_back(wkb_array);
  return wkb_array_vec;
}

TEST(INDEX_TEST, test1) {
  IndexTree index;

  std::vector<std::string> roads;
  roads.push_back("LINESTRING (0 0,2 0)");
  roads.push_back("LINESTRING (5 0,5 5)");

  std::vector<std::string> gps_points;
  gps_points.push_back("POINT (1.0001 0.0001)");

  auto gps_points_binary_vec = wkb(gps_points);

  auto roads_binary_vec = wkb(roads);
  index.Append(roads_binary_vec);
//  index.near_road2(gps_points_binary_vec);
  index.near_road(gps_points_binary_vec, 1000);
}