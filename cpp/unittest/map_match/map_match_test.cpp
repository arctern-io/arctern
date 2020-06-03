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

#include "map_match/map_match.h"

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

std::vector<OGRGeometry*> construct_geometry(const std::vector<std::string>& wkt) {
  std::vector<OGRGeometry*> result;
  for (int32_t i = 0; i < wkt.size(); i++) {
    OGRGeometry* gps_point;
    OGRGeometryFactory::createFromWkt(wkt[i].c_str(), nullptr, &gps_point);
    result.push_back(gps_point);
  }

  return result;
}

void destory_geometry(std::vector<OGRGeometry*>& geos) {
  for (int32_t i = 0; i < geos.size(); i++) {
    OGRGeometryFactory::destroyGeometry(geos[i]);
  }
}

std::vector<double> min_distacne(std::vector<std::string>& roads,
                                 std::vector<std::string>& gps_points) {
  std::vector<double> result;
  auto roads_geo = construct_geometry(roads);
  auto gps_points_geo = construct_geometry(gps_points);

  for (int32_t i = 0; i < gps_points.size(); i++) {
    double distance = std::numeric_limits<double>::max();

    if (gps_points_geo[i] == nullptr) {
      result.push_back(0);
      continue;
    }

    for (int32_t j = 0; j < roads.size(); j++) {
      if (roads_geo[j] == nullptr) {
        continue;
      }
      if (gps_points_geo[i]->Distance(roads_geo[j]) <= distance) {
        distance = gps_points_geo[i]->Distance(roads_geo[j]);
      }
    }
    result.push_back(distance);
  }

  destory_geometry(roads_geo);
  destory_geometry(gps_points_geo);

  return result;
}

std::vector<std::string> nearest(std::vector<std::string> roads,
                                 std::vector<std::string> gps_points) {
  std::vector<std::string> result;
  auto roads_geo = construct_geometry(roads);
  auto gps_points_geo = construct_geometry(gps_points);

  std::string nearest_road;
  for (int32_t i = 0; i < gps_points.size(); i++) {
    double distance = std::numeric_limits<double>::max();
    for (int32_t j = 0; j < roads.size(); j++) {
      if (gps_points_geo[i]->Distance(roads_geo[j]) <= distance) {
        distance = gps_points_geo[i]->Distance(roads_geo[j]);
        nearest_road = roads[j];
      }
    }
    result.push_back(nearest_road);
  }

  destory_geometry(roads_geo);
  destory_geometry(gps_points_geo);

  return result;
}

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

TEST(MAP_MATCH_TEST, NEAREST_LOCATION_ON_ROAD1) {
  std::vector<std::string> roads;
  roads.push_back("LINESTRING (-73.9975944 40.7140611,-73.9974922 40.7139962)");
  roads.push_back("LINESTRING (-73.9980065 40.7138119,-73.9980743 40.7137811)");
  roads.push_back("LINESTRING (-73.9975554 40.7141073,-73.9975944 40.7140611)");
  roads.push_back("LINESTRING (-73.9978864 40.714317,-73.997674 40.7140968)");
  roads.push_back("LINESTRING (-73.997981 40.7136728,-73.9980743 40.7137811)");
  roads.push_back("LINESTRING (-73.9980743 40.7137811,-73.9984728 40.7136003)");
  roads.push_back("LINESTRING (-73.9611014 40.7608112,-73.9610636 40.7608639)");
  roads.push_back("LINESTRING (-73.9594166 40.7593773,-73.9593736 40.7593593)");
  roads.push_back("LINESTRING (-73.961609 40.7602969,-73.9615014 40.7602517)");
  roads.push_back("LINESTRING (-73.9615569 40.7601753,-73.9615014 40.7602517)");

  std::vector<std::string> gps_points;
  gps_points.push_back("POINT (-73.993003 40.747594)");
  gps_points.push_back("POINT (-73.959908 40.776353)");
  gps_points.push_back("POINT (-73.955183 40.773459)");
  gps_points.push_back("POINT (-73.985233 40.744682)");
  gps_points.push_back("POINT (-73.997969 40.682816)");
  gps_points.push_back("POINT (-73.996458 40.758197)");
  gps_points.push_back("POINT (-73.98824 40.74896)");
  gps_points.push_back("POINT (-73.985185 40.735828)");
  gps_points.push_back("POINT (-73.989726 40.767795)");
  gps_points.push_back("POINT (-73.992669 40.768327)");

  auto gps_points_binary_vec = wkb(gps_points);
  auto roads_binary_vec = wkb(roads);

  auto result = arctern::map_match::nearest_location_on_road(roads_binary_vec,
                                                             gps_points_binary_vec);
  auto compare_result = min_distacne(roads, gps_points);
  auto result_1 = std::static_pointer_cast<arrow::BinaryArray>(result[0]);
  assert(result_1->length() == gps_points_binary_vec[0]->length());

  for (int32_t i = 0; i < result_1->length(); i++) {
    if (result_1->GetView(i).size() == 0) continue;
    OGRGeometry* gps_point = nullptr;
    OGRGeometryFactory::createFromWkt(gps_points[i].c_str(), nullptr, &gps_point);
    OGRGeometry* projection_point = nullptr;
    OGRGeometryFactory::createFromWkb(result_1->GetString(i).c_str(), nullptr,
                                      &projection_point);
    auto projection_point1 = dynamic_cast<OGRPoint*>(projection_point);
    assert(projection_point1->Distance(gps_point) == compare_result[i]);
    OGRGeometryFactory::destroyGeometry(gps_point);
    OGRGeometryFactory::destroyGeometry(projection_point);
  }
}

TEST(MAP_MATCH_TEST, NEAREST_LOCATION_ON_ROAD2) {
  std::vector<std::string> roads;
  roads.push_back("LINESTRING (-73.9975944 40.7140611,-73.9974922 40.7139962)");
  roads.push_back("LINESTRING (-73.9980065 40.7138119,-73.9980743 40.7137811)");
  roads.push_back("LINESTRING (-73.9975554 40.7141073,-73.9975944 40.7140611)");
  roads.push_back("LINESTRING (-73.9978864 40.714317,-73.997674 40.7140968)");
  roads.push_back("LINESTRING (-73.997981 40.7136728,-73.9980743 40.7137811)");
  roads.push_back("LINESTRING (-73.9980743 40.7137811,-73.9984728 40.7136003)");
  roads.push_back("LINESTRING (-73.9611014 40.7608112,-73.9610636 40.7608639)");
  roads.push_back("LINESTRING (-73.9594166 40.7593773,-73.9593736 40.7593593)");
  roads.push_back("LINESTRING (-73.961609 40.7602969,-73.9615014 40.7602517)");
  roads.push_back("LINESTRING (-73.9615569 40.7601753,-73.9615014 40.7602517)");

  std::vector<std::string> gps_points;
  gps_points.push_back("POINT (-73.993003 40.747594)");
  gps_points.push_back("POINT (-73.959908 40.776353)");
  gps_points.push_back("POINT (-73.955183 40.773459)");
  gps_points.push_back("POINT (-73.985233 40.744682)");
  gps_points.push_back("POINT (-73.997969 40.682816)");
  gps_points.push_back("POINT (-73.996458 40.758197)");
  gps_points.push_back("POINT (-73.98824 40.74896)");
  gps_points.push_back("POINT (-73.985185 40.735828)");
  gps_points.push_back("POINT (-73.989726 40.767795)");
  gps_points.push_back("POINT (-73.992669 40.768327)");

  auto roads_binary_vec = wkb(roads);
  auto gps_points_binary_vec = wkb(gps_points);

  auto result = arctern::map_match::nearest_location_on_road(roads_binary_vec,
                                                             gps_points_binary_vec);
  auto compare_result = min_distacne(roads, gps_points);

  auto result_1 = std::static_pointer_cast<arrow::BinaryArray>(result[0]);
  assert(result_1->length() == gps_points_binary_vec[0]->length());

  for (int32_t i = 0; i < result_1->length(); i++) {
    OGRGeometry* gps_point = nullptr;
    OGRGeometryFactory::createFromWkt(gps_points[i].c_str(), nullptr, &gps_point);
    OGRGeometry* projection_point = nullptr;
    OGRGeometryFactory::createFromWkb(result_1->GetString(i).c_str(), nullptr,
                                      &projection_point);
    auto projection_point1 = dynamic_cast<OGRPoint*>(projection_point);
    assert(projection_point1->Distance(gps_point) == compare_result[i]);
    OGRGeometryFactory::destroyGeometry(gps_point);
    OGRGeometryFactory::destroyGeometry(projection_point);
  }
}

TEST(MAP_MATCH_TEST, NEAREST_ROAD) {
  std::vector<std::string> roads;
  roads.push_back("LINESTRING (-73.9975944 40.7140611,-73.9974922 40.7139962)");
  roads.push_back("LINESTRING (-73.9980065 40.7138119,-73.9980743 40.7137811)");
  roads.push_back("LINESTRING (-73.9975554 40.7141073,-73.9975944 40.7140611)");
  roads.push_back("LINESTRING (-73.9978864 40.714317,-73.997674 40.7140968)");
  roads.push_back("LINESTRING (-73.997981 40.7136728,-73.9980743 40.7137811)");
  roads.push_back("LINESTRING (-73.9980743 40.7137811,-73.9984728 40.7136003)");
  roads.push_back("LINESTRING (-73.9611014 40.7608112,-73.9610636 40.7608639)");
  roads.push_back("LINESTRING (-73.9594166 40.7593773,-73.9593736 40.7593593)");
  roads.push_back("LINESTRING (-73.961609 40.7602969,-73.9615014 40.7602517)");
  roads.push_back("LINESTRING (-73.9615569 40.7601753,-73.9615014 40.7602517)");

  std::vector<std::string> gps_points;
  gps_points.push_back("POINT (-73.993003 40.747594)");
  gps_points.push_back("POINT (-73.959908 40.776353)");
  gps_points.push_back("POINT (-73.955183 40.773459)");
  gps_points.push_back("POINT (-73.985233 40.744682)");
  gps_points.push_back("POINT (-73.997969 40.682816)");
  gps_points.push_back("POINT (-73.996458 40.758197)");
  gps_points.push_back("POINT (-73.98824 40.74896)");
  gps_points.push_back("POINT (-73.985185 40.735828)");
  gps_points.push_back("POINT (-73.989726 40.767795)");
  gps_points.push_back("POINT (-73.992669 40.768327)");

  auto gps_points_binary_vec = wkb(gps_points);

  auto roads_binary_vec = wkb(roads);

  auto result = arctern::map_match::nearest_road(roads_binary_vec, gps_points_binary_vec);
  auto compare_result = nearest(roads, gps_points);

  auto result_1 = std::static_pointer_cast<arrow::BinaryArray>(result[0]);
  assert(result_1->length() == gps_points_binary_vec[0]->length());

  for (int32_t i = 0; i < result_1->length(); i++) {
    OGRGeometry* nearest_road = nullptr;
    OGRGeometryFactory::createFromWkb(result_1->GetString(i).c_str(), nullptr,
                                      &nearest_road);
    char* str;
    OGR_G_ExportToWkt(nearest_road, &str);
    OGRGeometryFactory::destroyGeometry(nearest_road);
    assert(std::string(str) == compare_result[i]);
    CPLFree(str);
  }
}

TEST(MAP_MATCH_TEST, NEAR_ROAD) {
  std::vector<std::string> roads;
  roads.push_back("LINESTRING (-73.9975944 40.7140611,-73.9974922 40.7139962)");
  roads.push_back("LINESTRING (-73.9980065 40.7138119,-73.9980743 40.7137811)");
  roads.push_back("LINESTRING (-73.9975554 40.7141073,-73.9975944 40.7140611)");
  roads.push_back("LINESTRING (-73.9978864 40.714317,-73.997674 40.7140968)");
  roads.push_back("LINESTRING (-73.997981 40.7136728,-73.9980743 40.7137811)");
  roads.push_back("LINESTRING (-73.9980743 40.7137811,-73.9984728 40.7136003)");
  roads.push_back("LINESTRING (-73.9611014 40.7608112,-73.9610636 40.7608639)");
  roads.push_back("LINESTRING (-73.9594166 40.7593773,-73.9593736 40.7593593)");
  roads.push_back("LINESTRING (-73.961609 40.7602969,-73.9615014 40.7602517)");
  roads.push_back("LINESTRING (-73.9615569 40.7601753,-73.9615014 40.7602517)");

  std::vector<std::string> gps_points;
  gps_points.push_back("POINT (-73.961003 40.760594)");
  gps_points.push_back("POINT (-73.959908 40.776353)");
  gps_points.push_back("POINT (-73.955183 40.773459)");
  gps_points.push_back("POINT (-73.985233 40.744682)");
  gps_points.push_back("POINT (-73.997969 40.682816)");
  gps_points.push_back("POINT (-73.996458 40.758197)");
  gps_points.push_back("POINT (-73.98824 40.74896)");
  gps_points.push_back("POINT (-73.985185 40.735828)");
  gps_points.push_back("POINT (-73.989726 40.767795)");
  gps_points.push_back("POINT (-73.992669 40.768327)");

  auto gps_points_binary_vec = wkb(gps_points);

  auto roads_binary_vec = wkb(roads);
  auto result =
      arctern::map_match::near_road(roads_binary_vec, gps_points_binary_vec, 1000);
  auto compare_result = nearest(roads, gps_points);

  auto result_1 = std::static_pointer_cast<arrow::BinaryArray>(result[0]);
  assert(result_1->length() == gps_points_binary_vec[0]->length());
  std::vector<bool> out_std = {true,  false, false, false, false,
                               false, false, false, false, false};
  int offset = 0;
  for (int j = 0; j < result.size(); j++) {
    auto values = std::static_pointer_cast<arrow::BooleanArray>(result[j]);
    auto size = result[j]->length();
    auto type = result[j]->type_id();
    assert(type == arrow::Type::BOOL);
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(values->Value(i), out_std[offset++]);
    }
  }
}
