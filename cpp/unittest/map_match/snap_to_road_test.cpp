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
#include <gtest/gtest.h>
#include <iostream>

#include "map_match/snap_road.h"

std::vector<double> min_distacne(std::vector<std::string> roads,
                                 std::vector<std::string> gps_points) {
    std::vector<double> result;
    OGRGeometry *gps_point = nullptr;
    for (int32_t i = 0; i < gps_points.size(); i++) {
        auto err_code = OGRGeometryFactory::createFromWkt(gps_points[i].c_str(),
                                                          nullptr,
                                                          &gps_point);
        if (err_code != OGRERR_NONE) throw nullptr;
        auto gps_point1 = dynamic_cast<OGRPoint *>(gps_point);
        double distance = 1000000000;
        for (int32_t j = 0; j < roads.size(); j++) {
            OGRGeometry *road = nullptr;
            err_code = OGRGeometryFactory::createFromWkt(roads[j].c_str(), nullptr, &road);
            if (err_code != OGRERR_NONE) throw nullptr;
            auto road1 = dynamic_cast<OGRLineString *>(road);

            if (gps_point1->Distance(road1) <= distance) {
                distance = gps_point1->Distance(road1);
            }
        }
        result.push_back(distance);
    }
    delete gps_point;
    return result;
}

TEST(MAP_MATCH_TEST, SNAP_TO_ROAD) {
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

    auto compare_result = min_distacne(roads, gps_points);

    arrow::BinaryBuilder builder;
    for (int32_t i = 0; i < roads.size(); i++) {
        OGRGeometry *gps_point = nullptr;
        auto err_code = OGRGeometryFactory::createFromWkt(gps_points[i].c_str(),
                                                          nullptr,
                                                          &gps_point);
        if (err_code != OGRERR_NONE) throw nullptr;
        auto gps_point1 = dynamic_cast<OGRPoint *>(gps_point);
        auto wkb_size = gps_point1->WkbSize();
        auto wkb = static_cast<unsigned char *>(CPLMalloc(wkb_size));
        OGR_G_ExportToWkb(gps_point1, OGRwkbByteOrder::wkbNDR, wkb);
        builder.Append(wkb, wkb_size);
    }
    std::shared_ptr<arrow::Array> gps_points_binary;
    builder.Finish(&gps_points_binary);
    std::vector<std::shared_ptr<arrow::Array>> gps_points_binary_vec;
    gps_points_binary_vec.push_back(gps_points_binary);

    for (int32_t i = 0; i < roads.size(); i++) {
        OGRGeometry *road = nullptr;
        auto err_code = OGRGeometryFactory::createFromWkt(roads[i].c_str(),
                                                          nullptr,
                                                          &road);
        if (err_code != OGRERR_NONE) throw nullptr;
        auto road1 = dynamic_cast<OGRLineString *>(road);
        auto wkb_size = road1->WkbSize();
        auto wkb = static_cast<unsigned char *>(CPLMalloc(wkb_size));
        OGR_G_ExportToWkb(road1, OGRwkbByteOrder::wkbNDR, wkb);
        builder.Append(wkb, wkb_size);
    }

    std::shared_ptr<arrow::Array> roads_binary;
    builder.Finish(&roads_binary);
    std::vector<std::shared_ptr<arrow::Array>> roads_binary_vec;
    roads_binary_vec.push_back(roads_binary);

    auto result = arctern::snap::snap_to_road(roads_binary_vec,
                                              gps_points_binary_vec,
                                              1);
    auto result_1 = std::static_pointer_cast<arrow::BinaryArray>(result[0]);

    for (int32_t i = 0; i < compare_result.size(); i++) {
        OGRGeometry *gps_point = nullptr;
        auto err_code = OGRGeometryFactory::createFromWkt(gps_points[i].c_str(),
                                                          nullptr,
                                                          &gps_point);
        if (err_code != OGRERR_NONE) throw nullptr;
        auto gps_point1 = dynamic_cast<OGRPoint *>(gps_point);

        OGRGeometry *projection_point = nullptr;
        err_code = OGRGeometryFactory::createFromWkb(result_1->GetString(i).c_str(),
                                                     nullptr,
                                                     &projection_point);
        if (err_code != OGRERR_NONE) throw nullptr;
        auto projection_point1 = dynamic_cast<OGRPoint *>(projection_point);
        assert(projection_point1->Distance(gps_point1) == compare_result[i]);
    }
}