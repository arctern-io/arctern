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

#include <string>

#include "map_match/snap_road.h"

namespace arctern {
namespace snap {

struct Projection {
    unsigned char *point_str;
    int32_t size;
    double distance;
};

Projection projection_to_edge(const std::string &road_str,
        const std::string &gps_point_str) {
    double min_distance = 1000000;
    auto nearest_point = std::make_shared<OGRPoint>();

    OGRGeometry *road = nullptr;
    auto err_code = OGRGeometryFactory::createFromWkb(road_str.c_str(),
                                               nullptr,
                                               &road);
    if (err_code != OGRERR_NONE) throw nullptr;
    auto road1 = dynamic_cast<OGRLineString *>(road);

    OGRGeometry *gps_point = nullptr;

    err_code = OGRGeometryFactory::createFromWkb(gps_point_str.c_str(),
                                                 nullptr,
                                                 &gps_point);

    if (err_code != OGRERR_NONE) throw nullptr;
    auto gps_point1 = dynamic_cast<OGRPoint *>(gps_point);
    double x = gps_point1->getX();
    double y = gps_point1->getY();

    for (int32_t i = 0; i < (road1->getNumPoints() - 1); i++) {
        double x1 = road1->getX(i);
        double y1 = road1->getY(i);
        double x2 = road1->getX(i + 1);
        double y2 = road1->getY(i + 1);
        double L2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        if (L2 == 0.0) {
            throw nullptr;
        }
        double x1_x = x - x1;
        double y1_y = y - y1;
        double x1_x2 = x2 - x1;
        double y1_y2 = y2 - y1;
        double ratio = (x1_x * x1_x2 + y1_y * y1_y2) / L2;
        ratio = (ratio > 1) ? 1 : ratio;
        ratio = (ratio < 0) ? 0 : ratio;
        double prj_x = x1 + ratio * (x1_x2);
        double prj_y = y1 + ratio * (y1_y2);

        double distance = (x - prj_x) * (x - prj_x) + (y - prj_y) * (y - prj_y);

        if (min_distance >= distance) {
            min_distance = distance;
            nearest_point->setX(prj_x);
            nearest_point->setY(prj_y);
        }
    }

    Projection projection;
    auto wkb_size = nearest_point->WkbSize();
    auto wkb = static_cast<unsigned char *>(CPLMalloc(wkb_size));
    OGR_G_ExportToWkb(nearest_point.get(), OGRwkbByteOrder::wkbNDR, wkb);
    projection.point_str = wkb;
    projection.distance = min_distance;
    projection.size = wkb_size;

    OGRGeometryFactory::destroyGeometry(road1);
    OGRGeometryFactory::destroyGeometry(gps_point1);

    return projection;
}

Projection nearest_projection(const std::vector<std::shared_ptr<arrow::Array>> &roads,
        const std::string &gps_point_str) {
    double min_distance = 10000000;
    Projection result;
    for (int32_t i = 0; i < roads.size(); i++) {
        auto roads_str = std::static_pointer_cast<arrow::BinaryArray>(roads[i]);
        for (int32_t j = 0; j < roads[i]->length(); j++) {
            auto projection_point = projection_to_edge(roads_str->GetString(j),
                                                       gps_point_str);
            if (min_distance >= projection_point.distance) {
                min_distance = projection_point.distance;
                result = projection_point;
            }
        }
    }

    return result;
}


std::vector<std::shared_ptr<arrow::Array>> snap_to_road(
        const std::vector<std::shared_ptr<arrow::Array>> &roads,
        const std::vector<std::shared_ptr<arrow::Array>> &gps_points,
        int32_t num_thread
        ) {
    std::vector<std::vector<Projection>> projections_str;
    std::vector<std::shared_ptr<arrow::Array>> result;

    for (int32_t i = 0; i < gps_points.size(); i++) {
        std::vector<Projection> gps_projection_str(gps_points[i]->length());
        projections_str.push_back(gps_projection_str);
    }
    
    for (int32_t i = 0; i < gps_points.size(); i++) {
        auto gps_points_str = std::static_pointer_cast<arrow::BinaryArray>(gps_points[i]);
        #pragma omp parallel for num_threads(num_thread)
        for (int32_t j = 0; j < gps_points[i]->length(); j++) {
            projections_str[i][j] = nearest_projection(roads,
                                                       gps_points_str->GetString(j));
        }
    }

    arrow::BinaryBuilder builder;
    for (int32_t i = 0; i < gps_points.size(); i++) {
        std::shared_ptr<arrow::BinaryArray> projection_str;
        for (int32_t j = 0; j < gps_points[i]->length(); j++) {
            builder.Append(projections_str[i][j].point_str, projections_str[i][j].size);
        }
        builder.Finish(&projection_str);
        result.emplace_back(projection_str);
    }

    return result;
}


} // namespace snap
} // namespace arctern
