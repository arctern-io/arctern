//
// Created by mike on 2/10/20.
//

#include <gtest/gtest.h>
#include "gis/gis_definitions.h"
#include "functor/st_point.h"
#include <cmath>
#include <functor/st_distance.h>
#include "test_helper.h"

using namespace zilliz;
using namespace zilliz::gis;
using namespace zilliz::gis::cuda;

TEST(FunctorPoint, naive) {
    vector<double> xs{1, 2, 3, 4, 5};
    vector<double> ys{0, 1, 2, 3, 4};
    GeometryVector left_points;
    GeometryVector right_points;
    ST_point(xs.data(), ys.data(), xs.size(), left_points);
    for(auto &x: xs) {
        x = -x;
    }
    for(auto &y: ys) {
        y = -y;
    }
    ST_point(xs.data(), ys.data(), xs.size(), right_points);
    vector<double> distance(xs.size());
    ST_distance(left_points, right_points, distance.data());
    for(int i = 0; i < xs.size(); ++i) {
        auto std = (xs[i] * xs[i] + ys[i] * ys[i]) * 4;
        auto res = distance[i] * distance[i];
        ASSERT_DOUBLE_EQ(res, std);
    }
}
