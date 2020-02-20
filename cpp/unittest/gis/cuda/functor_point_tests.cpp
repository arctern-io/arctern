//
// Created by mike on 2/10/20.
//

#include <gtest/gtest.h>

#include <cmath>

#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/functor/st_distance.h"
#include "gis/cuda/functor/st_point.h"
#include "test_helper.h"

using namespace zilliz;
using namespace zilliz::gis;
using namespace zilliz::gis::cuda;

TEST(FunctorPoint, naive) {
  vector<double> xs{1, 2, 3, 4, 5};
  vector<double> ys{0, 1, 2, 3, 4};
  GeometryVector left_points;
  GeometryVector right_points;
  ST_Point(xs.data(), ys.data(), (int)xs.size(), left_points);
  for (auto& x : xs) {
    x = -x;
  }
  for (auto& y : ys) {
    y = -y;
  }
  ST_Point(xs.data(), ys.data(), (int)xs.size(), right_points);
  vector<double> distance(xs.size());
  ST_Distance(left_points, right_points, distance.data());
  for (size_t i = 0; i < xs.size(); ++i) {
    auto std = (xs[i] * xs[i] + ys[i] * ys[i]) * 4;
    auto res = distance[i] * distance[i];
    ASSERT_DOUBLE_EQ(res, std);
  }
}
