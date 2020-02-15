//
// Created by mike on 2/10/20.
//

#include <gtest/gtest.h>
#include "gis/gis_definitions.h"
#include "functor/st_point.h"
#include <cmath>
#include "test_helper.h"

using namespace zilliz;
using namespace zilliz::gis;
using namespace zilliz::gis::cuda;

TEST(FunctorPoint, naive) {
    vector<double> xs{1, 2, 3, 4, 5};
    vector<double> ys{0, 1, 2, 3, 4};
    GeometryVector result;
    ST_point(xs.data(), ys.data(), xs.size(), result);
    auto x = 1 + 1;
}
