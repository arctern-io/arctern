//
// Created by mike on 2/10/20.
//

#include <gtest/gtest.h>
#include "common/gis_definitions.h"
#include "functor/st_area.h"
#include <cmath>
#include "test_helper.h"

using namespace zilliz;
using namespace zilliz::gis;
using namespace zilliz::gis::cuda;

TEST(FunctorArea, naive) {
    ASSERT_TRUE(true);
    // TODO use gdal to convert better good
    // https://gis.stackexchange.com/questions/294597/st-area-and-srid-4326
    // POLYGON((38.31 36.85999999,38.31 36.90999999,38.38 36.90999999,38.38 36.85999999,38.31
    // 36.85999999))


    // Polygon((
    auto raw_data = hexstring_to_binary(
        "01030000000100000004000000000000000000084000000000000008400000000000000840000000"
        "00000010400000000000001040000000000000104000000000000010400000000000000840");

    int n = 3;
    GeometryVector gvec;
    gvec.WkbDecodeInitalize();
    for (int i = 0; i < n; ++i) {
        gvec.WkbDecodeAppend(raw_data.data());
    }
    gvec.WkbDecodeFinalize();

    vector<double> result(n);
    ST_Area(gvec, result.data());
    for (int i = 0; i < n; ++i) {
        auto std = 1;
        ASSERT_DOUBLE_EQ(result[i], std);
    }
}
