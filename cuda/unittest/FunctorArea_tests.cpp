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
using namespace zilliz::gis::cpp;

TEST(FunctorArea, naive) {
    ASSERT_TRUE(true);
    // TODO use gdal to convert better good
    // https://gis.stackexchange.com/questions/294597/st-area-and-srid-4326
    // POLYGON((38.31 36.85999999,38.31 36.90999999,38.38 36.90999999,38.38 36.85999999,38.31
    // 36.85999999))
    auto raw_data = hexstring_to_binary(
        "0103000000010000000500000048e17a14ae2743401fcecb7a146e424048e17a14ae274340863432"
        "e17a744240713d0ad7a3304340863432e17a744240713d0ad7a33043401fcecb7a146e424048e17a"
        "14ae2743401fcecb7a146e4240");

    int n = 3;
    GeometryVector gvec;
    gvec.decodeFromWKB_initialize();
    for (int i = 0; i < n; ++i) {
        gvec.decodeFromWKB_append(raw_data.data());
    }
    gvec.decodeFromWKB_finalize();

    auto left_ctx = gvec.create_gpuctx();
    vector<double> result(n);
    ST_area(gvec, result.data());
    for (int i = 0; i < n; ++i) {
        auto std = 34625394.4708328;
        ASSERT_DOUBLE_EQ(result[i], std);
    }
}
