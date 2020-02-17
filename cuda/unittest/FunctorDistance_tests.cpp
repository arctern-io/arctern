//
// Created by mike on 2/10/20.
//

#include <gtest/gtest.h>
#include "gis/gis_definitions.h"
#include "functor/st_distance.h"
#include <cmath>
#include "test_helper.h"

using namespace zilliz;
using namespace zilliz::gis;
using namespace zilliz::gis::cuda;

TEST(FunctorDistance, naive) {
    ASSERT_TRUE(true);
    // TODO use gdal to convert better good

    // POINT(3 1), copy from WKB WKT convertor
    //    uint8_t data_left[] = {0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    //                           0x00, 0x00, 0x00, 0x00, 0x08, 0x40, 0x00, 0x00, 0x00,
    //                           0x00, 0x00, 0x00, 0xf0, 0x3f};
    auto vec_left = hexstring_to_binary("01010000000000000000000840000000000000f03f");
    auto data_left = vec_left.data();
    char data[1 + 4 + 16];
    int num = 5;

    uint8_t byte_order = 0x1;
    memcpy(data + 0, &byte_order, sizeof(byte_order));
    uint32_t point_tag = 1;
    memcpy(data + 1, &point_tag, sizeof(point_tag));

    GeometryVector gvec_left;
    GeometryVector gvec_right;
    gvec_left.WkbDecodeInitalize();
    gvec_right.WkbDecodeInitalize();

    for (int i = 0; i < num; ++i) {
        double x = i;
        double y = i + 1;
        static_assert(sizeof(x) == 8, "wtf");
        memcpy(data + 5, &x, sizeof(x));
        memcpy(data + 5 + 8, &y, sizeof(y));

        gvec_left.WkbDecodeAppend(data_left);
        gvec_right.WkbDecodeAppend(data);
    }
    gvec_left.WkbDecodeFinalize();
    gvec_right.WkbDecodeFinalize();
    vector<double> result(5);
    ST_Distance(gvec_left, gvec_right, result.data());
    for (int i = 0; i < num; ++i) {
        auto std = sqrt(pow(i - 3, 2) + pow(i + 1 - 1, 2));
        ASSERT_DOUBLE_EQ(result[i], std);
    }
}
