//
// Created by mike on 2/10/20.
//

#include <gtest/gtest.h>
#include "common/gis_definitions.h"
#include "functor/st_distance.h"
#include <cmath>
#include "test_helper.h"

using namespace zilliz;
using namespace zilliz::gis;
using namespace zilliz::gis::cpp;

//TEST(FunctorArea, naive) {
//    ASSERT_TRUE(true);
//    // TODO use gdal to convert better good
//    // POINT(3 1), copy from WKB WKT convertor
//
//    char data[1 + 4 + 16];
//    static_assert(sizeof(data_left) == sizeof(data), "wtf");
//    int num = 5;
//
//    uint8_t byte_order = 0x1;
//    memcpy(data + 0, &byte_order, sizeof(byte_order));
//    uint32_t point_tag = 1;
//    memcpy(data + 1, &point_tag, sizeof(point_tag));
//
//    GeometryVector gvec_left;
//    GeometryVector gvec_right;
//    gvec_left.decodeFromWKB_initialize();
//    gvec_right.decodeFromWKB_initialize();
//
//    for (int i = 0; i < num; ++i) {
//        double x = i;
//        double y = i + 1;
//        static_assert(sizeof(x) == 8, "wtf");
//        memcpy(data + 5, &x, sizeof(x));
//        memcpy(data + 5 + 8, &y, sizeof(y));
//
//        gvec_left.decodeFromWKB_append((char*)data_left);
//        gvec_right.decodeFromWKB_append(data);
//    }
//    gvec_left.decodeFromWKB_finalize();
//    gvec_right.decodeFromWKB_finalize();
//    auto left_ctx = gvec_left.create_gpuctx();
//    vector<double> result(5);
//    ST_area(gvec_left, gvec_right, result.data());
//    for(int i = 0; i < num; ++i) {
//        auto std = sqrt(pow(i - 3, 2)  + pow(i + 1 - 1, 2));
//        ASSERT_DOUBLE_EQ(result[i], std);
//    }
//}
