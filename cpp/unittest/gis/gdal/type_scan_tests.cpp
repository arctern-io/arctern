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

#include <arrow/api.h>
#include <arrow/array.h>
#include <gtest/gtest.h>
#include <ogr_geometry.h>
#include <ctime>
#include <iostream>

#include "arrow/gis_api.h"
#include "utils/check_status.h"
#include "gis/gdal/type_scan.cpp"


#define COMMON_TEST_CASES                                                              \
  auto p1 = "POINT (0 1)";                                                             \
                                                                                       \
  auto p2 = "LINESTRING (0 0, 0 1)";                                                   \
  auto p3 = "LINESTRING (0 0, 1 0)";                                                   \
  auto p4 = "LINESTRING (0 0, 1 1)";                                                   \
  auto p5 = "LINESTRING (0 0, 0 1, 1 1)";                                              \
  auto p6 = "LINESTRING (0 0, 1 0, 1 1)";                                              \
  auto p7 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";                                         \
                                                                                       \
  auto p8 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";                                     \
  auto p9 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";                                     \
  auto p10 = "POLYGON ((0 0, 1 0, 1 1, 0 0))";                                         \
  auto p11 = "POLYGON ((0 0, 0 1, 1 1, 0 0))";                                         \
                                                                                       \
  auto p12 = "MULTIPOINT (0 0, 1 0, 1 2)";                                             \
  auto p13 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";                                        \
                                                                                       \
  auto p14 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1) )";                        \
  auto p15 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1) )";                        \
  auto p16 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )"; \
                                                                                       \
  auto p17 = "MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )";                                 \
  auto p18 = "MULTIPOLYGON ( ((0 0, 1 1, 1 0,0 0)) )";                                 \
                                                                                       \
  auto p19 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";     \
  auto p20 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 4 0, 4 1, 0 1, 0 0)), ((0 0, 0 4, 4 4, 4 0, 0 0)) )";     \
  auto p21 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 0 4, 4 4, 4 0, 0 0)) )";     \
  auto p22 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 4 0, 4 4, 0 4, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0)) )";     \
  auto p23 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 1, 4 1, 4 0, 0 0)), ((0 0, 4 0, 4 4, 0 4, 0 0)) )";     \
  auto p24 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 1, 1 1, 1 0, 0 0)), ((0 0, 4 0, 4 4, 0 4, 0 0)) )";     \
                                                                                       \
  auto p25 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 4 0, 4 4, 0 4, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)) )";     \
  auto p26 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 4 0, 4 4, 0 4, 0 0)) )";     \
  auto p27 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 1 1, 1 0, 0 0)) )";     \
  auto p28 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 1, 1 1, 1 0, 0 0)), ((0 0, 0 4, 4 4, 4 0, 0 0)) )";     \
                                                                                       \
  auto p29 = "MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 0)), ((0 0, 1 4, 1 0, 0 0)) )";        \
  auto p30 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 0 0)), ((0 0, 1 -8, 1 1, 0 1, 0 0)) )";  \
  auto p31 = "MULTIPOLYGON ( ((0 0, 1 -8, 1 1, 0 1, 0 0)), ((0 0, 4 4, 0 4, 0 0)) )";  \
  auto p32 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 1 -8, 1 1, 0 1, 0 0)), ((0 0, 4 4, 0 4, 0 0)),((0 0, 0 "  \
      "-2, -3 4, 0 2, 0 0)) )";                                                        \
  auto p33 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 1 -8, 1 1, 0 1, 0 0)), ((0 0, 4 4, 0 4, 0 0)),((0 0, 0 "  \
      "-2, 3 4, 0 2, 0 0)) )";

#define CONSTRUCT_COMMON_TEST_CASES    \
  arrow::StringBuilder builder;        \
  std::shared_ptr<arrow::Array> input; \
  builder.Append(std::string(p1));     \
  builder.Append(std::string(p2));     \
  builder.Append(std::string(p3));     \
  builder.Append(std::string(p4));     \
  builder.Append(std::string(p5));     \
  builder.Append(std::string(p6));     \
  builder.Append(std::string(p7));     \
  builder.Append(std::string(p8));     \
  builder.Append(std::string(p9));     \
  builder.Append(std::string(p10));    \
  builder.Append(std::string(p11));    \
  builder.Append(std::string(p12));    \
  builder.Append(std::string(p13));    \
  builder.Append(std::string(p14));    \
  builder.Append(std::string(p15));    \
  builder.Append(std::string(p16));    \
  builder.Append(std::string(p17));    \
  builder.Append(std::string(p18));    \
  builder.Append(std::string(p19));    \
  builder.Append(std::string(p20));    \
  builder.Append(std::string(p21));    \
  builder.Append(std::string(p22));    \
  builder.Append(std::string(p23));    \
  builder.Append(std::string(p24));    \
  builder.Append(std::string(p25));    \
  builder.Append(std::string(p26));    \
  builder.Append(std::string(p27));    \
  builder.Append(std::string(p28));    \
  builder.Append(std::string(p29));    \
  builder.Append(std::string(p30));    \
  builder.Append(std::string(p31));    \
  builder.Append(std::string(p32));    \
  builder.Append(std::string(p33));    \
  builder.Finish(&input);

using WkbTypes = zilliz::gis::WkbTypes;
using GroupedWkbTypes = zilliz::gis::GroupedWkbTypes;

TEST(type_scan_test, single_type_scan) {
    COMMON_TEST_CASES;
    CONSTRUCT_COMMON_TEST_CASES;
    zilliz::gis::gdal::TypeScannerForWkt scanner(input);    
    scanner.mutable_types().push_back(WkbTypes::kPoint);
    scanner.mutable_types().push_back(WkbTypes::kLineString);
    scanner.mutable_types().push_back(WkbTypes::kPolygon);
    scanner.mutable_types().push_back(WkbTypes::kMultiPoint);
    scanner.mutable_types().push_back(WkbTypes::kMultiLineString);
    scanner.mutable_types().push_back(WkbTypes::kMultiPolygon);
    auto type_masks = scanner.Scan();
    ASSERT_EQ(type_masks->is_unique_type, false);
    ASSERT_EQ(type_masks->is_unique_grouped_types, false);
    {
        auto &mask = type_masks->type_masks[WkbTypes::kPoint]; 
        for (int i = 0; i < mask.size(); i++) {
            if (i == 0) {
                ASSERT_EQ(mask[i], true);
            } else {
                ASSERT_EQ(mask[i], false);
            }
        }
    }
    {
        auto &mask = type_masks->type_masks[WkbTypes::kLineString]; 
        for (int i = 0; i < mask.size(); i++) {
            if (i > 0 && i < 7) {
                ASSERT_EQ(mask[i], true);
            } else {
                ASSERT_EQ(mask[i], false);
            }
        }
    }
    {
        auto &mask = type_masks->type_masks[WkbTypes::kPolygon]; 
        for (int i = 0; i < mask.size(); i++) {
            if (i > 6 && i < 11) {
                ASSERT_EQ(mask[i], true);
            } else {
                ASSERT_EQ(mask[i], false);
            }
        }
    }
    {
        auto &mask = type_masks->type_masks[WkbTypes::kMultiPoint]; 
        for (int i = 0; i < mask.size(); i++) {
            if (i > 10 && i < 13) {
                ASSERT_EQ(mask[i], true);
            } else {
                ASSERT_EQ(mask[i], false);
            }
        }
    }
    {
        auto &mask = type_masks->type_masks[WkbTypes::kMultiLineString]; 
        for (int i = 0; i < mask.size(); i++) {
            if (i > 12 && i < 16) {
                ASSERT_EQ(mask[i], true);
            } else {
                ASSERT_EQ(mask[i], false);
            }
        }
    }
    {
        auto &mask = type_masks->type_masks[WkbTypes::kMultiPolygon]; 
        for (int i = 0; i < mask.size(); i++) {
            if (i > 15 && i < 33) {
                ASSERT_EQ(mask[i], true);
            } else {
                ASSERT_EQ(mask[i], false);
            }
        }
    }
    {
        auto &mask = type_masks->type_masks[WkbTypes::kUnknown]; 
        for (int i = 0; i < mask.size(); i++) {
            ASSERT_EQ(mask[i], false);
        }
    }
}

TEST(type_scan_test, unknown_type) {
    COMMON_TEST_CASES;
    CONSTRUCT_COMMON_TEST_CASES;
    zilliz::gis::gdal::TypeScannerForWkt scanner(input);    
    scanner.mutable_types().push_back(WkbTypes::kLineString);
    scanner.mutable_types().push_back(WkbTypes::kMultiPoint);
    scanner.mutable_types().push_back(WkbTypes::kMultiPolygon);
    auto type_masks = scanner.Scan();
    ASSERT_EQ(type_masks->is_unique_type, false);
    ASSERT_EQ(type_masks->is_unique_grouped_types, false);

    auto &mask = type_masks->type_masks[WkbTypes::kUnknown]; 
    for (int i = 0; i < mask.size(); i++) {
        if ((i == 0) || (i > 6 && i < 11) || (i > 12 && i < 16)) {
            ASSERT_EQ(mask[i], true);
        } else {
            ASSERT_EQ(mask[i], false);
        }
    }
}

TEST(type_scan_test, unique_type) {
    COMMON_TEST_CASES;
    CONSTRUCT_COMMON_TEST_CASES;
    {
        zilliz::gis::gdal::TypeScannerForWkt scanner(input);    
        auto type_masks = scanner.Scan();
        ASSERT_EQ(type_masks->is_unique_type, true);
        ASSERT_EQ(type_masks->unique_type, WkbTypes::kUnknown);
        ASSERT_EQ(type_masks->is_unique_grouped_types, false);
        ASSERT_EQ(type_masks->type_masks.size(), 0);
    }

    {
        zilliz::gis::gdal::TypeScannerForWkt scanner(input);    
        scanner.mutable_types().push_back(WkbTypes::kMultiPolygon);
        auto type_masks = scanner.Scan();
        ASSERT_EQ(type_masks->is_unique_type, false);
        ASSERT_EQ(type_masks->is_unique_grouped_types, false);
    }
    {
      auto p2 = "LINESTRING (0 0, 0 1)";          
      auto p3 = "LINESTRING (0 0, 1 0)";          
      auto p4 = "LINESTRING (0 0, 1 1)";          
      auto p5 = "LINESTRING (0 0, 0 1, 1 1)";     
      auto p6 = "LINESTRING (0 0, 1 0, 1 1)";     
      auto p7 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
      arrow::StringBuilder builder;        
      std::shared_ptr<arrow::Array> input; 
      builder.Append(std::string(p2));     
      builder.Append(std::string(p3));     
      builder.Append(std::string(p4));     
      builder.Append(std::string(p5));     
      builder.Append(std::string(p6));     
      builder.Append(std::string(p7));     
      builder.Finish(&input);
        zilliz::gis::gdal::TypeScannerForWkt scanner(input);    
        scanner.mutable_types().push_back(WkbTypes::kLineString);
        auto type_masks = scanner.Scan();
        ASSERT_EQ(type_masks->is_unique_type, true);
        ASSERT_EQ(type_masks->is_unique_grouped_types, false);
        ASSERT_EQ(type_masks->unique_type, WkbTypes::kLineString);
        ASSERT_EQ(type_masks->type_masks.size(), 0);
    }
}

TEST(type_scan_test, grouped_type) {
    COMMON_TEST_CASES;
    CONSTRUCT_COMMON_TEST_CASES;
    zilliz::gis::gdal::TypeScannerForWkt scanner(input);    
    GroupedWkbTypes type1({WkbTypes::kPoint,WkbTypes::kLineString});
    GroupedWkbTypes type2({WkbTypes::kPolygon,WkbTypes::kMultiPoint});
    scanner.mutable_grouped_types().push_back(type1);
    scanner.mutable_grouped_types().push_back(type2);
    auto type_masks = scanner.Scan();
    ASSERT_EQ(type_masks->is_unique_type, false);
    ASSERT_EQ(type_masks->is_unique_grouped_types, false);

    {
        auto &mask = type_masks->grouped_type_masks[type1]; 
        for (int i = 0; i < mask.size(); i++) {
            if (i < 7) {
                ASSERT_EQ(mask[i], true);
            } else {
                ASSERT_EQ(mask[i], false);
            }
        }
    }
    {
        auto &mask = type_masks->grouped_type_masks[type2]; 
        for (int i = 0; i < mask.size(); i++) {
            if (i > 6 && i < 13) {
                ASSERT_EQ(mask[i], true);
            } else {
                ASSERT_EQ(mask[i], false);
            }
        }
    }
}

TEST(type_scan_test, unique_grouped_type) {
  auto p8 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";   
  auto p9 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";   
  auto p10 = "POLYGON ((0 0, 1 0, 1 1, 0 0))";       
  auto p11 = "POLYGON ((0 0, 0 1, 1 1, 0 0))";       
                                                     
  auto p12 = "MULTIPOINT (0 0, 1 0, 1 2)";           
  auto p13 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";      

  arrow::StringBuilder builder;        
  std::shared_ptr<arrow::Array> input; 
  builder.Append(std::string(p8));     
  builder.Append(std::string(p9));     
  builder.Append(std::string(p10));    
  builder.Append(std::string(p11));    
  builder.Append(std::string(p12));    
  builder.Append(std::string(p13));    
  builder.Finish(&input);

    zilliz::gis::gdal::TypeScannerForWkt scanner(input);    
    GroupedWkbTypes type({WkbTypes::kPolygon,WkbTypes::kMultiPoint});
    scanner.mutable_grouped_types().push_back(type);
    auto type_masks = scanner.Scan();
    ASSERT_EQ(type_masks->is_unique_type, false);
    ASSERT_EQ(type_masks->is_unique_grouped_types, true);
    ASSERT_EQ(type_masks->unique_grouped_types, type);
}

