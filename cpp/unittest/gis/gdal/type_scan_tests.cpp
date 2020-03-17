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
#include <random>

#include "arrow/gis_api.h"
#include "gis/dispatch/wkt_type_scanner.h"
#include "gis/gdal/geometry_cases.h"
#include "utils/check_status.h"

using GroupedWkbTypes = arctern::gis::dispatch::GroupedWkbTypes;
using WkbTypes = arctern::gis::WkbTypes;

TEST(type_scan, single_type_scan) {
  XYSpaceWktCases cases;
  auto geo_cases = cases.GetAllCases();
  arctern::gis::dispatch::TypeScannerForWkt scanner(geo_cases);
  scanner.mutable_types().push_back({WkbTypes::kPoint});
  scanner.mutable_types().push_back({WkbTypes::kLineString});
  scanner.mutable_types().push_back({WkbTypes::kPolygon});
  scanner.mutable_types().push_back({WkbTypes::kMultiPoint});
  scanner.mutable_types().push_back({WkbTypes::kMultiLineString});
  scanner.mutable_types().push_back({WkbTypes::kMultiPolygon});
  auto type_masks = scanner.Scan();
  ASSERT_EQ(type_masks->is_unique_type, false);

  for (auto type : scanner.types()) {
    auto& mask = type_masks->get_masks(type);
    auto uid = type_masks->get_encode_uid(type);
    auto range = cases.GetCaseIndexRange(*type.begin());
    auto encode_uids = type_masks->encode_uids;
    for (int i = 0; i < mask.size(); i++) {
      if (i >= range.first && i < range.second) {
        ASSERT_EQ(mask[i], true);
        ASSERT_EQ(encode_uids[i], uid);
      } else {
        ASSERT_EQ(mask[i], false);
        ASSERT_NE(encode_uids[i], uid);
      }
    }
    auto count = type_masks->get_counts(type);
    ASSERT_EQ(count, range.second - range.first);
  }
  {
    GroupedWkbTypes type = {WkbTypes::kUnknown};
    auto& mask = type_masks->get_masks(type);
    for (int i = 0; i < mask.size(); i++) {
      ASSERT_EQ(mask[i], false);
    }
  }
}

TEST(type_scan, unknown_type) {
  XYSpaceWktCases cases;
  auto geo_cases = cases.GetAllCases();
  arctern::gis::dispatch::TypeScannerForWkt scanner(geo_cases);
  scanner.mutable_types().push_back({WkbTypes::kLineString});
  scanner.mutable_types().push_back({WkbTypes::kMultiPoint});
  scanner.mutable_types().push_back({WkbTypes::kMultiPolygon});
  auto type_masks = scanner.Scan();
  ASSERT_EQ(type_masks->is_unique_type, false);

  GroupedWkbTypes type = {WkbTypes::kUnknown};
  auto range0 = cases.GetCaseIndexRange(WkbTypes::kPoint);
  auto range1 = cases.GetCaseIndexRange(WkbTypes::kPolygon);
  auto range2 = cases.GetCaseIndexRange(WkbTypes::kMultiLineString);
  auto& encode_uids = type_masks->encode_uids;
  auto uid = type_masks->get_encode_uid(type);
  auto& mask = type_masks->get_masks(type);
  for (int i = 0; i < mask.size(); i++) {
    if ((i >= range0.first && i < range0.second) ||
        (i >= range1.first && i < range1.second) ||
        (i >= range2.first && i < range2.second)) {
      ASSERT_EQ(mask[i], true);
      ASSERT_EQ(encode_uids[i], uid);
    } else {
      ASSERT_EQ(mask[i], false);
      ASSERT_NE(encode_uids[i], uid);
    }
  }
}

TEST(type_scan, unique_type) {
  XYSpaceWktCases cases;
  auto geo_cases = cases.GetAllCases();
  {
    arctern::gis::dispatch::TypeScannerForWkt scanner(geo_cases);
    auto type_masks = scanner.Scan();
    GroupedWkbTypes type = {WkbTypes::kUnknown};
    ASSERT_EQ(type_masks->is_unique_type, true);
    ASSERT_EQ(type_masks->unique_type, type);
  }

  {
    arctern::gis::dispatch::TypeScannerForWkt scanner(geo_cases);
    scanner.mutable_types().push_back({WkbTypes::kMultiPolygon});
    auto type_masks = scanner.Scan();
    GroupedWkbTypes type = {WkbTypes::kMultiPolygon};
    ASSERT_EQ(type_masks->is_unique_type, false);
  }
  {
    auto geo_cases2 = cases.GetCases({WkbTypes::kLineString});
    arctern::gis::dispatch::TypeScannerForWkt scanner(geo_cases2);
    scanner.mutable_types().push_back({WkbTypes::kLineString});
    auto type_masks = scanner.Scan();
    GroupedWkbTypes type = {WkbTypes::kLineString};
    ASSERT_EQ(type_masks->is_unique_type, true);
    ASSERT_EQ(type_masks->unique_type, type);
    ASSERT_EQ(type_masks->dict.size(), 0);
  }
}

TEST(type_scan, grouped_type) {
  XYSpaceWktCases cases;
  auto geo_cases = cases.GetAllCases();
  arctern::gis::dispatch::TypeScannerForWkt scanner(geo_cases);
  GroupedWkbTypes type1({WkbTypes::kPoint, WkbTypes::kLineString});
  GroupedWkbTypes type2({WkbTypes::kPolygon, WkbTypes::kMultiPoint});
  scanner.mutable_types().push_back(type1);
  scanner.mutable_types().push_back(type2);
  auto type_masks = scanner.Scan();
  ASSERT_EQ(type_masks->is_unique_type, false);
  auto& encode_uids = type_masks->encode_uids;

  {
    const auto& mask = type_masks->get_masks(type1);
    auto uid = type_masks->get_encode_uid(type1);
    auto range0 = cases.GetCaseIndexRange(WkbTypes::kPoint);
    auto range1 = cases.GetCaseIndexRange(WkbTypes::kLineString);
    for (int i = 0; i < mask.size(); i++) {
      if ((i >= range0.first && i < range0.second) ||
          (i >= range1.first && i < range1.second)) {
        ASSERT_EQ(mask[i], true);
        ASSERT_EQ(encode_uids[i], uid);
      } else {
        ASSERT_EQ(mask[i], false);
        ASSERT_NE(encode_uids[i], uid);
      }
    }
  }
  {
    const auto& mask = type_masks->get_masks(type2);
    auto uid = type_masks->get_encode_uid(type2);
    auto range0 = cases.GetCaseIndexRange(WkbTypes::kPolygon);
    auto range1 = cases.GetCaseIndexRange(WkbTypes::kMultiPoint);
    for (int i = 0; i < mask.size(); i++) {
      if ((i >= range0.first && i < range0.second) ||
          (i >= range1.first && i < range1.second)) {
        ASSERT_EQ(mask[i], true);
        ASSERT_EQ(encode_uids[i], uid);
      } else {
        ASSERT_EQ(mask[i], false);
        ASSERT_NE(encode_uids[i], uid);
      }
    }
  }
}

TEST(type_scan, unique_grouped_type) {
  XYSpaceWktCases cases;
  auto geo_cases = cases.GetCases({WkbTypes::kPolygon, WkbTypes::kMultiPoint});

  arctern::gis::dispatch::TypeScannerForWkt scanner(geo_cases);
  GroupedWkbTypes type({WkbTypes::kPolygon, WkbTypes::kMultiPoint});
  scanner.mutable_types().push_back(type);
  auto type_masks = scanner.Scan();
  ASSERT_EQ(type_masks->is_unique_type, true);
  ASSERT_EQ(type_masks->unique_type, type);
  ASSERT_TRUE(type_masks->encode_uids.empty());
  ASSERT_TRUE(type_masks->dict.empty());
}

TEST(type_scan, merge_and_split) {
  using arctern::gis::dispatch::WktArrayMerge;
  using arctern::gis::dispatch::WktArraySplit;
  using std::string;
  using std::vector;
  std::vector<std::string> strs{"one", "two",  "",   "$a", "four",
                                "#",   "five", "$b", "$c", ""};
  std::vector<bool> masks;
  arrow::StringBuilder builder;
  for (auto str : strs) {
    if (str == "#") {
      builder.AppendNull();
    } else {
      builder.Append(str);
    }
    masks.push_back(!str.empty() && str[0] == '$');
  }
  std::shared_ptr<arrow::StringArray> input;
  builder.Finish(&input);
  auto tmps = WktArraySplit(input, masks);
  vector<string> false_strs = {"one", "two", "", "four", "#", "five", ""};
  vector<string> true_strs = {"$a", "$b", "$c"};
  auto checker = [](std::shared_ptr<arrow::Array> left_raw, vector<string> right) {
    auto left = std::static_pointer_cast<arrow::StringArray>(left_raw);
    ASSERT_EQ(left->length(), right.size());
    for (auto i = 0; i < right.size(); ++i) {
      auto str = right[i];
      if (str == "#") {
        ASSERT_TRUE(left->IsNull(i));
      } else {
        ASSERT_FALSE(left->IsNull(i));
        ASSERT_EQ(left->GetString(i), str) << i;
      }
    }
  };
  checker(tmps[0], false_strs);
  checker(tmps[1], true_strs);
  auto output = WktArrayMerge(tmps, masks);
  checker(output, strs);
}
