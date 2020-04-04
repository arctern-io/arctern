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
#include "gis/dispatch/dispatch.h"
#include "gis/dispatch/wkb_type_scanner.h"
#include "gis/dispatch/wkt_type_scanner.h"
#include "gis/gdal/geometry_cases.h"
#include "gis/test_common/transforms.h"
#include "utils/arrow_alias.h"
#include "utils/check_status.h"

namespace dispatch = arctern::gis::dispatch;
using GroupedWkbTypes = arctern::gis::dispatch::GroupedWkbTypes;
using WkbTypes = arctern::gis::WkbTypes;

template <typename Tuple>
class TypeScan : public ::testing::Test {
 public:
  using XYSpaceCases = std::tuple_element_t<0, Tuple>;
  using TypeScanner = std::tuple_element_t<1, Tuple>;
  using ArrayType = std::tuple_element_t<2, Tuple>;
};

using TypesContainer = ::testing::Types<
    std::tuple<XYSpaceWkbCases, dispatch::WkbTypeScanner, arrow::BinaryArray>,
    std::tuple<XYSpaceWktCases, dispatch::WktTypeScanner, arrow::StringArray> >;
TYPED_TEST_SUITE(TypeScan, TypesContainer);

TYPED_TEST(TypeScan, single_type_scan) {
  typename TestFixture::XYSpaceCases cases;
  auto geo_cases = cases.GetAllCases();
  typename TestFixture::TypeScanner scanner(geo_cases);
  scanner.mutable_types().push_back({WkbTypes::kPoint});
  scanner.mutable_types().push_back({WkbTypes::kLineString});
  scanner.mutable_types().push_back({WkbTypes::kPolygon});
  scanner.mutable_types().push_back({WkbTypes::kMultiPoint});
  scanner.mutable_types().push_back({WkbTypes::kMultiLineString});
  scanner.mutable_types().push_back({WkbTypes::kMultiPolygon});
  auto type_masks = scanner.Scan();
  ASSERT_EQ(type_masks->is_unique_type, false);

  for (auto type : scanner.types()) {
    const auto& mask = type_masks->get_mask(type);
    auto range = cases.GetCaseIndexRange(*type.begin());
    for (int i = 0; i < mask.size(); i++) {
      if (i >= range.first && i < range.second) {
        ASSERT_EQ(mask[i], true);
      } else {
        ASSERT_EQ(mask[i], false);
      }
    }
    auto count = type_masks->get_count(type);
    ASSERT_EQ(count, range.second - range.first);
  }
  {
    GroupedWkbTypes type = {WkbTypes::kUnknown};
    const auto& mask = type_masks->get_mask(type);
    for (int i = 0; i < mask.size(); i++) {
      ASSERT_EQ(mask[i], false);
    }
  }
}

TYPED_TEST(TypeScan, unknown_type) {
  typename TestFixture::XYSpaceCases cases;
  auto geo_cases = cases.GetAllCases();
  typename TestFixture::TypeScanner scanner(geo_cases);
  scanner.mutable_types().push_back({WkbTypes::kLineString});
  scanner.mutable_types().push_back({WkbTypes::kMultiPoint});
  scanner.mutable_types().push_back({WkbTypes::kMultiPolygon});
  auto type_masks = scanner.Scan();
  ASSERT_EQ(type_masks->is_unique_type, false);

  GroupedWkbTypes type = {WkbTypes::kUnknown};
  auto range0 = cases.GetCaseIndexRange(WkbTypes::kPoint);
  auto range1 = cases.GetCaseIndexRange(WkbTypes::kPolygon);
  auto range2 = cases.GetCaseIndexRange(WkbTypes::kMultiLineString);
  const auto& mask = type_masks->get_mask(type);
  for (int i = 0; i < mask.size(); i++) {
    if ((i >= range0.first && i < range0.second) ||
        (i >= range1.first && i < range1.second) ||
        (i >= range2.first && i < range2.second)) {
      ASSERT_EQ(mask[i], true);
    } else {
      ASSERT_EQ(mask[i], false);
    }
  }
}

TYPED_TEST(TypeScan, unique_type) {
  typename TestFixture::XYSpaceCases cases;
  auto geo_cases = cases.GetAllCases();
  {
    typename TestFixture::TypeScanner scanner(geo_cases);
    auto type_masks = scanner.Scan();
    GroupedWkbTypes type = {WkbTypes::kUnknown};
    ASSERT_EQ(type_masks->is_unique_type, true);
    ASSERT_EQ(type_masks->unique_type, type);
  }

  {
    typename TestFixture::TypeScanner scanner(geo_cases);
    scanner.mutable_types().push_back({WkbTypes::kMultiPolygon});
    auto type_masks = scanner.Scan();
    GroupedWkbTypes type = {WkbTypes::kMultiPolygon};
    ASSERT_EQ(type_masks->is_unique_type, false);
  }
  {
    auto geo_cases2 = cases.GetCases({WkbTypes::kLineString});
    typename TestFixture::TypeScanner scanner(geo_cases2);
    scanner.mutable_types().push_back({WkbTypes::kLineString});
    auto type_masks = scanner.Scan();
    GroupedWkbTypes type = {WkbTypes::kLineString};
    ASSERT_EQ(type_masks->is_unique_type, true);
    ASSERT_EQ(type_masks->unique_type, type);
    ASSERT_EQ(type_masks->dict.size(), 0);
  }
}

TYPED_TEST(TypeScan, grouped_type) {
  typename TestFixture::XYSpaceCases cases;
  auto geo_cases = cases.GetAllCases();
  typename TestFixture::TypeScanner scanner(geo_cases);
  GroupedWkbTypes type1({WkbTypes::kPoint, WkbTypes::kLineString});
  GroupedWkbTypes type2({WkbTypes::kPolygon, WkbTypes::kMultiPoint});
  scanner.mutable_types().push_back(type1);
  scanner.mutable_types().push_back(type2);
  auto type_masks = scanner.Scan();
  ASSERT_EQ(type_masks->is_unique_type, false);

  {
    const auto& mask = type_masks->get_mask(type1);
    auto range0 = cases.GetCaseIndexRange(WkbTypes::kPoint);
    auto range1 = cases.GetCaseIndexRange(WkbTypes::kLineString);
    for (int i = 0; i < mask.size(); i++) {
      if ((i >= range0.first && i < range0.second) ||
          (i >= range1.first && i < range1.second)) {
        ASSERT_EQ(mask[i], true);
      } else {
        ASSERT_EQ(mask[i], false);
      }
    }
  }
  {
    const auto& mask = type_masks->get_mask(type2);
    auto range0 = cases.GetCaseIndexRange(WkbTypes::kPolygon);
    auto range1 = cases.GetCaseIndexRange(WkbTypes::kMultiPoint);
    for (int i = 0; i < mask.size(); i++) {
      if ((i >= range0.first && i < range0.second) ||
          (i >= range1.first && i < range1.second)) {
        ASSERT_EQ(mask[i], true);
      } else {
        ASSERT_EQ(mask[i], false);
      }
    }
  }
}

TYPED_TEST(TypeScan, unique_grouped_type) {
  typename TestFixture::XYSpaceCases cases;
  auto geo_cases = cases.GetCases({WkbTypes::kPolygon, WkbTypes::kMultiPoint});

  typename TestFixture::TypeScanner scanner(geo_cases);
  GroupedWkbTypes type({WkbTypes::kPolygon, WkbTypes::kMultiPoint});
  scanner.mutable_types().push_back(type);
  auto type_masks = scanner.Scan();
  ASSERT_EQ(type_masks->is_unique_type, true);
  ASSERT_EQ(type_masks->unique_type, type);
  ASSERT_TRUE(type_masks->dict.empty());
}

TYPED_TEST(TypeScan, merge_and_split) {
  using arctern::gis::dispatch::GenericArrayMerge;
  using arctern::gis::dispatch::GenericArraySplit;
  using std::string;
  using std::vector;

  using ArrayType = typename TestFixture::ArrayType;
  using BuilderType = typename arctern::GetArrowBuilderType<ArrayType>;

  std::vector<std::string> strs{"one", "two",  "",   "$a", "four",
                                "#",   "five", "$b", "$c", ""};
  std::vector<bool> masks;
  BuilderType builder;
  for (auto str : strs) {
    if (str == "#") {
      builder.AppendNull();
    } else {
      builder.Append(str);
    }
    masks.push_back(!str.empty() && str[0] == '$');
  }
  std::shared_ptr<ArrayType> input;
  builder.Finish(&input);
  auto tmps = GenericArraySplit(input, masks);
  vector<string> false_strs = {"one", "two", "", "four", "#", "five", ""};
  vector<string> true_strs = {"$a", "$b", "$c"};
  auto checker = [](std::shared_ptr<arrow::Array> left_raw, vector<string> right) {
    auto left = std::static_pointer_cast<ArrayType>(left_raw);
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
  auto output = GenericArrayMerge<ArrayType>({tmps[0], tmps[1]}, masks);
  checker(output, strs);
}

TYPED_TEST(TypeScan, dispatch) {
  using std::string;
  using std::vector;
  using ArrayType = typename TestFixture::ArrayType;
  vector<string> cases_raw = {
      "MultiLineString Z((0 0 0, 1 1 1), (0 0 0, 1 1 1, 2 2 2))",
      "MultiPoint Empty",
      "LineString(0 0, 0 1)",
      "Point(0 0)",
      "MultiPolygon Empty",
  };

  auto cases = arctern::gis::StrsTo<ArrayType>(cases_raw);

  GroupedWkbTypes type1 = {WkbTypes::kPoint, WkbTypes::kMultiPoint,
                           WkbTypes::kMultiLineString};
  GroupedWkbTypes type2 = {WkbTypes::kPoint, WkbTypes::kLineString,
                           WkbTypes::kMultiLineString};

  dispatch::MaskResult mask_result(cases, type1);
  mask_result.AppendFilter(cases, type2);
  auto checker_gen = [](int n) {
    return [n](std::shared_ptr<ArrayType> wkb) {
      EXPECT_EQ(wkb->length(), n);
      return wkb;
    };
  };

  dispatch::UnaryExecute<ArrayType>(mask_result, checker_gen(4), checker_gen(1), cases);
}
