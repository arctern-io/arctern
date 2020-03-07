/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License,Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <arrow/api.h>
#include <arrow/array.h>
#include <gtest/gtest.h>
#include <ogr_geometry.h>
#include <ctime>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/gis_api.h"
#include "gis/wkb_types.h"
#include "utils/check_status.h"

using WkbTypes = arctern::gis::WkbTypes;

class XYSpaceWktCases {
 public:
  XYSpaceWktCases();
  std::shared_ptr<arrow::StringArray> GetAllCases();
  std::shared_ptr<arrow::StringArray> GetCases(const std::vector<WkbTypes>& wkb_types);
  int GetCaseCount(const std::vector<WkbTypes>& wkb_types);
  std::pair<int, int> GetCaseIndexRange(WkbTypes wkb_type);

 private:
  std::vector<std::shared_ptr<arrow::StringArray>> cases_;
  std::vector<std::pair<int, int>> index_ranges_;
};

inline XYSpaceWktCases::XYSpaceWktCases()
    : cases_(uint32_t(WkbTypes::kMultiPolygon) + 1),
      index_ranges_(uint32_t(WkbTypes::kMultiPolygon) + 1) {
  int index_begin = 0;
  int index_end = 0;

  {
    arrow::StringBuilder builder;

    builder.Append("POINT (0 1)");

    std::shared_ptr<arrow::StringArray> cases;
    builder.Finish(&cases);
    auto type = uint32_t(WkbTypes::kPoint);
    cases_[type] = cases;
    index_end = index_begin + cases->length();
    index_ranges_[type] = {index_begin, index_end};
    index_begin = index_end;
  }

  {
    arrow::StringBuilder builder;

    builder.Append("LINESTRING (0 0,0 1)");
    builder.Append("LINESTRING (0 0,1 0)");
    builder.Append("LINESTRING (0 0,1 1)");
    builder.Append("LINESTRING (0 0,0 1,1 1)");
    builder.Append("LINESTRING (0 0,1 0,1 1)");
    builder.Append("LINESTRING (0 0,1 0,1 1,0 0)");

    std::shared_ptr<arrow::StringArray> cases;
    builder.Finish(&cases);
    auto type = uint32_t(WkbTypes::kLineString);
    cases_[type] = cases;
    index_end = index_begin + cases->length();
    index_ranges_[type] = {index_begin, index_end};
    index_begin = index_end;
  }

  {
    arrow::StringBuilder builder;

    builder.Append("POLYGON ((0 0,1 0,1 1,0 1,0 0))");
    builder.Append("POLYGON ((0 0,0 1,1 1,1 0,0 0))");
    builder.Append("POLYGON ((0 0,1 0,1 1,0 0))");
    builder.Append("POLYGON ((0 0,0 1,1 1,0 0))");

    std::shared_ptr<arrow::StringArray> cases;
    builder.Finish(&cases);
    auto type = uint32_t(WkbTypes::kPolygon);
    cases_[type] = cases;
    index_end = index_begin + cases->length();
    index_ranges_[type] = {index_begin, index_end};
    index_begin = index_end;
  }

  {
    arrow::StringBuilder builder;

    builder.Append("MULTIPOINT (0 0,1 0,1 2)");
    builder.Append("MULTIPOINT (0 0,1 0,1 2,1 2)");

    std::shared_ptr<arrow::StringArray> cases;
    builder.Finish(&cases);
    auto type = uint32_t(WkbTypes::kMultiPoint);
    cases_[type] = cases;
    index_end = index_begin + cases->length();
    index_ranges_[type] = {index_begin, index_end};
    index_begin = index_end;
  }

  {
    arrow::StringBuilder builder;

    builder.Append("MULTILINESTRING ((0 0,1 2),(0 0,1 0,1 1))");
    builder.Append("MULTILINESTRING ((0 0,1 2),(0 0,1 0,1 1))");
    builder.Append("MULTILINESTRING ((0 0,1 2),(0 0,1 0,1 1),(-1 2,3 4,9 -3,-4 100))");

    std::shared_ptr<arrow::StringArray> cases;
    builder.Finish(&cases);
    auto type = uint32_t(WkbTypes::kMultiLineString);
    cases_[type] = cases;
    index_end = index_begin + cases->length();
    index_ranges_[type] = {index_begin, index_end};
    index_begin = index_end;
  }

  {
    arrow::StringBuilder builder;

    builder.Append("MULTIPOLYGON (((0 0,1 0,1 1,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,1 1,1 0,0 0)))");

    builder.Append("MULTIPOLYGON (((0 0,0 4,4 4,4 0,0 0)),((0 0,4 0,4 1,0 1,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,4 0,4 1,0 1,0 0)),((0 0,0 4,4 4,4 0,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,1 0,1 1,0 1,0 0)),((0 0,0 4,4 4,4 0,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0)),((0 0,0 1,4 1,4 0,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,0 1,4 1,4 0,0 0)),((0 0,4 0,4 4,0 4,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,0 1,1 1,1 0,0 0)),((0 0,4 0,4 4,0 4,0 0)))");

    builder.Append("MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0)),((0 0,1 0,1 1,0 1,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,1 0,1 1,0 1,0 0)),((0 0,4 0,4 4,0 4,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,0 4,4 4,4 0,0 0)),((0 0,0 1,1 1,1 0,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,0 1,1 1,1 0,0 0)),((0 0,0 4,4 4,4 0,0 0)))");

    builder.Append("MULTIPOLYGON (((0 0,1 0,1 1,0 0)),((0 0,1 4,1 0,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,0 4,4 4,0 0)),((0 0,1 -8,1 1,0 1,0 0)))");
    builder.Append("MULTIPOLYGON (((0 0,1 -8,1 1,0 1,0 0)),((0 0,4 4,0 4,0 0)))");
    builder.Append(
        "MULTIPOLYGON (((0 0,1 -8,1 1,0 1,0 0)),((0 0,4 4,0 4,0 0)),((0 0,0 "
        "-2,-3 4,0 2,0 0)))");
    builder.Append(
        "MULTIPOLYGON (((0 0,1 -8,1 1,0 1,0 0)),((0 0,4 4,0 4,0 0)),((0 0,0 "
        "-2,3 4,0 2,0 0)))");

    std::shared_ptr<arrow::StringArray> cases;
    builder.Finish(&cases);
    auto type = uint32_t(WkbTypes::kMultiPolygon);
    cases_[type] = cases;
    index_end = index_begin + cases->length();
    index_ranges_[type] = {index_begin, index_end};
    index_begin = index_end;
  }

  {
    std::vector<uint32_t> types = {
        uint32_t(WkbTypes::kPoint),           uint32_t(WkbTypes::kLineString),
        uint32_t(WkbTypes::kPolygon),         uint32_t(WkbTypes::kMultiPoint),
        uint32_t(WkbTypes::kMultiLineString), uint32_t(WkbTypes::kMultiPolygon)};

    arrow::StringBuilder builder;
    for (auto type : types) {
      auto& cases = *cases_[type];
      for (auto i = 0; i < cases.length(); i++) {
        builder.Append(cases.GetString(i));
      }
    }
    std::shared_ptr<arrow::StringArray> cases;
    builder.Finish(&cases);
    cases_[0] = cases;
  }
}

inline std::shared_ptr<arrow::StringArray> XYSpaceWktCases::GetAllCases() {
  return cases_[0];
}

inline std::shared_ptr<arrow::StringArray> XYSpaceWktCases::GetCases(
    const std::vector<WkbTypes>& wkb_types) {
  if (wkb_types.size() == 1) {
    return cases_[uint32_t(wkb_types.front())];
  } else {
    arrow::StringBuilder builder;
    for (auto type : wkb_types) {
      auto& cases = *cases_[uint32_t(type)];
      for (auto i = 0; i < cases.length(); i++) {
        builder.Append(cases.GetString(i));
      }
    }
    std::shared_ptr<arrow::StringArray> cases;
    builder.Finish(&cases);
    return cases;
  }
}

inline int XYSpaceWktCases::GetCaseCount(const std::vector<WkbTypes>& wkb_types) {
  uint64_t count = 0;
  for (auto type : wkb_types) {
    count += cases_[uint32_t(type)]->length();
  }
  return count;
}

inline std::pair<int, int> XYSpaceWktCases::GetCaseIndexRange(WkbTypes wkb_type) {
  return index_ranges_[uint32_t(wkb_type)];
}
