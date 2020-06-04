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
#include "gis/gdal/format_conversion.h"
#include "gis/gdal/geometry_cases.h"
#include "utils/check_status.h"

TEST(format_conversion, wkt_wkb) {
  XYSpaceWktCases cases;
  auto origin_wkt = cases.GetAllCases();
  auto wkb = arctern::gis::gdal::WktToWkb(origin_wkt);
  auto wkt =
      std::static_pointer_cast<arrow::StringArray>(arctern::gis::gdal::WkbToWkt(wkb));

  ASSERT_EQ(origin_wkt->length(), wkb->length());
  ASSERT_EQ(origin_wkt->length(), wkt->length());

  for (auto i = 0; i < wkt->length(); i++) {
    ASSERT_EQ(origin_wkt->GetString(i), wkt->GetString(i));
  }
}
