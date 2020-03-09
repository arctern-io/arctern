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

#include "gis/gdal/format_conversion.h"
#include "utils/check_status.h"

#include <assert.h>
#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <memory>
#include <string>

namespace arctern {
namespace gis {
namespace gdal {

std::shared_ptr<arrow::Array> WkbToWkt(const std::shared_ptr<arrow::Array>& wkb) {
  auto wkb_array = std::static_pointer_cast<arrow::BinaryArray>(wkb);
  auto len = wkb_array->length();
  arrow::StringBuilder builder;
  OGRGeometry* geo = nullptr;
  char* wkt = nullptr;

  for (int i = 0; i < len; ++i) {
    auto str = wkb_array->GetString(i);
    CHECK_GDAL(OGRGeometryFactory::createFromWkb(str.c_str(), nullptr, &geo));
    CHECK_GDAL(OGR_G_ExportToWkt((void*)geo, &wkt));
    builder.Append(wkt);
    OGRGeometryFactory::destroyGeometry(geo);
    CPLFree(wkt);
  }

  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> WktToWkb(const std::shared_ptr<arrow::Array>& wkt) {
  auto wkt_array = std::static_pointer_cast<arrow::StringArray>(wkt);
  auto len = wkt_array->length();
  arrow::BinaryBuilder builder;
  OGRGeometry* geo = nullptr;

  for (int i = 0; i < len; ++i) {
    CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt_array->GetString(i).c_str(), nullptr,
                                                 &geo));
    auto wkb_size = OGR_G_WkbSize(geo);
    if (wkb_size) {
      GByte* wkb = new GByte[wkb_size];
      CHECK_GDAL(OGR_G_ExportToWkb((void*)geo, wkbNDR, wkb));
      builder.Append(wkb, wkb_size);
      delete[] wkb;
    } else {
      builder.AppendNull();
    }
    OGRGeometryFactory::destroyGeometry(geo);
  }

  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
