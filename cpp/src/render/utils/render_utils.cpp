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
#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <memory>
#include <string>
#include <vector>

#include "arrow/render_api.h"
#include "render/utils/render_utils.h"
#include "utils/check_status.h"

namespace arctern {
namespace render {

std::vector<OGRGeometry*> GeometryExtraction(
    const std::vector<std::shared_ptr<arrow::Array>>& arrs) {
  int total_size = 0;
  for (const auto& arr : arrs) {
    total_size += arr->length();
  }
  std::vector<OGRGeometry*> geos_res(total_size);

  int index = 0;
  for (const auto& arr : arrs) {
    assert(arr->type_id() == arrow::Type::BINARY);
    auto wkb_geometries = std::static_pointer_cast<arrow::BinaryArray>(arr);
    for (int i = 0; i < wkb_geometries->length(); i++) {
      OGRGeometry* geo = nullptr;
      auto err_code = OGRGeometryFactory::createFromWkb(
          wkb_geometries->GetString(i).c_str(), nullptr, &geo);
      if (err_code || geo == nullptr) {
        geos_res[index] = nullptr;
      } else {
        geos_res[index] = geo;
      }
      index++;
    }
  }

  return geos_res;
}

std::vector<std::string> WkbExtraction(
    const std::vector<std::shared_ptr<arrow::Array>>& arrs) {
  int total_size = 0;
  for (const auto& arr : arrs) {
    total_size += arr->length();
  }
  std::vector<std::string> wkb_res(total_size);

  int index = 0;
  for (const auto& arr : arrs) {
    assert(arr->type_id() == arrow::Type::BINARY);
    auto wkb_geometries = std::static_pointer_cast<arrow::BinaryArray>(arr);
    for (int i = 0; i < wkb_geometries->length(); i++) {
      wkb_res[index] = wkb_geometries->GetString(i);
      index++;
    }
  }

  return wkb_res;
}

std::vector<std::shared_ptr<arrow::Array>> GeometryExport(
    const std::vector<OGRGeometry*>& geos, int arrays_size) {
  int size_per_array = geos.size() / arrays_size;
  arrays_size = geos.size() % arrays_size == 0 ? arrays_size : arrays_size + 1;
  std::vector<std::shared_ptr<arrow::Array>> arrays(arrays_size);

  for (int i = 0; i < arrays_size; i++) {
    arrow::BinaryBuilder builder;

    for (int j = i * size_per_array; j < geos.size() && j < (i + 1) * size_per_array;
         j++) {
      auto sz = geos[j]->WkbSize();
      std::vector<char> str(sz);
      auto err_code = geos[j]->exportToWkb(OGRwkbByteOrder::wkbNDR, (uint8_t*)str.data());
      if (err_code != OGRERR_NONE) {
        std::string err_msg =
            "failed to export to wkt, error code = " + std::to_string(err_code);
        throw std::runtime_error(err_msg);
      }
      CHECK_ARROW(builder.Append(str.data(), str.size()));
      OGRGeometryFactory::destroyGeometry(geos[j]);
    }
    std::shared_ptr<arrow::Array> array;
    CHECK_ARROW(builder.Finish(&array));
    arrays[i] = array;
  }

  return arrays;
}

void pointXY_from_wkt_with_transform(const std::string& wkt, double& x, double& y,
                                     void* poCT) {
  OGRGeometry* res_geo = nullptr;
  CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt.c_str(), nullptr, &res_geo));
  CHECK_GDAL(OGR_G_Transform(res_geo, (OGRCoordinateTransformation*)poCT));
  auto rst_pointer = reinterpret_cast<OGRPoint*>(res_geo);
  x = rst_pointer->getX();
  y = rst_pointer->getY();
  OGRGeometryFactory::destroyGeometry(res_geo);
}

void pointXY_from_wkt(const std::string& wkt, double& x, double& y) {
  OGRGeometry* res_geo = nullptr;
  CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt.c_str(), nullptr, &res_geo));
  auto rst_pointer = reinterpret_cast<OGRPoint*>(res_geo);
  x = rst_pointer->getX();
  y = rst_pointer->getY();
  OGRGeometryFactory::destroyGeometry(res_geo);
}

}  // namespace render
}  // namespace arctern
