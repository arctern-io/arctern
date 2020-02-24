#include "wkb_transforms.h"

#include <ogr_api.h>
#include <ogrsf_frmts.h>

#include <cassert>
#include <string>
#include <iostream>
#include <vector>
namespace zilliz {
namespace gis {
namespace cuda {

std::vector<char> Wkt2Wkb(const std::string& geo_wkt) {
  OGRGeometry* geo = nullptr;
  {
    auto err_code = OGRGeometryFactory::createFromWkt(geo_wkt.c_str(), nullptr, &geo);
    assert(err_code == OGRERR_NONE);
  }
  auto sz = geo->WkbSize();
  std::vector<char> result(sz);
  {
    auto err_code = geo->exportToWkb(OGRwkbByteOrder::wkbNDR, (uint8_t*)result.data());
    assert(err_code == OGRERR_NONE);
  }
  OGRGeometryFactory::destroyGeometry(geo);
  return result;
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
