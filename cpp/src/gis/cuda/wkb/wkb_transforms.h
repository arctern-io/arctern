#pragma once
#include <vector>
namespace zilliz {
namespace gis {
namespace cuda {
std::vector<char> Wkt2Wkb(const char* geo_wkt);
}
}  // namespace gis
}  // namespace zilliz
