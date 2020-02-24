#pragma once
#include <string>
#include <vector>
namespace zilliz {
namespace gis {
namespace cuda {
std::vector<char> Wkt2Wkb(const std::string& geo_wkt);

}
}  // namespace gis
}  // namespace zilliz
