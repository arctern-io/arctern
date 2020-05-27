#pragma once
#include <vector>

#include "utils/arrow_alias.h"

namespace arctern {
namespace gis {
namespace spatial_join {
std::vector<Int32ArrayPtr> ST_IndexedWithin(const std::vector<WkbArrayPtr>& points,
                                            const std::vector<WkbArrayPtr>& polygons);

}  // namespace spatial_join
}  // namespace gis
}  // namespace arctern
