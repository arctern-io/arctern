#pragma once
#include <vector>

#include "utils/arrow_alias.h"
#include "index/index.h"

namespace arctern {
namespace gis {
namespace spatial_join {

using IndexType = arctern::index::IndexType;
using IndexNode = arctern::index::IndexNode;

//std::vector<Int32ArrayPtr> ST_IndexedWithin(const std::vector<WkbArrayPtr>& points,
//                                            const std::vector<WkbArrayPtr>& polygons,
//                                            const std::string index_type = "RTREE");

std::vector<std::shared_ptr<arrow::Array>> ST_IndexedWithin(
        const std::vector<std::shared_ptr<arrow::Array>> &points,
        const std::vector<std::shared_ptr<arrow::Array>> &polygons,
        const std::string index_type);

}  // namespace spatial_join
}  // namespace gis
}  // namespace arctern
