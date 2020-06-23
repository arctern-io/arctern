#pragma once

#include <memory>
#include <string>
#include <vector>

#include "arrow/api.h"

namespace arctern {
namespace geo_indexing {

class GeosIndex {
public:
  GeosIndex();

  void append(const std::vector<std::shared_ptr<arrow::Array>>& geos);

  std::vector<std::shared_ptr<arrow::Array>> near_road(
          const std::vector<std::shared_ptr<arrow::Array>>& gps_points,
          const double distance);
};


}
}