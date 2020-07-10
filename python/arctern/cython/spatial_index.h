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

  std::vector<std::shared_ptr<arrow::Array>> nearest_location_on_road(
          const std::vector<std::shared_ptr<arrow::Array>>& gps_points);

  std::vector<std::shared_ptr<arrow::Array>> nearest_road(
          const std::vector<std::shared_ptr<arrow::Array>>& gps_points);

  std::vector<std::shared_ptr<arrow::Array>> ST_IndexedWithin(
          const std::vector<std::shared_ptr<arrow::Array>>& points);

  std::vector<std::shared_ptr<arrow::Array>> query(
          const std::vector<std::shared_ptr<arrow::Array>>& inputs);
};

}
}
