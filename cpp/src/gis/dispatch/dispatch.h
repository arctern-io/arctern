#pragma once
#include <set>

#include "utils/arrow_alias.h"

namespace arctern {
namespace gis {
namespace dispatch {

// split into [false_array, true_array]
std::array<std::shared_ptr<arrow::Array>, 2> WktArraySplit(
    const std::shared_ptr<arrow::Array>& geometries, const std::vector<bool>& mask);

// merge [false_array, true_array]
std::shared_ptr<arrow::Array> WktArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask);

// merge [false_array, true_array]
std::shared_ptr<arrow::Array> DoubleArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask);

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern

#include "gis/dispatch/dispatch.impl.h"
