#pragma once
#include <memory>

#include "gis/cuda/mock/arrow/api.h"
namespace arctern::gis {
using ArrayPtr = std::shared_ptr<arrow::Array>;
using WktArrayPtr = std::shared_ptr<arrow::StringArray>;
using WkbArrayPtr = std::shared_ptr<arrow::BinaryArray>;
using DoubleArrayPtr = std::shared_ptr<arrow::DoubleArray>;
using IntArrayPtr = std::shared_ptr<arrow::>;



}
