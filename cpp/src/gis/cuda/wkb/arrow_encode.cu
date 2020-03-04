#include <exception>

#include "gis/cuda/wkb/conversions.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
//
}


namespace internal {
// return size of total data length in bytes
int ExportWkbFillOffsets(const GeometryVector& vec, WkbArrowContext& output) {
  //
}


void ExportWkbFillValues(const GeometryVector& vec, WkbArrowContext& output);

}  // namespace internal

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
