#pragma once
#include "gis/cuda/mock/arrow/api.h"
#include "gis/cuda/wkb/conversions.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace internal {
struct WkbArrowContext {
  char* values;
  int* offsets;
  int size;

 public:
  DEVICE_RUNNABLE inline char* get_wkb_ptr(int index) { return values + index; }
  DEVICE_RUNNABLE inline const char* get_wkb_ptr(int index) const { return values + index; }
  DEVICE_RUNNABLE inline int null_counts() const { return 0 * size; }
};

GeometryVector CreateGeometryVectorFromWkbImpl(const WkbArrowContext& input);

// return size of total data length in bytes
int ExportWkbFillOffsets(const GeometryVector& vec, WkbArrowContext& output);

void ExportWkbFillValues(const GeometryVector& vec, WkbArrowContext& output);

}  // namespace internal

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
