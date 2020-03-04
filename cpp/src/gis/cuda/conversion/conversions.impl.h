#pragma once
#include "conversions.h"
#include "gis/cuda/mock/arrow/api.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace internal {
struct WkbArrowContext {
  char* values;
  int* offsets;
  int size;

 public:
  DEVICE_RUNNABLE inline char* get_wkb_ptr(int index) { return values + offsets[index]; }
  DEVICE_RUNNABLE inline const char* get_wkb_ptr(int index) const {
    return values + offsets[index];
  }
  DEVICE_RUNNABLE inline int null_counts() const { return 0 * size; }
};

GeometryVector ArrowWkbToGeometryVectorImpl(const WkbArrowContext& input);

// return size of total data length in bytes
void ExportWkbFillOffsets(const GpuContext& input, WkbArrowContext& output,
                          int* value_length);

void ExportWkbFillValues(const GpuContext& input, WkbArrowContext& output);

}  // namespace internal

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
