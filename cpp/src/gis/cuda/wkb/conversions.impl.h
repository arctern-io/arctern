#pragma once
#include "gis/cuda/mock/arrow/api.h"
#include "gis/cuda/wkb/conversions.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace internal {
struct WkbArrowContext {
  char* data;
  int* offsets;
  int size;

 public:
  DEVICE_RUNNABLE inline char* get_wkb_ptr(int index) { return data + index; }
  DEVICE_RUNNABLE inline const char* get_wkb_ptr(int index) const { return data + index; }
  DEVICE_RUNNABLE inline int null_counts() const { return 0 * size; }
};

GeometryVector CreateGeometryVectorFromWkbImpl(WkbArrowContext input);

}  // namespace internal

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
