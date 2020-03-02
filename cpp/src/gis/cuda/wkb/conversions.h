#pragma once
#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/mock/arrow/api.h"
namespace zilliz {
namespace gis {
namespace cuda {

struct ArrowContext {
  char* data;
  uint32_t* offsets;
  int size;

 public:
  DEVICE_RUNNABLE inline char* get_wkb_ptr(int index) { return data + index; }
  DEVICE_RUNNABLE inline const char* get_wkb_ptr(int index) const { return data + index; }
  DEVICE_RUNNABLE int null_counts() { return 0 * size; }
};

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
